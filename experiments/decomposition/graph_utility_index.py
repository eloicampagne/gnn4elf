"""Can a metric on the data predict what the edges are worth?

The node-count sweep separates two effects that a per-node reference confounds:
sharing weights across nodes, and passing messages along edges. The first swings
with the collection and the depth of the record; the second tracks the number of
nodes and little else. A predictor is therefore useful only if it predicts the
second, so the target here is the edge gain, shared minus graph, and never the
total gain.

Three candidate indices, each with a null model, all computed without training a
single GNN.

I_res  A polynomial graph convolution is a low-pass filter on the Laplacian, so
       it can only remove what is smooth along the edges in the error a local
       model leaves. Ridge per node gives that residual in closed form; the
       normalised Dirichlet energy of the residual, referred to the same energy
       under a random permutation of the node labels, says whether the residual
       is smoother on the true graph than on a shuffled one.

mst_w  Mean MST edge weight on the correlation distance, referred to the same
       quantity on series shifted circularly by a random lag. The shift destroys
       the dependence between series and preserves every autocorrelation, so a
       ratio well below one means the cloud of series has a geometry.

mst_cv Coefficient of variation of the MST edge weights: communities joined by a
       few long bridges give a high value, a homogeneous cloud a flat one.
"""

import json
import os
import sys

import numpy as np
import torch
from scipy.sparse.csgraph import minimum_spanning_tree

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from _paths import ROOT, DATA, CACHE  # noqa: E402,F401  (puts every experiment folder on sys.path)
import ltsf_nodes as E  # noqa: E402
import weave_nodes as W  # noqa: E402

OUT = os.path.join(ROOT, "results_ltsf_nodes")
N_PERM = 64
RIDGE = 1e-2
_PANEL = {}


# ------------------------------------------------------------------ ridge residual
def _solve(xt, yt, scale, mult):
    N, F = xt.shape[1], xt.shape[2]
    g = torch.einsum("tnf,tng->nfg", xt, xt)
    b = torch.einsum("tnf,tnh->nfh", xt, yt)
    lam = mult * scale
    eye = torch.eye(F).expand(N, F, F)
    return torch.linalg.solve(g + lam[:, None, None] * eye, b)


def ridge_residual(x, y, n_tr, n_ev):
    """Per-node ridge fitted on the training window, residual on the evaluation window.

    x is [T, N, F] and y is [T, N] or [T, N, H]. A bias column is appended. The
    fit is per node, which is the reference the sweep uses, so the residual is
    what node-specific weights on own features alone leave over.

    The penalty is picked on the last fifth of the training window, which matters
    where the record is short: forty-eight days carrying fifty-seven variables
    send a fixed small penalty into a negative R2, and the residual is then not
    the error of any sensible local model.
    """
    if y.ndim == 2:
        y = y[..., None]
    T, N, F = x.shape
    xb = torch.cat([x, torch.ones(T, N, 1)], dim=-1)          # [T, N, F+1]
    n_in = max(4, n_tr // 5)
    xa, ya = xb[:n_tr - n_in], y[:n_tr - n_in]
    xv, yv = xb[n_tr - n_in:n_tr], y[n_tr - n_in:n_tr]
    scale = torch.einsum("tnf,tnf->nf", xa, xa).mean(1).clamp(min=1e-8)
    best, best_m = None, None
    for m in (1e-4, 1e-3, 1e-2, 1e-1, 1.0, 10.0):
        w = _solve(xa, ya, scale, m)
        e = float(((yv - torch.einsum("tnf,nfh->tnh", xv, w)) ** 2).mean())
        if np.isfinite(e) and (best is None or e < best):
            best, best_m = e, m
    scale_full = torch.einsum("tnf,tnf->nf", xb[:n_tr], xb[:n_tr]).mean(1).clamp(min=1e-8)
    # The refit uses the whole window, which can be worse conditioned than the
    # inner fit that chose the penalty; a non-finite solve is walked up the grid.
    grid = [m for m in (1e-4, 1e-3, 1e-2, 1e-1, 1.0, 10.0) if m >= best_m]
    for m in grid:
        w = _solve(xb[:n_tr], y[:n_tr], scale_full, m)
        r = y[n_ev:] - torch.einsum("tnf,nfh->tnh", xb[n_ev:], w)
        if torch.isfinite(r).all():
            return r, m
    return r, grid[-1]


# ------------------------------------------------------------------ graph indices
def norm_laplacian(edge_index, n):
    A = np.zeros((n, n))
    ei = edge_index.numpy()
    A[ei[0], ei[1]] = 1.0
    A = np.maximum(A, A.T)
    np.fill_diagonal(A, 0.0)
    d = A.sum(1)
    dm = np.where(d > 0, d ** -0.5, 0.0)
    return np.eye(n) - (dm[:, None] * A * dm[None, :])


def gram(r):
    """[N, N] second-moment matrix of a [T, N, H] signal, all time steps pooled."""
    R = r.permute(1, 0, 2).reshape(r.shape[1], -1).numpy().astype(np.float64)
    return R @ R.T


def i_res(r, L, rng):
    """Sum_t r_t' L r_t is tr(L G), so a permutation of node labels costs N^2.

    Two readings of the same null. The ratio 1 - s / E[s] is an effect size and
    is not comparable across node counts, since the spread of the null narrows
    as N grows. The standardised score (E[s] - s) / sd[s] is the permutation
    test itself and does not carry that dependence.
    """
    G = gram(r)
    den = max(float(np.trace(G)), 1e-12)
    s = float((L * G.T).sum()) / den
    n = L.shape[0]
    null = np.array([float((L[np.ix_(p, p)] * G.T).sum()) / den
                     for p in (rng.permutation(n) for _ in range(N_PERM))])
    mu, sd = float(null.mean()), float(null.std(ddof=1))
    return 1.0 - s / max(mu, 1e-12), (mu - s) / max(sd, 1e-12), s, mu


def corr_distance(v):
    """v is [T, N] over the training window."""
    c = np.corrcoef(v.T)
    c = np.nan_to_num(c, nan=0.0)
    return np.sqrt(np.clip(2.0 * (1.0 - c), 0.0, None))


def mst_stats(v, rng):
    """Mean MST edge weight referred to a circular-shift null, and its spread."""
    d = corr_distance(v)
    m = minimum_spanning_tree(d).toarray()
    w = m[m > 0]
    if len(w) == 0:
        return float("nan"), float("nan")
    null = []
    for _ in range(8):
        sh = np.stack([np.roll(v[:, j], int(rng.integers(len(v)))) for j in range(v.shape[1])], 1)
        mn = minimum_spanning_tree(corr_distance(sh)).toarray()
        wn = mn[mn > 0]
        if len(wn):
            null.append(wn.mean())
    ratio = w.mean() / np.mean(null) if null else float("nan")
    return float(ratio), float(w.std() / max(w.mean(), 1e-12))


# ------------------------------------------------------------------- per cell work
def cell_electricity(n_nodes, train_hours, seed):
    key = train_hours
    if key not in _PANEL:
        _PANEL[key] = E.load_panel(train_hours)
    x, y, n_tr, n_va = _PANEL[key]
    rng = np.random.default_rng(1000 + seed)
    keep = sorted(rng.choice(x.shape[1], size=min(n_nodes, x.shape[1]), replace=False).tolist())
    x, y = x[:, keep], y[:, keep]
    x, y, _, _ = E.standardise(x, y, n_tr)
    ei, _ = E.knn_graph(y[:n_tr])
    return x, y, n_tr, n_va, ei, y[:n_tr].numpy()


def cell_weave(n_nodes, train_days, seed):
    z = W.panel()
    rng = np.random.default_rng(1000 + seed)
    keep = sorted(rng.choice(z["Xtr"].shape[1], size=min(n_nodes, z["Xtr"].shape[1]),
                             replace=False).tolist())
    xtr = torch.tensor(z["Xtr"][-train_days:, keep, :])
    ytr = torch.tensor(z["Ytr"][-train_days:, keep, :])
    xte = torch.tensor(z["Xte"][:, keep, :])
    yte = torch.tensor(z["Yte"][:, keep, :])
    n_va = max(8, len(xtr) // 5)
    x = torch.cat([xtr, xte]); y = torch.cat([ytr, yte])
    n_tr = len(xtr) - n_va
    n_ev = len(xtr)
    mx, sx = x[:n_tr].mean(0), x[:n_tr].std(0)
    my, sy = y[:n_tr].mean(0), y[:n_tr].std(0)
    dx, dy = sx < 1e-8, sy < 1e-8
    x = (x - mx) / torch.where(dx, torch.ones_like(sx), sx)
    y = (y - my) / torch.where(dy, torch.ones_like(sy), sy)
    x[:, dx] = 0.0
    ei = W.edges_from(z["A"], keep)
    return x, y, n_tr, n_ev, ei, y[:n_tr].mean(-1).numpy()


def indices(coll, n_nodes, window, seed):
    build = cell_electricity if coll == "elec" else cell_weave
    x, y, n_tr, n_ev, ei, vtrain = build(n_nodes, window, seed)
    n = x.shape[1]
    L = norm_laplacian(ei, n)
    r, lam = ridge_residual(x, y, n_tr, n_ev)
    rng = np.random.default_rng(7000 + seed)
    ires, z_res, s_true, s_null = i_res(r, L, rng)
    mw, mcv = mst_stats(vtrain, rng)
    # How much of the target the local model already reconstructs, on the same window.
    ytail = (y[n_ev:] if y.ndim == 3 else y[n_ev:, :, None])
    r2 = 1.0 - float((r ** 2).sum() / max(float((ytail ** 2).sum()), 1e-12))
    return {"i_res": ires, "z_res": z_res, "s_true": s_true, "s_null": s_null,
            "mst_w": mw, "mst_cv": mcv, "local_r2": r2, "lam": lam, "n": n}


# --------------------------------------------------------------------------- driver
def measured_gains():
    """Edge gain, shared minus best graph, per collection, window, node count and seed."""
    out = {}
    for coll, path in (("elec", "electricity_nodes.json"), ("weave", "weave_nodes.json")):
        d = json.load(open(os.path.join(OUT, path)))
        for k, v in d.items():
            n, w, mo, s = k.split("|")
            out.setdefault((coll, int(w), int(n), int(s)), {})[mo] = v["nrmse"]
    gains = {}
    for k, m in out.items():
        g = [m[c] for c in ("gnn_SAGEConv", "gnn_ChebConv") if c in m]
        if "shared" in m and "per_node" in m and g:
            gains[k] = {"edge": m["shared"] - min(g),
                        "share": m["per_node"] - m["shared"],
                        "total": m["per_node"] - min(g)}
    return gains


if __name__ == "__main__":
    gains = measured_gains()
    path = os.path.join(OUT, "graph_utility_index.json")
    res = json.load(open(path)) if os.path.exists(path) else {}
    todo = [k for k in sorted(gains) if "|".join(map(str, k)) not in res]
    print(f"{len(res)} en cache, {len(todo)} cellules a mesurer", flush=True)
    for k in todo:
        coll, w, n, s = k
        try:
            v = indices(coll, n, w, s)
            v.update(gains[k])
            res["|".join(map(str, k))] = v
            json.dump(res, open(path, "w"))
            print(f"  [{coll:5} w={w:4} N={n:3} s={s}] I_res {v["i_res"]:+.4f}  z {v["z_res"]:+6.2f}  "
                  f"mst_w {v['mst_w']:.3f}  mst_cv {v['mst_cv']:.3f}  "
                  f"R2loc {v['local_r2']:+.3f}  gain_aretes {v['edge']:+.4f}", flush=True)
        except Exception as e:
            print(f"  [{coll} {w} {n} {s}] ECHEC {type(e).__name__}: {e}", flush=True)
    print("=== termine ===", flush=True)
