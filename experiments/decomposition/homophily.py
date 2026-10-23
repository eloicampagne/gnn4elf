"""Homophily for a regression panel, and what the node count is a proxy for.

Edge homophily counts the fraction of edges joining two nodes of the same class,
which needs a class. Its continuous analogue is Moran's I, the spatial
autocorrelation of a signal on a graph, whose null expectation is -1/(N-1) and
which is one minus the normalised Dirichlet energy up to the centring. Two
readings again: on the target, where the answer is known in advance to be near
one and useless, and on what a per-node ridge leaves over, which is the only
version that says whether a neighbour supplies something.

Second question in the same pass. The sweep varies the node count by drawing a
subset, so what grows with N is not only the number of series: drawing ten nodes
out of three hundred and twenty leaves five "nearest" neighbours that are not
near at all, while drawing all of them makes the k-nearest-neighbour graph
genuinely local. Mean edge correlation against a random-pair baseline separates
the two readings of the node-count effect.
"""

import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from _paths import ROOT, DATA, CACHE  # noqa: E402,F401  (puts every experiment folder on sys.path)
import graph_utility_index as G  # noqa: E402

OUT = G.OUT


def moran(sig, A):
    """Moran's I of a [T, N] or [T, N, H] signal, pooled over time and horizon."""
    s = sig.reshape(sig.shape[0], sig.shape[1], -1) if sig.ndim == 3 else sig[:, :, None]
    s = s - s.mean(axis=1, keepdims=True)                 # centred across nodes
    w = A.sum()
    if w == 0:
        return float("nan")
    num = np.einsum("tnh,nm,tmh->", s, A, s)
    den = (s ** 2).sum()
    return float(sig.shape[1] * num / (w * max(den, 1e-12)))


def edge_corr(v, A, rng, n_rand=2000):
    """Mean correlation over retained edges, and over random pairs for reference."""
    c = np.nan_to_num(np.corrcoef(v.T), nan=0.0)
    iu = np.triu(A, 1) > 0
    e = c[iu]
    n = c.shape[0]
    if n < 2:
        return float("nan"), float("nan")
    i = rng.integers(0, n, n_rand); j = rng.integers(0, n, n_rand)
    keep = i != j
    return float(e.mean()) if e.size else float("nan"), float(c[i[keep], j[keep]].mean())


def dense(ei, n):
    A = np.zeros((n, n))
    a = ei.numpy()
    A[a[0], a[1]] = 1.0
    A = np.maximum(A, A.T)
    np.fill_diagonal(A, 0.0)
    return A


def one(coll, window, n_nodes, seed):
    build = G.cell_electricity if coll == "elec" else G.cell_weave
    x, y, n_tr, n_ev, ei, vtrain = build(n_nodes, window, seed)
    n = x.shape[1]
    A = dense(ei, n)
    r, _ = G.ridge_residual(x, y, n_tr, n_ev)
    yv = y.numpy() if y.ndim == 2 else y.numpy()
    rng = np.random.default_rng(9000 + seed)
    ec, rc = edge_corr(vtrain, A, rng)
    # Residual correlation over the same edges, the conditional version of homophily.
    rr = r.reshape(r.shape[0], r.shape[1], -1).permute(1, 0, 2).reshape(n, -1).numpy()
    rec, rrc = edge_corr(rr.T, A, rng)
    return {"moran_y": moran(yv[n_ev:], A), "moran_res": moran(r.numpy(), A),
            "edge_corr": ec, "rand_corr": rc,
            "edge_corr_res": rec, "rand_corr_res": rrc,
            "null": -1.0 / (n - 1), "n": n}


if __name__ == "__main__":
    idx = json.load(open(os.path.join(OUT, "graph_utility_index.json")))
    keys = sorted(idx)
    path = os.path.join(OUT, "homophily.json")
    res = json.load(open(path)) if os.path.exists(path) else {}
    todo = [k for k in keys if k not in res]
    print(f"{len(res)} en cache, {len(todo)} a mesurer", flush=True)
    for k in todo:
        coll, w, n, s = k.split("|")
        try:
            res[k] = one(coll, int(w), int(n), int(s))
            json.dump(res, open(path, "w"))
            v = res[k]
            print(f"  [{k:20}] Moran y {v['moran_y']:+.3f}  res {v['moran_res']:+.3f}  "
                  f"corr arete {v['edge_corr']:+.3f} (hasard {v['rand_corr']:+.3f})  "
                  f"res {v['edge_corr_res']:+.3f}", flush=True)
        except Exception as e:
            print(f"  [{k}] ECHEC {type(e).__name__}: {e}", flush=True)
    print("=== termine ===", flush=True)
