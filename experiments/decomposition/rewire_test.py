"""Does the graph's contribution come from its structure, or from its sparsity?

The observational evidence is exhausted: two collections, and the number of nodes
collinear with every candidate index. What remains is an intervention. At fixed
collection, fixed node count and fixed node draw, the k-nearest-neighbour graph
is replaced by a degree-preserving random rewiring. The rewiring destroys the
correspondence between the graph and the signal and leaves N, the degree
sequence, the parameter count and the training budget untouched.

Two outcomes, both informative. If the gain survives the rewiring, message
passing acts as a regulariser and not as a carrier of structure, and no metric on
the data can ever predict it. If the gain collapses, the permutation test on the
residual measures something real and the decision rule stops resting on a count
of nodes.

The rewired residual smoothness is recorded alongside, which is the manipulation
check: it should sit at zero by construction.
"""

import json
import os
import sys
import time

import networkx as nx
import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from _paths import ROOT, DATA, CACHE  # noqa: E402,F401  (puts every experiment folder on sys.path)
import graph_utility_index as G  # noqa: E402
import ltsf_nodes as E  # noqa: E402

OUT = G.OUT
HIDDEN = E.HIDDEN


def rewire(ei, n, seed):
    """Degree-preserving double edge swaps on the undirected simple graph."""
    a = ei.numpy()
    g = nx.Graph()
    g.add_nodes_from(range(n))
    g.add_edges_from({(int(i), int(j)) for i, j in zip(a[0], a[1]) if i != j})
    m = g.number_of_edges()
    if m >= 2:
        try:
            nx.double_edge_swap(g, nswap=10 * m, max_tries=200 * m, seed=seed)
        except nx.NetworkXAlgorithmError:
            pass                                   # too constrained to swap; kept as is
    src, dst = [], []
    for i, j in g.edges():
        src += [i, j]; dst += [j, i]
    if not src:
        src, dst = list(range(n)), list(range(n))
    return torch.tensor([src, dst], dtype=torch.long)


def one(coll, window, n_nodes, conv, seed, arm):
    build = G.cell_electricity if coll == "elec" else G.cell_weave
    x, y, n_tr, n_ev, ei, vtrain = build(n_nodes, window, seed)
    n = x.shape[1]
    if arm == "rewired":
        ei = rewire(ei, n, 5000 + seed)
    h_out = y.shape[2] if y.ndim == 3 else 1
    torch.manual_seed(seed)
    m = E.Graph(x.shape[2], HIDDEN, conv, ei, None, out=h_out)
    pred, val = E.fit(m, x, y, n_tr, n_ev, seed)
    tgt = y[n_ev:]
    err = ((pred - tgt) ** 2).mean((0, 2)).sqrt() if tgt.ndim == 3 else \
          ((pred - tgt) ** 2).mean(0).sqrt()
    r, _ = G.ridge_residual(x, y, n_tr, n_ev)
    _, z, _, _ = G.i_res(r, G.norm_laplacian(ei, n), np.random.default_rng(7000 + seed))
    return {"nrmse": float(err.mean()), "val": val, "z_res": z, "n": n}


def best_conv():
    """Operator chosen per cell on the true graph, seeds averaged, not per seed."""
    acc = {}
    for coll, path in (("elec", "electricity_nodes.json"), ("weave", "weave_nodes.json")):
        d = json.load(open(os.path.join(OUT, path)))
        for k, v in d.items():
            nn_, w, mo, s = k.split("|")
            if mo.startswith("gnn_"):
                acc.setdefault((coll, int(w), int(nn_)), {}).setdefault(mo, []).append(v["nrmse"])
    return {k: min(m, key=lambda c: np.mean(m[c])).split("_")[1] for k, m in acc.items()}


if __name__ == "__main__":
    convs = best_conv()
    idx = json.load(open(os.path.join(OUT, "graph_utility_index.json")))
    cells = sorted({tuple(k.split("|")[:3]) for k in idx})
    # One seed across every cell first, then the next: a full picture arrives after
    # the first pass instead of after the last. Inside a pass the cheap cells run
    # first, cost being carried mostly by the node count and the training window.
    cells = sorted(cells, key=lambda t: (int(t[2]) * int(t[1])))
    keys = []
    for s in (0, 1, 2):
        for coll, w, n in cells:
            c = convs.get((coll, int(w), int(n)))
            if c is None:
                continue
            for arm in ("rewired", "true"):     # true re-run under the identical harness
                keys.append(f"{coll}|{w}|{n}|{c}|{s}|{arm}")

    path = os.path.join(OUT, "rewire_test.json")
    res = json.load(open(path)) if os.path.exists(path) else {}
    todo = [k for k in keys if k not in res]
    print(f"{len(res)} en cache, {len(todo)} a calculer", flush=True)
    for k in todo:
        coll, w, n, c, s, arm = k.split("|")
        t0 = time.time()
        try:
            res[k] = one(coll, int(w), int(n), c, int(s), arm)
            json.dump(res, open(path, "w"))
            print(f"  [{k:34}] nRMSE {res[k]['nrmse']:.4f}  z {res[k]['z_res']:+7.2f}"
                  f"  ({time.time()-t0:.0f}s)", flush=True)
        except Exception as e:
            print(f"  [{k:34}] ECHEC {type(e).__name__}: {e}", flush=True)
    print("=== termine ===", flush=True)
