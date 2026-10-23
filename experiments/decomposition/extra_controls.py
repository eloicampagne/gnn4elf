"""Two controls asked by a referee, on the 30-hour WEAVE-UK cell of the decomposition.

(A) Rewiring draws. eval_metrics.py rewires the simple graph once per training
seed (draw seed 5000 + seed), so the draw and the initialisation are confounded
and the edge-identity result rests on three graphs. Here ten independent draws
(draw seeds 9000 + d) are each crossed with the three training seeds, for the
GNN and for the per-node MLP reading the rewired neighbour mean.

(B) Graph estimated inside the window. The decomposition reuses the adjacency
cached from the full training record. Here the k-nearest-neighbour correlation
graph is re-estimated from the 30-hour training window alone (first horizon of
each origin, i.e. the half-hourly series), with mutual pairs either kept twice,
as in the main table, or stored once.

Same cell, splits, inputs, convolution (ChebConv), width and training routine as
eval_metrics.py; nothing else changes.
"""

import json
import os
import sys
import time

import numpy as np
import torch
from torch_geometric.utils import coalesce

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from _paths import ROOT, DATA, CACHE  # noqa: E402,F401  (puts every experiment folder on sys.path)
import ltsf_nodes as E  # noqa: E402
import eval_metrics as V  # noqa: E402
import rewire_test as R  # noqa: E402

OUT = os.path.join(E.OUT, "extra_controls.json")
COLL, WIN, CONV = "weave", 60, "ChebConv"
HIDDEN = E.HIDDEN


def neighbour_input(xz, ei, n):
    src, dst = ei[0], ei[1]
    nb = torch.zeros_like(xz).index_add_(1, dst, xz[:, src])
    deg = torch.zeros(n, dtype=xz.dtype).index_add_(0, dst, torch.ones(dst.numel(), dtype=xz.dtype))
    return torch.cat([xz, nb / deg.clamp(min=1.0).view(1, n, 1)], dim=-1)


def fit_score(model_kind, ei, seed):
    xz, yz, my, sy, n_tr, n_ev, _ = V.cell(COLL, WIN, seed)
    n, f, h_out = xz.shape[1], xz.shape[2], yz.shape[2]
    torch.manual_seed(seed)
    if model_kind == "gnn":
        m = E.Graph(f, HIDDEN, CONV, ei, None, out=h_out)
    else:                                            # per-node MLP with neighbour mean
        xz = neighbour_input(xz, ei, n)
        m = E.PerNode(n, 2 * f, HIDDEN, out=h_out)
    pred, val = E.fit(m, xz, yz, n_tr, n_ev, seed)
    d = (pred - yz[n_ev:]) ** 2
    return {"err": d.mean((0, 2)).sqrt().tolist(), "val": val}


def window_graph():
    xz, yz, my, sy, n_tr, n_ev, _ = V.cell(COLL, WIN, 0)
    ei, _ = E.knn_graph(yz[:n_tr, :, 0])            # duplicated mutual pairs, as edges_from does
    return ei, coalesce(ei, num_nodes=xz.shape[1])


def jobs():
    xz, *_ , ei_cached = V.cell(COLL, WIN, 0)
    n = xz.shape[1]
    simple = coalesce(ei_cached, num_nodes=n)
    out = []
    for d in range(10):                              # (A)
        g = R.rewire(simple, n, 9000 + d)
        for s in (0, 1, 2):
            out.append((f"A|rewired|{d}|{s}", "gnn", g, s))
            out.append((f"A|pnn_rewired|{d}|{s}", "pnn", g, s))
    win_dup, win_simple = window_graph()             # (B)
    for s in (0, 1, 2):
        out.append((f"B|gnn_win_dup|{s}", "gnn", win_dup, s))
        out.append((f"B|gnn_win_simple|{s}", "gnn", win_simple, s))
        out.append((f"B|pnn_win_dup|{s}", "pnn", win_dup, s))
    return out


if __name__ == "__main__":
    res = json.load(open(OUT)) if os.path.exists(OUT) else {}
    todo = [j for j in jobs() if j[0] not in res]
    # cheap per-node runs first, then (B), then the rewiring GNNs
    todo.sort(key=lambda j: (j[1] == "gnn", not j[0].startswith("B")))
    print(f"{len(res)} en cache, {len(todo)} a calculer", flush=True)
    for key, kind, ei, s in todo:
        t0 = time.time()
        try:
            res[key] = fit_score(kind, ei, s)
            json.dump(res, open(OUT, "w"))
            print(f"  [{key:24}] nRMSE {np.mean(res[key]['err']):.4f}  ({time.time()-t0:.0f}s)", flush=True)
        except Exception as e:
            print(f"  [{key:24}] ECHEC {type(e).__name__}: {e}", flush=True)
    print("=== termine ===", flush=True)
