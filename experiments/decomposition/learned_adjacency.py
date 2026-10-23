"""A learned adjacency in the fixed-capacity protocol, and the same net without it.

Section 2 names MTGNN and ADLGNN as the works bearing directly on the comparisons
run here, says their advantage over fixed-graph networks is unclear, and calls a
systematic comparison worth running. The comparison is not in the manuscript, so
the deflationary reading rests on models that all receive the graph rather than
build one. This script runs the missing arm.

Two models are fitted per cell, identical in every respect but one: MTGNN with
its graph learner on, and the same network with `gcn_true=False`, which keeps the
dilated temporal trunk and removes the message passing. The pair is what makes
the test clean. A comparison of MTGNN against the manuscript's SAGE would confound
the learned adjacency with a different trunk, a different capacity and a different
regularisation; the matched pair confounds nothing, and it answers the question the
paper actually raises: whether a graph the model discovers for itself buys what a
graph handed to it does not.

The protocol is eval_metrics.py's, cell for cell: the same panels, splits, features
and targets, the same three seeds, the same optimiser, patience and batch size. The
F features of a node are laid on MTGNN's time axis with one input channel, which is
what they are, and the model emits the H targets of the cell in one shot. Capacity is
cut from the published defaults so the parameter count sits near the width-64 models
of the fixed-capacity table rather than an order above them; the counts are reported
with the errors so the reader can judge what the comparison holds fixed.

Dropout stays at MTGNN's own default: it is part of the model, and both arms carry it.
"""

import json
import os
import sys
import time

import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from _paths import ROOT, DATA, CACHE  # noqa: E402,F401  (puts every experiment folder on sys.path)
HERE = ROOT
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, "external", "MTGNN"))

import ltsf_nodes as E  # noqa: E402
import eval_metrics as V  # noqa: E402
from net import gtnet  # noqa: E402

PATH = os.path.join(E.OUT, "learned_adjacency.json")
CELLS = [("weave", 481), ("weave", 60), ("elec", 1440), ("elec", 264)]
SEEDS = (0, 1, 2)

# Cut from the published defaults (conv 32 / residual 32 / skip 64 / end 128,
# 3 layers, node_dim 40) so the count lands near the width-64 models.
CAP = dict(conv_channels=16, residual_channels=16, skip_channels=32,
           end_channels=64, layers=2, node_dim=16, subgraph_size=10,
           gcn_depth=2, propalpha=0.05, tanhalpha=3, dropout=0.3)


class MTGNNWrap(torch.nn.Module):
    """[B, N, F] -> [B, N, H], the interface every model in the protocol uses."""

    def __init__(self, n, f, h_out, gcn, seed):
        super().__init__()
        torch.manual_seed(seed)
        self.h_out = h_out
        self.net = gtnet(gcn_true=gcn, buildA_true=gcn, num_nodes=n,
                         device=torch.device("cpu"), predefined_A=None,
                         seq_length=f, in_dim=1, out_dim=h_out, **CAP)

    def forward(self, x):                                   # x [B, N, F]
        o = self.net(x.unsqueeze(1))                        # [B, 1, N, F] -> [B, H, N, 1]
        o = o.squeeze(-1).permute(0, 2, 1)                  # [B, N, H]
        return o.squeeze(-1) if self.h_out == 1 else o


def run(coll, window, gcn, seed):
    xz, yz, my, sy, n_tr, n_ev, _ = V.cell(coll, window, seed)
    n, f = xz.shape[1], xz.shape[2]
    h_out = yz.shape[2] if yz.ndim == 3 else 1
    m = MTGNNWrap(n, f, h_out, gcn, seed)
    t0 = time.time()
    pred, val = E.fit(m, xz, yz, n_tr, n_ev, seed)
    d = (pred - yz[n_ev:]) ** 2
    err = d.mean((0, 2)).sqrt() if d.ndim == 3 else d.mean(0).sqrt()
    return {"nrmse": float(err.mean()), "val": val, "n": n,
            "params": sum(p.numel() for p in m.parameters()),
            "fit_s": round(time.time() - t0, 1)}


if __name__ == "__main__":
    torch.set_num_threads(int(os.environ.get("NTHREADS", "8")))
    res = json.load(open(PATH)) if os.path.exists(PATH) else {}
    # Seed outermost: a run stopped early still covers every cell and both arms.
    keys = [f"{c}|{w}|{a}|{s}" for s in SEEDS for c, w in CELLS
            for a in ("graph", "nograph")]
    todo = [k for k in keys if k not in res]
    print(f"{len(res)} en cache, {len(todo)} a calculer", flush=True)
    for k in todo:
        c, w, a, s = k.split("|")
        t0 = time.time()
        try:
            res[k] = run(c, int(w), a == "graph", int(s))
            json.dump(res, open(PATH, "w"))
            v = res[k]
            print(f"  [{k:24}] nRMSE {v['nrmse']:.4f}  {v['params']:>9,d} p  "
                  f"({time.time()-t0:.0f}s)", flush=True)
        except Exception as e:
            print(f"  [{k:24}] ECHEC {type(e).__name__}: {e}", flush=True)

    print("\n=== learned adjacency, nRMSE, mean over three seeds ===", flush=True)
    print(f"{'cell':14}{'MTGNN + learned graph':>24}{'MTGNN, graph off':>20}{'difference':>14}")
    for c, w in CELLS:
        def mu(a):
            ks = [f"{c}|{w}|{a}|{s}" for s in SEEDS]
            return np.mean([res[k]["nrmse"] for k in ks]) if all(k in res for k in ks) else np.nan
        g, ng = mu("graph"), mu("nograph")
        print(f"{c+'|'+str(w):14}{g:>24.4f}{ng:>20.4f}{ng-g:>+14.4f}")
    print("\n=== termine ===", flush=True)
