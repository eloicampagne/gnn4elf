"""A recurrent baseline in the fixed-capacity protocol.

The manuscript carries no LSTM, GRU or temporal-convolution model anywhere, which
is the first half of the round-one request for baselines that represent practice.
The lag block a node reads is a sequence, and reading it with a feedforward layer
discards that; a recurrent reader is the cheapest model that does not.

The features of a cell split cleanly. On WEAVE-UK the first forty-eight columns are
the half-hourly lags and the last nine are the cyclical instant and the weekday; on
electricity the first seven are the lags of the protocol and the last four the
cyclical hour and weekday. The lags go to a GRU as a sequence of length L with one
channel, the calendar features join its final state, and a linear head emits the
targets of the cell. The recurrence is shared across nodes, so this is the shared
MLP's recurrent counterpart and it enters the table at the same width of sixty-four.

Everything else is eval_metrics.py's: same panels, splits, three seeds, optimiser,
patience and batch size.
"""

import json
import os
import sys
import time

import numpy as np
import torch
from torch import nn

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from _paths import ROOT, DATA, CACHE  # noqa: E402,F401  (puts every experiment folder on sys.path)
import ltsf_nodes as E  # noqa: E402
import eval_metrics as V  # noqa: E402

PATH = os.path.join(E.OUT, "recurrent_baseline.json")
CELLS = [("elec", 264), ("elec", 1440), ("weave", 60), ("weave", 481)]
SEEDS = (0, 1, 2)
NLAGS = {"elec": len(E.LAGS), "weave": 48}


class SharedGRU(nn.Module):
    """One recurrence for every node: lags as a sequence, calendar at the head."""

    def __init__(self, n_lags, n_cal, h, out=1):
        super().__init__()
        self.n_lags, self.out = n_lags, out
        self.gru = nn.GRU(1, h, batch_first=True)
        self.head = nn.Sequential(nn.Linear(h + n_cal, h), nn.ReLU(), nn.Linear(h, out))

    def forward(self, x):                                   # x [B, N, F]
        b, n, f = x.shape
        lag = x[..., :self.n_lags].reshape(b * n, self.n_lags, 1)
        cal = x[..., self.n_lags:].reshape(b * n, f - self.n_lags)
        _, hN = self.gru(lag)
        o = self.head(torch.cat([hN[-1], cal], dim=-1)).view(b, n, self.out)
        return o.squeeze(-1) if self.out == 1 else o


def run(coll, window, seed):
    xz, yz, my, sy, n_tr, n_ev, _ = V.cell(coll, window, seed)
    n, f = xz.shape[1], xz.shape[2]
    h_out = yz.shape[2] if yz.ndim == 3 else 1
    torch.manual_seed(seed)
    m = SharedGRU(NLAGS[coll], f - NLAGS[coll], E.HIDDEN, out=h_out)
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
    keys = [f"{c}|{w}|{s}" for s in SEEDS for c, w in CELLS]
    todo = [k for k in keys if k not in res]
    print(f"{len(res)} en cache, {len(todo)} a calculer", flush=True)
    for k in todo:
        c, w, s = k.split("|")
        t0 = time.time()
        try:
            res[k] = run(c, int(w), int(s))
            json.dump(res, open(PATH, "w"))
            v = res[k]
            print(f"  [{k:18}] nRMSE {v['nrmse']:.4f}  {v['params']:>9,d} p  "
                  f"({time.time()-t0:.0f}s)", flush=True)
        except Exception as e:
            print(f"  [{k:18}] ECHEC {type(e).__name__}: {e}", flush=True)

    ref = json.load(open(os.path.join(E.OUT, "eval_metrics.json")))
    print("\n=== shared GRU against the fixed-capacity table, nRMSE ===", flush=True)
    print(f"{'cell':14}{'shared GRU':>12}{'shared MLP':>12}{'per-node MLP':>14}{'SAGE':>10}")
    for c, w in CELLS:
        ks = [f"{c}|{w}|{s}" for s in SEEDS]
        gru = np.mean([res[k]["nrmse"] for k in ks]) if all(k in res for k in ks) else np.nan
        def r(model):
            kk = [f"{c}|{w}|{model}|{s}" for s in SEEDS]
            return np.mean([ref[k]["nrmse"] for k in kk]) if all(k in ref for k in kk) else np.nan
        print(f"{c+'|'+str(w):14}{gru:>12.4f}{r('shared'):>12.4f}{r('per_node'):>14.4f}{r('graph'):>10.4f}")
    print("\n=== termine ===", flush=True)
