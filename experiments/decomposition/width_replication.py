"""The fixed-capacity decomposition at two further widths.

Holding every model at one hidden layer of width sixty-four stops a difference of
width from standing in for the constraint under measurement. It does not show
that the increments the decomposition reports survive a change of capacity, and a
graph model plausibly needs more capacity than a per-node MLP to turn neighbour
features into a gain. This script repeats the four cells of the fixed-capacity
table at widths thirty-two and one hundred and twenty-eight, everything else held
at what eval_metrics.py used, and reports the two increments of the commuting
square at each width.

If the sharing and edge increments keep their sign and rough size across the
three widths, the width-64 reading is a property of the constraint. If they do
not, it is a property of the capacity, and the section's conclusion is narrower
than it is stated.
"""

import json
import os
import sys
import time

import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from _paths import ROOT, DATA, CACHE  # noqa: E402,F401  (puts every experiment folder on sys.path)
import ltsf_nodes as E  # noqa: E402
import eval_metrics as V  # noqa: E402
from increment_intervals import neighbour_mean, CELLS, MODELS, CONV, SEEDS  # noqa: E402

PATH = os.path.join(E.OUT, "width_replication.json")
WIDTHS = (32, 128)


def run(coll, window, model, seed, hidden):
    xz, yz, my, sy, n_tr, n_ev, ei = V.cell(coll, window, seed)
    n, f = xz.shape[1], xz.shape[2]
    h_out = yz.shape[2] if yz.ndim == 3 else 1

    torch.manual_seed(seed)
    if model == "per_node":
        m = E.PerNode(n, f, hidden, out=h_out)
    elif model == "per_node_nbr":
        xz = torch.cat([xz, neighbour_mean(xz, ei, n)], dim=-1)
        m = E.PerNode(n, 2 * f, hidden, out=h_out)
    elif model == "shared":
        m = E.Shared(f, hidden, out=h_out)
    elif model == "global":
        m = E.Global(n, f, hidden, out=h_out)
    else:
        m = E.Graph(f, hidden, CONV[(coll, window)], ei, None, out=h_out)

    t0 = time.time()
    pred, val = E.fit(m, xz, yz, n_tr, n_ev, seed)
    d = (pred - yz[n_ev:]) ** 2
    err = d.mean((0, 2)).sqrt() if d.ndim == 3 else d.mean(0).sqrt()
    return {"nrmse": float(err.mean()), "val": val, "n": n,
            "params": sum(p.numel() for p in m.parameters()),
            "fit_s": round(time.time() - t0, 1)}


if __name__ == "__main__":
    torch.set_num_threads(4)
    res = json.load(open(PATH)) if os.path.exists(PATH) else {}
    keys = [f"{c}|{w}|{m}|{h}|{s}" for h in WIDTHS for c, w in CELLS
            for m in MODELS for s in SEEDS]
    todo = [k for k in keys if k not in res]
    print(f"{len(res)} en cache, {len(todo)} a calculer", flush=True)
    for k in todo:
        c, w, m, h, s = k.split("|")
        t0 = time.time()
        try:
            res[k] = run(c, int(w), m, int(s), int(h))
            json.dump(res, open(PATH, "w"))
            print(f"  [{k:30}] nRMSE {res[k]['nrmse']:.4f}  "
                  f"{res[k]['params']:>10,d} p  ({time.time()-t0:.0f}s)", flush=True)
        except Exception as e:
            print(f"  [{k:30}] ECHEC {type(e).__name__}: {e}", flush=True)

    print("\n=== increments by width, nRMSE, mean over three seeds ===", flush=True)
    hdr = f"{'cell':14}{'arrow':26}" + "".join(f"{h:>12}" for h in (32, 64, 128))
    print(hdr)
    ref = json.load(open(os.path.join(E.OUT, "eval_metrics.json")))
    unsh = json.load(open(os.path.join(E.OUT, "unshared_graph.json")))
    for c, w in CELLS:
        def mean_at(model, h):
            if h == 64:
                src, key = (unsh, f"{c}|{w}|per_node_nbr") if model == "per_node_nbr" \
                    else (ref, f"{c}|{w}|{model}")
                ks = [f"{key}|{s}" for s in SEEDS]
                return np.mean([src[k]["nrmse"] for k in ks]) if all(k in src for k in ks) else np.nan
            ks = [f"{c}|{w}|{model}|{h}|{s}" for s in SEEDS]
            return np.mean([res[k]["nrmse"] for k in ks]) if all(k in res for k in ks) else np.nan
        for name, a, b in [("sharing", "per_node", "shared"),
                           ("edges, unshared", "per_node", "per_node_nbr"),
                           ("edges, shared", "shared", "graph"),
                           ("total", "per_node", "graph")]:
            row = "".join(f"{mean_at(a,h)-mean_at(b,h):>+12.4f}" for h in (32, 64, 128))
            print(f"{c+'|'+str(w):14}{name:26}{row}")
    print("\n=== termine ===", flush=True)
