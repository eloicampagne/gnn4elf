"""Intervals on the increments of the fixed-capacity decomposition.

The manuscript reports every increment of Section 6.4 as a difference of two
means over three seeds, to four decimal places, with no interval. Some of those
differences are smaller than the seed dispersion printed beside them. This script
attaches an interval to each, by the one resampling that matches the statistic
being reported.

nRMSE averages a per-node error over the nodes of the panel, so the node is the
unit the average is taken over and the unit to resample. For each cell of the
fixed-capacity table the five models are fitted on the same split with the same
three seeds, the per-node root mean squared error is kept for every node, the
three seeds are averaged inside each node, and the nodes are then resampled with
replacement. Every model in a draw is scored on the same resampled nodes, so the
comparison stays paired and the seed pairing is preserved.

Reported for each cell: the four arrows of the commuting square, each with a
percentile interval, and the fraction of draws in which the arrow keeps its sign.
An arrow whose interval covers zero is one the manuscript should not decompose.

Protocol, split, graph and width are eval_metrics.py's, which is the script that
produced the electricity block of the table.
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

OUT = E.OUT
HIDDEN = E.HIDDEN
ERR = os.path.join(OUT, "increment_errors.json")
CI = os.path.join(OUT, "increment_intervals.json")

CELLS = [("elec", 264), ("elec", 1440), ("weave", 60), ("weave", 481)]
MODELS = ["per_node", "per_node_nbr", "shared", "global", "graph_id", "graph"]
CONV = {("elec", 264): "SAGEConv", ("elec", 1440): "ChebConv",
        ("weave", 60): "ChebConv", ("weave", 481): "SAGEConv"}
SEEDS = (0, 1, 2)
NBOOT = 5000


def neighbour_mean(x, ei, n):
    src, dst = ei[0], ei[1]
    out = torch.zeros(x.shape[0], n, x.shape[2], dtype=x.dtype)
    out.index_add_(1, dst, x[:, src])
    deg = torch.zeros(n, dtype=x.dtype).index_add_(
        0, dst, torch.ones(dst.numel(), dtype=x.dtype))
    return out / deg.clamp(min=1.0).view(1, n, 1)


def per_node_rmse(coll, window, model, seed):
    """Root mean squared error of every node, in the standardised space nRMSE uses."""
    xz, yz, my, sy, n_tr, n_ev, ei = V.cell(coll, window, seed)
    n, f = xz.shape[1], xz.shape[2]
    h_out = yz.shape[2] if yz.ndim == 3 else 1

    torch.manual_seed(seed)                      # before construction, see M8
    if model == "per_node":
        m = E.PerNode(n, f, HIDDEN, out=h_out)
    elif model == "per_node_nbr":
        xz = torch.cat([xz, neighbour_mean(xz, ei, n)], dim=-1)
        m = E.PerNode(n, 2 * f, HIDDEN, out=h_out)
    elif model == "shared":
        m = E.Shared(f, HIDDEN, out=h_out)
    elif model == "global":
        m = E.Global(n, f, HIDDEN, out=h_out)
    elif model == "graph_id":
        # Same backbone and convolution as the graph network, no edge: no propagation.
        m = E.Graph(f, HIDDEN, CONV[(coll, window)], torch.zeros((2, 0), dtype=torch.long),
                    None, out=h_out)
    else:
        m = E.Graph(f, HIDDEN, CONV[(coll, window)], ei, None, out=h_out)

    t0 = time.time()
    pred, val = E.fit(m, xz, yz, n_tr, n_ev, seed)
    true = yz[n_ev:]
    d = (pred - true) ** 2
    err = d.mean((0, 2)).sqrt() if d.ndim == 3 else d.mean(0).sqrt()
    return (err.tolist(), sum(p.numel() for p in m.parameters()),
            round(time.time() - t0, 1))


def collect():
    res = json.load(open(ERR)) if os.path.exists(ERR) else {}
    keys = [f"{c}|{w}|{m}|{s}" for c, w in CELLS for m in MODELS for s in SEEDS]
    todo = [k for k in keys if k not in res]
    print(f"{len(res)} en cache, {len(todo)} a calculer", flush=True)
    for k in todo:
        c, w, m, s = k.split("|")
        t0 = time.time()
        try:
            err, params, fit_s = per_node_rmse(c, int(w), m, int(s))
            res[k] = {"err": err, "params": params, "fit_s": fit_s}
            json.dump(res, open(ERR, "w"))
            print(f"  [{k:28}] nRMSE {np.mean(err):.4f}  {params:>10,d} p  "
                  f"({time.time()-t0:.0f}s)", flush=True)
        except Exception as e:
            print(f"  [{k:28}] ECHEC {type(e).__name__}: {e}", flush=True)
    return res


ARROWS = [("sharing", "per_node", "shared"),
          ("edges, unshared", "per_node", "per_node_nbr"),
          ("edges, shared", "shared", "graph"),
          ("backbone, no edge", "shared", "graph_id"),
          ("edges, same backbone", "graph_id", "graph"),
          ("total, per-node to graph", "per_node", "graph")]


def intervals(res):
    rng = np.random.default_rng(0)
    out = {}
    for c, w in CELLS:
        E_ = {}
        for m in MODELS:
            ks = [f"{c}|{w}|{m}|{s}" for s in SEEDS]
            if not all(k in res for k in ks):
                break
            E_[m] = np.mean([res[k]["err"] for k in ks], axis=0)   # seeds averaged per node
        if len(E_) < len(MODELS):
            continue
        n = len(next(iter(E_.values())))
        draws = rng.integers(0, n, size=(NBOOT, n))
        cell = {"n_nodes": n,
                "nrmse": {m: float(v.mean()) for m, v in E_.items()},
                "params": {m: res[f"{c}|{w}|{m}|0"]["params"] for m in MODELS},
                "fit_s": {m: float(np.mean([res[f"{c}|{w}|{m}|{s}"]["fit_s"]
                                            for s in SEEDS])) for m in MODELS},
                "arrows": {}}
        for name, a, b in ARROWS:
            point = float(E_[a].mean() - E_[b].mean())
            bs = E_[a][draws].mean(1) - E_[b][draws].mean(1)
            lo, hi = np.percentile(bs, [2.5, 97.5])
            cell["arrows"][name] = {
                "point": point, "lo": float(lo), "hi": float(hi),
                "se": float(bs.std(ddof=1)),
                "sign_kept": float((np.sign(bs) == np.sign(point)).mean()),
                "covers_zero": bool(lo <= 0.0 <= hi)}
        out[f"{c}|{w}"] = cell
    json.dump(out, open(CI, "w"), indent=1)
    return out


if __name__ == "__main__":
    res = collect()
    out = intervals(res)
    print("\n=== increments, 95% percentile interval over a node bootstrap "
          f"({NBOOT} draws, seeds averaged inside each node) ===", flush=True)
    for cell, v in out.items():
        print(f"\n{cell}   {v['n_nodes']} nodes")
        for name, a in v["arrows"].items():
            flag = "  covers 0" if a["covers_zero"] else ""
            print(f"  {name:26} {a['point']:+.4f}  [{a['lo']:+.4f}, {a['hi']:+.4f}]"
                  f"  sign kept {a['sign_kept']:.0%}{flag}")
    print("\n=== termine ===", flush=True)
