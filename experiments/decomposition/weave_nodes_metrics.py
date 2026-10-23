"""The node-count sweep, scored three ways, with the seed controlling the fit.

This file regenerates results_ltsf_nodes/weave_nodes_metrics.json, which supplies
the WEAVE-UK block of the manuscript's fixed-capacity table and the whole of its
node-count table, and which had no generating script in the repository.

Two things differ from weave_nodes.py, and only two. The sweep here also fits the
unshared control that reads the neighbour mean, at the full panel where the
manuscript reports it. And the generator is seeded before the model is built:
fit() seeds after construction, and PerNode draws its weights in __init__, so
under weave_nodes.py the initialisation followed whatever state the process held
and therefore how full the JSON cache was when the run started. Comparing
weave_nodes.json against weave_nodes_metrics.json on their shared keys shows what
that cost: the two disagree by up to 0.077 nRMSE.

Protocol is otherwise weave_nodes.py's, cell for cell: the same draws, the same
five retained neighbours, the same split, the same width of sixty-four, the same
three repetitions. The three scores are the per-node nRMSE the sweep reports and
the system RMSE and MAPE of eval_metrics.py, the RMSE carried to kWh.

Results are written to a new file. Nothing that backs a published table is
overwritten; the comparison between the two is the point.
"""

import json
import os
import sys
import time

import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from _paths import ROOT, DATA, CACHE  # noqa: E402,F401  (puts every experiment folder on sys.path)
from ltsf_nodes import PerNode, Shared, Global, Graph, fit  # noqa: E402
import weave_nodes as W  # noqa: E402

OUT = W.OUT
HIDDEN = W.HIDDEN
PATH = os.path.join(OUT, "weave_nodes_metrics_seeded.json")


def neighbour_mean(x, ei, n):
    """Mean of each node's neighbours' features, as unshared_graph.py computes it.

    Edges are stored source-first, so a node's neighbours are the sources of the
    edges whose target it is. Isolated nodes receive zeros.
    """
    src, dst = ei[0], ei[1]
    out = torch.zeros(x.shape[0], n, x.shape[2], dtype=x.dtype)
    out.index_add_(1, dst, x[:, src])
    deg = torch.zeros(n, dtype=x.dtype).index_add_(
        0, dst, torch.ones(dst.numel(), dtype=x.dtype))
    return out / deg.clamp(min=1.0).view(1, n, 1)


def score(pred_z, true_z, my, sy):
    """Per-node nRMSE in the standardised space, system RMSE and MAPE in the original one."""
    nrmse = float(((pred_z - true_z) ** 2).mean((0, 2)).sqrt().mean())
    p, t = pred_z * sy + my, true_z * sy + my
    p = p.permute(0, 2, 1).reshape(-1, p.shape[1])
    t = t.permute(0, 2, 1).reshape(-1, t.shape[1])
    ps, ts = p.sum(1), t.sum(1)
    rmse = float(((ts - ps) ** 2).mean().sqrt()) / 1000.0          # Wh -> kWh
    mape = float(((ts - ps) / ts.clamp(min=1e-8)).abs().mean() * 100)
    return {"nrmse": nrmse, "rmse": rmse, "mape": mape}


def run(n_nodes, train_days, model_name, seed):
    z = W.panel()
    rng = np.random.default_rng(1000 + seed)
    n_all = z["Xtr"].shape[1]
    keep = sorted(rng.choice(n_all, size=min(n_nodes, n_all), replace=False).tolist())

    xtr = torch.tensor(z["Xtr"][-train_days:, keep, :])
    ytr = torch.tensor(z["Ytr"][-train_days:, keep, :])
    xte, yte = torch.tensor(z["Xte"][:, keep, :]), torch.tensor(z["Yte"][:, keep, :])

    n_va = max(8, len(xtr) // 5)
    x, y = torch.cat([xtr, xte]), torch.cat([ytr, yte])
    n_tr, n_val_end = len(xtr) - n_va, len(xtr)

    mx, sx = x[:n_tr].mean(0), x[:n_tr].std(0)
    my, sy = y[:n_tr].mean(0), y[:n_tr].std(0)
    dead_x, dead_y = sx < 1e-8, sy < 1e-8
    x = (x - mx) / torch.where(dead_x, torch.ones_like(sx), sx)
    y = (y - my) / torch.where(dead_y, torch.ones_like(sy), sy)
    x[:, dead_x] = 0.0

    n, f, h_out = x.shape[1], x.shape[2], y.shape[2]
    ei = W.edges_from(z["A"], keep)

    # Before construction, not after: this is the whole difference from weave_nodes.py.
    torch.manual_seed(seed)

    if model_name == "per_node":
        m = PerNode(n, f, HIDDEN, out=h_out)
    elif model_name == "per_node_nbr":
        x = torch.cat([x, neighbour_mean(x, ei, n)], dim=-1)
        m = PerNode(n, 2 * f, HIDDEN, out=h_out)
    elif model_name == "shared":
        m = Shared(f, HIDDEN, out=h_out)
    elif model_name == "global":
        m = Global(n, f, HIDDEN, out=h_out)
    else:
        m = Graph(f, HIDDEN, model_name.split("_")[1], ei, None, out=h_out)

    t0 = time.time()
    pred, val = fit(m, x, y, n_tr, n_val_end, seed)
    out = score(pred, y[n_val_end:], my, sy)
    out.update(val=val, n=n, days=int(n_tr),
               params=sum(p.numel() for p in m.parameters()),
               fit_s=round(time.time() - t0, 1))
    return out


if __name__ == "__main__":
    MODELS = ["per_node", "shared", "global", "gnn_SAGEConv", "gnn_ChebConv"]
    keys = []
    for w in (481, 60):
        for n in (10, 40, 160, 500):
            for s in (0, 1, 2):
                for mo in MODELS + (["per_node_nbr"] if n == 500 else []):
                    keys.append(f"{n}|{w}|{mo}|{s}")

    res = json.load(open(PATH)) if os.path.exists(PATH) else {}
    todo = [k for k in keys if k not in res]
    print(f"{len(res)} en cache, {len(todo)} a calculer", flush=True)
    for k in todo:
        n, w, mo, s = k.split("|")
        t0 = time.time()
        try:
            res[k] = run(int(n), int(w), mo, int(s))
            json.dump(res, open(PATH, "w"))
            v = res[k]
            print(f"  [{k:26}] nRMSE {v['nrmse']:.4f}  RMSE {v['rmse']:8.2f}  "
                  f"MAPE {v['mape']:6.2f}%  {v['params']:>9,d} p  ({time.time()-t0:.0f}s)", flush=True)
        except Exception as e:
            print(f"  [{k:26}] ECHEC {type(e).__name__}: {e}", flush=True)
    print("=== termine ===", flush=True)
