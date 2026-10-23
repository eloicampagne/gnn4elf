"""Re-scores the full-panel cells with the metrics the manuscript uses.

The sweep reports nRMSE: each node's target is centred and reduced by its own
training mean and standard deviation, the root mean squared error is taken over
the test window for each node, and the node errors are averaged. The manuscript
scores at system level instead, summing the nodes first and scoring the sum,

    RMSE = sqrt( (1/T) sum_t ( sum_i y_it - sum_i yhat_it )^2 )
    MAPE = (1/T) sum_t | ( sum_i y_it - sum_i yhat_it ) / sum_i y_it |

which is a different quantity, not a rescaling: the per-node figure penalises
errors that cancel across nodes, the system figure does not. Neither can be
derived from the other, so the forecasts are recomputed and scored three ways.
Predictions are returned to the original unit before scoring.

Only the full panels are re-scored, three hundred and nineteen hourly clients and
five hundred feeders, since those are the cells that carry no subsampling and are
the ones a chapter could quote.
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
import rewire_test as R  # noqa: E402
import weave_nodes as W  # noqa: E402

OUT = E.OUT
HIDDEN = E.HIDDEN
_PANEL = {}


def cell(coll, window, seed):
    """Full panel, standardised inputs, and the moments needed to undo it."""
    if coll == "elec":
        if window not in _PANEL:
            _PANEL[window] = E.load_panel(window)
        x, y, n_tr, n_ev = _PANEL[window]
        keep = list(range(x.shape[1]))
        xz, yz, my, sy = E.standardise(x, y, n_tr)
        ei, _ = E.knn_graph(yz[:n_tr])
        return xz, yz, my, sy, n_tr, n_ev, ei
    z = W.panel()
    keep = list(range(z["Xtr"].shape[1]))
    xtr = torch.tensor(z["Xtr"][-window:]); ytr = torch.tensor(z["Ytr"][-window:])
    xte = torch.tensor(z["Xte"]); yte = torch.tensor(z["Yte"])
    n_va = max(8, len(xtr) // 5)
    x = torch.cat([xtr, xte]); y = torch.cat([ytr, yte])
    n_tr, n_ev = len(xtr) - n_va, len(xtr)
    mx, sx = x[:n_tr].mean(0), x[:n_tr].std(0)
    my, sy = y[:n_tr].mean(0), y[:n_tr].std(0)
    dx, dy = sx < 1e-8, sy < 1e-8
    xz = (x - mx) / torch.where(dx, torch.ones_like(sx), sx)
    yz = (y - my) / torch.where(dy, torch.ones_like(sy), sy)
    xz[:, dx] = 0.0
    return xz, yz, my, sy, n_tr, n_ev, W.edges_from(z["A"], keep)


def score(pred_z, true_z, my, sy):
    """Three readings of the same forecast, the first two at system level."""
    p = pred_z * sy + my                                   # back to the original unit
    t = true_z * sy + my
    if p.ndim == 3:                                        # [T, N, H] -> [T*H, N]
        p = p.permute(0, 2, 1).reshape(-1, p.shape[1])
        t = t.permute(0, 2, 1).reshape(-1, t.shape[1])
    ps, ts = p.sum(1), t.sum(1)                            # national aggregate
    rmse_sys = float(((ts - ps) ** 2).mean().sqrt())
    mape_sys = float(((ts - ps) / ts.clamp(min=1e-8)).abs().mean() * 100)
    rmse_node = float(((p - t) ** 2).mean(0).sqrt().mean())
    mape_node = float((((t - p) / t.clamp(min=1e-8)).abs().mean(0) * 100).median())
    nrmse = float(((pred_z - true_z) ** 2).mean(0).sqrt().mean()) if pred_z.ndim == 2 \
        else float(((pred_z - true_z) ** 2).mean((0, 2)).sqrt().mean())
    return {"rmse_sys": rmse_sys, "mape_sys": mape_sys,
            "rmse_node": rmse_node, "mape_node": mape_node, "nrmse": nrmse}


def run(coll, window, model, seed, conv):
    xz, yz, my, sy, n_tr, n_ev, ei = cell(coll, window, seed)
    n, f = xz.shape[1], xz.shape[2]
    h_out = yz.shape[2] if yz.ndim == 3 else 1
    if model in ("rewired", "pnn_rewired"):
        ei = R.rewire(ei, n, 5000 + seed)
    if model in ("graph_coal", "pnn_coal"):
        from torch_geometric.utils import coalesce
        ei = coalesce(ei, num_nodes=n)          # mutual neighbours counted once, as in the rewired graph
    torch.manual_seed(seed)
    if model == "per_node":
        m = E.PerNode(n, f, HIDDEN, out=h_out)
    elif model == "shared":
        m = E.Shared(f, HIDDEN, out=h_out)
    elif model in ("pnn_coal", "pnn_rewired"):
        # Per-node model reading the mean of its (simple-graph) neighbours, true or rewired.
        src, dst = ei[0], ei[1]
        nb = torch.zeros_like(xz).index_add_(1, dst, xz[:, src])
        deg = torch.zeros(n, dtype=xz.dtype).index_add_(0, dst, torch.ones(dst.numel(), dtype=xz.dtype))
        xz = torch.cat([xz, nb / deg.clamp(min=1.0).view(1, n, *([1] * (xz.ndim - 2)))], dim=-1)
        m = E.PerNode(n, 2 * f, HIDDEN, out=h_out)
    elif model == "graph_id":
        # Same backbone and convolution as the graph network, no edge: no propagation.
        m = E.Graph(f, HIDDEN, conv, torch.zeros((2, 0), dtype=torch.long), None, out=h_out)
    else:
        m = E.Graph(f, HIDDEN, conv, ei, None, out=h_out)
    pred, val = E.fit(m, xz, yz, n_tr, n_ev, seed)
    return dict(score(pred, yz[n_ev:], my, sy), val=val, n=n)


if __name__ == "__main__":
    CELLS = (("elec", 264, "SAGEConv"), ("weave", 481, "SAGEConv"),
             ("weave", 60, "ChebConv"), ("elec", 1440, "ChebConv"))
    MODELS = ("shared", "per_node", "graph", "rewired", "graph_id", "graph_coal", "pnn_coal", "pnn_rewired")
    keys = [f"{c}|{w}|{mo}|{s}" for s in (0, 1, 2) for c, w, cv in CELLS for mo in MODELS]
    convs = {(c, w): cv for c, w, cv in CELLS}

    path = os.path.join(OUT, "eval_metrics.json")
    res = json.load(open(path)) if os.path.exists(path) else {}
    todo = [q for q in keys if q not in res]
    print(f"{len(res)} en cache, {len(todo)} a calculer", flush=True)
    for q in todo:
        coll, w, mo, s = q.split("|")
        t0 = time.time()
        try:
            res[q] = run(coll, int(w), mo, int(s), convs[(coll, int(w))])
            json.dump(res, open(path, "w"))
            v = res[q]
            print(f"  [{q:28}] RMSE_sys {v['rmse_sys']:10.1f}  MAPE_sys {v['mape_sys']:6.2f}%"
                  f"  nRMSE {v['nrmse']:.4f}  ({time.time()-t0:.0f}s)", flush=True)
        except Exception as e:
            print(f"  [{q:28}] ECHEC {type(e).__name__}: {e}", flush=True)
    print("=== termine ===", flush=True)
