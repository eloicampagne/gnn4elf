"""Strong local references for the fixed-capacity decomposition.

A referee objects that the per-node MLP of the decomposition is over-parameterised
for the windows it is fitted on (thousands of weights per node, a few dozen
origins), so that its defeat by sharing measures its variance as much as the
value of sharing. This script answers with references that cannot overfit in the
same way, on the same cells, splits, inputs and score as increment_intervals.py:

  ridge_pn       per-node ridge, penalty chosen node by node on the validation window
  ridge_pn_nbr   the same with the neighbour mean of the features appended
  ridge_sh       one ridge shared by all nodes, penalty chosen on validation
  ridge_sh_nbr   the same with the neighbour mean appended
  persist        last observed value at the same position of the daily cycle
                 (lag 24 h on Electricity, D-1 on WEAVE-UK)

Every score is the per-node RMSE in the standardised space nRMSE uses, averaged
over nodes. Nothing here trains a network; the fits are closed form.
"""

import json
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from _paths import ROOT, DATA, CACHE  # noqa: E402,F401  (puts every experiment folder on sys.path)
import ltsf_nodes as E  # noqa: E402
import eval_metrics as V  # noqa: E402
import weave_nodes as W  # noqa: E402
from increment_intervals import CELLS, neighbour_mean  # noqa: E402

OUT = os.path.join(E.OUT, "local_references.json")
GRID = (1e-4, 1e-3, 1e-2, 1e-1, 1.0, 10.0, 100.0, 1000.0)


def per_node_err(pred, true):
    d = (pred - true) ** 2
    return (d.mean((0, 2)) if d.ndim == 3 else d.mean(0)).sqrt()


def ridge_per_node(x, y, n_tr, n_ev):
    """Node-specific ridge; each node keeps the penalty that minimises its own validation error."""
    if y.ndim == 2:
        y = y[..., None]
    T, N, F = x.shape
    xb = torch.cat([x, torch.ones(T, N, 1)], dim=-1).double()
    yb = y.double()
    xt, yt = xb[:n_tr], yb[:n_tr]
    g = torch.einsum("tnf,tng->nfg", xt, xt)
    b = torch.einsum("tnf,tnh->nfh", xt, yt)
    scale = torch.einsum("tnf,tnf->nf", xt, xt).mean(1).clamp(min=1e-8)
    eye = torch.eye(F + 1, dtype=torch.float64).expand(N, F + 1, F + 1)
    best_err = torch.full((N,), float("inf"), dtype=torch.float64)
    best_w = torch.zeros(N, F + 1, yb.shape[-1], dtype=torch.float64)
    for m in GRID:
        w = torch.linalg.solve(g + (m * scale)[:, None, None] * eye, b)
        pv = torch.einsum("tnf,nfh->tnh", xb[n_tr:n_ev], w)
        e = ((pv - yb[n_tr:n_ev]) ** 2).mean((0, 2))
        better = torch.isfinite(e) & (e < best_err)
        best_err = torch.where(better, e, best_err)
        best_w[better] = w[better]
    pred = torch.einsum("tnf,nfh->tnh", xb[n_ev:], best_w)
    return per_node_err(pred, yb[n_ev:]).float()


def ridge_shared(x, y, n_tr, n_ev):
    """One ridge for all nodes, penalty chosen on the pooled validation error."""
    if y.ndim == 2:
        y = y[..., None]
    T, N, F = x.shape
    xb = torch.cat([x, torch.ones(T, N, 1)], dim=-1).double()
    yb = y.double()
    xt = xb[:n_tr].reshape(-1, F + 1)
    yt = yb[:n_tr].reshape(-1, yb.shape[-1])
    g, b = xt.T @ xt, xt.T @ yt
    scale = float((xt ** 2).sum(0).mean())
    best, best_w = float("inf"), None
    for m in GRID:
        w = torch.linalg.solve(g + m * scale * torch.eye(F + 1, dtype=torch.float64), b)
        e = float(((xb[n_tr:n_ev] @ w - yb[n_tr:n_ev]) ** 2).mean())
        if np.isfinite(e) and e < best:
            best, best_w = e, w
    return per_node_err(xb[n_ev:] @ best_w, yb[n_ev:]).float()


def persistence(coll, window, yz, my, sy, n_ev):
    """Same slot of the previous daily cycle, standardised with the training moments."""
    if coll == "elec":
        x_raw, _, _, _ = E.load_panel(window)
        p = (x_raw[:, :, 3] - my) / sy                       # lag 24 h
        return per_node_err(p[n_ev:], yz[n_ev:])
    z = W.panel()
    x_raw = torch.tensor(np.concatenate([z["Xtr"][-window:], z["Xte"]]))
    lags = x_raw[:, :, :48]                                   # newest first: lag j is y_{t-1-j}
    p = lags.flip(-1)                                         # horizon k reads y_{t+k-48}
    p = (p - my) / sy
    return per_node_err(p[n_ev:], yz[n_ev:])


def main():
    res = {}
    for coll, window in CELLS:
        xz, yz, my, sy, n_tr, n_ev, ei = V.cell(coll, window, 0)
        n = xz.shape[1]
        xn = torch.cat([xz, neighbour_mean(xz, ei, n)], dim=-1)
        cell = {
            "ridge_pn": ridge_per_node(xz, yz, n_tr, n_ev),
            "ridge_pn_nbr": ridge_per_node(xn, yz, n_tr, n_ev),
            "ridge_sh": ridge_shared(xz, yz, n_tr, n_ev),
            "ridge_sh_nbr": ridge_shared(xn, yz, n_tr, n_ev),
            "persist": persistence(coll, window, yz, my, sy, n_ev),
        }
        key = f"{coll}|{window}"
        res[key] = {k: v.tolist() for k, v in cell.items()}
        print(key, {k: round(float(v.mean()), 4) for k, v in cell.items()}, flush=True)
    json.dump(res, open(OUT, "w"))


if __name__ == "__main__":
    main()
