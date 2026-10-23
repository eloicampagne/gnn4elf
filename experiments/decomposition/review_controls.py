"""Two controls asked by a second referee, on the decomposition of the large panels.

(A) The fourth corner of the MLP square. The decomposition reads sharing and
neighbourhood access on models of different families (MLPs and a residual GNN).
Within the MLP family three corners exist (per-node, per-node + neighbour mean,
shared); here the shared MLP reading the neighbour mean completes the square, on
the four cells of increment_intervals.py with the same inputs, graph, splits,
width and training routine.

(B) Other periods. The 11-day Electricity cell ends with the record (July 2019).
The same cell is repeated on four disjoint windows ending 30, 60, 90 and 120 days
earlier, for the models of the decomposition plus the shared MLP with neighbours,
so that the increments can be compared across periods.

Per-node errors are stored, in the standardised space nRMSE uses, for the
seed-and-node bootstrap of seed_node_intervals.py.
"""

import json
import os
import sys
import time

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from _paths import ROOT, DATA, CACHE  # noqa: E402,F401  (puts every experiment folder on sys.path)
import ltsf_nodes as E  # noqa: E402
import eval_metrics as V  # noqa: E402
from increment_intervals import CELLS, CONV, SEEDS, neighbour_mean  # noqa: E402

OUT = os.path.join(E.OUT, "review_controls.json")
HIDDEN = E.HIDDEN
SHIFTS_DAYS = (30, 60, 90, 120)
MODELS_B = ["per_node", "per_node_nbr", "shared", "shared_nbr", "graph_id", "graph"]


def load_panel_shifted(train_hours, shift_hours):
    """E.load_panel with the window ending shift_hours before the end of the record."""
    df = pd.read_csv(E.CSV, parse_dates=["date"]).set_index("date")
    df = df.drop(columns=[c for c in ("OT",) if c in df.columns])
    span = max(E.LAGS) + train_hours + E.H_VAL + E.H_TEST
    end = len(df) - shift_hours
    df = df.iloc[end - span:end]
    df = df.loc[:, df.std() > 0]
    v = df.to_numpy(dtype=np.float32)
    idx = np.arange(max(E.LAGS), len(df))
    lags = np.stack([v[idx - l] for l in E.LAGS], axis=-1)
    hour = df.index.hour.to_numpy()[idx]
    dow = df.index.dayofweek.to_numpy()[idx]
    cal = np.stack([np.sin(2 * np.pi * hour / 24), np.cos(2 * np.pi * hour / 24),
                    np.sin(2 * np.pi * dow / 7), np.cos(2 * np.pi * dow / 7)], axis=-1)
    cal = np.repeat(cal[:, None, :], v.shape[1], axis=1).astype(np.float32)
    x = np.concatenate([lags, cal], axis=-1)
    y = v[idx]
    n_tr = len(idx) - E.H_VAL - E.H_TEST
    return torch.tensor(x), torch.tensor(y), n_tr, n_tr + E.H_VAL, str(df.index[-1].date())


def cell_shifted(shift_days):
    x, y, n_tr, n_ev, end = load_panel_shifted(264, 24 * shift_days)
    xz, yz, my, sy = E.standardise(x, y, n_tr)
    ei, _ = E.knn_graph(yz[:n_tr])
    return xz, yz, n_tr, n_ev, ei, end


def fit_model(model, xz, yz, n_tr, n_ev, ei, conv, seed):
    n, f = xz.shape[1], xz.shape[2]
    h_out = yz.shape[2] if yz.ndim == 3 else 1
    torch.manual_seed(seed)
    if model == "per_node":
        m = E.PerNode(n, f, HIDDEN, out=h_out)
    elif model == "per_node_nbr":
        xz = torch.cat([xz, neighbour_mean(xz, ei, n)], dim=-1)
        m = E.PerNode(n, 2 * f, HIDDEN, out=h_out)
    elif model == "shared":
        m = E.Shared(f, HIDDEN, out=h_out)
    elif model == "shared_nbr":
        xz = torch.cat([xz, neighbour_mean(xz, ei, n)], dim=-1)
        m = E.Shared(2 * f, HIDDEN, out=h_out)
    elif model == "graph_id":
        m = E.Graph(f, HIDDEN, conv, torch.zeros((2, 0), dtype=torch.long), None, out=h_out)
    else:
        m = E.Graph(f, HIDDEN, conv, ei, None, out=h_out)
    pred, _ = E.fit(m, xz, yz, n_tr, n_ev, seed)
    d = (pred - yz[n_ev:]) ** 2
    return (d.mean((0, 2)) if d.ndim == 3 else d.mean(0)).sqrt().tolist()


def jobs():
    out = [("A", c, w, s) for s in SEEDS for c, w in CELLS]                       # (A) first, cheap
    out += [("B", d, m, s) for d in SHIFTS_DAYS for s in SEEDS for m in MODELS_B]  # (B)
    return out


if __name__ == "__main__":
    res = json.load(open(OUT)) if os.path.exists(OUT) else {}
    cache = {}
    for job in jobs():
        if job[0] == "A":
            _, c, w, s = job
            key = f"A|{c}|{w}|shared_nbr|{s}"
        else:
            _, d, mo, s = job
            key = f"B|elec|264|shift{d}|{mo}|{s}"
        if key in res:
            continue
        t0 = time.time()
        try:
            if job[0] == "A":
                xz, yz, my, sy, n_tr, n_ev, ei = V.cell(c, w, s)
                err = fit_model("shared_nbr", xz, yz, n_tr, n_ev, ei, CONV[(c, w)], s)
                res[key] = {"err": err}
            else:
                if d not in cache:
                    cache[d] = cell_shifted(d)
                xz, yz, n_tr, n_ev, ei, end = cache[d]
                err = fit_model(mo, xz, yz, n_tr, n_ev, ei, CONV[("elec", 264)], s)
                res[key] = {"err": err, "end": end, "n": len(err)}
            json.dump(res, open(OUT, "w"))
            print(f"  [{key:36}] nRMSE {np.mean(err):.4f}  ({time.time()-t0:.0f}s)", flush=True)
        except Exception as e:
            print(f"  [{key:36}] ECHEC {type(e).__name__}: {e}", flush=True)
    print("=== termine ===", flush=True)
