"""The two models the favourable regime never received.

The French diagnostic reads its verdict off four models on the two axes of the
taxonomy, and off one control that no shared convolution realizes: node-specific
weights fed both a node's own features and the mean of its neighbours'. On twelve
regions that control settles the question, 871 MW against 876 for the same model
deprived of neighbours, so once the sharing handicap is gone the neighbourhood
adds nothing.

The three hundred and twenty clients and the five hundred feeders, the regime the
chapter calls favourable, were measured with three models only: per-node, shared,
and the graph network. The decomposition available there is therefore

    E(per-node) - E(graph) = [E(per-node) - E(shared)] + [E(shared) - E(graph)],

whose second term is what edges add once the weights are already common. It is a
conditional quantity, and on WEAVE-UK the first term carries almost everything, so
the reversal recorded on those collections may be a property of weight sharing and
not of the graph. This script runs the complementary path,

    E(per-node) - E(per-node + neighbour mean),

which is the effect of the edges with sharing held absent, and the global
perceptron on the full panels, which is the access axis the earlier table stopped
at forty nodes.

Everything else is held at what eval_metrics.py used: the same cells, the same
convolutions, the same three seeds, the same splits, the same width, and the same
three scores. Nothing here is a new protocol; the point is that the models be
comparable to the ones already reported.

The neighbour mean is a fixed linear map, so it is applied to the whole feature
tensor once and the model that reads it is E.PerNode over 2f inputs. The control
therefore costs what the per-node reference costs.
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
import rewire_test as R  # noqa: E402


# --------------------------------------------------------------------- panel
# pandas parses the date column through np_datetime, which segfaults on this
# interpreter. The column is read as text and converted with numpy instead;
# 1 January 1970 was a Thursday, hence the offset that puts Monday at zero and
# reproduces pandas' dayofweek exactly.
def _load_panel(train_hours):
    df = pd.read_csv(E.CSV)
    dates = np.array(df.pop("date"), dtype="datetime64[s]")
    df = df.drop(columns=[c for c in ("OT",) if c in df.columns])
    span = max(E.LAGS) + train_hours + E.H_VAL + E.H_TEST
    df, dates = df.iloc[-span:], dates[-span:]
    df = df.loc[:, df.std() > 0]
    v = df.to_numpy(dtype=np.float32)

    idx = np.arange(max(E.LAGS), len(df))
    lags = np.stack([v[idx - l] for l in E.LAGS], axis=-1)
    hh = dates.astype("datetime64[h]").astype(np.int64)
    hour = (hh % 24)[idx]
    dow = (((hh // 24) + 3) % 7)[idx]
    cal = np.stack([np.sin(2 * np.pi * hour / 24), np.cos(2 * np.pi * hour / 24),
                    np.sin(2 * np.pi * dow / 7), np.cos(2 * np.pi * dow / 7)], axis=-1)
    cal = np.repeat(cal[:, None, :], v.shape[1], axis=1).astype(np.float32)

    x = np.concatenate([lags, cal], axis=-1)
    y = v[idx]
    n_tr = len(idx) - E.H_VAL - E.H_TEST
    return torch.tensor(x), torch.tensor(y), n_tr, n_tr + E.H_VAL


E.load_panel = _load_panel
import eval_metrics as V  # noqa: E402  (imports ltsf_nodes, whose loader is now patched)

OUT = E.OUT
HIDDEN = E.HIDDEN


def neighbour_mean(x, ei, n):
    """Mean of each node's neighbours' features, the aggregation a convolution performs.

    Edges are stored source-first, so the neighbours of node i are the sources of
    the edges whose target is i. Isolated nodes receive zeros, which the model
    reads as an absent neighbourhood rather than as its own features repeated.
    """
    src, dst = ei[0], ei[1]
    out = torch.zeros(x.shape[0], n, x.shape[2], dtype=x.dtype)
    out.index_add_(1, dst, x[:, src])
    deg = torch.zeros(n, dtype=x.dtype).index_add_(
        0, dst, torch.ones(dst.numel(), dtype=x.dtype))
    return out / deg.clamp(min=1.0).view(1, n, 1)


def run(coll, window, model, seed, conv):
    xz, yz, my, sy, n_tr, n_ev, ei = V.cell(coll, window, seed)
    n, f = xz.shape[1], xz.shape[2]
    h_out = yz.shape[2] if yz.ndim == 3 else 1
    if model == "per_node_nbr_rewired":
        ei = R.rewire(ei, n, 5000 + seed)
    torch.manual_seed(seed)
    if model.startswith("per_node_nbr"):
        # The control doubles the input width of a model that already holds one
        # weight set per node, so a loss on the shortest window could be the price
        # of those parameters and not the fault of the neighbours. The rewired
        # variant pays exactly the same price over a degree-matched random
        # neighbourhood, which separates the two readings.
        xz = torch.cat([xz, neighbour_mean(xz, ei, n)], dim=-1)
        m = E.PerNode(n, 2 * f, HIDDEN, out=h_out)
    elif model == "global":
        m = E.Global(n, f, HIDDEN, out=h_out)
    else:
        raise ValueError(model)
    pred, val = E.fit(m, xz, yz, n_tr, n_ev, seed)
    return dict(V.score(pred, yz[n_ev:], my, sy), val=val, n=n)


if __name__ == "__main__":
    CELLS = (("elec", 264, "SAGEConv"), ("weave", 481, "SAGEConv"),
             ("weave", 60, "SAGEConv"), ("elec", 1440, "ChebConv"))
    MODELS = ("per_node_nbr", "global", "per_node_nbr_rewired")
    keys = [f"{c}|{w}|{mo}|{s}" for s in (0, 1, 2) for c, w, cv in CELLS for mo in MODELS]
    convs = {(c, w): cv for c, w, cv in CELLS}

    path = os.path.join(OUT, "unshared_graph.json")
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
            print(f"  [{q:30}] RMSE_sys {v['rmse_sys']:10.1f}  MAPE_sys {v['mape_sys']:6.2f}%"
                  f"  nRMSE {v['nrmse']:.4f}  ({time.time()-t0:.0f}s)", flush=True)
        except Exception as e:
            print(f"  [{q:30}] ECHEC {type(e).__name__}: {e}", flush=True)
    print("=== termine ===", flush=True)
