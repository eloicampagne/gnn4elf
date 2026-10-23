"""Does the twelve-node verdict hold when the graph grows?

The French diagnostic compares a per-node reference against graph models on
twelve regions, which is too few for the count of nodes to be separated from
everything else. Here the same comparison runs on the electricity collection of
the cascaded-transfer paper, with the number of series drawn down from three
hundred and twenty to ten and every other quantity held fixed.

Protocol of that paper: the target is predicted from lags 1, 2, 3, 24, 25, 48 and
168 together with a cyclical encoding of the hour and the weekday, eleven days of
training and four days of test at the end of the record, series with a constant
target dropped. Two days of validation are inserted before the test, which the
paper does not need and this comparison does: the reference picks its width there
and the graph its operator, and selecting one on validation while the other is
fixed is what made the earlier reading unfair.

The four models differ by two properties and nothing else, whether a node sees
its neighbours and whether the weights are shared across nodes. Node-specific
weights are held in one batched parameter rather than in N separate models, so
the reference costs what the shared model costs at three hundred and twenty
nodes instead of three hundred and twenty times more.
"""

import json
import os
import sys
import time

import numpy as np
import pandas as pd
import torch
from torch import nn

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from _paths import ROOT, DATA, CACHE  # noqa: E402,F401  (puts every experiment folder on sys.path)
from graphtoolbox.models import myGNN  # noqa: E402
import torch_geometric.nn.conv as pyg_conv  # noqa: E402

CSV = os.path.join(DATA, "electricity", "electricity.csv")
OUT = os.path.join(ROOT, "results_ltsf_nodes")
os.makedirs(OUT, exist_ok=True)

LAGS = [1, 2, 3, 24, 25, 48, 168]
H_TEST, H_VAL = 4 * 24, 2 * 24
EPOCHS, PATIENCE, LR, BATCH = 300, 30, 1e-3, 32
HIDDEN = 64
DEV = torch.device("cpu")


# --------------------------------------------------------------------------- data
def load_panel(train_hours):
    """Feature and target tensors of shape [T, N, F] and [T, N], plus the split points."""
    df = pd.read_csv(CSV, parse_dates=["date"]).set_index("date")
    df = df.drop(columns=[c for c in ("OT",) if c in df.columns])
    span = max(LAGS) + train_hours + H_VAL + H_TEST
    df = df.iloc[-span:]
    df = df.loc[:, df.std() > 0]                      # constant series dropped
    v = df.to_numpy(dtype=np.float32)                 # [T, N]

    idx = np.arange(max(LAGS), len(df))
    lags = np.stack([v[idx - l] for l in LAGS], axis=-1)          # [T', N, 7]
    hour = df.index.hour.to_numpy()[idx]
    dow = df.index.dayofweek.to_numpy()[idx]
    cal = np.stack([np.sin(2 * np.pi * hour / 24), np.cos(2 * np.pi * hour / 24),
                    np.sin(2 * np.pi * dow / 7), np.cos(2 * np.pi * dow / 7)], axis=-1)
    cal = np.repeat(cal[:, None, :], v.shape[1], axis=1).astype(np.float32)  # [T', N, 4]

    x = np.concatenate([lags, cal], axis=-1)
    y = v[idx]
    n_tr = len(idx) - H_VAL - H_TEST
    return torch.tensor(x), torch.tensor(y), n_tr, n_tr + H_VAL


def standardise(x, y, n_tr):
    """Per-node moments taken on the training window alone."""
    mx, sx = x[:n_tr].mean(0), x[:n_tr].std(0)
    my, sy = y[:n_tr].mean(0), y[:n_tr].std(0)
    dead_x, dead_y = sx < 1e-8, sy < 1e-8
    xz = (x - mx) / torch.where(dead_x, torch.ones_like(sx), sx)
    yz = (y - my) / torch.where(dead_y, torch.ones_like(sy), sy)
    xz[:, dead_x] = 0.0
    return xz, yz, my, sy


def knn_graph(y_train, k=5):
    """Correlation over the training window, kept to the k strongest neighbours."""
    c = np.corrcoef(y_train.T.numpy())
    np.fill_diagonal(c, -np.inf)
    n = c.shape[0]
    src, dst, w = [], [], []
    for i in range(n):
        for j in np.argsort(-c[i])[:min(k, n - 1)]:
            src += [j, i]; dst += [i, j]; w += [max(c[i, j], 0.0)] * 2
    ei = torch.tensor([src, dst], dtype=torch.long)
    return ei, torch.tensor(w, dtype=torch.float32)


# ------------------------------------------------------------------------- models
class PerNode(nn.Module):
    """Node-specific weights, own features only, all nodes in one batched tensor."""

    def __init__(self, n, f, h, out=1):
        super().__init__()
        self.out = out
        self.w1 = nn.Parameter(torch.randn(n, f, h) * (1 / f) ** 0.5)
        self.b1 = nn.Parameter(torch.zeros(n, h))
        self.w2 = nn.Parameter(torch.randn(n, h, out) * (1 / h) ** 0.5)
        self.b2 = nn.Parameter(torch.zeros(n, out))

    def forward(self, x):                                  # x [B, N, F]
        h = torch.relu(torch.einsum("bnf,nfh->bnh", x, self.w1) + self.b1)
        o = torch.einsum("bnh,nho->bno", h, self.w2) + self.b2
        return o.squeeze(-1) if self.out == 1 else o


class Shared(nn.Module):
    """One weight set for every node, own features only."""

    def __init__(self, f, h, out=1):
        super().__init__()
        self.out = out
        self.net = nn.Sequential(nn.Linear(f, h), nn.ReLU(), nn.Linear(h, out))

    def forward(self, x):
        o = self.net(x)
        return o.squeeze(-1) if self.out == 1 else o


class Global(nn.Module):
    """One model reading every node's features at once."""

    def __init__(self, n, f, h, out=1):
        super().__init__()
        self.n, self.out = n, out
        self.net = nn.Sequential(nn.Linear(n * f, h), nn.ReLU(), nn.Linear(h, n * out))

    def forward(self, x):
        o = self.net(x.flatten(1))
        return o if self.out == 1 else o.view(-1, self.n, self.out)


class Graph(nn.Module):
    """GraphToolbox's myGNN over the correlation graph."""

    def __init__(self, f, h, conv, ei, ew, out=1):
        super().__init__()
        self.out = out
        self.gnn = myGNN(in_channels=f, num_layers=2, hidden_channels=h, out_channels=out,
                         conv_class=getattr(pyg_conv, conv), conv_kwargs={"K": 3} if conv == "ChebConv" else {},
                         heads=1)
        self.register_buffer("ei", ei)
        if ew is not None:
            self.register_buffer("ew", ew)

    def forward(self, x):                                  # x [B, N, F]
        o = torch.stack([self.gnn(x[b], self.ei) for b in range(x.shape[0])])
        return o.squeeze(-1) if self.out == 1 else o


# ------------------------------------------------------------------------ training
def fit(model, x, y, n_tr, n_va, seed):
    torch.manual_seed(seed)
    opt = torch.optim.Adam(model.parameters(), lr=LR)
    lf = nn.MSELoss()
    xt, yt = x[:n_tr], y[:n_tr]
    xv, yv = x[n_tr:n_va], y[n_tr:n_va]
    best, state, bad = float("inf"), None, 0
    for _ in range(EPOCHS):
        model.train()
        perm = torch.randperm(len(xt))
        for k in range(0, len(xt), BATCH):
            i = perm[k:k + BATCH]
            opt.zero_grad(); lf(model(xt[i]), yt[i]).backward(); opt.step()
        model.eval()
        with torch.no_grad():
            vl = float(lf(model(xv), yv))
        if vl < best - 1e-6:
            best, state, bad = vl, {k: v.clone() for k, v in model.state_dict().items()}, 0
        else:
            bad += 1
            if bad >= PATIENCE:
                break
    model.load_state_dict(state)
    model.eval()
    with torch.no_grad():
        pred = model(x[n_va:])
    return pred, best


def run(n_nodes, train_hours, model_name, seed):
    x, y, n_tr, n_va = load_panel(train_hours)
    rng = np.random.default_rng(1000 + seed)
    keep = sorted(rng.choice(x.shape[1], size=min(n_nodes, x.shape[1]), replace=False).tolist())
    x, y = x[:, keep], y[:, keep]
    x, y, my, sy = standardise(x, y, n_tr)
    n, f = x.shape[1], x.shape[2]

    if model_name == "per_node":
        m = PerNode(n, f, HIDDEN)
    elif model_name == "shared":
        m = Shared(f, HIDDEN)
    elif model_name == "global":
        m = Global(n, f, HIDDEN)
    else:
        ei, ew = knn_graph(y[:n_tr])
        m = Graph(f, HIDDEN, model_name.split("_")[1], ei, ew)

    pred, val = fit(m.to(DEV), x, y, n_tr, n_va, seed)
    # Per-node RMSE on the node's own scale, then normalised by its training spread,
    # so a large client does not decide the average on its own.
    err = ((pred - y[n_va:]) ** 2).mean(0).sqrt()
    return {"nrmse": float(err.mean()), "per_node": err.tolist(), "val": val, "n": n}


if __name__ == "__main__":
    MODELS = ["per_node", "shared", "gnn_SAGEConv", "gnn_ChebConv"]
    keys = []
    for n in (10, 20, 40, 80, 160, 320):
        for s in (0, 1, 2):
            for mo in MODELS + (["global"] if n <= 40 else []):
                keys.append(f"{n}|264|{mo}|{s}")
    # Second training window, to separate "many nodes" from "short history".
    for n in (40, 320):
        for s in (0, 1, 2):
            for mo in MODELS:
                keys.append(f"{n}|1440|{mo}|{s}")

    path = os.path.join(OUT, "electricity_nodes.json")
    res = json.load(open(path)) if os.path.exists(path) else {}
    todo = [k for k in keys if k not in res]
    print(f"{len(res)} en cache, {len(todo)} à calculer", flush=True)
    for k in todo:
        n, hrs, mo, s = k.split("|")
        t0 = time.time()
        try:
            res[k] = run(int(n), int(hrs), mo, int(s))
            json.dump(res, open(path, "w"))
            print(f"  [{k:26}] nRMSE {res[k]['nrmse']:.4f}  ({time.time()-t0:.0f}s)", flush=True)
        except Exception as e:
            print(f"  [{k:26}] ECHEC {type(e).__name__}: {e}", flush=True)
    print("=== terminé ===", flush=True)
