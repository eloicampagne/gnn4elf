"""The same node-count comparison on WEAVE-UK, five hundred feeders.

Electricity gives the count of series up to three hundred and twenty on hourly
client demand. WEAVE gives five hundred feeders at the half hour, with a day of
forty-eight slots as the target and an adjacency already built for the
cascaded-transfer experiments, so the two collections differ in domain,
resolution and graph while asking the same question.

The cache holds four hundred and eighty-one training days and one hundred and
forty-five test days, which allows the training window to be varied as well: the
French diagnostic reads its verdict at one thousand and ninety days and reverses
part of it at fifty-four, so both are worth revisiting where the nodes are many.

Seeding note, 31 August 2026. Until today this script left the weight initialisation
to fit(), which seeds after the model is built, so the seed did not control it. Two
runs of the sweep disagree by up to 0.077 nRMSE for that reason: compare
results_ltsf_nodes/weave_nodes.json with weave_nodes_metrics.json on the same keys.
The tables of the manuscript were produced before the fix below and are not
reproduced by running this file now. Rerunning the sweep from an empty cache and
re-deriving those tables is the correct repair; it is deferred, and the manuscript
states instead that differences of the order of the reported dispersion are not
claimed as orderings.
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
from ltsf_nodes import PerNode, Shared, Global, Graph, fit  # noqa: E402

CACHE = os.path.join(CACHE, "weave500_tasks.npz")
OUT = os.path.join(ROOT, "results_ltsf_nodes")
os.makedirs(OUT, exist_ok=True)
HIDDEN = 64
_Z = None


def panel():
    global _Z
    if _Z is None:
        _Z = np.load(CACHE, allow_pickle=True)
    return _Z


def edges_from(A, keep, k=5):
    """The cached adjacency restricted to the drawn nodes, kept to k neighbours."""
    W = A[np.ix_(keep, keep)].copy()
    np.fill_diagonal(W, 0.0)
    src, dst = [], []
    for i in range(len(keep)):
        for j in np.argsort(-W[i])[:min(k, len(keep) - 1)]:
            if W[i, j] > 0:
                src += [j, i]; dst += [i, j]
    if not src:                       # a draw can isolate every node
        src, dst = list(range(len(keep))), list(range(len(keep)))
    return torch.tensor([src, dst], dtype=torch.long)


def run(n_nodes, train_days, model_name, seed):
    z = panel()
    rng = np.random.default_rng(1000 + seed)
    n_all = z["Xtr"].shape[1]
    keep = sorted(rng.choice(n_all, size=min(n_nodes, n_all), replace=False).tolist())

    xtr = torch.tensor(z["Xtr"][-train_days:, keep, :])
    ytr = torch.tensor(z["Ytr"][-train_days:, keep, :])
    xte = torch.tensor(z["Xte"][:, keep, :])
    yte = torch.tensor(z["Yte"][:, keep, :])

    # The cache carries no validation split; the last fifth of the training days
    # is held out, which keeps the selection out of the test window.
    n_va = max(8, len(xtr) // 5)
    x = torch.cat([xtr, xte]); y = torch.cat([ytr, yte])
    n_tr = len(xtr) - n_va
    n_val_end = len(xtr)

    mx, sx = x[:n_tr].mean(0), x[:n_tr].std(0)
    my, sy = y[:n_tr].mean(0), y[:n_tr].std(0)
    dead_x, dead_y = sx < 1e-8, sy < 1e-8
    x = (x - mx) / torch.where(dead_x, torch.ones_like(sx), sx)
    y = (y - my) / torch.where(dead_y, torch.ones_like(sy), sy)
    x[:, dead_x] = 0.0

    # fit() seeds the generator, but only after the caller has built the model, and
    # PerNode draws its weights with torch.randn in __init__. Left to fit(), the
    # initialisation would follow whatever state the process happens to hold, which
    # depends on how many cells ran before it and therefore on how full the JSON
    # cache was when the script started. Seed here so the seed controls the weights.
    torch.manual_seed(seed)

    n, f, h_out = x.shape[1], x.shape[2], y.shape[2]
    if model_name == "per_node":
        m = PerNode(n, f, HIDDEN, out=h_out)
    elif model_name == "shared":
        m = Shared(f, HIDDEN, out=h_out)
    elif model_name == "global":
        m = Global(n, f, HIDDEN, out=h_out)
    else:
        m = Graph(f, HIDDEN, model_name.split("_")[1], edges_from(z["A"], keep), None, out=h_out)

    pred, val = fit(m, x, y, n_tr, n_val_end, seed)
    err = ((pred - y[n_val_end:]) ** 2).mean((0, 2)).sqrt()
    return {"nrmse": float(err.mean()), "val": val, "n": n, "days": int(n_tr)}


if __name__ == "__main__":
    MODELS = ["per_node", "shared", "gnn_SAGEConv", "gnn_ChebConv"]
    keys = []
    for n in (10, 40, 160, 500):
        for s in (0, 1, 2):
            for mo in MODELS + (["global"] if n <= 40 else []):
                keys.append(f"{n}|481|{mo}|{s}")
    for n in (40, 500):                       # short window, the regime that reverses
        for s in (0, 1, 2):
            for mo in MODELS:
                keys.append(f"{n}|60|{mo}|{s}")

    path = os.path.join(OUT, "weave_nodes.json")
    res = json.load(open(path)) if os.path.exists(path) else {}
    todo = [k for k in keys if k not in res]
    print(f"{len(res)} en cache, {len(todo)} à calculer", flush=True)
    for k in todo:
        n, d, mo, s = k.split("|")
        t0 = time.time()
        try:
            res[k] = run(int(n), int(d), mo, int(s))
            json.dump(res, open(path, "w"))
            print(f"  [{k:26}] nRMSE {res[k]['nrmse']:.4f}  ({time.time()-t0:.0f}s)", flush=True)
        except Exception as e:
            print(f"  [{k:26}] ECHEC {type(e).__name__}: {e}", flush=True)
    print("=== terminé ===", flush=True)
