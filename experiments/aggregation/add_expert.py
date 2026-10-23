"""Add one expert to a cell without retraining the others.

The cells of aggreg_experts.py are reproducible: the panel, the split, the
standardisation and the seed are functions of the dataset and the setting alone,
so an architecture trained afterwards lands on the same forecasts it would have
produced inside the original run. This script trains one and appends its column
to experts_<dataset>_<setting>.csv and to nodal_<dataset>_<setting>.npz.

    python add_expert.py rfrance single GatedGraph GatedGraphConv dtw 2 256 1e-3

The configuration is the one chosen on the validation months by try_convs.py,
never on the test year.
"""

import os
import sys
import time

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from _paths import ROOT, DATA, CACHE  # noqa: E402,F401  (puts every experiment folder on sys.path)
import aggreg_experts as A  # noqa: E402


def build_cell(dataset, setting):
    """Exactly what aggreg_experts.main builds, up to the training call."""
    xn, cal, y, nodes, dates = A.read_real()
    itr_all = (dates >= np.datetime64(A.TRAIN_FROM)) & (dates < np.datetime64(A.TRAIN_TO))
    if dataset in {"synth", "synth_structured"}:
        A.NODE_FEATS = A.FEATS_SYNTH
        noise = "graph" if dataset == "synth_structured" else A.SYNTH_NOISE
        xn, y = A.make_synth(xn, cal, y, dates, itr_all, noise, A.SEED)
    x, tgt, when = A.build(setting, xn, cal, y)
    itr, iva, ite = A.split(dates, when, setting)
    mu, sd = x[itr].mean(0), x[itr].std(0)
    sd = np.where(sd < 1e-8, 1.0, sd)
    xz = torch.tensor((x - mu) / sd, dtype=torch.float32)
    my, sy = tgt[itr].mean(0), tgt[itr].std(0)
    sy = np.where(sy < 1e-8, 1.0, sy)
    yz = torch.tensor((tgt - my) / sy, dtype=torch.float32)
    return xz, yz, tgt, my, sy, itr, iva, ite, len(nodes)


def main(dataset, setting, name, cfg):
    xz, yz, tgt, my, sy, itr, iva, ite, n = build_cell(dataset, setting)
    h_out = tgt.shape[2] if tgt.ndim == 3 else 1
    t0 = time.time()
    torch.manual_seed(A.SEED)
    m = A.GraphNet(xz.shape[2], cfg, out=h_out)
    v = A.fit(m, xz, yz, itr, iva, cfg["lr"], A.SEED)
    p = A.predict(m, xz[ite]) * sy + my
    nat = A.national(p, setting)
    truth = A.national(tgt[ite], setting)
    print(f"{dataset}/{setting} {name}: val {v:.4f}  national RMSE "
          f"{np.sqrt(((nat - truth) ** 2).mean()):8.1f} MW ({time.time() - t0:.0f}s)",
          flush=True)

    csv = os.path.join(A.OUT, f"experts_{dataset}_{setting}.csv")
    df = pd.read_csv(csv, index_col=0)
    load = df.pop("load")
    df[name] = nat
    df["load"] = load
    df.to_csv(csv)

    # The per-node file exists only where a map is drawn, that is on the French
    # cells; the synthetic panel carries no geography and keeps the sums alone.
    npz = os.path.join(A.OUT, f"nodal_{dataset}_{setting}.npz")
    if os.path.exists(npz):
        z = dict(np.load(npz, allow_pickle=True))
        col = p.transpose(0, 2, 1).reshape(-1, p.shape[1]) if p.ndim == 3 else p
        z["experts"] = np.append(z["experts"], name)
        z["pred"] = np.concatenate([z["pred"], col[:, :, None]], 2)
        np.savez_compressed(npz, **z)
    print("  colonnes ecrites dans", os.path.basename(csv),
          "et " + os.path.basename(npz) if os.path.exists(npz) else "", flush=True)


if __name__ == "__main__":
    ds, st, name, conv, adj, layers, hidden, lr = sys.argv[1:9]
    cfg = dict(conv=conv, adj=adj, layers=int(layers), hidden=int(hidden), lr=float(lr))
    main(ds, st, name, cfg)
