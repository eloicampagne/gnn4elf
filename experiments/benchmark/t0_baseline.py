"""t0-gamma (The Forecasting Company) as a covariate-aware foundation-model baseline.

Same protocol as the Chronos-2 rows of extra_baselines.py: zero-shot on the aggregate
load, one 48-step forecast per day-aligned test origin, the covariates known at the
origin (weather and calendar on REGIONS-FR, calendar on WEAVE-UK) passed over the
context and the horizon; point forecast = median quantile. t0-gamma reads at most
1024 steps in one pass, so the context is the last min(1024, origin) steps.

Two settings are scored, fixed before scoring: `default` (no post-processing) and
`recommended` (the four adjustments the model card recommends enabling).

    python t0_baseline.py rfrance
    python t0_baseline.py weave
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
from extra_baselines import H, HERE, OUT, WEAVE, WEATHER, align, calendar, metrics  # noqa: E402

CTX = 1024
SETTINGS = {"default": {},
            "recommended": dict(trim_leading_missing=True, symmetric_averaging=True,
                                clip_nonnegative=True, snap_integers=True)}

_MODEL = None


def model():
    global _MODEL
    if _MODEL is None:
        from t0 import T0Forecaster
        dev = "mps" if torch.backends.mps.is_available() else "cpu"
        _MODEL = T0Forecaster.from_pretrained("theforecastingcompany/t0-gamma").eval().to(dev)
    return _MODEL


def forecast(series, covs, start, n_days, **kw):
    """One median forecast per origin start + d*H, context right-aligned on the origin."""
    m, out = model(), []
    F = np.stack(list(covs.values())) if covs else None              # [F, T]
    for d in range(n_days):
        o = start + d * H
        a = max(0, o - CTX)
        ctx = torch.tensor(series[a:o], dtype=torch.float32)[None]  # [1, T]
        fc = None if F is None else torch.tensor(F[:, a:o + H], dtype=torch.float32)[None]
        with torch.no_grad():
            q = m.predict(ctx, horizon=H, quantile_levels=[0.5], future_covariates=fc, **kw)
        q = q.quantiles if hasattr(q, "quantiles") else q
        out.append(q.float().cpu().numpy()[0, :, 0])
    return np.concatenate(out)


def run(name, series, covs, start, true):
    n_days = len(true) // H
    res, store = {}, {"true": true}
    for s, kw in SETTINGS.items():
        tic = time.time()
        p = forecast(series, covs, start, n_days, **kw)
        res[f"t0-gamma ({s})"] = metrics(p, true); store[f"t0-gamma_{s}"] = p
        print(name, s, res[f"t0-gamma ({s})"], f"({time.time() - tic:.0f}s)", flush=True)
    json.dump({"metrics": res, "context": CTX}, open(os.path.join(OUT, f"{name}_t0.json"), "w"), indent=1)
    np.savez_compressed(os.path.join(OUT, f"{name}_t0_preds.npz"), **store)


def data_rfrance():
    """National load, covariates, first test origin and test targets of REGIONS-FR."""
    true = np.load(os.path.join(HERE, "results_rfrance_v2", "baselines_preds.npz"))["true"]
    df = pd.concat([pd.read_csv(os.path.join(HERE, "data", "rfrance", f)) for f in ("train2.csv", "test2.csv")])
    df["date"] = pd.to_datetime(df["date"])
    df = df.drop_duplicates(["date", "Region"]).sort_values(["date", "Region"])
    load = df.pivot(index="date", columns="Region", values="load").sort_index().dropna()
    nat = load.sum(axis=1)
    cov = df.groupby("date")[WEATHER + ["JourFerie", "Posan"]].mean().reindex(nat.index)
    start = align(nat.to_numpy(), true)
    covs = {k: cov[k].to_numpy(float) for k in WEATHER + ["JourFerie"]}
    covs.update(calendar(nat.index))
    covs["posan_sin"] = np.sin(2 * np.pi * cov["Posan"].to_numpy()); covs["posan_cos"] = np.cos(2 * np.pi * cov["Posan"].to_numpy())
    return nat.to_numpy(float), covs, start, true


def data_weave():
    """Summed load, calendar covariates, first test origin and test targets of WEAVE-UK."""
    true = np.load(os.path.join(HERE, "results_weave_500", "baselines_preds.npz"))["true"]
    df = pd.concat([pd.read_csv(os.path.join(WEAVE, f), usecols=["date", "id_unique", "consumption"])
                    for f in ("train_weave_500.csv", "test_weave_500.csv")])
    df["date"] = pd.to_datetime(df["date"], utc=True)
    nat = df.pivot_table(index="date", columns="id_unique", values="consumption").sort_index().sum(axis=1)
    start = align(nat.to_numpy(), true)
    return nat.to_numpy(float), calendar(nat.index), start, true


if __name__ == "__main__":
    name = sys.argv[1]
    run(name, *{"rfrance": data_rfrance, "weave": data_weave}[name]())
