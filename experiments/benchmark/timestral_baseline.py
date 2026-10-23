"""Timestral (Mistral AI) as a covariate-aware foundation-model baseline.

Same protocol as t0_baseline.py and the Chronos-2 rows of extra_baselines.py: zero-shot
on the aggregate load, one 48-step forecast per day-aligned test origin, the covariates
known at the origin passed over the context and the horizon; point forecast = median.
Package defaults are kept (symmetric averaging and non-negative clipping on).

Timestral reads up to 32768 steps, so the context length is itself a variable: the
full-context run uses min(32768, origin) steps, the short run the 1024 steps t0-gamma reads.

    python timestral_baseline.py rfrance [209M|123M|28M|5M]
    python timestral_baseline.py weave   [209M|...]
"""

import json
import os
import sys
import time

import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from _paths import ROOT, DATA, CACHE  # noqa: E402,F401  (puts every experiment folder on sys.path)
from extra_baselines import H, OUT, metrics  # noqa: E402
from t0_baseline import data_rfrance, data_weave  # noqa: E402

CONTEXTS = {"1024": 1024}   # the full 32768-step context costs about 1 min per origin on CPU (362 origins); not run


def forecast(model, series, covs, start, n_days, ctx_len):
    F = np.stack(list(covs.values())) if covs else None              # [F, T]
    out = []
    for d in range(n_days):
        o = start + d * H
        a = max(0, o - ctx_len)
        ctx = series[a:o][None].astype(np.float32)                  # [1, T]
        kc = None if F is None else F[:, a:o + H][None].astype(np.float32)
        with torch.no_grad():
            fc = model.forecast(ctx, prediction_length=H, known_covariates=kc)
        out.append(np.asarray(fc.median, dtype=float).reshape(-1)[:H])
    return np.concatenate(out)


if __name__ == "__main__":
    from timestral import TimestralForecaster
    name, size = sys.argv[1], (sys.argv[2] if len(sys.argv) > 2 else "209M")
    series, covs, start, true = {"rfrance": data_rfrance, "weave": data_weave}[name]()
    model = TimestralForecaster.from_pretrained(f"mistralai/Timestral-{size}", device="cpu")   # scaler needs float64, absent on MPS
    res, store = {}, {"true": true}
    for c, L in CONTEXTS.items():
        tic = time.time()
        p = forecast(model, series, covs, start, len(true) // H, L)
        res[f"Timestral-{size} (context {c})"] = metrics(p, true); store[f"Timestral-{size}_{c}"] = p
        print(name, size, c, res[f"Timestral-{size} (context {c})"], f"({time.time() - tic:.0f}s)", flush=True)
    json.dump({"metrics": res}, open(os.path.join(OUT, f"{name}_timestral_{size}.json"), "w"), indent=1)
    np.savez_compressed(os.path.join(OUT, f"{name}_timestral_{size}_preds.npz"), **store)
