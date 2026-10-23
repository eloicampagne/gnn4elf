"""t0-gamma plus a linear correction of its residuals on REGIONS-FR.

A plain AR on the residuals only reacts the day after an error, so the correction is an
ARX: for each half-hour h, the residual of day D is regressed on the residual of day D-1
at h (known at the origin), and on calendar indicators known in advance: holiday on D,
on D-1 and on D+1, day type, and the day after a clock change. It is fitted on the
t0-gamma residuals of 2018 (validation year) and applied unchanged to 2019, so the test
year plays no part in the fit.

    python fm_residual_correction.py      # needs the t0env (tfc-t0) for the 2018 forecasts
"""

import json
import os

import numpy as np
import pandas as pd

import t0_baseline as T
from extra_baselines import H, HERE, OUT, metrics

ALPHA = 1.0                                            # ridge penalty, fixed before scoring


def daily_calendar(days):
    df = pd.concat([pd.read_csv(os.path.join(HERE, "data", "rfrance", f), usecols=["date", "Region", "JourFerie", "DayType"])
                    for f in ("train2.csv", "test2.csv")])
    df["date"] = pd.to_datetime(df["date"])
    cal = df.groupby(df["date"].dt.normalize()).agg(hol=("JourFerie", "max"), dt=("DayType", "first"))
    allday = pd.date_range(days.min() - pd.Timedelta(days=1), days.max() + pd.Timedelta(days=1))
    hol = cal["hol"].reindex(allday).fillna(0).astype(float)
    X = pd.DataFrame(index=days)
    X["hol"], X["hol_prev"], X["hol_next"] = hol.reindex(days).values, hol.shift(1).reindex(days).values, hol.shift(-1).reindex(days).values
    dst = [pd.Timestamp(f"{y}-{m}-01") + pd.offsets.LastWeekOfMonth(weekday=6) + pd.Timedelta(days=1)
           for y in (2018, 2019) for m in (3, 10)]
    X["after_dst"] = days.normalize().isin(pd.DatetimeIndex(dst)).astype(float)
    dt = cal["dt"].reindex(days).values
    for v in (0, 1, 5, 6, 14, 30, 33):
        X[f"dt_{v}"] = (dt == v).astype(float)
    return X


def main():
    series, covs, start, true = T.data_rfrance()
    d2 = pd.concat([pd.read_csv(os.path.join(HERE, "data", "rfrance", f), usecols=["date", "Region", "load"])
                    for f in ("train2.csv", "test2.csv")])
    d2["date"] = pd.to_datetime(d2["date"]); d2 = d2.drop_duplicates(["date", "Region"])
    ix = d2.pivot(index="date", columns="Region", values="load").sort_index().dropna().index
    s18 = ix.searchsorted(pd.Timestamp("2018-01-01"))
    n18, n19 = (start - s18) // H, len(true) // H
    p18 = T.forecast(series, covs, s18, n18)                              # t0-gamma, default setting
    p19 = np.load(os.path.join(OUT, "rfrance_t0_preds.npz"))["t0-gamma_default"]
    r18 = series[s18:start] - p18
    r = np.concatenate([r18, true - p19]).reshape(-1, H)                  # residuals, 2018 then 2019
    days = ix[s18::H][:n18 + n19]
    C = daily_calendar(days).to_numpy()
    prev = np.vstack([np.zeros(H), r[:-1]])                               # residual of the previous day
    fit, test = np.arange(1, n18), np.arange(n18, n18 + n19)
    corr = np.zeros((n19, H))
    for h in range(H):
        X = np.column_stack([np.ones(len(days)), prev[:, h], C])
        A, b = X[fit], r[fit, h]
        w = np.linalg.solve(A.T @ A + ALPHA * np.diag([0] + [1] * (X.shape[1] - 1)), A.T @ b)
        corr[:, h] = X[test] @ w
    p = p19 + corr.reshape(-1)
    res = {"t0-gamma": metrics(p19, true), "t0-gamma + ARX residual correction": metrics(p, true)}
    print(json.dumps(res, indent=1))
    json.dump(res, open(os.path.join(OUT, "rfrance_t0_arx.json"), "w"), indent=1)
    np.save(os.path.join(OUT, "rfrance_t0_arx_pred.npy"), p)


if __name__ == "__main__":
    main()
