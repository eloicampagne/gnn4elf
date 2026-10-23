"""Two covariate-aware baselines for the architecture benchmark.

The round-one associate editor asked for an ARIMA with exogenous calendar and
weather inputs and for a time-series foundation model that reads covariates. This
script adds both to the benchmark of benchmark_rfrance.py and benchmark_weave500.py,
on the same test days and stride, scored on the summed (national) forecast.

The reported scheme is fixed before scoring, to match the rows already in the table:

  * ARIMA-X, REGIONS-FR only: one SARIMAX per half-hour on the daily national
    subseries, exogenous inputs = observed weather (averaged over regions) and
    calendar; order chosen by auto_arima (AIC) on 2015-2017, parameters estimated
    on 2015-2018, then frozen and applied to 2019 for one-day-ahead forecasts.
    WEAVE-UK has no weather and 9 training days, so its ARIMA-X (arimax_weave) is a
    single model on the half-hourly summed load with daily Fourier terms and a weekend
    indicator, its design chosen on the validation days (23-24 Feb).
    The three nonlinear REGIONS-FR designs of arimax_select were added after the linear
    design had been scored on 2019; the choice among the four is made on 2018 only.
  * Chronos-2, both collections: zero-shot on the aggregate load, one 48-step
    forecast per day-aligned origin, the covariates known at the origin passed as
    past and future covariates (weather and calendar on REGIONS-FR, calendar on
    WEAVE-UK); point forecast = median quantile.

Bottom-up variants (one model per region or feeder, then summed) are computed as
side information and stored under their own keys.

    python3 extra_baselines.py rfrance
    python3 extra_baselines.py arimax          # REGIONS-FR design selection on 2018
    python3 extra_baselines.py weave
    python3 extra_baselines.py arimax_weave    # WEAVE-UK ARIMA-X
    python3 extra_baselines.py multi rfrance   # side information
"""

import json
import os
import sys
import time
import warnings

import numpy as np
import pandas as pd
import torch

warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from _paths import ROOT, DATA, CACHE  # noqa: E402,F401  (puts every experiment folder on sys.path)
HERE = ROOT
WEAVE = os.path.join(DATA, "weave")
H = 48
OUT = os.path.join(HERE, "results_extra_baselines")
os.makedirs(OUT, exist_ok=True)

WEATHER = ["temp", "nebu", "wind", "tempMax", "tempMin", "temp_liss_fort", "temp_liss_faible"]


def metrics(pred, true):
    pred, true = np.asarray(pred, float), np.asarray(true, float)
    return float(np.sqrt(np.mean((pred - true) ** 2))), float(100 * np.mean(np.abs((pred - true) / true)))


def align(series, true):
    """Index at which `true` starts inside `series` (exact match on the summed load)."""
    n = len(true)
    for i in range(len(series) - n + 1):
        if abs(series[i] - true[0]) < 1e-6 and np.allclose(series[i:i + n], true):
            return i
    raise ValueError("test window not found in the series")


def calendar(dates):
    d = pd.DatetimeIndex(dates)
    inst = (d.hour * 2 + d.minute // 30).to_numpy()
    wd = d.dayofweek.to_numpy()
    return {"inst_sin": np.sin(2 * np.pi * inst / H), "inst_cos": np.cos(2 * np.pi * inst / H),
            "wd_sin": np.sin(2 * np.pi * wd / 7), "wd_cos": np.cos(2 * np.pi * wd / 7)}


def chronos2(tasks, batch_size=64):
    from chronos.chronos2 import Chronos2Pipeline
    dev = "mps" if torch.backends.mps.is_available() else "cpu"
    pipe = Chronos2Pipeline.from_pretrained("amazon/chronos-2", device_map=dev)
    qs = list(pipe.quantiles)
    mid = qs.index(min(qs, key=lambda q: abs(q - 0.5)))
    out = pipe.predict(tasks, prediction_length=H, batch_size=batch_size)
    return np.stack([o[0, mid].float().cpu().numpy() for o in out])          # [n_tasks, H]


# ---------------------------------------------------------------- REGIONS-FR
def rfrance():
    true = np.load(os.path.join(HERE, "results_rfrance_v2", "baselines_preds.npz"))["true"]
    df = pd.concat([pd.read_csv(os.path.join(HERE, "data", "rfrance", f)) for f in ("train2.csv", "test2.csv")])
    df["date"] = pd.to_datetime(df["date"])
    df = df.drop_duplicates(["date", "Region"]).sort_values(["date", "Region"])
    load = df.pivot(index="date", columns="Region", values="load").sort_index()
    load = load.dropna()
    nat = load.sum(axis=1)
    cov = df.groupby("date")[WEATHER + ["JourFerie", "Posan"]].mean().reindex(nat.index)
    dates = nat.index
    start = align(nat.to_numpy(), true)
    n_days = len(true) // H
    assert dates[start].hour == 0 and dates[start].minute == 0, "origins must fall on day boundaries"
    res, store = {}, {"true": true}

    # ARIMA-X on the national series, one model per half-hour on its daily subseries
    from pmdarima import auto_arima
    from statsmodels.tsa.statespace.sarimax import SARIMAX
    cal = calendar(dates)
    wd = pd.get_dummies(pd.DatetimeIndex(dates).dayofweek, prefix="wd", drop_first=True).to_numpy(float)
    X = np.column_stack([cov[WEATHER].to_numpy(), cov["JourFerie"].to_numpy(), wd,
                         np.sin(2 * np.pi * cov["Posan"].to_numpy()), np.cos(2 * np.pi * cov["Posan"].to_numpy())])
    y = nat.to_numpy(float)
    t_fit_end = np.searchsorted(dates, pd.Timestamp("2018-01-01"))   # order selection on 2015-2017
    t_est_end = start                                                  # estimation on 2015-2018
    t0 = np.searchsorted(dates, pd.Timestamp("2015-01-01"))
    pred = np.zeros(len(true)); orders = {}
    tic = time.time()
    for h in range(H):
        idx = np.arange(t0, len(y))
        idx = idx[((idx - start) % H) == h]                            # half-hour h of every day
        yi, Xi = y[idx], X[idx]
        mu, sd = Xi[idx < t_est_end].mean(0), Xi[idx < t_est_end].std(0) + 1e-9
        Xi = (Xi - mu) / sd
        sel = idx < t_fit_end
        try:
            order = auto_arima(yi[sel], X=Xi[sel], seasonal=False, stepwise=True, max_p=3, max_q=3, max_d=1,
                               information_criterion="aic", error_action="ignore", suppress_warnings=True).order
        except Exception:
            order = (1, 0, 1)
        orders[h] = list(order)
        est = idx < t_est_end
        trend = "c" if order[1] == 0 else "n"
        fit = SARIMAX(yi[est], exog=Xi[est], order=order, trend=trend).fit(disp=False, maxiter=200)
        full = fit.apply(yi, exog=Xi, refit=False)
        one_step = full.get_prediction(start=int(est.sum()), end=len(yi) - 1, dynamic=False).predicted_mean
        tst = np.arange(n_days)
        pred[tst * H + h] = np.asarray(one_step)[:n_days]
        print(f"  ARIMA-X h={h:2d} order={order}  ({time.time() - tic:.0f}s)", flush=True)
    res["ARIMA-X"] = metrics(pred, true); store["ARIMA-X"] = pred
    print("ARIMA-X", res["ARIMA-X"], flush=True)

    # Chronos-2 on the national series, covariates known at the origin
    keys = WEATHER + ["JourFerie"]
    feat = {k: cov[k].to_numpy(float) for k in keys}
    feat.update(calendar(dates))
    feat["posan_sin"] = np.sin(2 * np.pi * cov["Posan"].to_numpy()); feat["posan_cos"] = np.cos(2 * np.pi * cov["Posan"].to_numpy())

    def tasks_for(series, covs):
        out = []
        for d in range(n_days):
            o = start + d * H
            out.append({"target": series[:o].astype(np.float32),
                        "past_covariates": {k: v[:o].astype(np.float32) for k, v in covs.items()},
                        "future_covariates": {k: v[o:o + H].astype(np.float32) for k, v in covs.items()}})
        return out

    tic = time.time()
    p = chronos2(tasks_for(y, feat)).reshape(-1)
    res["Chronos-2"] = metrics(p, true); store["Chronos-2"] = p
    print("Chronos-2", res["Chronos-2"], f"({time.time() - tic:.0f}s)", flush=True)

    # side information: Chronos-2 bottom-up, one series per region with its own weather
    tic = time.time()
    bu = np.zeros(len(true))
    for reg in load.columns:
        sub = df[df.Region == reg].set_index("date").reindex(dates)
        covs = {k: sub[k].to_numpy(float) for k in keys}
        covs.update(calendar(dates))
        covs["posan_sin"] = feat["posan_sin"]; covs["posan_cos"] = feat["posan_cos"]
        bu += chronos2(tasks_for(load[reg].to_numpy(float), covs)).reshape(-1)
    res["Chronos-2 (bottom-up)"] = metrics(bu, true); store["Chronos-2_bottom-up"] = bu
    print("Chronos-2 bottom-up", res["Chronos-2 (bottom-up)"], f"({time.time() - tic:.0f}s)", flush=True)

    json.dump({"metrics": res, "arimax_orders": orders}, open(os.path.join(OUT, "rfrance.json"), "w"), indent=1)
    np.savez_compressed(os.path.join(OUT, "rfrance_preds.npz"), **store)


# ---------------------------------------------------------------- WEAVE-UK
def weave():
    true = np.load(os.path.join(HERE, "results_weave_500", "baselines_preds.npz"))["true"]
    df = pd.concat([pd.read_csv(os.path.join(WEAVE, f), usecols=["date", "id_unique", "consumption"])
                    for f in ("train_weave_500.csv", "test_weave_500.csv")])
    df["date"] = pd.to_datetime(df["date"], utc=True)
    load = df.pivot_table(index="date", columns="id_unique", values="consumption").sort_index()
    nat = load.sum(axis=1)
    dates = nat.index
    start = align(nat.to_numpy(), true)
    n_days = len(true) // H
    feat = calendar(dates)
    res, store = {}, {"true": true}

    def tasks_for(series):
        out = []
        for d in range(n_days):
            o = start + d * H
            out.append({"target": series[:o].astype(np.float32),
                        "past_covariates": {k: v[:o].astype(np.float32) for k, v in feat.items()},
                        "future_covariates": {k: v[o:o + H].astype(np.float32) for k, v in feat.items()}})
        return out

    tic = time.time()
    p = chronos2(tasks_for(nat.to_numpy(float))).reshape(-1)
    res["Chronos-2"] = metrics(p, true); store["Chronos-2"] = p
    print("Chronos-2", res["Chronos-2"], f"({time.time() - tic:.0f}s)", flush=True)

    tic = time.time()
    tasks = [t for c in load.columns for t in tasks_for(load[c].to_numpy(float))]
    bu = chronos2(tasks).reshape(len(load.columns), n_days * H).sum(0)
    res["Chronos-2 (bottom-up)"] = metrics(bu, true); store["Chronos-2_bottom-up"] = bu
    print("Chronos-2 bottom-up", res["Chronos-2 (bottom-up)"], f"({time.time() - tic:.0f}s)", flush=True)

    json.dump({"metrics": res}, open(os.path.join(OUT, "weave.json"), "w"), indent=1)
    np.savez_compressed(os.path.join(OUT, "weave_preds.npz"), **store)


# ---------------------------------------------------------------- ARIMA-X exogenous design
def arimax_select():
    """Choose the exogenous design of ARIMA-X on the common validation year, then test once.

    A linear temperature term misses the heating response of French load, so three
    nonlinear designs are compared with the first. Parameters are estimated on
    2015-2017 and scored on the one-day-ahead national forecasts of 2018, the
    validation year shared by every model of the benchmark; the retained design is
    re-estimated on 2015-2018 and applied to 2019. The ARMA orders are those chosen
    on 2015-2017 by rfrance().
    """
    from statsmodels.tsa.statespace.sarimax import SARIMAX
    true = np.load(os.path.join(HERE, "results_rfrance_v2", "baselines_preds.npz"))["true"]
    import re
    orders = {int(h): tuple(int(x) for x in o.split(","))
              for h, o in re.findall(r"ARIMA-X h=\s*(\d+) order=\(([^)]*)\)", open(os.path.join(OUT, "rfrance.log")).read())}
    assert len(orders) == H, "orders of rfrance() missing"
    df = pd.concat([pd.read_csv(os.path.join(HERE, "data", "rfrance", f)) for f in ("train2.csv", "test2.csv")])
    df["date"] = pd.to_datetime(df["date"])
    df = df.drop_duplicates(["date", "Region"]).sort_values(["date", "Region"])
    load = df.pivot(index="date", columns="Region", values="load").sort_index().dropna()
    nat = load.sum(axis=1)
    cov = df.groupby("date")[WEATHER + ["JourFerie", "Posan", "TDChauff1", "TDChauff2"]].mean().reindex(nat.index)
    dates = nat.index
    y = nat.to_numpy(float)
    start = align(y, true)
    n_days = len(true) // H
    wd = pd.get_dummies(pd.DatetimeIndex(dates).dayofweek, prefix="wd", drop_first=True).to_numpy(float)
    base = np.column_stack([cov[WEATHER].to_numpy(), cov["JourFerie"].to_numpy(), wd,
                            np.sin(2 * np.pi * cov["Posan"].to_numpy()), np.cos(2 * np.pi * cov["Posan"].to_numpy())])
    td = cov[["TDChauff1", "TDChauff2"]].to_numpy(float)
    dd = np.column_stack([np.maximum(15 - cov["temp"].to_numpy(), 0), np.maximum(15 - cov["temp_liss_fort"].to_numpy(), 0),
                          np.maximum(cov["temp"].to_numpy() - 20, 0)])
    designs = {"A: linear weather": base, "B: + operator heating terms": np.column_stack([base, td]),
               "C: + degree days": np.column_stack([base, dd]), "D: + both": np.column_stack([base, td, dd])}
    t0 = np.searchsorted(dates, pd.Timestamp("2015-01-01"))
    t_val, t_test = np.searchsorted(dates, pd.Timestamp("2018-01-01")), start
    val_days = np.arange(t_val, t_test)
    val_days = val_days[(val_days - start) % H == 0]

    def run(X, est_end, first, n):
        pred = np.zeros(n * H)
        for h in range(H):
            idx = np.arange(t0, len(y)); idx = idx[((idx - start) % H) == h]
            Xi = X[idx]; est = idx < est_end
            mu, sd = Xi[est].mean(0), Xi[est].std(0) + 1e-9
            Xi = (Xi - mu) / sd
            order = orders[h]
            fit = SARIMAX(y[idx][est], exog=Xi[est], order=order, trend="c" if order[1] == 0 else "n").fit(disp=False, maxiter=200)
            one = np.asarray(fit.apply(y[idx], exog=Xi, refit=False).get_prediction(dynamic=False).predicted_mean)
            k = np.searchsorted(idx, first)
            pred[np.arange(n) * H + h] = one[k:k + n]
        return pred

    val_true = y[val_days[0]:val_days[0] + len(val_days) * H]
    scores = {}
    for name, X in designs.items():
        tic = time.time()
        p = run(X, t_val, val_days[0], len(val_days))
        scores[name] = metrics(p, val_true)
        print(f"  validation 2018  {name:30s} RMSE {scores[name][0]:7.1f} MW  MAPE {scores[name][1]:.2f}%  ({time.time() - tic:.0f}s)", flush=True)
    best = min(scores, key=lambda k: scores[k][0])
    p = run(designs[best], t_test, start, n_days)
    test = metrics(p, true)
    print(f"retained: {best}  ->  test 2019 RMSE {test[0]:.1f} MW  MAPE {test[1]:.2f}%", flush=True)
    json.dump({"validation_2018": scores, "retained": best, "test_2019": test},
              open(os.path.join(OUT, "rfrance_arimax_select.json"), "w"), indent=1)
    np.save(os.path.join(OUT, "rfrance_arimax_selected_pred.npy"), p)


def arimax_weave():
    """ARIMA-X on the summed WEAVE-UK load, calendar inputs only, 48-step forecasts per origin.

    The per-half-hour design of REGIONS-FR would leave 9 points per model, so a single
    model runs on the half-hourly aggregate. Orders are chosen by AIC on the training
    days (14-22 Feb), the exogenous design on the common validation days (23-24 Feb),
    and the retained design is re-estimated on 14-24 Feb and applied once to the test.
    """
    from pmdarima import auto_arima
    from statsmodels.tsa.statespace.sarimax import SARIMAX
    true = np.load(os.path.join(HERE, "results_weave_500", "baselines_preds.npz"))["true"]
    df = pd.concat([pd.read_csv(os.path.join(WEAVE, f), usecols=["date", "id_unique", "consumption"])
                    for f in ("train_weave_500.csv", "test_weave_500.csv")])
    df["date"] = pd.to_datetime(df["date"], utc=True)
    nat = df.groupby("date").consumption.sum().sort_index()
    dates = nat.index
    y = nat.to_numpy(float)
    start = align(y, true)
    t_val = np.searchsorted(dates, pd.Timestamp("2024-02-23", tz="UTC"))
    assert start == np.searchsorted(dates, pd.Timestamp("2024-02-25", tz="UTC"))
    d = pd.DatetimeIndex(dates)
    inst = (d.hour * 2 + d.minute // 30).to_numpy()
    weekend = np.asarray(d.dayofweek >= 5, float)

    def fourier(K):
        return np.column_stack([f(2 * np.pi * k * inst / H) for k in range(1, K + 1) for f in (np.sin, np.cos)])

    designs = {"A: Fourier K=3 + weekend": (np.column_stack([fourier(3), weekend]), False),
               "B: Fourier K=6 + weekend": (np.column_stack([fourier(6), weekend]), False),
               "C: A + seasonal AR(48)": (np.column_stack([fourier(3), weekend]), True)}
    order = auto_arima(y[:t_val], X=designs["A: Fourier K=3 + weekend"][0][:t_val], seasonal=False, stepwise=True,
                       max_p=3, max_q=3, max_d=1, information_criterion="aic",
                       error_action="ignore", suppress_warnings=True).order
    print("order", order, flush=True)

    def run(X, seas, est_end, first, n):
        mod = SARIMAX(y[:est_end], exog=X[:est_end], order=order, trend="c" if order[1] == 0 else "n",
                      seasonal_order=(1, 0, 0, H) if seas else (0, 0, 0, 0))
        fit = mod.fit(disp=False, maxiter=200)
        pred = []
        for k in range(n):
            o = first + k * H
            res = fit.apply(y[:o], exog=X[:o], refit=False)
            pred.append(np.asarray(res.forecast(H, exog=X[o:o + H])))
        return np.concatenate(pred)

    scores = {}
    for name, (X, seas) in designs.items():
        tic = time.time()
        p = run(X, seas, t_val, t_val, 2)
        scores[name] = metrics(p, y[t_val:t_val + 2 * H])
        print(f"  validation 23-24 Feb  {name:28s} RMSE {scores[name][0] / 1000:7.1f} kWh  "
              f"MAPE {scores[name][1]:.2f}%  ({time.time() - tic:.0f}s)", flush=True)
    best = min(scores, key=lambda k: scores[k][0])
    X, seas = designs[best]
    p = run(X, seas, start, start, len(true) // H)
    test = metrics(p, true)
    print(f"retained: {best}  ->  test RMSE {test[0] / 1000:.1f} kWh  MAPE {test[1]:.2f}%", flush=True)
    json.dump({"order": list(order), "validation": scores, "retained": best, "test": test},
              open(os.path.join(OUT, "weave_arimax_select.json"), "w"), indent=1)
    np.save(os.path.join(OUT, "weave_arimax_pred.npy"), p)


# ---------------------------------------------------------------- multivariate Chronos-2
def multivariate(coll):
    """Chronos-2 reading all nodes of an origin as one multivariate series, then summed.

    Information flows between nodes inside the model, with no graph given. One task per
    origin, so no task ever sees the targets of another origin. Side information only.
    """
    if coll == "rfrance":
        true = np.load(os.path.join(HERE, "results_rfrance_v2", "baselines_preds.npz"))["true"]
        df = pd.concat([pd.read_csv(os.path.join(HERE, "data", "rfrance", f)) for f in ("train2.csv", "test2.csv")])
        df["date"] = pd.to_datetime(df["date"])
        df = df.drop_duplicates(["date", "Region"]).sort_values(["date", "Region"])
        load = df.pivot(index="date", columns="Region", values="load").sort_index().dropna()
        cov = df.groupby("date")[WEATHER + ["JourFerie", "Posan"]].mean().reindex(load.index)
        feat = {k: cov[k].to_numpy(float) for k in WEATHER + ["JourFerie"]}
        feat["posan_sin"] = np.sin(2 * np.pi * cov["Posan"].to_numpy()); feat["posan_cos"] = np.cos(2 * np.pi * cov["Posan"].to_numpy())
    else:
        true = np.load(os.path.join(HERE, "results_weave_500", "baselines_preds.npz"))["true"]
        df = pd.concat([pd.read_csv(os.path.join(WEAVE, f), usecols=["date", "id_unique", "consumption"])
                        for f in ("train_weave_500.csv", "test_weave_500.csv")])
        df["date"] = pd.to_datetime(df["date"], utc=True)
        load = df.pivot_table(index="date", columns="id_unique", values="consumption").sort_index()
        feat = {}
    dates = load.index
    feat.update(calendar(dates))
    Y = load.to_numpy(float).T                                                 # [V, T]
    start = align(Y.sum(0), true)
    n_days = len(true) // H
    tasks = []
    for d in range(n_days):
        o = start + d * H
        tasks.append({"target": Y[:, :o].astype(np.float32),
                      "past_covariates": {k: v[:o].astype(np.float32) for k, v in feat.items()},
                      "future_covariates": {k: v[o:o + H].astype(np.float32) for k, v in feat.items()}})
    from chronos.chronos2 import Chronos2Pipeline
    dev = "mps" if torch.backends.mps.is_available() else "cpu"
    pipe = Chronos2Pipeline.from_pretrained("amazon/chronos-2", device_map=dev)
    qs = list(pipe.quantiles); mid = qs.index(min(qs, key=lambda q: abs(q - 0.5)))
    tic = time.time()
    out = pipe.predict(tasks, prediction_length=H, batch_size=1)
    p = np.concatenate([o[:, mid].float().cpu().numpy().sum(0) for o in out])
    m = metrics(p, true)
    print(f"Chronos-2 multivariate {coll}", m, f"({time.time() - tic:.0f}s)", flush=True)
    path = os.path.join(OUT, f"{coll}_multivariate.json")
    json.dump({"metrics": {"Chronos-2 (multivariate)": m}}, open(path, "w"), indent=1)
    np.save(os.path.join(OUT, f"{coll}_multivariate_pred.npy"), p)


if __name__ == "__main__":
    if sys.argv[1] == "multi":
        multivariate(sys.argv[2])
    elif sys.argv[1] == "arimax":
        arimax_select()
    elif sys.argv[1] == "arimax_weave":
        arimax_weave()
    else:
        {"rfrance": rfrance, "weave": weave}[sys.argv[1]]()
