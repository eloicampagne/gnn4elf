"""Where the foundation models lose to supervised models on REGIONS-FR.

Reads the stored national forecasts (t0_baseline.py, timestral_baseline.py and the
XGBoost b-global row of benchmark_rfrance.py) and splits the 2019 test RMSE by day
type (holidays, the Monday after a clock change, other days) and by month.

    python3 fm_diagnostics.py
"""

import os

import numpy as np
import pandas as pd

from extra_baselines import H, HERE, OUT, WEATHER, align

df = pd.concat([pd.read_csv(os.path.join(HERE, "data", "rfrance", f)) for f in ("train2.csv", "test2.csv")])
df["date"] = pd.to_datetime(df["date"])
df = df.drop_duplicates(["date", "Region"])
nat = df.pivot(index="date", columns="Region", values="load").sort_index().dropna().sum(axis=1)
base = np.load(os.path.join(HERE, "results_rfrance_v2", "baselines_preds.npz"))
true = base["true"]
start = align(nat.to_numpy(), true)
days = nat.index[start::H][:len(true) // H]
hol = df.groupby("date")["JourFerie"].max().reindex(nat.index).to_numpy()[start::H][:len(days)] > 0
dst = np.isin(days.normalize(), pd.to_datetime(["2019-04-01", "2019-10-28"]))
other = ~hol & ~dst

preds = {"XGBoost (b-global)": base["XGBoost_(b-global)"].sum(2).mean(0).reshape(-1)}
for f, k, name in [("rfrance_t0_preds.npz", "t0-gamma_default", "t0-gamma"),
                   ("rfrance_timestral_209M_preds.npz", "Timestral-209M_1024", "Timestral-209M")]:
    path = os.path.join(OUT, f)
    if os.path.exists(path):
        preds[name] = np.load(path)[k]

daily = {n: np.sqrt(((p - true).reshape(-1, H) ** 2).mean(1)) for n, p in preds.items()}
rmse = lambda e, m: np.sqrt((e[m] ** 2).mean())
rows = [{"model": n, "all": rmse(e, np.ones_like(hol)), f"holidays ({hol.sum()})": rmse(e, hol),
         f"after clock change ({dst.sum()})": rmse(e, dst), f"other days ({other.sum()})": rmse(e, other)}
        for n, e in daily.items()]
print(pd.DataFrame(rows).set_index("model").round(0).to_string())
month = pd.DataFrame({n: [rmse(e, days.month == m) for m in range(1, 13)] for n, e in daily.items()},
                     index=range(1, 13))
print("\nRMSE by month (MW)\n" + month.round(0).to_string())
