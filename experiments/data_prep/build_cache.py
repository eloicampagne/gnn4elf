"""Cache the twelve French regional load tasks as plain arrays.

One task per region. The loading path replicates rerun_ff_local.py, itself a
transcription of expe.ipynb, so the numbers stay comparable to the per-region
MLP and to the GNN baselines: same feature list, same train/val/test split, same
per-node MinMax scaling fitted on train only.

One deliberate departure. rerun_ff_local.py defaults to stride 1, which makes the
forecast origin fall at any half-hour and forces a single 48-output head to
represent 48 phase-shifted mappings. That handicap is documented in its own
docstring and is what cripples the published feed-forward baseline. We use
stride 48, so the origin is always a day boundary and output k always means slot
k of the day, which is the task the GNN is given.

Run once: python3 build_cache.py
"""

import os
import sys
from pathlib import Path

import numpy as np
import torch
from sklearn.preprocessing import MinMaxScaler

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from _paths import ROOT, DATA, CACHE  # noqa: E402  (puts every experiment folder on sys.path)
GNN4ELF = Path(ROOT)
os.chdir(GNN4ELF)
sys.path.insert(0, str(GNN4ELF))

from graphtoolbox.data.dataset import DataClass, GraphDataset

OUT_CHANNELS = 48
STRIDE = 48
OUT = Path(CACHE)

DATA_KWARGS = {
    "node_var": "Region",
    "dummies": ["Instant", "JourSemaine", "DayType", "offset"],
    "day_inf_train": "2015-01-01", "day_sup_train": "2018-01-01",
    "day_inf_val": "2018-01-01", "day_sup_val": "2018-12-31",
    "day_inf_test": "2019-01-01", "day_sup_test": "2019-12-31",
}
DATASET_KWARGS = {
    "batch_size": 32, "adj_matrix": "gl3sr",
    "features_base": ["temp", "nebu", "wind", "tempMax", "tempMin", "Posan", "Instant",
                      "JourSemaine", "JourFerie", "offset", "DayType", "Weekend",
                      "temp_liss_fort", "temp_liss_faible"] + [f"load_l{t}" for t in range(1, 49)],
    "target_base": "load",
}


def sequences(X, y, horizon=OUT_CHANNELS, stride=STRIDE):
    idx = range(0, len(X) - horizon + 1, stride)
    return (np.stack([X[i] for i in idx]),
            np.stack([y[i:i + horizon] for i in idx]).reshape(len(list(idx)), horizon))


def main():
    OUT.mkdir(exist_ok=True)
    data = DataClass(path_train="./data/rfrance/train2.csv",
                     path_test="./data/rfrance/test2.csv",
                     data_kwargs=DATA_KWARGS, folder_config=".")
    gds = GraphDataset(data=data, period="train",
                       graph_folder="./graph_representations_rfrance",
                       dataset_kwargs=DATASET_KWARGS, out_channels=OUT_CHANNELS)
    features = list(gds.features)
    nodes = list(data.nodes)
    print(f"{len(nodes)} regions, {len(features)} features", flush=True)

    store = {"nodes": np.array(nodes), "features": np.array(features)}
    for node in nodes:
        def take(df, cols):
            return df[df[data.node_var] == node].reset_index(drop=True)[cols].values

        splits = {}
        raw = {k: (take(d, features), take(d, ["load"]))
               for k, d in (("train", data.df_train), ("val", data.df_val),
                            ("test", data.df_test))}
        sx = MinMaxScaler().fit(raw["train"][0])
        sy = MinMaxScaler().fit(raw["train"][1])
        for k, (xx, yy) in raw.items():
            a, b = sequences(sx.transform(xx), sy.transform(yy))
            splits[k] = (a.astype(np.float32), b.astype(np.float32))
        for k, (a, b) in splits.items():
            store[f"{node}|{k}|X"] = a
            store[f"{node}|{k}|y"] = b
        # store the target scaler so predictions can be returned in MW
        store[f"{node}|ymin"] = np.asarray(sy.data_min_, dtype=np.float64)
        store[f"{node}|yscale"] = np.asarray(sy.data_range_, dtype=np.float64)

    n0 = store[f"{nodes[0]}|train|X"].shape
    print(f"per-region sequences: train {n0[0]}, "
          f"val {store[f'{nodes[0]}|val|X'].shape[0]}, "
          f"test {store[f'{nodes[0]}|test|X'].shape[0]}, d={n0[1]}", flush=True)
    np.savez_compressed(OUT / "rfrance_tasks.npz", **store)
    print("written", OUT / "rfrance_tasks.npz", flush=True)


if __name__ == "__main__":
    main()
