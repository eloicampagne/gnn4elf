"""Architecture benchmark of the GNN chapter, rerun on the 500 WEAVE-UK feeders.

The benchmark of the manuscript (tab:combined-results, tab:aggregation-results,
tab:gs1and2) was first run on the 28-substation panel of arXiv:2507.03690v3
(results_weave/, expe.ipynb). The rest of the chapter reads the 500 feeders of
WEAVE-UK, so this script reruns the benchmark on that collection, with the
protocol of the notebook except where the panel size forces a change:

  * data: train_weave_500 / test_weave_500, train 14-22 Feb 2024, validation
    23-24 Feb, test 25-28 Feb; every sample is a day-aligned origin and the
    model returns the 48 half-hours of the next day (stride 48);
  * features: the 48 load lags, the half-hour instant (one-hot) and the weekday;
    no node identity enters, so the sharing axis keeps its meaning;
  * graphs: dense constructions are out of reach at 500 nodes, so each candidate
    (correlation, DTW, precision) keeps its 5 strongest neighbours per node,
    symmetrised, on the training window only;
  * GNNs: per-convolution Optuna search over graph, depth, width, learning rate,
    batch size and convolution-specific parameters, selected on validation RMSE
    of the summed forecast; then 3 seeds at 300 epochs, best-validation epoch;
  * baselines and aggregation as in expe.ipynb (shared and per-node MLP over
    10 seeds, XGBoost b-global and b-local over 3 seeds, GAM, ARIMA, Chronos-Bolt,
    persistence D-1 and D-7; uniform and ML-Poly mixtures over seeds and
    architectures, top-level and bottom-up).

    python3 benchmark_weave500.py prep
    python3 benchmark_weave500.py hpo --conv GraphSAGE --trials 40
    python3 benchmark_weave500.py train --conv GraphSAGE
    python3 benchmark_weave500.py baselines
    python3 benchmark_weave500.py aggregate
"""

import argparse
import json
import os
import shutil
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
os.chdir(HERE)

WEAVE = os.path.join(DATA, "weave")
PATH_TRAIN = os.path.join(WEAVE, "train_weave_500.csv")
PATH_TEST = os.path.join(WEAVE, "test_weave_500.csv")
GRAPH_DIR = os.path.join(HERE, "graph_representations_weave_500")
RES_DIR = os.path.join(HERE, "results_weave_500")
CKPT_TMP = os.path.join(HERE, "checkpoints_weave_500")
H = 48
K_NN = 5
GRAPHS = ("correlation", "dtw", "precision")

DATA_KWARGS = {
    "node_var": "id_unique",
    "dummies": ["instant"],
    "day_inf_train": "2024-02-14",
    "day_sup_train": "2024-02-23",
    "day_inf_val": "2024-02-23",
    "day_sup_val": "2024-02-25",
    "day_inf_test": "2024-02-25",
    "day_sup_test": "2024-02-29",
}
FEATURES = ["instant", "weekday"] + [f"consumption_l{t}" for t in range(1, H + 1)]
DATASET_KWARGS = {"adj_matrix": "correlation", "features_base": FEATURES,
                  "target_base": "consumption"}


# ---------------------------------------------------------------- data
_CACHE = {}


def datasets():
    if "ds" in _CACHE:
        return _CACHE["ds"]
    from graphtoolbox.data.dataset import DataClass, GraphDataset
    data = DataClass(path_train=PATH_TRAIN, path_test=PATH_TEST,
                     data_kwargs=DATA_KWARGS, folder_config=".")
    kw = dict(graph_folder=GRAPH_DIR, dataset_kwargs=DATASET_KWARGS, out_channels=H)
    tr = GraphDataset(data=data, period="train", **kw)
    va = GraphDataset(data=data, period="val", scalers_feat=tr.scalers_feat,
                      scalers_target=tr.scalers_target, **kw)
    te = GraphDataset(data=data, period="test", scalers_feat=tr.scalers_feat,
                      scalers_target=tr.scalers_target, **kw)
    _CACHE["ds"] = (data, tr, va, te)
    return _CACHE["ds"]


def train_matrix():
    """Node x time load matrix on the training window, in the node order of DataClass."""
    df = pd.read_csv(PATH_TRAIN, usecols=["date", "id_unique", "consumption"])
    df["date"] = pd.to_datetime(df["date"], utc=True)
    lo = pd.Timestamp(DATA_KWARGS["day_inf_train"], tz="UTC")
    hi = pd.Timestamp(DATA_KWARGS["day_sup_train"], tz="UTC")
    df = df[(df.date >= lo) & (df.date < hi)]
    piv = df.pivot(index="date", columns="id_unique", values="consumption")
    piv = piv.reindex(columns=np.sort(piv.columns))
    return piv.columns.to_numpy(), piv.to_numpy().T.astype(float)


# ---------------------------------------------------------------- graphs
def _knn_sym(S, k=K_NN):
    """Keep the k largest scores of each row, symmetrise by the maximum."""
    N = S.shape[0]
    S = S.copy()
    np.fill_diagonal(S, -np.inf)
    W = np.zeros((N, N))
    for i in range(N):
        for j in np.argsort(-S[i])[:k]:
            w = max(S[i, j], 0.0)
            W[i, j] = max(W[i, j], w)
            W[j, i] = max(W[j, i], w)
    return W


def _dtw_matrix(Z, band=H):
    from numba import njit, prange

    @njit(parallel=True, cache=False)
    def run(Z, band):
        N, T = Z.shape
        D = np.zeros((N, N))
        for i in prange(N):
            for j in range(i + 1, N):
                prev = np.full(T + 1, np.inf)
                prev[0] = 0.0
                for a in range(1, T + 1):
                    cur = np.full(T + 1, np.inf)
                    lo = max(1, a - band)
                    hi = min(T, a + band)
                    for b in range(lo, hi + 1):
                        c = (Z[i, a - 1] - Z[j, b - 1]) ** 2
                        m = prev[b]
                        if cur[b - 1] < m:
                            m = cur[b - 1]
                        if prev[b - 1] < m:
                            m = prev[b - 1]
                        cur[b] = c + m
                    prev = cur
                D[i, j] = np.sqrt(prev[T])
                D[j, i] = D[i, j]
        return D

    return run(Z, band)


def prep():
    from sklearn.covariance import LedoitWolf
    nodes, C = train_matrix()
    N, T = C.shape
    print(f"{N} feeders, {T} training half-hours", flush=True)
    sd = C.std(1, keepdims=True)
    dead = (sd < 1e-9).ravel()
    Z = (C - C.mean(1, keepdims=True)) / np.where(sd < 1e-9, 1.0, sd)

    out = {}
    R = (Z @ Z.T) / T
    np.fill_diagonal(R, 0.0)
    R[dead, :] = 0.0; R[:, dead] = 0.0
    out["correlation"] = _knn_sym(R)

    t0 = time.time()
    D = _dtw_matrix(Z)
    print(f"DTW matrix in {time.time() - t0:.0f}s", flush=True)
    med = np.median(D[np.triu_indices(N, 1)])
    S = np.exp(-D / med)
    S[dead, :] = 0.0; S[:, dead] = 0.0
    out["dtw"] = _knn_sym(S)

    P = LedoitWolf().fit(Z.T).precision_
    d = np.sqrt(np.clip(np.diag(P), 1e-12, None))
    Pc = -P / np.outer(d, d)                       # partial correlations
    np.fill_diagonal(Pc, 0.0)
    Pc[dead, :] = 0.0; Pc[:, dead] = 0.0
    out["precision"] = _knn_sym(np.abs(Pc))

    for name, W in out.items():
        os.makedirs(os.path.join(GRAPH_DIR, name), exist_ok=True)
        np.savetxt(os.path.join(GRAPH_DIR, name, "W.txt"), W, delimiter=" ", fmt="%.6g")
        deg = (W > 0).sum(1)
        print(f"{name}: {(W > 0).sum()} directed edges, median degree {np.median(deg):.0f}",
              flush=True)
    with open(os.path.join(GRAPH_DIR, "nodes.json"), "w") as f:
        json.dump([str(n) for n in nodes], f)
    print(f"{int(dead.sum())} constant series isolated", flush=True)


# ---------------------------------------------------------------- GNNs
def conv_map():
    from torch_geometric.nn.conv import (GCNConv, SAGEConv, GATConv, GATv2Conv,
                                         TransformerConv, TAGConv, ChebConv, APPNP)
    return {"GCNConv": GCNConv, "GraphSAGE": SAGEConv, "GATConv": GATConv,
            "GATv2Conv": GATv2Conv, "TransformerConv": TransformerConv,
            "TAGConv": TAGConv, "ChebConv": ChebConv, "APPNP": APPNP}


CONV_SPACE = {
    "GATConv": {"heads": [1, 2, 4]},
    "GATv2Conv": {"heads": [1, 2, 4]},
    "TransformerConv": {"heads": [1, 2, 4]},
    "TAGConv": {"K": [1, 2, 3, 4]},
    "ChebConv": {"K": [1, 2, 3, 4, 5]},
    "APPNP": {"K": [2, 4, 6, 8, 10], "alpha": [0.05, 0.1, 0.15, 0.2, 0.3]},
}


def build_model(conv_name, params, in_channels):
    from graphtoolbox.models.gnn import myGNN
    from torch_geometric.nn.conv import SAGEConv
    conv = conv_map()[conv_name]
    conv_kwargs = {k: params[k] for k in ("heads", "K", "alpha") if k in params}
    extra = {"project": True} if conv is SAGEConv else {}
    return myGNN(in_channels=in_channels, num_layers=params["num_layers"],
                 hidden_channels=params["hidden_channels"], out_channels=H,
                 conv_class=conv, conv_kwargs=conv_kwargs | extra | {"add_self_loops": False})


def fit_gnn(conv_name, params, seed, epochs, patience, tag):
    from graphtoolbox.training.trainer import Trainer, DEVICE
    from torch_geometric import seed_everything
    data, tr, va, te = datasets()
    for ds in (tr, va, te):
        ds._set_adj_matrix(adj_matrix=params["adj_matrix"])
    seed_everything(seed)
    model = build_model(conv_name, params, tr.num_node_features).to(DEVICE)
    trainer = Trainer(model=model, dataset_train=tr, dataset_val=va, dataset_test=te,
                      batch_size=params["batch_size"], return_attention=False,
                      model_kwargs={"lr": params["lr"], "num_epochs": epochs}, lam_reg=0,
                      reconcile=False)
    sdir = os.path.join(CKPT_TMP, tag)
    shutil.rmtree(sdir, ignore_errors=True)
    pred_te, target_te, _, _ = trainer.train(plot_loss=False, force_training=True, save=False,
                                             patience=patience, saving_directory=sdir)
    from torch_geometric.loader import DataLoader as PyGDataLoader
    val_rmse, pred_va, target_va = trainer._predict(PyGDataLoader(va, shuffle=False), split="val")
    return float(val_rmse), pred_te.detach().cpu(), target_te.detach().cpu()


def system_metrics(pred, target):
    p = np.asarray(pred).sum(axis=0) if np.ndim(pred) == 2 else np.asarray(pred)
    t = np.asarray(target).sum(axis=0) if np.ndim(target) == 2 else np.asarray(target)
    rmse = float(np.sqrt(np.mean((p - t) ** 2)))
    mape = float(100 * np.mean(np.abs((p - t) / t)))
    return rmse, mape


def hpo(conv_name, n_trials, epochs=200, patience=40):
    import optuna
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    space = CONV_SPACE.get(conv_name, {})
    log = []

    def objective(trial):
        p = {"adj_matrix": trial.suggest_categorical("adj_matrix", list(GRAPHS)),
             "num_layers": trial.suggest_int("num_layers", 1, 3),
             "hidden_channels": trial.suggest_int("hidden_channels", 32, 256, log=True),
             "lr": trial.suggest_float("lr", 1e-4, 1e-2, log=True),
             "batch_size": trial.suggest_categorical("batch_size", [2, 4, 8])}
        for k, vals in space.items():
            p[k] = trial.suggest_categorical(k, vals)
        if conv_name in ("GATConv", "GATv2Conv", "TransformerConv"):
            p["hidden_channels"] = max(p["heads"], p["hidden_channels"] // p["heads"] * p["heads"])
        t0 = time.time()
        try:
            v, _, _ = fit_gnn(conv_name, p, seed=42, epochs=epochs, patience=patience,
                              tag=f"hpo_{conv_name}")
        except Exception as e:  # an invalid configuration scores infinitely badly
            print(f"  trial {trial.number}: {type(e).__name__}: {e}", flush=True)
            return float("inf")
        log.append({**p, "val_rmse": v, "seconds": time.time() - t0})
        print(f"  trial {trial.number}: val RMSE {v:.1f} ({time.time() - t0:.0f}s) {p}", flush=True)
        return v

    study = optuna.create_study(direction="minimize", sampler=optuna.samplers.TPESampler(seed=42))
    study.optimize(objective, n_trials=n_trials)
    best = dict(study.best_params)
    if conv_name in ("GATConv", "GATv2Conv", "TransformerConv"):
        best["hidden_channels"] = max(best["heads"], best["hidden_channels"] // best["heads"] * best["heads"])
    os.makedirs(RES_DIR, exist_ok=True)
    cfg_path = os.path.join(HERE, "configs", "config_weave_500.json")
    cfg = json.load(open(cfg_path)) if os.path.exists(cfg_path) else {}
    cfg[conv_name] = {"adj_matrix": best.pop("adj_matrix"), "params": best,
                      "val_rmse": study.best_value, "n_trials": n_trials}
    json.dump(cfg, open(cfg_path, "w"), indent=2)
    json.dump(log, open(os.path.join(RES_DIR, f"hpo_{conv_name}.json"), "w"), indent=1)
    print(f"{conv_name}: best val RMSE {study.best_value:.1f} -> {cfg[conv_name]}", flush=True)


def train(conv_name, n_seeds=3, epochs=300):
    cfg = json.load(open(os.path.join(HERE, "configs", "config_weave_500.json")))[conv_name]
    params = {"adj_matrix": cfg["adj_matrix"], **cfg["params"]}
    folder = os.path.join(RES_DIR, conv_name)
    os.makedirs(folder, exist_ok=True)
    rows = []
    for i in range(n_seeds):
        v, pred, target = fit_gnn(conv_name, params, seed=42 + i, epochs=epochs,
                                  patience=epochs, tag=f"final_{conv_name}_{i}")
        torch.save({"predictions": pred}, os.path.join(folder, f"{conv_name}_run{i}.pt"))
        torch.save(target, os.path.join(RES_DIR, "target_nodes.pt"))
        r, m = system_metrics(pred.numpy(), target.numpy())
        rows.append({"seed": 42 + i, "val_rmse": v, "rmse": r, "mape": m})
        print(f"{conv_name} seed {42 + i}: test RMSE {r:.1f} MAPE {m:.2f}%", flush=True)
    json.dump(rows, open(os.path.join(folder, "metrics.json"), "w"), indent=1)
    rr = np.array([x["rmse"] for x in rows]); mm = np.array([x["mape"] for x in rows])
    print(f"{conv_name}: RMSE {rr.mean():.1f} ± {rr.std(ddof=1):.1f}, MAPE {mm.mean():.2f} ± {mm.std(ddof=1):.2f}",
          flush=True)


# ---------------------------------------------------------------- baselines
def snapshots():
    """Day-aligned samples shared by every baseline: scaled features, raw targets."""
    data, tr, va, te = datasets()
    def stack(ds):
        X = np.stack([d.x.numpy() for d in ds.pyg_data])          # [S, N, F]
        Y = np.stack([d.y.numpy() for d in ds.pyg_data])          # [S, N, 48]
        return X.astype(np.float32), Y.astype(np.float64)
    return stack(tr), stack(va), stack(te)


def national(Y):
    """[S, N, 48] -> national half-hourly series of length S*48."""
    return Y.sum(axis=1).reshape(-1)


def mlp_fit(Xtr, Ytr, Xva, Yva, Xte, cfg, seed, per_node, epochs=150, patience=12):
    """Shared MLP on stacked nodes, or one MLP per node trained in parallel.

    Targets are standardised per node on the training samples. The per-node
    variant holds one weight set per node (batched), so nodes do not share any
    parameter; early stopping reads the summed validation loss.
    """
    torch.manual_seed(seed); np.random.seed(seed)
    S, N, F = Xtr.shape
    mu = Ytr.mean(axis=(0, 2), keepdims=True); sd = Ytr.std(axis=(0, 2), keepdims=True) + 1e-9
    f = lambda a: torch.tensor(a, dtype=torch.float32)
    xtr, xva, xte = f(Xtr), f(Xva), f(Xte)
    ytr, yva = f((Ytr - mu) / sd), f((Yva - mu) / sd)
    Hh, L = cfg["hidden_channels"], cfg["num_layers"]
    if per_node:
        dims = [F] + [Hh] * (L - 1) + [H]
        Ws = [torch.nn.Parameter(torch.randn(N, a, b) / np.sqrt(a)) for a, b in zip(dims[:-1], dims[1:])]
        bs = [torch.nn.Parameter(torch.zeros(N, b)) for b in dims[1:]]
        params = Ws + bs
        def fwd(x):                                     # x [B, N, F]
            h = x
            for i, (W, b) in enumerate(zip(Ws, bs)):
                h = torch.einsum("bnf,nfh->bnh", h, W) + b
                if i < len(Ws) - 1:
                    h = torch.relu(h)
            return h
    else:
        from ff_model import FF
        net = FF(in_channels=F, hidden_channels=Hh, out_channels=H, num_layers=L)
        params = list(net.parameters())
        fwd = net
    opt = torch.optim.Adam(params, lr=cfg["lr"])
    best, best_state, bad = np.inf, None, 0
    bsz = cfg["batch_size"]
    for ep in range(epochs):
        perm = torch.randperm(S)
        for i in range(0, S, bsz):
            idx = perm[i:i + bsz]
            loss = ((fwd(xtr[idx]) - ytr[idx]) ** 2).mean()
            opt.zero_grad(); loss.backward(); opt.step()
        with torch.no_grad():
            v = float(((fwd(xva) - yva) ** 2).mean())
        if v < best:
            best, bad = v, 0
            best_state = [p.detach().clone() for p in params]
        else:
            bad += 1
            if bad >= patience:
                break
    with torch.no_grad():
        for p, q in zip(params, best_state):
            p.copy_(q)
        pred = fwd(xte).numpy() * sd + mu
        pva = fwd(xva).numpy() * sd + mu
    return pred, pva


def run_mlps(tr, va, te, n_seeds=10):
    (Xtr, Ytr), (Xva, Yva), (Xte, Yte) = tr, va, te
    out = {}
    rng = np.random.default_rng(42)
    cands = {"hidden_channels": [64, 128, 192, 256], "num_layers": [2, 3],
             "lr": [3e-4, 1e-3, 2e-3], "batch_size": [2, 4, 8]}
    for name, per_node in (("Shared MLP", False), ("Per-node MLP", True)):
        best, best_cfg = np.inf, None
        for t in range(8):
            cfg = {k: (float(rng.choice(v)) if k == "lr" else int(rng.choice(v))) for k, v in cands.items()}
            _, pva = mlp_fit(Xtr, Ytr, Xva, Yva, Xte, cfg, 42, per_node)
            r, _ = system_metrics(national(pva), national(Yva))
            if r < best:
                best, best_cfg = r, cfg
        preds = []
        for s in range(n_seeds):
            p, _ = mlp_fit(Xtr, Ytr, Xva, Yva, Xte, best_cfg, 42 + s, per_node)
            preds.append(p)
        out[name] = {"cfg": best_cfg, "preds": preds}
        print(f"{name}: cfg {best_cfg}", flush=True)
    return out


def run_xgb(tr, va, te, n_seeds=3):
    import xgboost as xgb
    (Xtr, Ytr), (Xva, Yva), (Xte, Yte) = tr, va, te
    S, N, F = Xtr.shape
    mu = Ytr.mean(axis=(0, 2), keepdims=True); sd = Ytr.std(axis=(0, 2), keepdims=True) + 1e-9
    ytr_s = (Ytr - mu) / sd; yva_s = (Yva - mu) / sd
    space = lambda rng: {
        "learning_rate": float(rng.choice([0.03, 0.05, 0.1, 0.2])),
        "max_depth": int(rng.choice([4, 6, 8, 10])),
        "min_child_weight": float(rng.choice([1.0, 3.0, 6.0, 10.0])),
        "subsample": float(rng.choice([0.7, 0.85, 1.0])),
        "colsample_bytree": float(rng.choice([0.6, 0.8, 1.0])),
        "reg_alpha": float(rng.choice([0.0, 1e-4, 1e-3, 1e-2])),
        "reg_lambda": float(rng.choice([0.1, 1.0, 5.0, 10.0])),
        "n_estimators": int(rng.choice([120, 180, 260, 340]))}
    out = {}
    # b-global: one model per horizon on the stacked nodes
    Xs = Xtr.reshape(-1, F); Ys = ytr_s.reshape(-1, H)
    Xv = Xva.reshape(-1, F); Yv = yva_s.reshape(-1, H)
    rng = np.random.default_rng(42); best, best_cfg = np.inf, None
    for t in range(10):
        cfg = space(rng)
        m = xgb.XGBRegressor(objective="reg:squarederror", tree_method="hist", random_state=42, **cfg)
        m.fit(Xs, Ys[:, H // 2]); v = float(np.sqrt(np.mean((m.predict(Xv) - Yv[:, H // 2]) ** 2)))
        if v < best:
            best, best_cfg = v, cfg
    preds = []
    for s in range(n_seeds):
        P = np.zeros((Xte.shape[0] * N, H))
        for h in range(H):
            m = xgb.XGBRegressor(objective="reg:squarederror", tree_method="hist",
                                 random_state=42 + s, **best_cfg)
            m.fit(Xs, Ys[:, h]); P[:, h] = m.predict(Xte.reshape(-1, F))
        preds.append(P.reshape(Xte.shape[0], N, H) * sd + mu)
    out["XGBoost (b-global)"] = {"cfg": best_cfg, "preds": preds}
    print(f"XGBoost b-global: cfg {best_cfg}", flush=True)
    # b-local: one model per node and horizon, defaults of the notebook
    loc_cfg = {"learning_rate": 0.1, "max_depth": 4, "min_child_weight": 1.0, "subsample": 1.0,
               "colsample_bytree": 1.0, "n_estimators": 120}
    preds = []
    for s in range(n_seeds):
        P = np.zeros((Xte.shape[0], N, H))
        for n in range(N):
            for h in range(H):
                m = xgb.XGBRegressor(objective="reg:squarederror", tree_method="hist",
                                     random_state=42 + s, n_jobs=1, **loc_cfg)
                m.fit(Xtr[:, n, :], ytr_s[:, n, h]); P[:, n, h] = m.predict(Xte[:, n, :])
        preds.append(P * sd + mu)
        print(f"XGBoost b-local seed {s} done", flush=True)
    out["XGBoost (b-local)"] = {"cfg": loc_cfg, "preds": preds}
    return out


def raw_frames():
    tr = pd.read_csv(PATH_TRAIN); te = pd.read_csv(PATH_TEST)
    df = pd.concat([tr, te]); df["date"] = pd.to_datetime(df["date"], utc=True)
    return df.sort_values(["id_unique", "date"])


def split(df, a, b):
    lo = pd.Timestamp(DATA_KWARGS[a], tz="UTC"); hi = pd.Timestamp(DATA_KWARGS[b], tz="UTC")
    return df[(df.date >= lo) & (df.date < hi)]


def run_gam_arima_persistence_chronos():
    from pygam import LinearGAM, f as fac, l
    df = raw_frames()
    trv = split(df, "day_inf_train", "day_sup_val")      # train + validation history
    trn = split(df, "day_inf_train", "day_sup_train")
    tst = split(df, "day_inf_test", "day_sup_test")
    out = {}
    nat = lambda d: d.groupby("date").agg(consumption=("consumption", "sum"),
                                         consumption_l48=("consumption_l48", "sum"),
                                         weekday=("weekday", "first"),
                                         instant=("instant", "first")).reset_index()
    ntr, nte = nat(trn), nat(tst)
    # GAM b-global: one model per half-hour on the national series
    pred = np.zeros(len(nte))
    for h in range(H):
        mtr = ntr.instant.values == h; mte = nte.instant.values == h
        g = LinearGAM(fac(0) + l(1)).fit(ntr.loc[mtr, ["weekday", "consumption_l48"]].values.astype(float),
                                        ntr.loc[mtr, "consumption"].values.astype(float))
        pred[mte] = g.predict(nte.loc[mte, ["weekday", "consumption_l48"]].values.astype(float))
    out["GAM (b-global)"] = {"national": pred}
    # GAM b-local: one model per node and half-hour
    nodes = np.sort(trn.id_unique.unique()); pred = np.zeros(len(nte))
    for n in nodes:
        a = trn[trn.id_unique == n]; b = tst[tst.id_unique == n].sort_values("date")
        pn = np.zeros(len(b))
        for h in range(H):
            ma = a.instant.values == h; mb = b.instant.values == h
            g = LinearGAM(l(0) + l(1)).fit(a.loc[ma, ["weekday", "consumption_l48"]].values.astype(float),
                                          a.loc[ma, "consumption"].values.astype(float))
            pn[mb] = g.predict(b.loc[mb, ["weekday", "consumption_l48"]].values.astype(float))
        pred += pn
    out["GAM (b-local)"] = {"national": pred}
    print("GAM done", flush=True)
    # persistence on the national series
    hist = trv.groupby("date").consumption.sum().values
    true = tst.groupby("date").consumption.sum().values
    full = np.concatenate([hist, true]); start = len(hist)
    out["Persistence--1 d"] = {"national": full[start - H:start - H + len(true)]}
    out["Persistence--7 d"] = {"national": full[start - 7 * H:start - 7 * H + len(true)]}
    # ARIMA: 48 per-instant models on the national series, one-step updates, frozen parameters
    from statsmodels.tsa.arima.model import ARIMA
    try:
        from pmdarima import auto_arima
        order = auto_arima(hist, seasonal=False, stepwise=True, max_p=5, max_d=2, max_q=5,
                           error_action="ignore", suppress_warnings=True).order
    except Exception:
        order = (1, 0, 1)
    pred = np.zeros(len(true))
    for h in range(H):
        sub, subt = hist[h::H], true[h::H]
        try:
            fit = ARIMA(sub, order=order).fit()
            for t, obs in enumerate(subt):
                pred[h + t * H] = fit.get_forecast(1).predicted_mean[0]
                fit = fit.append([obs], refit=False)
        except Exception:
            pred[h::H] = np.concatenate([[sub[-1]], subt[:-1]])
    out["ARIMA"] = {"national": pred, "order": list(order)}
    print(f"ARIMA{order} done", flush=True)
    # Chronos-Bolt, zero-shot, national series, one forecast per day-aligned origin
    from chronos.chronos_bolt import ChronosBoltPipeline
    pipe = ChronosBoltPipeline.from_pretrained("amazon/chronos-bolt-small")
    pred = []
    for d in range(len(true) // H):
        ctx = torch.tensor(np.concatenate([hist, true[:d * H]]), dtype=torch.float32).unsqueeze(0)
        q = torch.as_tensor(pipe.predict(ctx, prediction_length=H))   # [1, Q, H]
        pred.append(q[0].median(dim=0).values.numpy() if q.ndim == 3 else q.reshape(-1)[:H].numpy())
    out["Chronos-Bolt"] = {"national": np.concatenate(pred)}
    print("Chronos-Bolt done", flush=True)
    return out, true


def baselines():
    tr, va, te = snapshots()
    true_nat = national(te[1])
    res = {}
    for k, v in run_mlps(tr, va, te).items():
        res[k] = v
    for k, v in run_xgb(tr, va, te).items():
        res[k] = v
    other, true2 = run_gam_arima_persistence_chronos()
    assert np.allclose(true2, true_nat), "baseline targets misaligned with the GNN test window"
    res.update(other)
    os.makedirs(RES_DIR, exist_ok=True)
    rows = {}
    store = {}
    for k, v in res.items():
        if "preds" in v:
            mets = [system_metrics(national(p), true_nat) for p in v["preds"]]
            r = np.array([m[0] for m in mets]); m_ = np.array([m[1] for m in mets])
            rows[k] = {"rmse": r.mean(), "rmse_sd": r.std(), "mape": m_.mean(), "mape_sd": m_.std(),
                       "n": len(mets), "cfg": v.get("cfg")}
            store[k] = np.stack(v["preds"])
        else:
            r, m_ = system_metrics(v["national"], true_nat)
            rows[k] = {"rmse": r, "mape": m_, "n": 1, **({"order": v["order"]} if "order" in v else {})}
            store[k] = v["national"]
        print(f"{k:22s} RMSE {rows[k]['rmse'] / 1000:8.1f} kWh  MAPE {rows[k]['mape']:.2f}%", flush=True)
    json.dump(rows, open(os.path.join(RES_DIR, "baselines.json"), "w"), indent=1, default=float)
    np.savez_compressed(os.path.join(RES_DIR, "baselines_preds.npz"), true=true_nat,
                        **{k.replace(" ", "_"): v for k, v in store.items()})


def mlp_hpo(n_trials=40, n_seeds=10):
    """Tune the two MLPs as the GNNs are tuned (TPE, same budget and epochs), then refit."""
    import optuna
    tr, va, te = snapshots()
    (Xtr, Ytr), (Xva, Yva), (Xte, Yte) = tr, va, te
    true_nat = national(Yte)
    rows = json.load(open(os.path.join(RES_DIR, "baselines.json")))
    store = dict(np.load(os.path.join(RES_DIR, "baselines_preds.npz")))
    hp = {}
    for name, per_node in (("Shared MLP", False), ("Per-node MLP", True)):
        def objective(trial):
            cfg = {"num_layers": trial.suggest_int("num_layers", 1, 3),
                   "hidden_channels": trial.suggest_int("hidden_channels", 32, 256, log=True),
                   "lr": trial.suggest_float("lr", 1e-4, 1e-2, log=True),
                   "batch_size": trial.suggest_categorical("batch_size", [2, 4, 8])}
            _, pva = mlp_fit(Xtr, Ytr, Xva, Yva, Xte, cfg, 42, per_node, epochs=200, patience=40)
            return system_metrics(national(pva), national(Yva))[0]
        study = optuna.create_study(direction="minimize", sampler=optuna.samplers.TPESampler(seed=42))
        study.optimize(objective, n_trials=n_trials)
        cfg = study.best_params
        preds = [mlp_fit(Xtr, Ytr, Xva, Yva, Xte, cfg, 42 + s, per_node, epochs=300, patience=300)[0]
                 for s in range(n_seeds)]
        mets = [system_metrics(national(q), true_nat) for q in preds]
        r = np.array([m[0] for m in mets]); m_ = np.array([m[1] for m in mets])
        rows[name] = {"rmse": r.mean(), "rmse_sd": r.std(), "mape": m_.mean(), "mape_sd": m_.std(),
                      "n": n_seeds, "cfg": cfg, "n_trials": n_trials}
        store[name.replace(" ", "_")] = np.stack(preds)
        hp[name] = {"params": cfg, "val_rmse": study.best_value, "n_trials": n_trials}
        print(f"{name}: cfg {cfg} RMSE {r.mean() / 1000:.1f} ± {r.std() / 1000:.1f} kWh  "
              f"MAPE {m_.mean():.2f} ± {m_.std():.2f}", flush=True)
    json.dump(rows, open(os.path.join(RES_DIR, "baselines.json"), "w"), indent=1, default=float)
    np.savez_compressed(os.path.join(RES_DIR, "baselines_preds.npz"), **store)
    json.dump(hp, open(os.path.join(RES_DIR, "hpo_mlp.json"), "w"), indent=1, default=float)


def xgb_hpo(n_trials=40, n_seeds=3):
    """Tune pooled XGBoost with the GNN budget (TPE, seed 42) on the mid-horizon model, then refit 48."""
    import optuna, xgboost as xgb
    tr, va, te = snapshots()
    (Xtr, Ytr), (Xva, Yva), (Xte, Yte) = tr, va, te
    S, N, F = Xtr.shape
    mu = Ytr.mean(axis=(0, 2), keepdims=True); sd = Ytr.std(axis=(0, 2), keepdims=True) + 1e-9
    Xs = Xtr.reshape(-1, F); Ys = ((Ytr - mu) / sd).reshape(-1, H)
    Xv = Xva.reshape(-1, F); Yv = ((Yva - mu) / sd).reshape(-1, H)
    def objective(trial):
        cfg = {"learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
               "max_depth": trial.suggest_int("max_depth", 3, 10),
               "min_child_weight": trial.suggest_float("min_child_weight", 1.0, 10.0, log=True),
               "subsample": trial.suggest_float("subsample", 0.6, 1.0),
               "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
               "reg_alpha": trial.suggest_float("reg_alpha", 1e-4, 1.0, log=True),
               "reg_lambda": trial.suggest_float("reg_lambda", 0.1, 10.0, log=True),
               "n_estimators": trial.suggest_int("n_estimators", 100, 400)}
        m = xgb.XGBRegressor(objective="reg:squarederror", tree_method="hist", random_state=42, **cfg)
        m.fit(Xs, Ys[:, H // 2])
        return float(np.sqrt(np.mean((m.predict(Xv) - Yv[:, H // 2]) ** 2)))
    study = optuna.create_study(direction="minimize", sampler=optuna.samplers.TPESampler(seed=42))
    study.optimize(objective, n_trials=n_trials)
    cfg = study.best_params
    true_nat = national(Yte)
    preds = []
    for s in range(n_seeds):
        P = np.zeros((Xte.shape[0] * N, H))
        for h in range(H):
            m = xgb.XGBRegressor(objective="reg:squarederror", tree_method="hist", random_state=42 + s, **cfg)
            m.fit(Xs, Ys[:, h]); P[:, h] = m.predict(Xte.reshape(-1, F))
        preds.append(P.reshape(Xte.shape[0], N, H) * sd + mu)
    mets = [system_metrics(national(q), true_nat) for q in preds]
    r = np.array([m[0] for m in mets]); m_ = np.array([m[1] for m in mets])
    rows = json.load(open(os.path.join(RES_DIR, "baselines.json")))
    store = dict(np.load(os.path.join(RES_DIR, "baselines_preds.npz")))
    rows["XGBoost (b-global)"] = {"rmse": r.mean(), "rmse_sd": r.std(), "mape": m_.mean(), "mape_sd": m_.std(),
                                  "n": n_seeds, "cfg": cfg, "n_trials": n_trials}
    store["XGBoost_(b-global)"] = np.stack(preds)
    json.dump(rows, open(os.path.join(RES_DIR, "baselines.json"), "w"), indent=1, default=float)
    np.savez_compressed(os.path.join(RES_DIR, "baselines_preds.npz"), **store)
    print(f"XGBoost b-global: cfg {cfg} RMSE {r.mean() / 1000:.1f} ± {r.std() / 1000:.1f} kWh  "
          f"MAPE {m_.mean():.2f} ± {m_.std():.2f}", flush=True)


# ---------------------------------------------------------------- aggregation
def mlpol(experts, y, step=H):
    """ML-Poly mixture with daily updates, uniform weights on the first day."""
    from opera import Mixture
    X = pd.DataFrame(experts)
    T, K = X.shape
    pred = np.empty(T)
    pred[:step] = X.iloc[:step].values @ (np.ones(K) / K)
    mix = Mixture(y=y[:step], experts=X.iloc[:step], model="MLpol", loss_type="mse")
    for s in range(step, T, step):
        e = min(s + step, T)
        pred[s:e] = np.ravel(mix.predict(new_experts=X.iloc[s:e]))
        mix.update(new_experts=X.iloc[s:e], new_y=y[s:e])
    return pred


def aggregate():
    cfg = json.load(open(os.path.join(HERE, "configs", "config_weave_500.json")))
    target = torch.load(os.path.join(RES_DIR, "target_nodes.pt")).numpy()     # [N, T]
    y = target.sum(axis=0)
    rows = {}
    arch_avg = {}
    for conv in cfg:
        folder = os.path.join(RES_DIR, conv)
        runs = [torch.load(os.path.join(folder, f), weights_only=False)["predictions"].numpy()
                for f in sorted(os.listdir(folder)) if f.endswith(".pt")]
        tops = [r.sum(axis=0) for r in runs]
        arch_avg[conv] = np.mean(runs, axis=0)
        rows[(conv, "Uniform")] = system_metrics(np.mean(tops, axis=0), y)
        rows[(conv, "Online")] = system_metrics(mlpol({f"s{i}": t for i, t in enumerate(tops)}, y), y)
    rows[("All", "Uniform")] = system_metrics(np.mean([a.sum(0) for a in arch_avg.values()], axis=0), y)
    rows[("All", "Top")] = system_metrics(mlpol({k: a.sum(0) for k, a in arch_avg.items()}, y), y)
    bottom = np.zeros_like(y)
    for n in range(target.shape[0]):
        bottom += mlpol({k: a[n] for k, a in arch_avg.items()}, target[n])
    rows[("All", "Bottom")] = system_metrics(bottom, y)
    for k, (r, m) in rows.items():
        print(f"{k[0]:16s} {k[1]:8s} RMSE {r / 1000:7.1f} kWh  MAPE {m:.2f}%", flush=True)
    json.dump({f"{a}|{b}": {"rmse": r, "mape": m} for (a, b), (r, m) in rows.items()},
              open(os.path.join(RES_DIR, "aggregation.json"), "w"), indent=1)


# ---------------------------------------------------------------- main
if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("stage")
    ap.add_argument("--conv", default=None)
    ap.add_argument("--trials", type=int, default=40)
    ap.add_argument("--seeds", type=int, default=3)
    a = ap.parse_args()
    if a.stage == "prep":
        prep()
    elif a.stage == "hpo":
        hpo(a.conv, a.trials)
    elif a.stage == "train":
        train(a.conv, a.seeds)
    elif a.stage == "baselines":
        baselines()
    elif a.stage == "mlp_hpo":
        mlp_hpo(a.trials)
    elif a.stage == "xgb_hpo":
        xgb_hpo(a.trials)
    elif a.stage == "aggregate":
        aggregate()
    elif a.stage == "time":
        p = {"adj_matrix": "correlation", "num_layers": 2, "hidden_channels": 64,
             "lr": 1e-3, "batch_size": 4}
        data, tr, va, te = datasets()
        print("samples", len(tr), len(va), len(te), "features", tr.num_node_features, flush=True)
        t0 = time.time()
        v, pred, target = fit_gnn(a.conv or "GraphSAGE", p, 42, 200, 200, "timing")
        print(f"val {v:.1f}, test {system_metrics(pred.numpy(), target.numpy())}, "
              f"{time.time() - t0:.0f}s", flush=True)
    else:
        raise SystemExit(f"unknown stage {a.stage}")
