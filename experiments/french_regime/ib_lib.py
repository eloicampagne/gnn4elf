# Généré depuis expe_inductive_bias.ipynb : cellules de définition seulement.
import matplotlib
matplotlib.use('Agg')

# ===== cellule 2 =====
import os, sys, gc, json, time, copy, itertools, traceback
import numpy as np
import pandas as pd
import torch
from torch import nn
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from _paths import ROOT  # noqa: E402  (puts every experiment folder on sys.path)
HERE = ROOT

from graphtoolbox.data.dataset import DataClass, GraphDataset
from graphtoolbox.models import myGNN
from graphtoolbox.training import Trainer, set_device
import torch_geometric.nn.conv as pyg_conv

set_device("cpu")   # 12-node graphs: MPS is ~5x slower than CPU here
GRAPHS = os.path.join(HERE, "graph_representations_rfrance")
OUT    = os.path.join(HERE, os.environ.get("IB_OUT", "results_inductive_bias"))
os.makedirs(OUT, exist_ok=True)

OUT_CHANNELS = 48
SEEDS        = [int(s) for s in os.environ.get("IB_SEEDS", "42,0,1").split(",")]
EPOCHS       = int(os.environ.get("IB_EPOCHS", 300))
PATIENCE     = int(os.environ.get("IB_PATIENCE", 75))
HIDDEN_GNN   = 64
NUM_LAYERS   = 2

# expe.ipynb cell 4, rfrance branch
data_kwargs = {
    "node_var": "Region",
    "dummies": ["Instant", "JourSemaine", "DayType", "offset"],
    "day_inf_train": "2015-01-01", "day_sup_train": "2018-01-01",
    "day_inf_val": "2018-01-01", "day_sup_val": "2018-12-31",
    "day_inf_test": "2019-01-01", "day_sup_test": "2019-12-31",
}
dataset_kwargs = {
    "batch_size": 32, "adj_matrix": "space",
    "features_base": ["temp", "nebu", "wind", "tempMax", "tempMin", "Posan", "Instant",
                      "JourSemaine", "JourFerie", "offset", "DayType", "Weekend",
                      "temp_liss_fort", "temp_liss_faible"] + [f"load_l{t}" for t in range(1, 49)],
    "target_base": "load",
}

# --- second feature arm: the 48 load lags swapped for the 48 target-slot temperatures ---
# computed_features receives the full frame sorted by [node_var, date], so a groupby shift(-k)
# lands the temperature of slot k of the day on the midnight row that GraphDataset reads.
def _temp_at(k):
    return lambda df: df.groupby("Region")["temp"].shift(-k)

BASE_30 = ["temp", "nebu", "wind", "tempMax", "tempMin", "Posan", "JourSemaine", "JourFerie",
           "offset", "DayType", "Weekend", "temp_liss_fort", "temp_liss_faible"]

data_kwargs_nolag = dict(data_kwargs)
data_kwargs_nolag["dummies"] = ["JourSemaine", "DayType", "offset"]   # Instant dummies are dead, see below
data_kwargs_nolag["computed_features"] = {f"temp_h{k}": _temp_at(k) for k in range(OUT_CHANNELS)}

dataset_kwargs_nolag = dict(dataset_kwargs)
dataset_kwargs_nolag["features_base"] = BASE_30 + [f"temp_h{k}" for k in range(OUT_CHANNELS)]

# --- third arm: both windows at once, past lags AND target-slot temperatures ---
data_kwargs_both = dict(data_kwargs_nolag)
dataset_kwargs_both = dict(dataset_kwargs)
dataset_kwargs_both["features_base"] = (BASE_30 + [f"temp_h{k}" for k in range(OUT_CHANNELS)]
                                        + [f"load_l{t}" for t in range(1, OUT_CHANNELS + 1)])

ARMS = {"lags":  (data_kwargs,       dataset_kwargs),
        "nolag": (data_kwargs_nolag, dataset_kwargs_nolag),
        "both":  (data_kwargs_both,  dataset_kwargs_both)}

# Every cached result carries the configuration that produced it. Without this a file written
# at one epoch budget is silently reused at another, and a single table ends up comparing models
# trained for different lengths -- which is exactly how a 150-epoch baseline came to be scored
# against a 300-epoch GNN here.
CACHE_STAMP = {"epochs": EPOCHS, "patience": PATIENCE, "seeds": SEEDS,
               "hidden_gnn": HIDDEN_GNN, "layers": NUM_LAYERS}

def _load_cache(p, name):
    if not os.path.exists(p):
        return {}
    blob = json.load(open(p))
    if "_config" not in blob:
        raise RuntimeError(f"{p} : fichier sans provenance, anterieur au marquage. Supprime-le.")
    if blob["_config"] != CACHE_STAMP:
        raise RuntimeError(
            f"{name} : le cache a ete calcule avec {blob['_config']},\n"
            f"la configuration actuelle est {CACHE_STAMP}.\n"
            f"Melanger deux budgets dans un meme tableau fausse la comparaison.\n"
            f"Supprime {p}, ou retablis la configuration precedente.")
    return blob["results"]

def _save_cache(p, res):
    json.dump({"_config": CACHE_STAMP, "results": res}, open(p, "w"), indent=1)

def cache(name, fn):
    """Run fn once, keep the result on disk. Delete the file in results_inductive_bias to redo."""
    p = os.path.join(OUT, name + ".json")
    r = _load_cache(p, name)
    if r:
        return r
    r = fn()
    _save_cache(p, r)
    return r

def cache_each(name, keys, fn, show=lambda v: ""):
    """Compute fn(key) for every key, saving after each one.

    Whole-phase caching would throw away an hour of training on any interruption, so the
    results file is rewritten entry by entry: rerunning the cell resumes where it stopped.
    A key that raises is reported and skipped rather than killing the sweep.
    """
    p = os.path.join(OUT, name + ".json")
    res = _load_cache(p, name)
    todo = [k for k in keys if k not in res]
    print(f"{name}: {len(res)} en cache, {len(todo)} a calculer", flush=True)
    failed = 0
    for k in todo:
        t0 = time.time()
        try:
            res[k] = fn(k)
            _save_cache(p, res)
            print(f"  [{k:38}] {show(res[k]):40} ({time.time()-t0:.0f}s)", flush=True)
        except Exception as e:
            failed += 1
            if failed == 1:          # the first failure carries the diagnosis; show it in full
                traceback.print_exc()
            print(f"  [{k:38}] ECHEC: {type(e).__name__}: {e}", flush=True)
    if not res:
        # Returning {} here would surface three cells later as an unrelated pandas error on an
        # empty frame, so fail where the cause is still visible.
        raise RuntimeError(f"{name}: aucune configuration n'a abouti ({failed} echecs). "
                           f"La trace ci-dessus donne la cause.")
    return res

# ===== cellule 4 =====
PREFECTURES = {   # (lat, lon) des préfectures de région, ordre alphabétique des noms de nœuds
    "Auvergne_Rhone_Alpes": (45.76, 4.83),        # Lyon
    "Bourgogne_Franche_Comte": (47.32, 5.04),     # Dijon
    "Bretagne": (48.11, -1.68),                   # Rennes
    "Centre_Val_de_Loire": (47.90, 1.90),         # Orleans
    "Grand_Est": (48.58, 7.75),                   # Strasbourg
    "Hauts_de_France": (50.63, 3.06),             # Lille
    "Ile_de_France": (48.86, 2.35),               # Paris
    "Normandie": (49.44, 1.10),                   # Rouen
    "Nouvelle_Aquitaine": (44.84, -0.58),         # Bordeaux
    "Occitanie": (43.60, 1.44),                   # Toulouse
    "Pays_de_la_Loire": (47.22, -1.55),           # Nantes
    "Provence_Alpes_Cote_d_Azur": (43.30, 5.37),  # Marseille
}

def haversine(a, b):
    R = 6371.0
    la1, lo1, la2, lo2 = map(np.radians, [a[0], a[1], b[0], b[1]])
    h = np.sin((la2 - la1) / 2) ** 2 + np.cos(la1) * np.cos(la2) * np.sin((lo2 - lo1) / 2) ** 2
    return 2 * R * np.arcsin(np.sqrt(h))

def build_gaussian(nodes, name="gaussian"):
    """W_ij = exp(-d_ij^2 / sigma^2) on the complete graph, sigma = sd of off-diagonal distances."""
    D = np.array([[haversine(PREFECTURES[a], PREFECTURES[b]) for b in nodes] for a in nodes])
    off = ~np.eye(len(nodes), dtype=bool)
    sigma = D[off].std()
    W = np.exp(-(D ** 2) / (sigma ** 2))
    np.fill_diagonal(W, 1.0)
    d = os.path.join(GRAPHS, name); os.makedirs(d, exist_ok=True)
    np.savetxt(os.path.join(d, "W.txt"), W)
    return W, D, sigma

_DATA = {}
def get_data(arm="lags"):
    if arm not in _DATA:
        _DATA[arm] = DataClass(path_train=os.path.join(HERE, "data/rfrance/train2.csv"),
                               path_test=os.path.join(HERE, "data/rfrance/test2.csv"),
                               data_kwargs=ARMS[arm][0], folder_config=HERE)
    return _DATA[arm]

_DS = {}
def resolve_adj(adj, arm):
    """The learned adjacency is arm-specific (learned end-to-end on that arm's features), so the
    virtual name 'learned' resolves to the frozen graph learned on the matching arm."""
    return f"learned_{arm}" if adj == "learned" else adj

def datasets(adj, arm="lags"):
    """Train/val/test GraphDataset triple on adjacency `adj`, scalers fitted on train."""
    if (adj, arm) in _DS:
        return _DS[(adj, arm)]
    real = resolve_adj(adj, arm)
    data = get_data(arm)
    dk = dict(ARMS[arm][1]); dk["adj_matrix"] = real
    common = dict(graph_folder=GRAPHS, dataset_kwargs=dk, out_channels=OUT_CHANNELS)
    tr = GraphDataset(data=data, period="train", **common)
    va = GraphDataset(data=data, period="val", scalers_feat=tr.scalers_feat,
                      scalers_target=tr.scalers_target, **common)
    te = GraphDataset(data=data, period="test", scalers_feat=tr.scalers_feat,
                      scalers_target=tr.scalers_target, **common)
    for ds in (tr, va, te):
        ds._set_adj_matrix(adj_matrix=real)
    _DS[(adj, arm)] = (tr, va, te)
    return _DS[(adj, arm)]

NODES = list(get_data().nodes)
W_gauss, D_km, SIGMA = build_gaussian(NODES)
print(f"sigma = {SIGMA:.0f} km   poids hors diagonale : "
      f"[{W_gauss[~np.eye(12,dtype=bool)].min():.3f}, {W_gauss[~np.eye(12,dtype=bool)].max():.3f}]")

# 'learned' is arm-specific: the row here reads the graph learned on the lags arm, with
# learned_nolag reported in the section-D comparison. Both must have been produced by the
# learn_graph.py step; a missing file drops the row rather than crashing the sweep.
ADJS = ["eye", "space", "gaussian", "dtw", "learned"]
ADJS = [a for a in ADJS
        if os.path.exists(os.path.join(GRAPHS, resolve_adj(a, "lags"), "W.txt"))]
rows = []
for a in ADJS:
    W = np.loadtxt(os.path.join(GRAPHS, resolve_adj(a, "lags"), "W.txt"))
    off = ~np.eye(12, dtype=bool)
    directed = not np.allclose(W, W.T)
    rows.append({"graphe": a, "dirige": directed,
                 "aretes hors diag": int((W[off] != 0).sum()), "sur": int(off.sum()),
                 "degre moyen": round(float((W[off] != 0).sum() / 12), 1),
                 "poids moyen": round(float(W[off][W[off] != 0].mean()) if (W[off] != 0).any() else 0.0, 3)})
pd.DataFrame(rows)

# ===== cellule 8 =====
CONVS = {
    "ChebConv":          dict(family="spectrale",     kwargs={"K": 3}),
    "SAGEConv":          dict(family="spatiale",      kwargs={}),
    "GATv2Conv":         dict(family="attentionnelle", kwargs={}),
    "ResGatedGraphConv": dict(family="relationnelle", kwargs={}),
}
pd.DataFrame([{"convolution": k, "famille": v["family"]} for k, v in CONVS.items()])

# ===== cellule 10 =====
def self_loop_edges(n=12):
    """E_ref: the 12 self-loops alone, i.e. the eye graph's edge_index."""
    i = torch.arange(n)
    return torch.stack([i, i])

def neighbour_share(model, test, ndays=20, naive=False):
    """Share of the convolution term contributed by genuine neighbours, per layer, averaged over days.

    naive=True reproduces the empty-edge reference, kept only to exhibit the self-loop artefact.
    """
    ref = torch.empty((2, 0), dtype=torch.long) if naive else self_loop_edges()
    out = []
    with torch.no_grad():
        for d in range(min(ndays, len(test))):
            g = test[d]
            x = model.node_encoder(g.x)
            sh = []
            for layer in model.layers:
                hin = layer.act(layer.norm(x))
                h = layer.conv(hin, g.edge_index)
                nb = h - layer.conv(hin, ref)
                sh.append(float(torch.norm(nb) / (torch.norm(h) + 1e-12)))
                x = x + h
            out.append(sh)
    return float(np.mean(out))

def rmse(p, y):
    return float(np.sqrt(np.mean((np.asarray(p) - np.asarray(y)) ** 2)))

# ===== cellule 12 =====
def run_gnn(adj, conv, seed, epochs=EPOCHS, hidden=HIDDEN_GNN, train_subset=None, measure=True,
            arm="lags"):
    """Train myGNN and return (national RMSE, neighbour share, #params)."""
    tr, va, te = datasets(adj, arm)
    if train_subset is not None:
        tr = copy.copy(tr)
        tr.pyg_data = [datasets(adj, arm)[0].pyg_data[i] for i in train_subset]
    set_device("cpu"); torch.manual_seed(seed); np.random.seed(seed); gc.collect()
    m = myGNN(in_channels=te.num_node_features, num_layers=NUM_LAYERS, hidden_channels=hidden,
              out_channels=OUT_CHANNELS, conv_class=getattr(pyg_conv, conv),
              conv_kwargs=CONVS[conv]["kwargs"], heads=1)
    npar = sum(p.numel() for p in m.parameters())
    t = Trainer(model=m, dataset_train=tr, dataset_val=va, dataset_test=te, batch_size=16,
                model_kwargs={"lr": 1e-3, "num_epochs": epochs}, reconcile=False, lam_reg=0)
    pred, target, *_ = t.train(plot_loss=False, force_training=True, save=False, patience=PATIENCE)
    r = rmse(pred.sum(0).cpu().numpy(), target.sum(0).cpu().numpy())
    if not measure:
        return r, None, npar
    return r, neighbour_share(m, te), npar

def _control():
    rows = []
    for conv in CONVS:
        tr, va, te = datasets("eye")
        set_device("cpu"); torch.manual_seed(42)
        m = myGNN(in_channels=te.num_node_features, num_layers=NUM_LAYERS, hidden_channels=HIDDEN_GNN,
                  out_channels=OUT_CHANNELS, conv_class=getattr(pyg_conv, conv),
                  conv_kwargs=CONVS[conv]["kwargs"], heads=1)
        rows.append({"convolution": conv,
                     "part naive (aretes vides)": round(neighbour_share(m, te, ndays=5, naive=True), 4),
                     "part corrigee (boucles seules)": round(neighbour_share(m, te, ndays=5), 4)})
    return rows

ctrl = pd.DataFrame(cache("control_metric", _control))
print("Graphe eye : aucun voisin n'existe. Toute part non nulle est un artefact.")
ctrl

# ===== cellule 14 =====
def stack(ds):
    return np.stack([d.x.numpy() for d in ds]), np.stack([d.y.numpy() for d in ds])

def _fit_mlp(a, yt, b, yv, c, hidden, epochs, seed, out_dim=OUT_CHANNELS):
    torch.manual_seed(seed)
    m = nn.Sequential(nn.Linear(a.shape[1], hidden), nn.ReLU(),
                      nn.Linear(hidden, hidden), nn.ReLU(), nn.Linear(hidden, out_dim))
    opt = torch.optim.Adam(m.parameters(), lr=1e-3); lf = nn.MSELoss()
    best, state, bad = 1e18, None, 0
    for _ in range(epochs):
        m.train(); perm = torch.randperm(len(a))
        for k in range(0, len(a), 64):
            i = perm[k:k + 64]
            opt.zero_grad(); lf(m(a[i]), yt[i]).backward(); opt.step()
        m.eval()
        with torch.no_grad():
            vl = float(lf(m(b), yv))
        if vl < best - 1e-6:
            best, state, bad = vl, {k: v.clone() for k, v in m.state_dict().items()}, 0
        else:
            bad += 1
            if bad >= PATIENCE:
                break
    m.load_state_dict(state); m.eval()
    with torch.no_grad():
        return m(c).numpy(), best, sum(p.numel() for p in m.parameters())

def _z(A, ref):
    """Standardise A with ref's moments.

    A feature that is constant within ref must be zeroed, not divided by ~0. At small training
    fractions the rare calendar dummies do go constant while still varying in test, and dividing
    by std+1e-6 then maps them to ~1e6, which is what made the MLP diverge on the sample curve.
    """
    mu, sg = ref.mean(0), ref.std(0)
    dead = sg < 1e-8
    sg = np.where(dead, 1.0, sg)
    Z = (A - mu) / sg
    Z[:, dead] = 0.0
    return torch.tensor(Z, dtype=torch.float32)

def run_mlp(kind, seed, hidden, epochs=EPOCHS, train_subset=None, arm="lags"):
    """kind in {per_node, shared, all_per_node}. Returns (national RMSE, val loss, #params)."""
    tr, va, te = datasets("eye", arm)
    Xtr, Ytr = stack(tr); Xva, Yva = stack(va); Xte, Yte = stack(te)
    if train_subset is not None:
        Xtr, Ytr = Xtr[train_subset], Ytr[train_subset]
    N = Xtr.shape[1]
    pred, vls, npar = np.zeros_like(Yte), [], 0

    if kind == "shared":
        A = Xtr.reshape(-1, Xtr.shape[2]); B = Xva.reshape(-1, Xva.shape[2]); Cc = Xte.reshape(-1, Xte.shape[2])
        ya = Ytr.reshape(-1, OUT_CHANNELS); yb = Yva.reshape(-1, OUT_CHANNELS)
        ym, ys = ya.mean(), ya.std() + 1e-6
        p, vl, npar = _fit_mlp(_z(A, A), torch.tensor((ya - ym) / ys, dtype=torch.float32),
                               _z(B, A), torch.tensor((yb - ym) / ys, dtype=torch.float32),
                               _z(Cc, A), hidden, epochs, seed)
        pred = (p * ys + ym).reshape(Yte.shape); vls = [vl]
    else:
        for i in range(N):
            if kind == "per_node":
                A, B, Cc = Xtr[:, i, :], Xva[:, i, :], Xte[:, i, :]
            elif kind == "all_per_node":
                A = Xtr.reshape(len(Xtr), -1); B = Xva.reshape(len(Xva), -1); Cc = Xte.reshape(len(Xte), -1)
            else:
                raise ValueError(kind)
            ym, ys = Ytr[:, i, :].mean(), Ytr[:, i, :].std() + 1e-6
            p, vl, np_i = _fit_mlp(_z(A, A), torch.tensor((Ytr[:, i, :] - ym) / ys, dtype=torch.float32),
                                   _z(B, A), torch.tensor((Yva[:, i, :] - ym) / ys, dtype=torch.float32),
                                   _z(Cc, A), hidden, epochs, seed)
            pred[:, i, :] = p * ys + ym; vls.append(vl); npar += np_i
    return rmse(pred.sum(1).ravel(), Yte.sum(1).ravel()), float(np.mean(vls)), npar
