"""Per-node nRMSE of the four French nodewise models at the full training sample.

The French comparison of the paper (tab:data-regime) reports national RMSE only. The
large-panel sections read a per-node error instead, each node's target standardised by
its training mean and standard deviation, so this script refits the four models of
ib_lib.py at 1090 days (same seeds, width, depth, epochs, patience) and scores both.
"""
import json, os, sys
import numpy as np, torch
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from _paths import ROOT, DATA, CACHE  # noqa: E402,F401  (puts every experiment folder on sys.path)
import ib_lib as L  # noqa: E402  (runs the notebook setup on import)
from graphtoolbox.models import myGNN
from graphtoolbox.training import Trainer, set_device
import torch_geometric.nn.conv as pyg_conv

OUT = os.path.join(L.HERE, "results_inductive_bias", "french_nrmse.json")
tr, va, te = L.datasets("eye")
Xtr, Ytr = L.stack(tr); Xte, Yte = L.stack(te)
SD = Ytr.transpose(1, 0, 2).reshape(Ytr.shape[1], -1).std(1)      # per node, training window

def nrmse_nodes(pred_nt, true_nt):
    return float(np.mean(np.sqrt(np.mean((pred_nt - true_nt) ** 2, axis=1)) / SD))

def mlp(kind, seed):
    Xva, Yva = L.stack(va); N = Xtr.shape[1]; pred = np.zeros_like(Yte)
    if kind == "shared":
        raise ValueError
    for i in range(N):
        A, B, C = ((Xtr[:, i, :], Xva[:, i, :], Xte[:, i, :]) if kind == "per_node" else
                   (Xtr.reshape(len(Xtr), -1), Xva.reshape(len(Xva), -1), Xte.reshape(len(Xte), -1)))
        ym, ys = Ytr[:, i, :].mean(), Ytr[:, i, :].std() + 1e-6
        p, _, _ = L._fit_mlp(L._z(A, A), torch.tensor((Ytr[:, i, :] - ym) / ys, dtype=torch.float32),
                             L._z(B, A), torch.tensor((Yva[:, i, :] - ym) / ys, dtype=torch.float32),
                             L._z(C, A), L.HIDDEN_GNN, L.EPOCHS, seed)
        pred[:, i, :] = p * ys + ym
    p_nt = pred.transpose(1, 0, 2).reshape(N, -1); t_nt = Yte.transpose(1, 0, 2).reshape(N, -1)
    return L.rmse(p_nt.sum(0), t_nt.sum(0)), nrmse_nodes(p_nt, t_nt)

def gnn(adj, seed):
    a, b, c = L.datasets(adj)
    set_device("cpu"); torch.manual_seed(seed); np.random.seed(seed)
    m = myGNN(in_channels=c.num_node_features, num_layers=L.NUM_LAYERS, hidden_channels=L.HIDDEN_GNN,
              out_channels=L.OUT_CHANNELS, conv_class=pyg_conv.SAGEConv, conv_kwargs={}, heads=1)
    t = Trainer(model=m, dataset_train=a, dataset_val=b, dataset_test=c, batch_size=16,
                model_kwargs={"lr": 1e-3, "num_epochs": L.EPOCHS}, reconcile=False, lam_reg=0)
    pred, target, *_ = t.train(plot_loss=False, force_training=True, save=False, patience=L.PATIENCE)
    p, y = pred.cpu().numpy(), target.cpu().numpy()
    return L.rmse(p.sum(0), y.sum(0)), nrmse_nodes(p, y)

res = json.load(open(OUT)) if os.path.exists(OUT) else {}
for seed in L.SEEDS:
    for name, fn in (("per_node", lambda s: mlp("per_node", s)), ("global", lambda s: mlp("all_per_node", s)),
                     ("identity_gnn", lambda s: gnn("eye", s)), ("sage_space", lambda s: gnn("space", s))):
        k = f"{name}|{seed}"
        if k in res: continue
        r, n = fn(seed); res[k] = {"rmse": r, "nrmse": n}
        json.dump(res, open(OUT, "w"), indent=1)
        print(f"{k:22s} RMSE {r:7.1f}  nRMSE {n:.4f}", flush=True)
for name in ("per_node", "global", "identity_gnn", "sage_space"):
    v = [res[f"{name}|{s}"] for s in L.SEEDS]
    r = np.array([x["rmse"] for x in v]); n = np.array([x["nrmse"] for x in v])
    print(f"{name:14s} RMSE {r.mean():7.1f} ({r.std(ddof=1):.1f})  nRMSE {n.mean():.4f} ({n.std(ddof=1):.4f})", flush=True)
print("=== termine ===", flush=True)
