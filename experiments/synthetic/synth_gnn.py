"""INFORMS-style synthetic study: when does a GNN beat an MLP, and is the gain from the MODEL
(weight sharing + equivariant neighbour aggregation) rather than mere access to the structure?

DGP (shared latent driver / denoising):
  clean temperature field    T*(t) ~ N(mu, s^2 C(rho)),  C spatial correlation from a graph
  observed temperature       temp_j(t) = T*_j(t) + xi_j(t),   xi ~ N(0, sig_obs^2 I)   (own noise)
  load                       load_j(t) = spline_j(T*_j(t)) + eta_j(t)                  (real temp->load curve)

The load depends on the CLEAN temperature; the model sees the NOISY one. Neighbours (correlated clean
temps, independent noise) let a node denoise its own reading. rho=0 -> no structure; sig_obs=0 -> own
reading already clean; both make the neighbourhood useless.

Models (feature = observed temp, target = load), national RMSE on held-out snapshots:
  per_node   : one net per node, own temp only            (no structure, no sharing)
  global     : one net per node, ALL nodes' temps          (structure via full access, no model bias)
  gnn_graph  : shared net + mean-aggregation over the graph (structure via the GNN inductive bias)
  gnn_eye    : shared net, no aggregation                   (sharing only, control)

Key contrast: gnn_graph vs global. If gnn_graph wins, the graph's benefit needs the MODEL, not just
access to neighbour features.
"""
import os, sys, time, json, gc, argparse
import numpy as np
import torch
from torch import nn
from scipy.interpolate import UnivariateSpline

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from _paths import ROOT  # noqa: E402
HERE = ROOT
DATA = os.path.join(HERE, "data/rfrance/train2.csv")
torch.set_num_threads(4)
np.seterr(all="ignore")   # silence spurious Apple-Accelerate BLAS matmul warnings (data verified finite)

# ---------------------------------------------------------------- real temp->load splines
_SPLINES = None
def real_splines():
    """One smoothing spline temp->load per region, fit on real data, normalised to zero-mean unit-std."""
    global _SPLINES
    if _SPLINES is not None:
        return _SPLINES
    import pandas as pd
    df = pd.read_csv(DATA, usecols=["Region", "temp", "load"])
    splines = []
    for reg, g in df.groupby("Region"):
        t = g["temp"].to_numpy(); y = g["load"].to_numpy()
        # bin by temperature, average load, fit a smooth spline on the binned curve
        order = np.argsort(t); t, y = t[order], y[order]
        edges = np.linspace(np.quantile(t, 0.01), np.quantile(t, 0.99), 40)
        mid = 0.5 * (edges[:-1] + edges[1:])
        binned = np.array([y[(t >= edges[i]) & (t < edges[i + 1])].mean() for i in range(len(mid))])
        ok = np.isfinite(binned)
        sp = UnivariateSpline(mid[ok], binned[ok], k=3, s=len(mid[ok]) * np.var(binned[ok]) * 0.05)
        splines.append({"sp": sp, "tmin": mid[ok].min(), "tmax": mid[ok].max(),
                        "ymu": float(np.mean(binned[ok])), "ysd": float(np.std(binned[ok]))})
    _SPLINES = splines
    return splines

def eval_spline(entry, temp):
    t = np.clip(temp, entry["tmin"], entry["tmax"])
    y = entry["sp"](t)
    return (y - entry["ymu"]) / (entry["ysd"] + 1e-9)   # standardised load curve

# ---------------------------------------------------------------- graph + spatial correlation
def geometric_graph(N, seed, k=6):
    """Random geometric graph: connect each node to its k nearest neighbours in [0,1]^2."""
    rng = np.random.default_rng(seed)
    pos = rng.random((N, 2))
    D = np.sqrt(((pos[:, None, :] - pos[None, :, :]) ** 2).sum(-1))
    A = np.zeros((N, N))
    for i in range(N):
        nn_i = np.argsort(D[i])[1:k + 1]
        A[i, nn_i] = 1.0; A[nn_i, i] = 1.0
    np.fill_diagonal(A, 0.0)
    return A, pos, D

def corr_matrix(D, rho):
    """Spatial correlation C from distances, scaled so typical neighbour correlation ~ rho.
    rho=0 -> identity (independent nodes)."""
    if rho <= 0:
        return np.eye(D.shape[0])
    d1 = np.median(D[D > 0])                 # typical near-neighbour distance
    ell = -d1 / np.log(max(rho, 1e-3))       # length so exp(-d1/ell)=rho
    C = np.exp(-D / ell)
    C = 0.5 * (C + C.T)
    C += 1e-4 * np.eye(D.shape[0])
    return C

# ---------------------------------------------------------------- data generation
def _sqrtm_psd(C):
    """Stable matrix square root via symmetric eigendecomposition (clip negative eigenvalues)."""
    w, V = np.linalg.eigh(0.5 * (C + C.T))
    w = np.clip(w, 0.0, None)
    return V @ np.diag(np.sqrt(w)) @ V.T

def generate(N, T, rho, sig_obs, seed, k=6):
    rng = np.random.default_rng(seed)
    A, pos, D = geometric_graph(N, seed, k=k)
    C = corr_matrix(D, rho)
    S = _sqrtm_psd(C)                                  # C = S S,  stable
    # clean temperature: spatially correlated field per snapshot, scaled to a realistic range
    z = rng.standard_normal((T, N)) @ S
    z = (z - z.mean()) / (z.std() + 1e-9)
    temp_clean = 12.0 + 6.0 * z                       # deg C, mean 12 sd ~6, spans splines' support
    temp_obs = temp_clean + rng.standard_normal((T, N)) * sig_obs
    sp = real_splines()
    load = np.empty((T, N), dtype=np.float32)
    for j in range(N):
        e = sp[j % len(sp)]
        load[:, j] = eval_spline(e, temp_clean[:, j]) + rng.standard_normal(T) * 0.05
    return (temp_obs.astype(np.float32), load, A.astype(np.float32))

# ---------------------------------------------------------------- models
class PerNode(nn.Module):
    """One MLP per node; input = own temp (global=False) or all temps (global=True)."""
    def __init__(self, N, hidden=32, glob=False):
        super().__init__()
        self.N, self.glob = N, glob
        din = N if glob else 1
        self.W1 = nn.Parameter(torch.randn(N, din, hidden) * (1.0 / max(1, din) ** 0.5))
        self.b1 = nn.Parameter(torch.zeros(N, hidden))
        self.W2 = nn.Parameter(torch.randn(N, hidden, 1) * (1.0 / hidden ** 0.5))
        self.b2 = nn.Parameter(torch.zeros(N, 1))
    def forward(self, x):                                 # x: [B, N] observed temps
        if self.glob:
            h = torch.einsum("bd,ndh->bnh", x, self.W1) + self.b1
        else:
            h = x.unsqueeze(-1) * self.W1[:, 0, :].unsqueeze(0) + self.b1   # own temp only, per-node weights
        h = torch.relu(h)
        out = torch.einsum("bnh,nho->bno", h, self.W2).squeeze(-1) + self.b2.squeeze(-1)
        return out                                        # [B, N]

class GNN(nn.Module):
    """Shared encoder + L rounds of mean aggregation over A (plus self), shared readout."""
    def __init__(self, A, hidden=32, layers=2, use_graph=True):
        super().__init__()
        N = A.shape[0]
        Ahat = A.copy(); np.fill_diagonal(Ahat, 1.0)
        if not use_graph:
            Ahat = np.eye(N)
        deg = Ahat.sum(1, keepdims=True); deg[deg == 0] = 1
        self.register_buffer("P", torch.tensor(Ahat / deg, dtype=torch.float32))   # row-normalised
        self.enc = nn.Linear(1, hidden)
        self.layers = layers
        self.lin = nn.ModuleList([nn.Linear(hidden, hidden) for _ in range(layers)])
        self.dec = nn.Linear(hidden, 1)
    def forward(self, x):                                  # x: [B, N]
        h = torch.relu(self.enc(x.unsqueeze(-1)))          # [B, N, H]
        for lin in self.lin:
            agg = torch.einsum("nm,bmh->bnh", self.P, h)   # neighbour mean
            h = torch.relu(lin(agg)) + h
        return self.dec(h).squeeze(-1)                     # [B, N]

def rmse(p, y):
    return float(np.sqrt(np.mean((p - y) ** 2)))

def train_eval(model, Xtr, Ytr, Xva, Yva, Xte, Yte, epochs=200, lr=1e-3, patience=30, seed=0):
    torch.manual_seed(seed)
    opt = torch.optim.Adam(model.parameters(), lr=lr); lf = nn.MSELoss()
    Xtr_, Ytr_ = torch.tensor(Xtr), torch.tensor(Ytr)
    Xva_, Yva_ = torch.tensor(Xva), torch.tensor(Yva)
    best, state, bad = 1e18, None, 0
    B = 128
    for ep in range(epochs):
        model.train(); perm = torch.randperm(len(Xtr_))
        for i in range(0, len(Xtr_), B):
            idx = perm[i:i + B]
            opt.zero_grad(); loss = lf(model(Xtr_[idx]), Ytr_[idx]); loss.backward(); opt.step()
        model.eval()
        with torch.no_grad():
            vl = float(lf(model(Xva_), Yva_))
        if vl < best - 1e-6:
            best, state, bad = vl, {k: v.clone() for k, v in model.state_dict().items()}, 0
        else:
            bad += 1
            if bad >= patience:
                break
    model.load_state_dict(state); model.eval()
    with torch.no_grad():
        p = model(torch.tensor(Xte)).numpy()
    return rmse(p, Yte)

def run_config(N, rho, sig_obs, frac, seed, T=1800, hidden=32):
    Xall, Yall, A = generate(N, T, rho, sig_obs, seed)
    ntr = int(0.7 * T); nva = int(0.15 * T)
    Xtr, Ytr = Xall[:ntr], Yall[:ntr]; Xva, Yva = Xall[ntr:ntr + nva], Yall[ntr:ntr + nva]
    Xte, Yte = Xall[ntr + nva:], Yall[ntr + nva:]
    k = max(16, int(frac * len(Xtr)))
    rng = np.random.default_rng(1000 + seed)
    sub = rng.choice(len(Xtr), size=k, replace=False)
    Xtr, Ytr = Xtr[sub], Ytr[sub]
    out = {}
    out["per_node"] = train_eval(PerNode(N, hidden, glob=False), Xtr, Ytr, Xva, Yva, Xte, Yte, seed=seed)
    out["global"]   = train_eval(PerNode(N, hidden, glob=True),  Xtr, Ytr, Xva, Yva, Xte, Yte, seed=seed)
    out["gnn_graph"] = train_eval(GNN(A, hidden, use_graph=True), Xtr, Ytr, Xva, Yva, Xte, Yte, seed=seed)
    out["gnn_eye"]   = train_eval(GNN(A, hidden, use_graph=False), Xtr, Ytr, Xva, Yva, Xte, Yte, seed=seed)
    return out

if __name__ == "__main__":
    t0 = time.time()
    print("PILOT: N=50, rho=0.7, sig_obs=3.0, frac=0.3, seed=0", flush=True)
    r = run_config(N=50, rho=0.7, sig_obs=3.0, frac=0.3, seed=0)
    print({k: round(v, 4) for k, v in r.items()}, f"  [{time.time()-t0:.0f}s]", flush=True)
    print("\ncontrol rho=0 (no structure): graph should NOT help", flush=True)
    r0 = run_config(N=50, rho=0.0, sig_obs=3.0, frac=0.3, seed=0)
    print({k: round(v, 4) for k, v in r0.items()}, flush=True)
    print("\ncontrol sig_obs=0 (clean own obs): graph should NOT help", flush=True)
    rc = run_config(N=50, rho=0.7, sig_obs=0.0, frac=0.3, seed=0)
    print({k: round(v, 4) for k, v in rc.items()}, flush=True)
