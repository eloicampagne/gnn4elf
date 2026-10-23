"""Experts for the aggregation figures of the manuscript, four cells.

Two datasets and two forecasting settings are crossed.

The French regional data carry whatever inter-node structure the grid and the
weather actually produce. The synthetic data are generated as the chapter
describes: a temperature field

    T_j(t) = a t + b_j (cos w_1 t + cos w_2 t),   b ~ N(mu, C),

with mu and C the mean and the covariance of the observed regional temperatures,
each region rescaled to its observed range, and a load obtained by passing that
temperature through a cubic spline fitted to the observed load of the region,

    L_j(t) = f_j(T_j(t)) + eps_j(t),   eps ~ N(0, Sigma).

Sigma = diag leaves the regions independent given their own temperature, which is
the case with no structure for a neighbourhood to carry. Sigma = rho(W) couples
the noise over the space graph.

The single-step setting predicts the load of an instant from the covariates of
that instant, with no load lag. The sequence-to-sequence setting predicts the
next forty-eight half-hours from the covariates of the issue instant and the
forty-eight past loads, training on windows of stride one and evaluating on
windows of stride forty-eight, as in the chapter.

Seven experts run in every cell: five graph networks carrying the convolution
and the adjacency of the tuned configuration, a weight-shared feedforward
network, and the operational additive model of gam_expert.py.

    python aggreg_experts.py rfrance single
    python aggreg_experts.py synth   single
    python aggreg_experts.py synth_structured single
    python aggreg_experts.py rfrance seq
    python aggreg_experts.py synth   seq

Each run writes results_aggreg/experts_<dataset>_<setting>.csv, one column per
expert plus the observed national load, over the test year.
"""

import os
import sys
import time

import numpy as np
import pandas as pd
import torch
from torch import nn

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from _paths import ROOT, DATA, CACHE  # noqa: E402,F401  (puts every experiment folder on sys.path)
import torch_geometric.nn.conv as pyg_conv  # noqa: E402
from graphtoolbox.models import myGNN  # noqa: E402
from torch_geometric.utils import dense_to_sparse  # noqa: E402

HERE = ROOT
OUT = os.path.join(HERE, "results_aggreg")
GRAPHS = os.path.join(HERE, "graph_representations_rfrance")
os.makedirs(OUT, exist_ok=True)

TRAIN_FROM, TRAIN_TO = "2017-01-01", "2018-07-01"
VAL_TO = "2019-01-01"
SEED = 0
EPOCHS, PATIENCE = 60, 8
HORIZON = 48                                               # steps predicted at once
SYNTH_NOISE = "indep"                                      # "indep" or "graph"
SYNTH_RHO = 0.9                                            # coupling of the graph noise
SYNTH_NOISE_SHARE = 0.15                                   # noise sd over signal sd

# Covariates of the operational load GAM. No load lag in the single-step
# setting, thermal inertia entering through the smoothed temperatures.
FEATS_REAL = ["temp", "nebu", "wind", "temp_liss_fort", "temp_liss_faible",
              "tempMax", "tempMin"]
# The generator produces a temperature and nothing else, so the synthetic panel
# keeps the temperature and what derives from it.
FEATS_SYNTH = ["temp", "temp_liss_fort", "temp_liss_faible", "tempMax", "tempMin"]
NODE_FEATS = FEATS_REAL                                    # set by main()

GNNS = {
    "GCN":  dict(conv="GCNConv",  adj="space",        layers=1, hidden=170, lr=3e-3),
    "SAGE": dict(conv="SAGEConv", adj="correlation",  layers=2, hidden=401, lr=4e-4),
    "GAT":  dict(conv="GATConv",  adj="gl3sr",        layers=3, hidden=364, lr=1e-3, heads=2),
    "Cheb": dict(conv="ChebConv", adj="distsplines",  layers=1, hidden=249, lr=6e-4, K=3),
    # TAG is left out of the four cells. Its three layers at width 329 never
    # trigger the early stop and cost between twenty-four minutes and nine hours
    # per cell, against five to nine megawatts on the mixture and four to nine on
    # what the graph experts contribute, measured on the cells that had it.
}
BATCH = 256


# --------------------------------------------------------------------- data
def read_real():
    """Observed panel: weather [T, N, F], calendar [T, 4], load [T, N]."""
    cols = ["date", "Region", "load", "Instant", "Posan", "DayType",
            "offset"] + FEATS_REAL
    tr = pd.read_csv(os.path.join(HERE, "data/rfrance/train2.csv"), usecols=cols)
    te = pd.read_csv(os.path.join(HERE, "data/rfrance/test2.csv"), usecols=cols)
    df = pd.concat([tr, te], ignore_index=True)
    # ISO timestamps sort lexicographically. Converting the NumPy array avoids
    # pandas' string-to-datetime path, which crashes with the current Python
    # 3.14 wheels on macOS.
    df = df.sort_values(["date", "Region"])
    nodes = np.sort(df["Region"].unique())
    dates = np.sort(df["date"].unique()).astype("datetime64[m]")
    T, N = len(dates), len(nodes)
    xn = df[FEATS_REAL].to_numpy(dtype=np.float64).reshape(T, N, len(FEATS_REAL))
    cal = df[["Instant", "Posan", "DayType", "offset"]] \
        .to_numpy(dtype=np.float64).reshape(T, N, 4)[:, 0, :]
    y = df["load"].to_numpy(dtype=np.float64).reshape(T, N)
    return xn, cal, y, nodes, dates


def spline_response(temp, load, lo, hi, bins=60):
    """Cubic response of load to temperature, read off binned training means."""
    from scipy.interpolate import CubicSpline
    edges = np.linspace(lo, hi, bins + 1)
    idx = np.clip(np.digitize(temp, edges) - 1, 0, bins - 1)
    centre, value = [], []
    for b in range(bins):
        m = idx == b
        if m.sum() >= 20:
            centre.append(0.5 * (edges[b] + edges[b + 1]))
            value.append(load[m].mean())
    c, v = np.array(centre), np.array(value)
    sp = CubicSpline(c, v)
    # Bins with too few observations leave the tails without knots; evaluating
    # there would let the cubic run away, so the argument is held at the range
    # the training data actually covered.
    return lambda t: sp(np.clip(t, c[0], c[-1]))


def make_synth(xn, cal, y, dates, itr, noise, seed=0):
    """Temperature field and load of the chapter's generator."""
    rng = np.random.default_rng(seed)
    n = y.shape[1]
    temp = xn[:, :, 0]
    mu, C = temp[itr].mean(0), np.cov(temp[itr].T)
    b = rng.multivariate_normal(mu, C)                     # one amplitude per region
    t = np.arange(len(dates), dtype=float)
    a = 0.5 / len(t)                                       # half a degree over the record
    cycle = np.cos(2 * np.pi * t / 48) + np.cos(2 * np.pi * t / (48 * 365.25))
    Tg = a * t[:, None] + b[None, :] * cycle[:, None]
    lo, hi = temp.min(0), temp.max(0)
    Tg = lo + (Tg - Tg.min(0)) * (hi - lo) / (Tg.max(0) - Tg.min(0))

    clean = np.zeros_like(Tg)
    for j in range(n):
        f = spline_response(temp[itr, j], y[itr, j], lo[j], hi[j])
        clean[:, j] = f(Tg[:, j])
    # The noise is set as a share of the generated signal. Taking instead the
    # residual of the real load around a temperature-only spline would put the
    # whole daily and weekly profile into eps and leave a panel that is almost
    # pure noise, on which no model can be told from another.
    sd = SYNTH_NOISE_SHARE * clean[itr].std(0)
    if noise == "indep":
        R = np.eye(n)
    else:
        A = np.loadtxt(os.path.join(GRAPHS, "space", "W.txt"))
        np.fill_diagonal(A, 0.0)
        d = np.maximum(A.sum(1), 1e-8)
        Ah = A / np.sqrt(np.outer(d, d))                   # symmetric normalisation
        S = np.linalg.inv(np.eye(n) - SYNTH_RHO * Ah)      # conditional autoregression
        s = np.sqrt(np.diag(S))
        R = S / np.outer(s, s)
    Sigma = R * np.outer(sd, sd)
    eps = rng.multivariate_normal(np.zeros(n), Sigma, size=len(t))
    ys = clean + eps

    # The generator gives a temperature; the smoothed temperatures and the daily
    # extremes are rebuilt from it with the operational coefficients.
    out = [Tg]
    for alpha in (0.997, 0.972):
        z = np.zeros_like(Tg)
        z[0] = Tg[0]
        for k in range(1, len(Tg)):
            z[k] = alpha * z[k - 1] + (1 - alpha) * Tg[k]
        out.append(z)
    day = np.arange(len(Tg)) // 48
    for red in (np.maximum, np.minimum):
        agg = pd.DataFrame(Tg).groupby(day).agg("max" if red is np.maximum else "min")
        out.append(agg.to_numpy()[day])
    return np.stack(out, 2), ys


def calendar_block(cal):
    """Half-hour indicators, yearly harmonics, day type and period indicators."""
    inst, posan = cal[:, 0].astype(int), cal[:, 1]
    dtype, off = cal[:, 2].astype(int), cal[:, 3].astype(int)
    enc = [(inst == k).astype(float) for k in range(48)]
    enc += [f(2 * np.pi * m * posan) for m in (1, 2, 3) for f in (np.sin, np.cos)]
    enc += [(dtype == k).astype(float) for k in np.unique(dtype)]
    enc += [(off == k).astype(float) for k in np.unique(off)]
    return np.stack(enc, 1)


def build(setting, xn, cal, y):
    """Model inputs and targets for the requested setting.

    single: x[t] holds the covariates of t, y[t] the load of t.
    seq:    x[t] holds those covariates and the forty-eight past loads of every
            node, y[t] the forty-eight loads from t on.
    """
    T, N = y.shape
    xc = np.repeat(calendar_block(cal)[:, None, :], N, 1)
    if setting == "single":
        return np.concatenate([xn, xc], 2), y, np.arange(T)
    # Row i is the window issued at t = HORIZON + i: it reads the loads of
    # t-1 down to t-HORIZON and predicts t up to t+HORIZON-1.
    keep = T - 2 * HORIZON + 1
    lags = np.stack([y[HORIZON - 1 - k: HORIZON - 1 - k + keep] for k in range(HORIZON)], 2)
    tgt = np.stack([y[HORIZON + k: HORIZON + k + keep] for k in range(HORIZON)], 2)
    x = np.concatenate([xn[HORIZON:HORIZON + keep], xc[HORIZON:HORIZON + keep], lags], 2)
    return x, tgt, np.arange(HORIZON, HORIZON + keep)


def split(dates, when, setting):
    """Training, validation and test masks over the rows build() returned."""
    d = dates[when]
    tr = (d >= np.datetime64(TRAIN_FROM)) & (d < np.datetime64(TRAIN_TO))
    va = (d >= np.datetime64(TRAIN_TO)) & (d < np.datetime64(VAL_TO))
    te = d >= np.datetime64(VAL_TO)
    if setting == "seq":
        # Non-overlapping evaluation windows, the first aligned on midnight.
        midnight = d.astype("datetime64[m]") == d.astype("datetime64[D]")
        start = int(np.argmax(te & midnight))
        keep = np.zeros(len(d), dtype=bool)
        keep[start::HORIZON] = True
        te = te & keep
    return tr, va, te


def graph(name):
    a = torch.tensor(np.loadtxt(os.path.join(GRAPHS, name, "W.txt")), dtype=torch.float32)
    return dense_to_sparse(a)


# ------------------------------------------------------------------- models
class Shared(nn.Module):
    """One weight set applied to every region's own covariates, no neighbourhood."""

    def __init__(self, f, h, out=1):
        super().__init__()
        self.out = out
        self.net = nn.Sequential(nn.Linear(f, h), nn.ReLU(), nn.Linear(h, h),
                                 nn.ReLU(), nn.Linear(h, out))

    def forward(self, x):                                  # x [B, N, F]
        o = self.net(x)
        return o.squeeze(-1) if self.out == 1 else o


class GraphNet(nn.Module):
    """GraphToolbox's myGNN, one graph per time step, batched block diagonally."""

    def __init__(self, f, cfg, out=1):
        super().__init__()
        self.out = out
        kw = {k: cfg[k] for k in ("K",) if k in cfg}
        self.gnn = myGNN(in_channels=f, num_layers=cfg["layers"],
                         hidden_channels=cfg["hidden"], out_channels=out,
                         conv_class=getattr(pyg_conv, cfg["conv"]),
                         conv_kwargs=kw, heads=cfg.get("heads", 1))
        ei, ew = graph(cfg["adj"])
        self.register_buffer("ei", ei)
        self.register_buffer("ew", ew)

    def forward(self, x):                                  # x [B, N, F]
        b, n, f = x.shape
        off = (torch.arange(b, device=x.device) * n).repeat_interleave(self.ei.shape[1])
        ei = self.ei.repeat(1, b) + off
        ew = self.ew.repeat(b)
        bv = torch.arange(b, device=x.device).repeat_interleave(n)
        o = self.gnn(x.reshape(b * n, f), ei, edge_weight=ew, batch=bv)
        return o.reshape(b, n) if self.out == 1 else o.reshape(b, n, self.out)


def fit(model, x, y, itr, iva, lr, seed):
    """Adam on the training months, early stopping on the validation months."""
    torch.manual_seed(seed)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    lf = nn.MSELoss()
    xt, yt, xv, yv = x[itr], y[itr], x[iva], y[iva]
    best, state, bad = float("inf"), None, 0
    for _ in range(EPOCHS):
        model.train()
        perm = torch.randperm(len(xt))
        for i in range(0, len(xt), BATCH):
            b = perm[i:i + BATCH]
            opt.zero_grad()
            lf(model(xt[b]), yt[b]).backward()
            opt.step()
        model.eval()
        with torch.no_grad():
            vl = float(np.mean([float(lf(model(xv[i:i + BATCH]), yv[i:i + BATCH]))
                                for i in range(0, len(xv), BATCH)]))
        if vl < best - 1e-6:
            best, state, bad = vl, {k: v.clone() for k, v in model.state_dict().items()}, 0
        else:
            bad += 1
            if bad >= PATIENCE:
                break
    model.load_state_dict(state)
    model.eval()
    return best


def predict(model, x):
    with torch.no_grad():
        return torch.cat([model(x[i:i + BATCH]) for i in range(0, len(x), BATCH)]).numpy()


def national(p, setting):
    """Sum over the nodes, windows laid end to end when the forecast is a window."""
    s = p.sum(1)
    return s.reshape(-1) if setting == "seq" else s


# ---------------------------------------------------------------------- run
def main(dataset, setting):
    global NODE_FEATS
    xn, cal, y, nodes, dates = read_real()
    itr_all = (dates >= np.datetime64(TRAIN_FROM)) & (dates < np.datetime64(TRAIN_TO))
    if dataset in {"synth", "synth_structured"}:
        NODE_FEATS = FEATS_SYNTH
        noise = "graph" if dataset == "synth_structured" else SYNTH_NOISE
        xn, y = make_synth(xn, cal, y, dates, itr_all, noise, SEED)
    x, tgt, when = build(setting, xn, cal, y)
    itr, iva, ite = split(dates, when, setting)
    print(f"{dataset}/{setting}: {len(nodes)} nodes, {x.shape[2]} features, "
          f"train {itr.sum()}, val {iva.sum()}, test {ite.sum()}", flush=True)

    mu, sd = x[itr].mean(0), x[itr].std(0)
    sd = np.where(sd < 1e-8, 1.0, sd)
    xz = torch.tensor((x - mu) / sd, dtype=torch.float32)
    my, sy = tgt[itr].mean(0), tgt[itr].std(0)
    sy = np.where(sy < 1e-8, 1.0, sy)
    yz = torch.tensor((tgt - my) / sy, dtype=torch.float32)
    h_out = tgt.shape[2] if tgt.ndim == 3 else 1
    f = xz.shape[2]
    truth = national(tgt[ite], setting)

    out, nodal = {}, {}

    def report(name, p, t0, val=None):
        v = f"val {val:.4f}  " if val is not None else " " * 11
        print(f"  {name:5} {v}national RMSE "
              f"{np.sqrt(((national(p, setting) - truth) ** 2).mean()):8.1f} MW "
              f"({time.time() - t0:.0f}s)", flush=True)
        # Windows laid end to end, so a node's forecast is one series either way.
        nodal[name] = p.transpose(0, 2, 1).reshape(-1, p.shape[1]) if p.ndim == 3 else p
        return national(p, setting)

    for name, cfg in GNNS.items():
        t0 = time.time()
        torch.manual_seed(SEED)
        m = GraphNet(f, cfg, out=h_out)
        v = fit(m, xz, yz, itr, iva, cfg["lr"], SEED)
        out[name] = report(name, predict(m, xz[ite]) * sy + my, t0, v)

    t0 = time.time()
    torch.manual_seed(SEED)
    m = Shared(f, 256, out=h_out)
    v = fit(m, xz, yz, itr, iva, 1e-3, SEED)
    out["MLP"] = report("MLP", predict(m, xz[ite]) * sy + my, t0, v)

    t0 = time.time()
    import gam_expert
    p = gam_expert.fit_gam(setting, xn, cal, y, when, itr | iva, ite, len(nodes))
    out["GAM"] = report("GAM", p, t0)

    idx = dates[when][ite]
    if setting == "seq":                                   # windows laid end to end
        idx = np.concatenate([dates[t:t + HORIZON] for t in when[ite]])
    df = pd.DataFrame(out, index=np.datetime_as_string(idx, unit="m"))
    df["load"] = truth
    df.index.name = "date"
    path = os.path.join(OUT, f"experts_{dataset}_{setting}.csv")
    df.to_csv(path)
    print("ecrit", path, df.shape, flush=True)

    # Per-node forecasts, which the national sum discards and the error maps and
    # the node-level aggregation both need.
    tn = tgt[ite]
    npz = os.path.join(OUT, f"nodal_{dataset}_{setting}.npz")
    np.savez_compressed(
        npz, nodes=np.array(nodes), experts=np.array(list(nodal)),
        pred=np.stack([nodal[k] for k in nodal], 2),
        true=tn.transpose(0, 2, 1).reshape(-1, tn.shape[1]) if tn.ndim == 3 else tn,
        dates=idx.astype("datetime64[m]"))
    print("ecrit", npz, flush=True)


if __name__ == "__main__":
    ds = sys.argv[1] if len(sys.argv) > 1 else "rfrance"
    st = sys.argv[2] if len(sys.argv) > 2 else "single"
    if len(sys.argv) > 3:
        SYNTH_NOISE = sys.argv[3]
    main(ds, st)
