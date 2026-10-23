"""LEConv and GatedGraphConv as candidate experts on the French single-step cell.

The sweep of the framework chapter puts LEConv at the head of fifty-one
convolutions on this collection, GatedGraphConv close behind, and neither carries
a tuned configuration in the benchmark table. Two configurations are tried for
each, the sweep's default width and a wider one, and the choice between them is
made on the validation months alone.
"""
import sys, time
import numpy as np, torch
import aggreg_experts as A

CANDS = {
    "LE-64":  dict(conv="LEConv", adj="dtw", layers=1, hidden=64,  lr=1e-3),
    "LE-256": dict(conv="LEConv", adj="dtw", layers=2, hidden=256, lr=1e-3),
    "GG-64":  dict(conv="GatedGraphConv", adj="dtw", layers=1, hidden=64,  lr=1e-3),
    "GG-256": dict(conv="GatedGraphConv", adj="dtw", layers=2, hidden=256, lr=1e-3),
}

if __name__ == "__main__":
    xn, cal, y, nodes, dates = A.read_real()
    x, tgt, when = A.build("single", xn, cal, y)
    itr, iva, ite = A.split(dates, when, "single")
    mu, sd = x[itr].mean(0), x[itr].std(0); sd = np.where(sd < 1e-8, 1., sd)
    xz = torch.tensor((x - mu) / sd, dtype=torch.float32)
    my, sy = tgt[itr].mean(0), tgt[itr].std(0)
    yz = torch.tensor((tgt - my) / sy, dtype=torch.float32)
    true = tgt[ite]
    for name, cfg in CANDS.items():
        t0 = time.time()
        try:
            torch.manual_seed(0)
            m = A.GraphNet(xz.shape[2], cfg, out=1)
            v = A.fit(m, xz, yz, itr, iva, cfg["lr"], 0)
            p = A.predict(m, xz[ite]) * sy + my
            mape = (np.abs((true - p) / np.clip(np.abs(true), 1e-8, None)).mean(0) * 100).mean()
            rmse = np.sqrt(((p.sum(1) - true.sum(1)) ** 2).mean())
            print(f"{name:7} val {v:.4f}  MAPE regional {mape:5.2f}%  "
                  f"national {rmse:7.1f} MW ({time.time()-t0:.0f}s)", flush=True)
        except Exception as e:
            print(f"{name:7} echec {type(e).__name__}: {str(e)[:90]}", flush=True)
