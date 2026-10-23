"""Sweep the four axes, one at a time from a base config, to map where the GNN is useful.
Caches to synth_sweep.json. Base: N=60, rho=0.7, sig_obs=3.0, frac=0.3."""
import json, os, time, sys
import numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from _paths import ROOT  # noqa: E402  (puts every experiment folder on sys.path)
import synth_gnn as G

OUT = os.path.join(ROOT, "results_synth_gnn_when_useful", "synth_sweep.json")
res = json.load(open(OUT)) if os.path.exists(OUT) else {}
SEEDS = [0, 1, 2]
BASE = dict(N=60, rho=0.7, sig_obs=3.0, frac=0.3)

AXES = {
    "N":       [12, 30, 60, 120, 250],
    "frac":    [0.05, 0.10, 0.25, 0.50, 1.00],
    "rho":     [0.0, 0.3, 0.5, 0.7, 0.9],
    "sig_obs": [0.0, 1.0, 2.0, 4.0, 6.0],
}

def do(cfg, seed):
    key = f"{cfg['N']}|{cfg['rho']}|{cfg['sig_obs']}|{cfg['frac']}|{seed}"
    if key in res:
        return
    t0 = time.time()
    r = G.run_config(N=cfg["N"], rho=cfg["rho"], sig_obs=cfg["sig_obs"], frac=cfg["frac"], seed=seed)
    res[key] = r
    json.dump(res, open(OUT, "w"), indent=1)
    print(f"{key:28} " + "  ".join(f"{k}={v:.3f}" for k, v in r.items()) + f"  [{time.time()-t0:.0f}s]", flush=True)

for axis, vals in AXES.items():
    print(f"\n=== sweeping {axis} (base {BASE}) ===", flush=True)
    for v in vals:
        cfg = dict(BASE); cfg[axis] = v
        for s in SEEDS:
            do(cfg, s)
print("\nDONE", flush=True)
