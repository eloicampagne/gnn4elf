"""Increment intervals that carry seed variability, and the decomposition against the strongest local reference.

increment_intervals.py averages each node's error over the three seeds before
resampling nodes, so its intervals ignore the seed-to-seed spread. Here each
bootstrap draw first resamples the three seeds of every model independently
(with replacement), then resamples nodes, paired across models. Nothing is
retrained: the per-node errors are read from increment_errors.json, and the
closed-form local references from local_references.json.

The second table recomputes the share of the per-node-to-GNN gap recovered by
sharing when the per-node reference is the stronger of the per-node MLP and the
per-node ridge in each cell.
"""

import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from _paths import ROOT, DATA, CACHE  # noqa: E402,F401  (puts every experiment folder on sys.path)
from increment_intervals import CELLS, SEEDS, ARROWS, ERR  # noqa: E402
import ltsf_nodes as E  # noqa: E402

LOCAL = os.path.join(E.OUT, "local_references.json")
OUT = os.path.join(E.OUT, "seed_node_intervals.json")
NBOOT = 5000


def main():
    res = json.load(open(ERR))
    loc = json.load(open(LOCAL))
    rng = np.random.default_rng(0)
    out = {}
    for c, w in CELLS:
        key = f"{c}|{w}"
        models = sorted({m for _, a, b in ARROWS for m in (a, b)})
        S = {m: np.array([res[f"{key}|{m}|{s}"]["err"] for s in SEEDS]) for m in models}  # [3, N]
        n = S[models[0]].shape[1]
        node_draws = rng.integers(0, n, size=(NBOOT, n))
        seed_draws = {m: rng.integers(0, len(SEEDS), size=(NBOOT, len(SEEDS))) for m in models}
        cell = {"n_nodes": n, "increments": {}}
        for name, a, b in ARROWS:
            point = float(S[a].mean() - S[b].mean())
            ea = S[a][seed_draws[a]].mean(1)                  # [NBOOT, N], seeds resampled
            eb = S[b][seed_draws[b]].mean(1)
            d = np.take_along_axis(ea - eb, node_draws, axis=1).mean(1)
            # node-only interval, for comparison with the manuscript
            d0 = (S[a].mean(0) - S[b].mean(0))[node_draws].mean(1)
            cell["increments"][name] = {
                "point": point,
                "ci_seed_node": [float(np.quantile(d, .025)), float(np.quantile(d, .975))],
                "ci_node_only": [float(np.quantile(d0, .025)), float(np.quantile(d0, .975))],
            }
        # strongest local reference
        mlp_pn = S["per_node"].mean()
        ridge_pn = float(np.mean(loc[key]["ridge_pn"]))
        ref_name, ref = ("ridge_pn", ridge_pn) if ridge_pn < mlp_pn else ("per_node_mlp", float(mlp_pn))
        gnn, sh = float(S["graph"].mean()), float(S["shared"].mean())
        cell["strong_reference"] = {
            "reference": ref_name, "ref": ref, "per_node_mlp": float(mlp_pn), "ridge_pn": ridge_pn,
            "persist": float(np.mean(loc[key]["persist"])),
            "shared_mlp": sh, "gnn": gnn,
            "total_gap": ref - gnn,
            "sharing_share": (ref - sh) / (ref - gnn) if ref - gnn > 0 else None,
            "ridge_sh": float(np.mean(loc[key]["ridge_sh"])),
            "ridge_pn_nbr": float(np.mean(loc[key]["ridge_pn_nbr"])),
            "ridge_sh_nbr": float(np.mean(loc[key]["ridge_sh_nbr"])),
        }
        out[key] = cell
        print(f"\n== {key}  (N={n})")
        for name, v in cell["increments"].items():
            lo, hi = v["ci_seed_node"]; lo0, hi0 = v["ci_node_only"]
            flag = "  includes 0" if lo <= 0 <= hi else ""
            print(f"  {name:26s} {v['point']:+.4f}  seed+node [{lo:+.4f}, {hi:+.4f}]"
                  f"   node only [{lo0:+.4f}, {hi0:+.4f}]{flag}")
        r = cell["strong_reference"]
        share = "n/a" if r["sharing_share"] is None else f"{100*r['sharing_share']:.0f}%"
        print(f"  strongest local ref = {r['reference']} {r['ref']:.4f} (MLP {r['per_node_mlp']:.4f},"
              f" ridge {r['ridge_pn']:.4f}, persistence {r['persist']:.4f});"
              f" gap to GNN {r['total_gap']:+.4f}; shared MLP recovers {share}")
    json.dump(out, open(OUT, "w"), indent=1)


if __name__ == "__main__":
    main()
