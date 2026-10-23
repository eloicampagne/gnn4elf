"""French regions: fixed-capacity curves against training volume, set beside the tuned models.

Left: national test RMSE of the four nodewise models at one fixed configuration (two hidden
layers or convolution blocks of width 64, no tuning) as the training set contracts
(tab:data-regime, three seeds). Right: at the full sample, each family at that fixed
configuration (hollow) and at its tuned configuration of tab:combined-results (filled),
with the pooled gradient-boosted model and the bottom-up graph mixture as references.
"""
import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

days = np.array([54, 109, 272, 545, 1090])
fixed = {  # mean, sd over three seeds (tab:data-regime)
    "Per-node MLP": ([1948, 1412, 1090, 958, 876], [69, 28, 41, 5, 3]),
    "Global MLP":   ([2432, 1687, 1179, 1014, 921], [269, 105, 25, 29, 3]),
    "Identity GNN": ([2038, 1763, 1292, 1113, 918], [103, 101, 17, 31, 7]),
    "SAGE GNN":     ([2080, 1738, 1206, 983, 891], [112, 55, 41, 27, 19]),
}
col = {"Per-node MLP": "#1b9e77", "Global MLP": "#7570b3", "Identity GNN": "#d95f02",
       "SAGE GNN": "#2c7fb8", "GCN": "#e7298a", "GAT": "#a6761d"}

# full sample: (fixed, tuned) national RMSE and tuned sd (tab:combined-results)
pairs = [("Per-node MLP", 876, 922, 10), ("SAGE GNN", 891, 925, 11),
         ("Shared MLP", None, 937, 26), ("GCN", None, 891, 26), ("GAT", None, 899, 37)]

fig, (a, b) = plt.subplots(1, 2, figsize=(10.5, 3.9), gridspec_kw={"width_ratios": [1.35, 1]})
for k, (m, s) in fixed.items():
    m, s = np.array(m), np.array(s)
    a.plot(days, m, "-o", color=col[k], ms=4, lw=1.6, label=k)
    a.fill_between(days, m - s, m + s, color=col[k], alpha=0.15, lw=0)
a.set_xscale("log"); a.set_yscale("log")
a.set_xticks(days); a.set_xticklabels([str(d) for d in days])
a.set_yticks([900, 1200, 1600, 2400]); a.set_yticklabels(["900", "1200", "1600", "2400"])
a.minorticks_off()
a.set_xlabel("training days (log scale)"); a.set_ylabel("national RMSE (MW, log scale)")
a.set_title("Fixed configuration (width 64, untuned)", fontsize=10)
a.legend(frameon=False, fontsize=8.5)
a.grid(alpha=0.3)

rows = ["Per-node MLP", "Shared MLP", "SAGE GNN", "GCN", "GAT"]
labels = ["Per-node MLP", "Shared (MLP / identity GNN)", "SAGE GNN", "GCN", "GAT"]
y = np.arange(len(rows))[::-1]
for yi, name in zip(y, rows):
    fx = {"Per-node MLP": 876, "SAGE GNN": 891, "Shared MLP": 918}.get(name)
    tu = {r[0]: (r[2], r[3]) for r in pairs}[name]
    c = col.get(name, col["Identity GNN"])
    if fx is not None:
        b.plot(fx, yi, "o", mfc="white", mec=c, mew=1.6, ms=7)
    b.errorbar(tu[0], yi, xerr=tu[1], fmt="o", color=c, ms=7, capsize=3, lw=1.2)
b.axvline(859, color="0.35", ls="--", lw=1.1)
b.text(859, len(rows) - 0.45, " XGBoost\n (tuned)", fontsize=7.5, color="0.3", va="top")
b.axvline(809, color="0.35", ls=":", lw=1.3)
b.text(809, len(rows) - 0.45, " graph\n mixture", fontsize=7.5, color="0.3", va="top")
b.set_yticks(y); b.set_yticklabels(labels, fontsize=9)
b.set_ylim(-0.6, len(rows) - 0.1)
b.set_xlabel("national RMSE at 1090 days (MW)")
b.set_title("Full sample: fixed (hollow) vs tuned (filled)", fontsize=10)
b.grid(axis="x", alpha=0.3)
fig.tight_layout()
out = os.path.join(os.path.dirname(os.path.abspath(__file__)), "out", "french_regime_tuned.pdf")
os.makedirs(os.path.dirname(out), exist_ok=True)
fig.savefig(out, bbox_inches="tight"); fig.savefig(out.replace(".pdf", ".png"), dpi=130, bbox_inches="tight")
print(out)
