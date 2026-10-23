"""Relate French regional latent distances to geographic distances."""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import spearmanr

ROOT = Path(__file__).resolve().parents[1]
# Written by GraphToolbox, examples/load/compare_latent_representations.py
ARCHIVE = ROOT / "results_interpretability" / "latent_representations_mlp_gnn.npz"
OUTPUT = ROOT / "figures" / "out" / "france_latent_geography.pdf"
METRICS = ROOT / "figures" / "out" / "france_latent_geography_metrics.json"
OUTPUT.parent.mkdir(exist_ok=True)

REGIONS = ("ARA", "BFC", "BRE", "CVL", "GES", "HDF", "IDF", "NAQ", "NOR", "OCC", "PAC", "PDL")
LATITUDE = np.array((45.7578137, 47.3215806, 48.3905283, 47.9027336,
                     48.5846140, 50.6365654, 48.8627250, 44.8412250,
                     49.4404591, 43.6044622, 43.2961743, 47.2186371))
LONGITUDE = np.array((4.8320114, 5.0414701, -4.4860088, 1.9086066,
                      7.7507127, 3.0635282, 2.2875920, -0.5800364,
                      1.0939658, 1.4442469, 5.3699525, -1.5541362))


def standardize(z):
    z = np.asarray(z, dtype=np.float64)
    scale = z.std(axis=0)
    keep = scale > 1e-6
    return (z[:, keep] - z[:, keep].mean(axis=0)) / scale[keep]


def geographic_distances():
    lat = np.radians(LATITUDE)
    lon = np.radians(LONGITUDE)
    dlat = lat[:, None] - lat
    dlon = lon[:, None] - lon
    a = np.sin(dlat / 2) ** 2 + np.cos(lat[:, None]) * np.cos(lat) * np.sin(dlon / 2) ** 2
    return 2 * 6371.0 * np.arcsin(np.sqrt(a))


def mean_same_day_distances(z, n_nodes):
    z = standardize(z)
    z = z.reshape(-1, n_nodes, z.shape[-1])
    return np.sqrt(np.square(z[:, :, None, :] - z[:, None, :, :]).sum(axis=-1)).mean(axis=0)


def mantel_spearman(geo, latent, permutations=9999, seed=42):
    tri = np.triu_indices_from(geo, k=1)
    observed = float(spearmanr(geo[tri], latent[tri]).statistic)
    rng = np.random.default_rng(seed)
    null = np.empty(permutations)
    for k in range(permutations):
        order = rng.permutation(len(geo))
        permuted_geo = geo[np.ix_(order, order)]
        null[k] = spearmanr(permuted_geo[tri], latent[tri]).statistic
    p_value = float((1 + np.count_nonzero(np.abs(null) >= abs(observed))) / (permutations + 1))
    return observed, p_value


def main():
    data = np.load(ARCHIVE)
    geo = geographic_distances()
    latent = {
        "Shared MLP": mean_same_day_distances(data["shared_mlp"], len(REGIONS)),
        "SAGE GNN": mean_same_day_distances(data["sage_gnn"], len(REGIONS)),
    }
    tri = np.triu_indices(len(REGIONS), k=1)
    metrics = {}

    fig, axes = plt.subplots(1, 2, figsize=(9.6, 3.7), sharex=True, sharey=True,
                             constrained_layout=True)
    colours = ("#3b6fd0", "#4e9a51")
    close_pairs = np.flatnonzero(geo[tri] < 200)
    for ax, (name, distances), colour in zip(axes, latent.items(), colours):
        x, y = geo[tri], distances[tri]
        rho, p_value = mantel_spearman(geo, distances)
        metrics[name.lower().replace(" ", "_")] = {
            "spearman_rho": rho,
            "node_label_permutation_p": p_value,
        }
        ax.scatter(x, y, s=23, color=colour, alpha=0.72, linewidths=0)
        slope, intercept = np.polyfit(x, y, 1)
        line_x = np.array((x.min(), x.max()))
        ax.plot(line_x, intercept + slope * line_x, linestyle="--", color="0.35", linewidth=1)
        label_offsets = ((4, 8), (4, -11), (4, 5), (4, -10), (4, 5))
        for pair_index, offset in zip(close_pairs, label_offsets):
            i, j = tri[0][pair_index], tri[1][pair_index]
            ax.annotate(
                f"{REGIONS[i]}–{REGIONS[j]}", (x[pair_index], y[pair_index]),
                xytext=offset, textcoords="offset points", fontsize=6.5, color="0.25",
            )
        ax.text(
            0.97, 0.95, f"Spearman $\\rho$ = {rho:.3f}\npermutation $p$ = {p_value:.3f}",
            transform=ax.transAxes, ha="right", va="top", fontsize=8,
            bbox=dict(boxstyle="round,pad=0.25", facecolor="white", edgecolor="0.8", alpha=0.9),
        )
        ax.set_title(name)
        ax.set_xlabel("Geographic distance (km)")
        ax.grid(color="0.9", linewidth=0.6)
    axes[0].set_ylabel("Mean same-day latent distance")
    METRICS.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    fig.savefig(OUTPUT, bbox_inches="tight")
    print(json.dumps(metrics, indent=2))
    print(OUTPUT)


if __name__ == "__main__":
    main()
