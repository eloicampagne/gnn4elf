"""Project matched shared-MLP and SAGE representations and quantify their geometry.

The activation archive is produced by
``GraphToolbox/examples/load/compare_latent_representations.py`` from the two
matched checkpoints.  t-SNE is fitted separately in each model space; only the
within-panel neighbourhood geometry is interpreted.
"""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from sklearn.manifold import TSNE

ROOT = Path(__file__).resolve().parents[1]
# Written by GraphToolbox, examples/load/compare_latent_representations.py
ARCHIVE = ROOT / "results_interpretability" / "latent_representations_mlp_gnn.npz"
OUTPUT = ROOT / "figures" / "out" / "mlp_gnn_tsne.pdf"
METRICS = ROOT / "figures" / "out" / "mlp_gnn_representation_metrics.json"
OUTPUT.parent.mkdir(exist_ok=True)

REGIONS = ("ARA", "BFC", "BRE", "CVL", "GES", "HDF", "IDF", "NAQ", "NOR", "OCC", "PAC", "PDL")
PERIODS = ("Jan--Mar", "Apr--Jun", "Jul--Sep", "Oct--Dec")


def standardize(z):
    z = np.asarray(z, dtype=np.float64)
    scale = z.std(axis=0)
    keep = scale > 1e-6
    z = (z[:, keep] - z[:, keep].mean(axis=0)) / scale[keep]
    return np.clip(z, -20, 20)


def linear_cka(x, y):
    x = x - x.mean(axis=0)
    y = y - y.mean(axis=0)
    xy = np.einsum("ni,nj->ij", x, y)
    xx = np.einsum("ni,nj->ij", x, x)
    yy = np.einsum("ni,nj->ij", y, y)
    return float((xy * xy).sum() / np.sqrt((xx * xx).sum() * (yy * yy).sum()))


def explained_centroid_variance(z, labels):
    """Share of latent variance explained by between-label centroids."""
    centre = z.mean(axis=0)
    total = np.square(z - centre).sum()
    between = 0.0
    for label in np.unique(labels):
        group = z[labels == label]
        between += len(group) * np.square(group.mean(axis=0) - centre).sum()
    return float(between / total)


def main():
    data = np.load(ARCHIVE)
    mlp = standardize(data["shared_mlp"])
    gnn = standardize(data["sage_gnn"])
    nodes = data["node"]
    seasons = data["season"]

    # Every third day, all regions: a balanced display of 1,452 observations.
    day = np.arange(len(nodes)) // len(REGIONS)
    idx = np.flatnonzero(day % 3 == 0)
    maps = {
        "Shared MLP": TSNE(n_components=2, perplexity=40, init="random",
                            learning_rate=200, max_iter=1500,
                            random_state=42).fit_transform(mlp[idx]),
        "SAGE GNN": TSNE(n_components=2, perplexity=40, init="random",
                          learning_rate=200, max_iter=1500,
                          random_state=42).fit_transform(gnn[idx]),
    }

    metrics = {
        "observations": int(len(nodes)),
        "displayed_observations": int(len(idx)),
        "linear_cka": linear_cka(mlp, gnn),
        "shared_mlp_node_variance_share": explained_centroid_variance(mlp, nodes),
        "sage_gnn_node_variance_share": explained_centroid_variance(gnn, nodes),
        "shared_mlp_period_variance_share": explained_centroid_variance(mlp, seasons),
        "sage_gnn_period_variance_share": explained_centroid_variance(gnn, seasons),
    }
    METRICS.write_text(json.dumps(metrics, indent=2), encoding="utf-8")

    region_colors = plt.get_cmap("tab20")(np.linspace(0, 1, len(REGIONS)))
    period_colors = plt.get_cmap("viridis")(np.linspace(0.08, 0.92, 4))
    fig, axes = plt.subplots(2, 2, figsize=(10.7, 7.1), constrained_layout=True)
    for col, name in enumerate(("Shared MLP", "SAGE GNN")):
        emb = maps[name]
        axes[0, col].scatter(emb[:, 0], emb[:, 1], c=region_colors[nodes[idx]],
                             s=6, alpha=0.52, linewidths=0)
        axes[1, col].scatter(emb[:, 0], emb[:, 1], c=period_colors[seasons[idx]],
                             s=6, alpha=0.52, linewidths=0)
        axes[0, col].set_title(name)
        for ax in axes[:, col]:
            ax.set_xlabel("t-SNE 1")
            ax.set_ylabel("t-SNE 2")
            ax.set_xticks([])
            ax.set_yticks([])
            for spine in ax.spines.values():
                spine.set_visible(False)

    axes[0, 0].text(-0.05, 0.5, "colored by region", rotation=90,
                    va="center", ha="right", transform=axes[0, 0].transAxes)
    axes[1, 0].text(-0.05, 0.5, "colored by period", rotation=90,
                    va="center", ha="right", transform=axes[1, 0].transAxes)
    axes[0, 1].legend(
        handles=[Line2D([0], [0], marker="o", linestyle="", color=region_colors[i],
                        label=label, markersize=5) for i, label in enumerate(REGIONS)],
        title="Region", ncol=2, fontsize=7, title_fontsize=8,
        loc="center left", bbox_to_anchor=(1.01, 0.5), frameon=False,
    )
    axes[1, 1].legend(
        handles=[Line2D([0], [0], marker="o", linestyle="", color=period_colors[i],
                        label=label, markersize=5) for i, label in enumerate(PERIODS)],
        title="Test period", fontsize=7, title_fontsize=8,
        loc="center left", bbox_to_anchor=(1.01, 0.5), frameon=False,
    )
    fig.savefig(OUTPUT, bbox_inches="tight")
    print(json.dumps(metrics, indent=2))
    print(OUTPUT)


if __name__ == "__main__":
    main()
