"""Project and compare the learned spaces of the WEAVE-500 MLP and GNN."""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from sklearn.cluster import SpectralClustering
from sklearn.manifold import TSNE

ROOT = Path(__file__).resolve().parents[1]
# Written by experiments/interpretability/extract_weave_latents.py
ARCHIVE = ROOT / "results_ltsf_nodes" / "weave500_latent_mlp_gnn.npz"
OUTPUT = ROOT / "figures" / "out" / "weave_mlp_gnn_tsne.pdf"
METRICS = ROOT / "figures" / "out" / "weave_mlp_gnn_representation_metrics.json"
OUTPUT.parent.mkdir(exist_ok=True)
PERIODS = ("First quarter", "Second quarter", "Third quarter", "Fourth quarter")


def standardize(z):
    z = np.asarray(z, dtype=np.float64)
    scale = z.std(axis=0)
    keep = scale > 1e-6
    return np.clip((z[:, keep] - z[:, keep].mean(axis=0)) / scale[keep], -20, 20)


def linear_cka(x, y):
    x = x - x.mean(axis=0)
    y = y - y.mean(axis=0)
    xy = x.T @ y
    xx = x.T @ x
    yy = y.T @ y
    return float(np.square(xy).sum() / np.sqrt(np.square(xx).sum() * np.square(yy).sum()))


def explained_centroid_variance(z, labels):
    centre = z.mean(axis=0)
    total = np.square(z - centre).sum()
    between = 0.0
    for label in np.unique(labels):
        group = z[labels == label]
        between += len(group) * np.square(group.mean(axis=0) - centre).sum()
    return float(between / total)


def edge_distance_ratio(z, adjacency):
    """Mean squared distance on five graph neighbours, relative to random pairs."""
    a = adjacency.copy()
    np.fill_diagonal(a, -np.inf)
    n_nodes = len(a)
    src = np.repeat(np.arange(n_nodes), 5)
    dst = np.concatenate([np.argsort(-a[i])[:5] for i in range(n_nodes)])
    zr = z.reshape(-1, n_nodes, z.shape[1])
    edge = np.square(zr[:, src] - zr[:, dst]).mean()
    rng = np.random.default_rng(42)
    random_dst = rng.integers(0, n_nodes - 1, len(src))
    random_dst += random_dst >= src
    random = np.square(zr[:, src] - zr[:, random_dst]).mean()
    return float(edge / random)


def main():
    data = np.load(ARCHIVE)
    mlp = standardize(data["shared_mlp"])
    gnn = standardize(data["sage_gnn"])
    nodes = data["node"]
    origins = data["origin"]
    periods = data["period"]
    adjacency = data["adjacency"]

    affinity = np.maximum(adjacency, adjacency.T)
    np.fill_diagonal(affinity, 0.0)
    clusters = SpectralClustering(
        n_clusters=8, affinity="precomputed", assign_labels="cluster_qr",
        random_state=42,
    ).fit_predict(affinity)
    observation_cluster = clusters[nodes]

    # All 500 feeders at every twelfth test origin: 6,500 matched observations.
    idx = np.flatnonzero(origins % 12 == 0)
    maps = {
        "Shared MLP": TSNE(
            n_components=2, perplexity=50, init="pca", learning_rate="auto",
            max_iter=1200, random_state=42,
        ).fit_transform(mlp[idx]),
        "SAGE GNN": TSNE(
            n_components=2, perplexity=50, init="pca", learning_rate="auto",
            max_iter=1200, random_state=42,
        ).fit_transform(gnn[idx]),
    }

    metrics = {
        "observations": int(len(nodes)),
        "displayed_observations": int(len(idx)),
        "linear_cka": linear_cka(mlp, gnn),
        "shared_mlp_feeder_variance_share": explained_centroid_variance(mlp, nodes),
        "sage_gnn_feeder_variance_share": explained_centroid_variance(gnn, nodes),
        "shared_mlp_period_variance_share": explained_centroid_variance(mlp, periods),
        "sage_gnn_period_variance_share": explained_centroid_variance(gnn, periods),
        "shared_mlp_edge_distance_ratio": edge_distance_ratio(mlp, adjacency),
        "sage_gnn_edge_distance_ratio": edge_distance_ratio(gnn, adjacency),
    }
    METRICS.write_text(json.dumps(metrics, indent=2), encoding="utf-8")

    cluster_colors = plt.get_cmap("tab10")(np.arange(8))
    period_colors = plt.get_cmap("viridis")(np.linspace(0.08, 0.92, 4))
    fig, axes = plt.subplots(2, 2, figsize=(10.7, 7.1), constrained_layout=True)
    for col, name in enumerate(("Shared MLP", "SAGE GNN")):
        emb = maps[name]
        axes[0, col].scatter(
            emb[:, 0], emb[:, 1], c=cluster_colors[observation_cluster[idx]],
            s=4, alpha=0.42, linewidths=0,
        )
        axes[1, col].scatter(
            emb[:, 0], emb[:, 1], c=period_colors[periods[idx]],
            s=4, alpha=0.42, linewidths=0,
        )
        axes[0, col].set_title(name)
        for ax in axes[:, col]:
            ax.set_xlabel("t-SNE 1")
            ax.set_ylabel("t-SNE 2")
            ax.set_xticks([])
            ax.set_yticks([])
            for spine in ax.spines.values():
                spine.set_visible(False)

    axes[0, 0].text(
        -0.05, 0.5, "colored by graph cluster", rotation=90,
        va="center", ha="right", transform=axes[0, 0].transAxes,
    )
    axes[1, 0].text(
        -0.05, 0.5, "colored by test period", rotation=90,
        va="center", ha="right", transform=axes[1, 0].transAxes,
    )
    axes[0, 1].legend(
        handles=[Line2D([0], [0], marker="o", linestyle="", color=cluster_colors[i],
                        label=f"Cluster {i + 1}", markersize=5) for i in range(8)],
        title="Graph cluster", ncol=2, fontsize=7, title_fontsize=8,
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
