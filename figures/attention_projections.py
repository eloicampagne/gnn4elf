"""Vector versions of the attention-projection figures in Chapter 2.

The exact attention dumps behind the six legacy panels are no longer present in
the project.  The extraction functions therefore digitise their coloured point
clouds once.  The resulting coordinates are cached under ``data/`` and every
element written to the output PDFs is a vector primitive.  The script removes
the misleading continuous colour bar from the binary synthetic experiment and
uses consistent labels and typography across panels.
"""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import colormaps
from matplotlib.colors import Normalize
from matplotlib.patches import Rectangle
import numpy as np

from style import ORANGE, PRUNE, apply


HERE = Path(__file__).resolve().parent
DATA = HERE / "data"
FIG = HERE / "out"
FIG.mkdir(exist_ok=True)
REAL_DATA = DATA / "attention_real_projections.npz"
SYNTH_DATA = DATA / "attention_synthetic_projections.npz"

# Pixel bounds of the plotting rectangles in the legacy 754 x 561 PNGs,
# followed by the data limits read from their axes.  Bounds are (left, top,
# right, bottom); the right and bottom endpoints are excluded.
SYNTH_SPECS = (
    ("PCA_synthe_two_coupling_true.png", (92, 36, 589, 500), (-0.75, 0.83, -0.63, 0.63)),
    ("PCA_hidden_couplage_layer0.png", (92, 4, 589, 500), (-0.42, 0.54, -0.45, 0.53)),
    ("PCA_hidden_couplage_layer1.png", (92, 36, 589, 500), (-1.48, 1.10, -0.78, 0.82)),
    ("PCA_hidden_couplage_layer2.png", (92, 4, 589, 500), (-1.14, 1.32, -0.90, 0.73)),
)

REAL_SPECS = (
    # Stop at the right spine of each axes.  The legacy colour bars begin at
    # columns 656 and 515; including them created the spurious vertical bands
    # that appeared in the first vectorisation.
    ("PCA_real_data_layer0_head0.png", (91, 36, 624, 536), (-2.25, 1.75, -1.70, 1.20)),
    ("umap_on_real_data_layer0_head0.png", (80, 35, 490, 575), (1.5, 18.5, -3.5, 13.5)),
)


def extract_legacy_synthetic():
    """Recover the four synthetic PCA point clouds from the legacy PNGs.

    No experimental values are inferred beyond the resolution of the original
    plots.  The operation is deterministic and exists because the underlying
    attention dumps are no longer present in the project.
    """
    palette = colormaps["plasma"](np.linspace(0, 1, 256))[:, :3]
    payload = {}
    for panel, (name, box, limits) in enumerate(SYNTH_SPECS):
        image = plt.imread(FIG / name)[..., :3]
        left, top, right, bottom = box
        crop = image[top:bottom, left:right]
        chroma = crop.max(axis=2) - crop.min(axis=2)
        mask = (chroma > 0.16) & (crop.max(axis=2) > 0.35)
        rows, cols = np.nonzero(mask)
        rgb = crop[rows, cols]

        # Recover the plasma coordinate, then collapse it to the two actual
        # coupling modes.  The old colour bar incorrectly presented a binary
        # label as a continuous variable and even named its midpoint "boundary".
        nearest = np.argmin(((rgb[:, None, :] - palette[None, :, :]) ** 2).sum(axis=2), axis=1)
        labels = (nearest >= 128).astype(np.int8)

        # Dense raster markers contribute several coloured pixels.  A stable
        # subsample keeps the geometry while avoiding a needlessly large PDF.
        rng = np.random.default_rng(2026 + panel)
        keep = []
        for label in (0, 1):
            idx = np.flatnonzero(labels == label)
            if len(idx) > 9000:
                idx = rng.choice(idx, 9000, replace=False)
            keep.append(idx)
        keep = np.sort(np.concatenate(keep))
        rows, cols, labels = rows[keep], cols[keep], labels[keep]

        xmin, xmax, ymin, ymax = limits
        x = xmin + cols / max(crop.shape[1] - 1, 1) * (xmax - xmin)
        y = ymax - rows / max(crop.shape[0] - 1, 1) * (ymax - ymin)
        payload[f"panel_{panel}"] = np.column_stack((x, y)).astype(np.float32)
        payload[f"labels_{panel}"] = labels

    DATA.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(SYNTH_DATA, **payload)


def extract_legacy_real():
    """Recover both real-data projections and their cyclic time coordinate."""
    palette = colormaps["hsv"](np.linspace(0, 1, 256))[:, :3]
    payload = {}
    for panel, (name, box, limits) in enumerate(REAL_SPECS):
        image = plt.imread(FIG / name)[..., :3]
        left, top, right, bottom = box
        crop = image[top:bottom, left:right]
        chroma = crop.max(axis=2) - crop.min(axis=2)
        mask = (chroma > 0.22) & (crop.max(axis=2) > 0.35)
        rows, cols = np.nonzero(mask)
        rgb = crop[rows, cols]
        time = np.argmin(
            ((rgb[:, None, :] - palette[None, :, :]) ** 2).sum(axis=2), axis=1
        ).astype(np.float32) / 255.0

        rng = np.random.default_rng(2030 + panel)
        if len(rows) > 18000:
            keep = np.sort(rng.choice(len(rows), 18000, replace=False))
            rows, cols, time = rows[keep], cols[keep], time[keep]
        xmin, xmax, ymin, ymax = limits
        x = xmin + cols / max(crop.shape[1] - 1, 1) * (xmax - xmin)
        y = ymax - rows / max(crop.shape[0] - 1, 1) * (ymax - ymin)
        payload[("pca", "umap")[panel]] = np.column_stack((x, y)).astype(np.float32)
        payload[("time_pca", "time_umap")[panel]] = time

    DATA.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(REAL_DATA, **payload)


def plot_real():
    if not REAL_DATA.exists():
        extract_legacy_real()
    data = np.load(REAL_DATA)
    pca, umap = data["pca"], data["umap"]

    fig, axes = plt.subplots(1, 2, figsize=(7.15, 3.0))
    cmap = colormaps["twilight_shifted"]
    norm = Normalize(0, 1)
    for ax, values, time, xlabel, ylabel in (
        (axes[0], pca, data["time_pca"], "Principal component 1", "Principal component 2"),
        (axes[1], umap, data["time_umap"], "UMAP 1", "UMAP 2"),
    ):
        scatter = ax.scatter(values[:, 0], values[:, 1], c=time, cmap=cmap,
                             norm=norm, s=1.0, alpha=0.55, linewidths=0)
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.grid(True)

    # A standard Matplotlib colour bar embeds a small raster in an otherwise
    # vector PDF.  Draw a segmented cyclic bar explicitly to keep the file pure
    # vector graphics.
    cax = fig.add_axes((0.915, 0.23, 0.018, 0.67))
    segments = 64
    for index in range(segments):
        cax.add_patch(Rectangle((0, index / segments), 1, 1 / segments,
                                facecolor=cmap((index + 0.5) / segments),
                                edgecolor="none"))
    cax.set_xlim(0, 1)
    cax.set_ylim(0, 1)
    cax.set_xticks([])
    cax.yaxis.tick_right()
    cax.yaxis.set_label_position("right")
    cax.set_yticks([0, 0.25, 0.5, 0.75, 1],
                   ["1 Jan", "1 Apr", "1 Jul", "1 Oct", "31 Dec"])
    cax.set_ylabel("Position in the year")
    axes[0].text(0.5, -0.25, "(a) PCA", ha="center", transform=axes[0].transAxes)
    axes[1].text(0.5, -0.25, "(b) UMAP", ha="center", transform=axes[1].transAxes)
    fig.subplots_adjust(left=0.08, right=0.90, bottom=0.23, top=0.98, wspace=0.34)
    fig.savefig(FIG / "attention_real.pdf")
    plt.close(fig)


def plot_synthetic():
    if not SYNTH_DATA.exists():
        extract_legacy_synthetic()
    data = np.load(SYNTH_DATA)
    fig = plt.figure(figsize=(7.15, 5.0))
    grid = fig.add_gridspec(2, 3, height_ratios=(1.05, 1), hspace=0.43, wspace=0.34)
    axes = [fig.add_subplot(grid[0, 1]), *(fig.add_subplot(grid[1, i]) for i in range(3))]
    colours = np.array([PRUNE, ORANGE])
    labels = ("Coupling mode A1", "Coupling mode A2")

    for panel, ax in enumerate(axes):
        points = data[f"panel_{panel}"]
        classes = data[f"labels_{panel}"]
        for label in (0, 1):
            select = classes == label
            ax.scatter(points[select, 0], points[select, 1], s=0.7,
                       color=colours[label], alpha=0.55, linewidths=0,
                       label=labels[label])
        ax.set_xlabel("Principal component 1")
        ax.set_ylabel("Principal component 2")
        ax.grid(True)

    panel_titles = (
        "(a) explicit coupling, layer 2",
        "(b) implicit coupling, layer 0",
        "(c) implicit coupling, layer 1",
        "(d) implicit coupling, layer 2",
    )
    for ax, title in zip(axes, panel_titles):
        ax.text(0.5, -0.28, title, ha="center", transform=ax.transAxes)
    handles, legend_labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, legend_labels, loc="upper center", ncol=2,
               frameon=False, bbox_to_anchor=(0.5, 0.995), markerscale=5)
    fig.subplots_adjust(left=0.08, right=0.985, bottom=0.12, top=0.91)
    fig.savefig(FIG / "attention_synthetic.pdf")
    plt.close(fig)


def main():
    apply()
    plot_real()
    plot_synthetic()
    print(FIG / "attention_real.pdf")
    print(FIG / "attention_synthetic.pdf")


if __name__ == "__main__":
    main()
