"""Expert aggregation, six cells (figures fig/expert_aggregation*.pdf).

Three datasets and two settings are crossed by ``experiments/aggregation/aggreg_experts.py``,
which writes one ``data/experts_<dataset>_<setting>.csv`` per cell: eight columns
of national forecasts, the sum of the twelve regional forecasts, plus the
observed national load, over the test year. The experts are six graph networks,
each with the convolution and the adjacency of its configuration, a
weight-shared feedforward network, and the operational additive model.

ML-Poly (Gaillard, Stoltz and van Erven, 2014) combines them online. In the
single-step setting the weights are updated at every half-hour, nothing being
committed ahead. In the sequence-to-sequence setting a whole day is committed at
once: the weights forecast the next forty-eight half-hours and the observed day
then updates them, which is the protocol the chapter's ensembles follow.

Both panels come from ``opera``'s own plotting functions, ``plot_weight`` for the
weight paths and ``avg_loss`` for the losses, the latter's bars converted from
mean squared error to its root so that they read in megawatts.

    python expert_aggregation.py                 # every cell that has a CSV
    python expert_aggregation.py rfrance single  # one cell

Dependencies: numpy, pandas, matplotlib, seaborn, opera
(``pip install git+https://github.com/Dralliag/opera-python.git``).
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from opera import Mixture
from opera.mixture import avg_loss, plot_weight

HERE = Path(__file__).resolve().parent
RESULTS = HERE.parent / "results_aggreg"
FIG = HERE / "out"
FIG.mkdir(exist_ok=True)
GRAPH = ["GCN", "SAGE", "GAT", "Cheb", "GGraph", "LE"]
# Columns some cells carry and no cell uses. TAG was dropped for its training
# cost, which reached nine hours in one cell, against five to nine megawatts on
# the mixture and four to nine on the graph contribution.
DROP = ["TAG"]
WEIGHTS = {}                      # mixture and expert names of each cell drawn
CELLS = [("rfrance", "single"), ("synth", "single"),
         ("synth_structured", "single"), ("rfrance", "seq"),
         ("synth", "seq"), ("synth_structured", "seq")]
TITLE = {("rfrance", "single"): "French regions\nSingle-step",
         ("synth", "single"): "Synthetic, independent\nSingle-step",
         ("synth_structured", "single"): "Synthetic, structured\nSingle-step",
         ("rfrance", "seq"): "French regions\nSequence-to-sequence",
         ("synth", "seq"): "Synthetic, independent\nSequence-to-sequence",
         ("synth_structured", "seq"): "Synthetic, structured\nSequence-to-sequence"}

# Graph experts in the manuscript's plum family, non-graph experts apart.
COLOR = {"GCN": "#4A0030", "GGraph": "#7A1550", "SAGE": "#9C3F73",
         "GAT": "#B76B92", "Cheb": "#D298B0", "LE": "#E9C6D6",
         "MLP": "#2C6FB5", "GAM": "#E08A2B"}


def run_mixture(experts, y, step):
    """ML-Poly over the test year, committing ``step`` half-hours at a time.

    With step = 1 the mixture's own record is the online prediction. With a
    longer commitment the weights are frozen inside a block, so the block
    forecasts are written back into the mixture's record, which is what
    ``avg_loss`` reads and what the reported error measures.
    """
    if step == 1:
        mix = Mixture(y=y, experts=experts, model="MLpol", loss_type="mse")
        return mix, np.ravel(mix.predictions)
    K = experts.shape[1]
    pred = np.empty(len(y))
    pred[:step] = experts.iloc[:step].to_numpy() @ (np.ones(K) / K)
    mix = Mixture(y=y[:step], experts=experts.iloc[:step], model="MLpol", loss_type="mse")
    for s in range(step, len(y), step):
        e = min(s + step, len(y))
        pred[s:e] = np.ravel(mix.predict(new_experts=experts.iloc[s:e]))
        mix.update(new_experts=experts.iloc[s:e], new_y=y[s:e])
    mix.predictions = pred.copy()
    return mix, pred


def rmse(p, y):
    return float(np.sqrt(np.mean((np.asarray(p) - np.asarray(y)) ** 2)))


def cell(dataset, setting):
    path = RESULTS / f"experts_{dataset}_{setting}.csv"
    if not path.exists():
        print(f"absent, ignore: {path.name}")
        return None
    step = 1 if setting == "single" else 48
    df = pd.read_csv(path, index_col=0, parse_dates=True)
    y = df.pop("load").to_numpy(dtype=float)
    df = df.drop(columns=[c for c in DROP if c in df.columns])
    names = list(df.columns)
    colors = np.array([matplotlib.colors.to_rgb(COLOR[c]) for c in names])

    mix, pred = run_mixture(df, y, step)
    _, pred_ng = run_mixture(df[[c for c in names if c not in GRAPH]], y, step)
    _, pred_g = run_mixture(df[[c for c in names if c in GRAPH]], y, step)

    res = {c: rmse(df[c], y) for c in names}
    res.update({"uniform": rmse(df.mean(axis=1), y), "MLpol": rmse(pred, y),
                "MLpol-nograph": rmse(pred_ng, y), "MLpol-graph": rmse(pred_g, y)})
    print(f"\n=== {TITLE[(dataset, setting)]} ===")
    for k, v in res.items():
        print(f"  {k:>14}  {v:9.1f} MW")
    print(f"  {'graph share':>14}  {res['MLpol-nograph'] - res['MLpol']:9.1f} MW "
          f"lost when the graph experts leave")
    print(f"  {'other share':>14}  {res['MLpol-graph'] - res['MLpol']:9.1f} MW "
          f"lost when the two non-graph experts leave")

    w = pd.DataFrame(mix.weights, index=df.index[:len(mix.weights)], columns=names)
    mix.index = w.index                                    # kept for the comparison grid
    WEIGHTS[(dataset, setting)] = (mix, names)
    monthly = w[GRAPH].sum(axis=1).groupby(w.index.month).mean()
    print("  monthly weight of the graph experts:",
          " ".join(f"{v:.2f}" for v in monthly))

    plt.rcParams.update({"font.size": 8, "axes.labelsize": 8,
                         "xtick.labelsize": 7, "ytick.labelsize": 7})
    fig, ax = plt.subplots(1, 2, figsize=(6.4, 2.7), width_ratios=(1.7, 1), dpi=200)

    plot_weight(ax[0], colors, mix, max_experts=len(names),
                title="Weights played by ML-Poly", ylabel="Weight")
    idx = w.index
    ticks = [i for i in range(len(idx)) if idx[i].day == 1 and idx[i].hour == 0
             and idx[i].minute == 0 and idx[i].month % 2 == 1]
    ax[0].set_xticks(ticks)
    ax[0].set_xticklabels([idx[i].strftime("%b") for i in ticks])
    ax[0].set_xlim(0, len(idx) - 1)
    ax[0].set_ylim(0, 1)
    ax[0].legend(ncol=4, fontsize=7, loc="upper center", frameon=False,
                 bbox_to_anchor=(0.5, -0.13))

    avg_loss(ax[1], colors, mix, max_experts=len(names),
             title="Error over the test year", ylabel="RMSE (MW)")
    for bar in ax[1].patches:                  # mean squared error -> its root
        bar.set_height(np.sqrt(bar.get_height()))
    ax[1].relim()
    ax[1].autoscale_view()
    ax[1].set_ylim(0, 1.35 * max(b.get_height() for b in ax[1].patches))
    ax[1].axhline(res["MLpol-nograph"], color="black", lw=0.9, ls=(0, (4, 2)))
    ax[1].annotate(f"ML-Poly on the two\nnon-graph experts, {res['MLpol-nograph']:.0f} MW",
                   xy=(0.24, res["MLpol-nograph"]), xycoords=("axes fraction", "data"),
                   xytext=(0.03, 0.97), textcoords="axes fraction",
                   fontsize=6.5, va="top",
                   arrowprops=dict(arrowstyle="-", lw=0.6, color="black",
                                   shrinkA=0, shrinkB=2))
    ax[1].tick_params(axis="x", labelsize=7)

    for a in ax:
        a.set_title(a.get_title(), fontsize=8.5)
        a.grid(alpha=0.3)
    fig.tight_layout()
    suffix = "" if (dataset, setting) == ("rfrance", "single") else f"_{dataset}_{setting}"
    out = FIG / f"expert_aggregation{suffix}.pdf"
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    print("  written", out.name)
    return res


def grid(weights):
    """The six weight paths, structure by column group, setting by row.

    The two panels whose nodes carry inter-node structure sit together on the
    left, the panel whose nodes are independent given their own covariates on the
    right, so the reader crosses the boundary once instead of twice.

    Eight experts in six shades of one family are not separable at this size, and
    every statement the text makes concerns the families, so the graph experts
    are summed into one band. The per-expert paths stay in the per-cell figures.

    The dashed line at 0.75 is the share the six graph experts hold among the
    eight by count alone. An allocation that cannot separate its experts returns
    that share, so a plum band sitting on the line reports indifference and not
    an accuracy the edges bought. The strip under each panel carries that
    difference on its own, positive where ML-Poly buys the graph experts more
    than counting them would give and negative where it buys them less, which
    the eye cannot read off the stack.
    """
    plt.rcParams.update({"font.size": 8, "axes.labelsize": 8,
                         "xtick.labelsize": 7, "ytick.labelsize": 7})
    # A spacer column carries the boundary between the two regimes: white space
    # reads better than a rule drawn across the panels.
    # Une colonne et une ligne d'espacement portent les separations : du blanc se
    # lit mieux qu'un filet tire au travers des panneaux.
    fig, allax = plt.subplots(5, 4, figsize=(7.8, 6.1), dpi=200,
                              gridspec_kw={"width_ratios": (1, 1, 0.16, 1),
                                           "height_ratios": (1, 0.34, 0.30, 1, 0.34),
                                           "hspace": 0.08})
    for a in allax[:, 2].tolist() + allax[2, :].tolist():
        a.set_visible(False)
    axes = allax[(0, 3), :][:, (0, 1, 3)]          # les empilements
    gaps = allax[(1, 4), :][:, (0, 1, 3)]          # les ecarts a la part de comptage
    order = [("rfrance", "single"), ("synth_structured", "single"),
             ("synth", "single"), ("rfrance", "seq"),
             ("synth_structured", "seq"), ("synth", "seq")]
    share = len(GRAPH) / (len(GRAPH) + 2)
    for ax, gx, key in zip(axes.ravel(), gaps.ravel(), order):
        if key not in weights:
            ax.set_visible(False); gx.set_visible(False)
            continue
        mix, names = weights[key]
        w = pd.DataFrame(mix.weights, columns=names)
        bands = [("Graph experts", w[[c for c in names if c in GRAPH]].sum(axis=1),
                  COLOR["GGraph"]),
                 ("Feedforward", w["MLP"], COLOR["MLP"]),
                 ("Additive model", w["GAM"], COLOR["GAM"])]
        ax.stackplot(range(len(w)), *[b[1].to_numpy() for b in bands],
                     colors=[b[2] for b in bands], labels=[b[0] for b in bands],
                     edgecolor="white", linewidth=0.2)
        ax.set_title(TITLE[key])
        first = ax is axes[0, 0] or ax is axes[1, 0]
        ax.set_ylabel("Weight" if first else "")
        if not first: ax.set_yticklabels([])          # meme echelle partout
        idx = pd.DatetimeIndex(mix.index)
        ticks = [i for i in range(len(idx)) if idx[i].day == 1 and idx[i].hour == 0
                 and idx[i].minute == 0 and idx[i].month % 3 == 1]
        ax.set_xticks(ticks)
        ax.set_xticklabels([idx[i].strftime("%b") for i in ticks])
        ax.set_xlim(0, len(idx) - 1)
        ax.set_ylim(0, 1)
        ax.axhline(share, color="0.25", lw=0.7, ls=(0, (4, 2)), zorder=5)
        ax.set_title(ax.get_title(), fontsize=8.5)
        ax.grid(alpha=0.3)
        ax.set_xticklabels([])
        ax.set_xlabel("")

        # L'ecart a la part de comptage, la seule quantite que la pile ne montre pas.
        g = bands[0][1].to_numpy() - share
        t = np.arange(len(g))
        gx.fill_between(t, 0, g, where=g >= 0, color=COLOR["GGraph"], linewidth=0)
        gx.fill_between(t, 0, g, where=g < 0, color="0.55", linewidth=0)
        gx.axhline(0, color="0.25", lw=0.7, ls=(0, (4, 2)))
        gx.set_xlim(0, len(idx) - 1)
        gx.set_ylim(-0.8, 0.3)
        gx.set_yticks((-0.75, 0, 0.25))
        gx.set_yticklabels(("$-0.75$", "$0$", "$+0.25$"))
        gfirst = gx is gaps[0, 0] or gx is gaps[1, 0]
        gx.set_ylabel("Gap" if gfirst else "", labelpad=1)
        if not gfirst: gx.set_yticklabels([])
        gx.set_xticks(ticks)
        gx.set_xticklabels([idx[i].strftime("%b") for i in ticks])
        gx.tick_params(axis="x", labelsize=7)
        gx.grid(alpha=0.3)
    handles, labels = axes[0, 0].get_legend_handles_labels()
    handles.append(matplotlib.lines.Line2D([], [], color="0.25", lw=0.7, ls=(0, (4, 2))))
    labels.append("count share of the graph experts")
    fig.legend(handles, labels, ncol=len(labels), fontsize=7, frameon=False,
               loc="lower center", bbox_to_anchor=(0.5, -0.02))
    fig.tight_layout(rect=(0, 0.03, 1, 0.95))
    xa = (axes[0, 0].get_position().x0 + axes[0, 1].get_position().x1) / 2
    fig.text(xa, 0.965, "inter-node structure present", ha="center", fontsize=8.5,
             fontweight="bold")
    fig.text((axes[0, 2].get_position().x0 + axes[0, 2].get_position().x1) / 2, 0.965,
             "no structure", ha="center", fontsize=8.5, fontweight="bold")
    out = FIG / "expert_aggregation_grid.pdf"
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    print("written", out.name)


def main():
    if len(sys.argv) > 2:
        cell(sys.argv[1], sys.argv[2])
        return
    table = {c: cell(*c) for c in CELLS}
    grid(WEIGHTS)
    rows = [(TITLE[c], r) for c, r in table.items() if r]
    if len(rows) > 1:
        print("\n=== summary, RMSE in MW ===")
        # The two panels sit at different load levels, so the two contributions
        # are also given as a share of the mixture's own error.
        print(f"{'cell':>42} {'best expert':>12} {'uniform':>9} {'ML-Poly':>9}"
              f" {'-graph':>14} {'-others':>14}")
        for name, r in rows:
            best = min(v for k, v in r.items() if k in COLOR)
            g, o = r["MLpol-nograph"] - r["MLpol"], r["MLpol-graph"] - r["MLpol"]
            print(f"{name:>42} {best:12.0f} {r['uniform']:9.0f} {r['MLpol']:9.0f}"
                  f" {g:8.0f} ({100 * g / r['MLpol']:3.0f}%)"
                  f" {o:8.0f} ({100 * o / r['MLpol']:3.0f}%)")


if __name__ == "__main__":
    main()
