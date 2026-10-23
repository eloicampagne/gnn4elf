"""Regional error maps for the French cells (fig/nodal_errors.pdf).

The aggregation figures score the national sum, where a region's overshoot can
cancel another's shortfall. This one keeps the regions apart: every panel is the
map GraphToolbox's ``plot_node_errors_map`` renders, one disc per region coloured
by the mean absolute percentage error of its own forecast over the test year.

Three forecasters are shown, the strongest graph network of the cell, the
additive model, and ML-Poly run node by node, which is the bottom aggregation of
the chapter: each region gets its own mixture and its own weights, and the
national forecast is the sum of the twelve.

Reads ``data/nodal_<dataset>_<setting>.npz``, written by
``experiments/aggregation/aggreg_experts.py``: per-node forecasts of every expert, the
observed load, and the region names.

    python nodal_errors.py            # single-step, the default cell
    python nodal_errors.py rfrance seq

Dependencies: numpy, pandas, matplotlib, basemap, opera, graphtoolbox.
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from opera import Mixture

from graphtoolbox.utils.visualizations import plot_node_errors_map

HERE = Path(__file__).resolve().parent
RESULTS = HERE.parent / "results_aggreg"
FIG = HERE / "out"
FIG.mkdir(exist_ok=True)
GRAPH = ["GCN", "SAGE", "GAT", "Cheb", "GGraph", "LE"]
DROP = ["TAG"]                    # dropped for its training cost, see expert_aggregation.py

# Coordinates of the reference city of each region, the positions the framework
# expects in its ``df_pos`` argument.
POS = pd.DataFrame({
    "REGION": ["Hauts_de_France", "Normandie", "Ile_de_France", "Grand_Est",
               "Bretagne", "Pays_de_la_Loire", "Centre_Val_de_Loire",
               "Bourgogne_Franche_Comte", "Nouvelle_Aquitaine",
               "Auvergne_Rhone_Alpes", "Occitanie", "Provence_Alpes_Cote_d_Azur"],
    "LATITUDE": [50.6365654, 49.4404591, 48.862725, 48.584614, 48.3905283,
                 47.2186371, 47.9027336, 47.3215806, 44.841225, 45.7578137,
                 43.6044622, 43.2961743],
    "LONGITUDE": [3.0635282, 1.0939658, 2.287592, 7.7507127, -4.4860088,
                  -1.5541362, 1.9086066, 5.0414701, -0.5800364, 4.8320114,
                  1.4442469, 5.3699525]})


# Official codes of the twelve metropolitan regions, the names being too long
# for the panels at the width the manuscript prints them.
ABBR = {"Auvergne_Rhone_Alpes": "ARA", "Bourgogne_Franche_Comte": "BFC",
        "Bretagne": "BRE", "Centre_Val_de_Loire": "CVL", "Grand_Est": "GES",
        "Hauts_de_France": "HDF", "Ile_de_France": "IDF", "Normandie": "NOR",
        "Nouvelle_Aquitaine": "NAQ", "Occitanie": "OCC",
        "Pays_de_la_Loire": "PDL", "Provence_Alpes_Cote_d_Azur": "PAC"}


def mape(pred, true):
    """Mean absolute percentage error per node, in percent."""
    return np.abs((true - pred) / np.clip(np.abs(true), 1e-8, None)).mean(0) * 100


def bottom_mixture(pred, true, step):
    """ML-Poly per node, the chapter's bottom aggregation."""
    out = np.zeros_like(true)
    for j in range(true.shape[1]):
        experts = pd.DataFrame(pred[:, j, :])
        if step == 1:
            m = Mixture(y=true[:, j], experts=experts, model="MLpol", loss_type="mse")
            out[:, j] = np.ravel(m.predictions)
            continue
        K = experts.shape[1]
        out[:step, j] = experts.iloc[:step].to_numpy() @ (np.ones(K) / K)
        m = Mixture(y=true[:step, j], experts=experts.iloc[:step],
                    model="MLpol", loss_type="mse")
        for s in range(step, len(true), step):
            e = min(s + step, len(true))
            out[s:e, j] = np.ravel(m.predict(new_experts=experts.iloc[s:e]))
            m.update(new_experts=experts.iloc[s:e], new_y=true[s:e, j])
    return out


def main(dataset="rfrance", setting="single"):
    z = np.load(RESULTS / f"nodal_{dataset}_{setting}.npz", allow_pickle=True)
    nodes, experts = list(z["nodes"]), list(z["experts"])
    pred, true = z["pred"], z["true"]
    keep = [i for i, e in enumerate(experts) if e not in DROP]
    experts, pred = [experts[i] for i in keep], pred[:, :, keep]
    step = 1 if setting == "single" else 48

    err = {e: mape(pred[:, :, k], true) for k, e in enumerate(experts)}
    best = min((e for e in experts if e in GRAPH), key=lambda e: err[e].mean())
    bottom = bottom_mixture(pred, true, step)
    err["ML-Poly, node by node"] = mape(bottom, true)
    nat = np.sqrt(((bottom.sum(1) - true.sum(1)) ** 2).mean())
    print(f"  national RMSE of the node-level mixture {nat:.1f} MW")

    print(f"{dataset}/{setting}: mean regional MAPE")
    for k, v in err.items():
        print(f"  {k:>22}  {v.mean():5.2f}%   worst {nodes[int(np.argmax(v))]}"
              f" {v.max():5.2f}%")

    panels = {f"{best}": err[best], "GAM": err["GAM"],
              "ML-Poly, node by node": err["ML-Poly, node by node"]}
    plot_node_errors_map(panels, nodes, POS, metric_label="MAPE (%)",
                         figsize=(3.4, 3.1), node_size=90)
    fig = plt.gcf()
    for ax in fig.axes[:len(panels)]:
        ax.set_title(ax.get_title(), fontsize=9)
    # The module writes the region's full name above its disc, which runs past
    # the frame on three panels of this width. The official codes fit, and the
    # caption carries the correspondence.
    # The module has already turned the region name into its display form,
    # underscores out and accents in, so the codes are keyed on that form.
    shown = {k.replace("_", " ").replace("Cote d Azur", "Côte d'Azur"): v
             for k, v in ABBR.items()}
    for txt in fig.findobj(matplotlib.text.Annotation):
        name, _, value = txt.get_text().partition("\n")
        txt.set_text(f"{shown.get(name, ABBR.get(name.replace(' ', '_'), name))}\n{value}")
        txt.set_fontsize(5.4)
        txt.set_clip_on(True)
        # Four labels move off the default position above their disc: Lille is
        # within a degree of the frame's northern edge, Paris and Orléans sit on
        # top of each other, and Lyon crowds the eastern border.
        side = {"HDF": (11, 0, "center", "left"), "CVL": (0, -12, "top", "center"),
                "ARA": (-11, 0, "center", "right")}.get(txt.get_text()[:3])
        if side:
            dx, dy, va, ha = side
            txt.set_position((dx, dy))
            txt.set_verticalalignment(va)
            txt.set_horizontalalignment(ha)
    suffix = "" if setting == "single" else f"_{setting}"
    out = FIG / f"nodal_errors{suffix}.pdf"
    fig.savefig(out, bbox_inches="tight", dpi=200)
    print("written", out.name)


if __name__ == "__main__":
    main(*(sys.argv[1:3] or ["rfrance", "single"]))
