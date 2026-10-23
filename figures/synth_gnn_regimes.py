"""La carte des regimes du panel synthetique (figure fig/synth_gnn_regimes.pdf).

Le generateur de ``experiments/synthetic/synth_gnn.py``
tire un champ de temperature propre, correle spatialement a la force rho, et le
fait observer a chaque noeud avec un bruit propre d'ecart-type sigma_obs. La
charge est la spline temperature-charge reelle de la region appliquee au champ
propre, si bien que la cible depend de ce que le noeud n'observe pas exactement.
Le trait d'un noeud est cette seule lecture bruitee, sans retard de charge.

Le reseau de graphe n'est aucune des convolutions de PyG. C'est un encodeur
partage suivi de deux tours d'agregation par la moyenne sur D^-1 (A + I), chacun
avec une connexion residuelle, puis d'une lecture partagee, largeur cachee 32. Le
controle d'identite remplace A + I par la matrice identite et ne laisse que le
partage des poids, ce qui en fait un MLP partage. Les deux modeles ont donc le
meme tronc et ne different que par l'operateur.

``synth_sweep.py`` balaie quatre axes autour de la configuration de base et ecrit
data/synth_sweep.json, trois graines par point.

    python synth_gnn_regimes.py
"""

import json
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import style

HERE = Path(__file__).resolve().parent
FIG = HERE / "out"
FIG.mkdir(exist_ok=True)

BASE = dict(N=60, rho=0.7, sig_obs=3.0, frac=0.3)
AXES = [("N", [12, 30, 60, 120, 250], "number of nodes $N$", True),
        ("frac", [0.05, 0.10, 0.25, 0.50, 1.00], "fraction of the training set", False),
        ("rho", [0.0, 0.3, 0.5, 0.7, 0.9], "correlation $\\rho$ of the clean field", False),
        ("sig_obs", [0.0, 1.0, 2.0, 4.0, 6.0], "own-observation noise $\\sigma_{\\mathrm{obs}}$", False)]
MODELS = [("per_node", "per-node MLP", style.GREEN, "s", "--"),
          ("global", "global MLP", style.ORANGE, "o", "-"),
          ("gnn_eye", "shared MLP (identity GNN)", style.GREY, "^", ":"),
          ("gnn_graph", "graph GNN (mean aggregation)", style.BLUE, "D", "-")]
SEEDS = (0, 1, 2)


def curves(res, axis, values):
    """Moyenne et ecart-type sur les graines, un point par valeur de l'axe."""
    out = {m: ([], []) for m, *_ in MODELS}
    for v in values:
        cfg = dict(BASE); cfg[axis] = v
        keys = [f"{cfg['N']}|{cfg['rho']}|{cfg['sig_obs']}|{cfg['frac']}|{s}" for s in SEEDS]
        keys = [k for k in keys if k in res]
        for m, *_ in MODELS:
            a = np.array([res[k][m] for k in keys])
            out[m][0].append(a.mean()); out[m][1].append(a.std())
    return {m: (np.array(mu), np.array(sd)) for m, (mu, sd) in out.items()}


def main():
    style.apply()
    res = json.load(open(HERE.parent / "results_synth_gnn_when_useful" / "synth_sweep.json"))
    fig, axes = plt.subplots(2, 2, figsize=(6.4, 3.5), dpi=200)
    for ax, (axis, values, label, logx) in zip(axes.ravel(), AXES):
        c = curves(res, axis, values)
        x = np.array(values, dtype=float)
        for m, name, col, mk, ls in MODELS:
            mu, sd = c[m]
            ax.plot(x, mu, ls, color=col, marker=mk, markersize=3.2, linewidth=1.2,
                    label=name, zorder=3 if m == "gnn_graph" else 2)
            ax.fill_between(x, mu - sd, mu + sd, color=col, alpha=0.13, linewidth=0)
        if logx:
            ax.set_xscale("log"); ax.set_xticks(x)
            ax.set_xticklabels([str(int(v)) for v in x])
        ax.set_xlabel(label)
        ax.set_ylim(bottom=0)
        ax.grid(axis="y", color="0.9", linewidth=0.6, zorder=0)
    for ax in axes[:, 0]:
        ax.set_ylabel("RMSE (standardized)")
    h, l = axes[0, 0].get_legend_handles_labels()
    fig.legend(h, l, loc="upper center", ncol=4, frameon=False,
               bbox_to_anchor=(0.5, 1.03), handlelength=1.6, columnspacing=1.1,
               fontsize=6.5)
    fig.tight_layout(rect=(0, 0, 1, 0.93), h_pad=0.6)
    fig.savefig(FIG / "synth_gnn_regimes.pdf")
    print("ecrit", FIG / "synth_gnn_regimes.pdf")


if __name__ == "__main__":
    main()
