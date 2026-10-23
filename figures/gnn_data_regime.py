"""National RMSE of the four nodewise models against the training window
(fig/gnn_data_regime.pdf).

Reads Table 2.7 (`tab:data-regime` in chapters/gnn.typ): national RMSE in
megawatts of the four nodewise models on the twelve French regions, as the
training set falls from 1090 to 54 days at fixed validation and test windows.
Every number is that table's; nothing is recomputed.

Left panel: the four errors. Right panel: the same table read as increments over
the per-node perceptron, which is the model holding the same covariates and none
of the graph's constraints. The access restriction is the difference between the
global and the per-node perceptron, neither of which shares weights; the sharing
cost is the difference between the identity graph network and the per-node
perceptron, both of which read a node's own features alone.
"""

from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

import style

HERE = Path(__file__).resolve().parent
OUT = HERE / "out" / "gnn_data_regime.pdf"
OUT.parent.mkdir(exist_ok=True)

DAYS = np.array([54, 109, 272, 545, 1090])
PCT = ["5%", "10%", "25%", "50%", "100%"]
RMSE = {
    "Per-node MLP": np.array([1948, 1412, 1090, 958, 876]),
    "Identity GNN": np.array([2038, 1763, 1292, 1113, 918]),
    "Graph GNN": np.array([2080, 1738, 1206, 983, 891]),
    "Global MLP": np.array([2432, 1687, 1179, 1014, 921]),
}
COL = {"Per-node MLP": "black", "Identity GNN": style.GREEN,
       "Graph GNN": style.BLUE, "Global MLP": style.ORANGE}
MARK = {"Per-node MLP": "o", "Identity GNN": "s", "Graph GNN": "^", "Global MLP": "D"}


def main():
    style.apply()
    fig, (a, b) = plt.subplots(1, 2, figsize=(6.6, 2.5),
                               gridspec_kw={"wspace": 0.28})

    for name, v in RMSE.items():
        a.plot(DAYS, v, marker=MARK[name], ms=3.2, color=COL[name], label=name,
               lw=1.4 if name == "Per-node MLP" else 1.1)
    a.set_xscale("log")
    a.set_yscale("log")
    a.set_xticks(DAYS)
    a.set_xticklabels([f"{d}" for d in DAYS])
    a.set_yticks([900, 1200, 1600, 2400])
    a.set_yticklabels(["900", "1200", "1600", "2400"])
    a.minorticks_off()
    a.set_xlabel("training days")
    a.set_ylabel("national RMSE (MW)")
    a.set_title("the four nodewise models")
    a.legend(frameon=False, handlelength=1.5, borderpad=0.1, loc="upper right")
    a.grid(axis="y", zorder=0)

    ref = RMSE["Per-node MLP"]
    inc = {
        "access restriction, from Global MLP": RMSE["Global MLP"] - ref,
        "edges, from Graph GNN": RMSE["Graph GNN"] - ref,
        "weight sharing, from Identity GNN": RMSE["Identity GNN"] - ref,
    }
    ic = {"access restriction, from Global MLP": style.ORANGE,
          "edges, from Graph GNN": style.BLUE,
          "weight sharing, from Identity GNN": style.GREEN}
    im = {"access restriction, from Global MLP": "D",
          "edges, from Graph GNN": "^",
          "weight sharing, from Identity GNN": "s"}
    b.axhline(0, color="black", lw=0.7)
    for name, v in inc.items():
        b.plot(DAYS, v, marker=im[name], ms=3.2, color=ic[name], label=name)
    b.annotate("484", (DAYS[0], inc["access restriction, from Global MLP"][0]),
               textcoords="offset points", xytext=(4, 3), fontsize=7,
               color=style.ORANGE)
    b.annotate("45", (DAYS[-1], inc["access restriction, from Global MLP"][-1]),
               textcoords="offset points", xytext=(-3, 9), fontsize=7,
               color=style.ORANGE, ha="right")
    b.set_xscale("log")
    b.set_xticks(DAYS)
    b.set_xticklabels([f"{d}\n{p}" for d, p in zip(DAYS, PCT)])
    b.minorticks_off()
    b.set_xlabel("training days")
    b.set_ylabel("cost over the per-node reference (MW)")
    b.set_title("gap to the per-node MLP")
    b.legend(frameon=False, handlelength=1.5, borderpad=0.1, loc="upper right")
    b.grid(axis="y", zorder=0)

    fig.savefig(OUT)
    print(f"wrote {OUT}")
    for name, v in inc.items():
        print(f"  {name:36s} " + "  ".join(f"{x:+5.0f}" for x in v))


if __name__ == "__main__":
    main()
