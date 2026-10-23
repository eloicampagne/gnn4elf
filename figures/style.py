"""Shared matplotlib settings for the manuscript figures.

The figures already in `fig/` are drawn with matplotlib's default DejaVu Sans, so
the family and the math font are left untouched here and only the sizes, the
spines and the palette are set. Colours are those of the CeTZ diagrams of the
second chapter, so a reader crossing from a drawn figure to a plotted one meets
the same palette.
"""

import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

FIG = Path(__file__).resolve().parent / "out"

BLUE = "#3b6fd0"
DARK = "#2b4c8f"
GREEN = "#4e9a51"
PRUNE = "#62003c"
ORANGE = "#d1791f"
GREY = "#696969"


def apply():
    plt.rcParams.update({
        "font.size": 8,
        "axes.labelsize": 8,
        "axes.titlesize": 8.5,
        "xtick.labelsize": 7.5,
        "ytick.labelsize": 7.5,
        "legend.fontsize": 7,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.linewidth": 0.6,
        "xtick.major.width": 0.6,
        "ytick.major.width": 0.6,
        "lines.linewidth": 1.2,
        "grid.linewidth": 0.4,
        "grid.color": "#cccccc",
        "pdf.fonttype": 42,
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.02,
    })


def parse_lang(argv=None):
    """Label language of a figure, read from ``--lang en`` (French by default).

    The two general introductions share the same plots, so every script writes a
    French and an English version of its figure from the same data.
    """
    argv = list(sys.argv[1:] if argv is None else argv)
    lang = "fr"
    for i, arg in enumerate(argv):
        if arg.startswith("--lang="):
            lang = arg.split("=", 1)[1]
        elif arg == "--lang":
            if i + 1 >= len(argv):
                raise SystemExit("--lang expects a value: fr or en")
            lang = argv[i + 1]
    if lang not in ("fr", "en"):
        raise SystemExit(f"unknown language {lang!r}: expected fr or en")
    return lang


def output(stem, lang, suffix=".pdf"):
    """Destination of a figure: ``fig/<stem>`` in French, ``fig/<stem>_en`` in English."""
    return FIG / f"{stem}{'' if lang == 'fr' else '_en'}{suffix}"
