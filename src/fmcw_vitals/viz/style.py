"""One place for the figure style so every plot in the repo looks consistent."""
from __future__ import annotations

import matplotlib as mpl

# Okabe-Ito colorblind-safe palette
COLORS = {
    "breath": "#0072B2",
    "heart": "#D55E00",
    "truth": "#009E73",
    "raw": "#8C8C8C",
    "accent": "#CC79A7",
}
CMAP = "magma"


def apply_style() -> None:
    mpl.rcParams.update(
        {
            "figure.figsize": (10, 4.5),
            "figure.dpi": 110,
            "savefig.dpi": 300,
            "savefig.bbox": "tight",
            "font.family": "DejaVu Sans",
            "font.size": 11,
            "axes.titlesize": 12,
            "axes.titleweight": "semibold",
            "axes.labelsize": 11,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": True,
            "grid.alpha": 0.25,
            "grid.linewidth": 0.6,
            "lines.linewidth": 1.4,
            "legend.frameon": False,
        }
    )
