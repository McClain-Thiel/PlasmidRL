"""Shared figure style for the paper notebooks.

Matches the camera-ready palette used by ``fig_distribution_grid.png``:
Real plasmids in green, the pretrained Base in dark navy, SFT in purple,
and the GRPO-trained RL model in red. ``set_paper_style()`` configures
matplotlib with paper-ready defaults (sans-serif, bold labels, no top /
right spines, light dashed grid, 200 DPI saves).
"""
from __future__ import annotations

from typing import Iterable

import matplotlib.pyplot as plt

PALETTE: dict[str, str] = {
    "Real": "#08306b",
    "Addgene": "#08306b",
    "Base": "#c6dbef",
    "SFT": "#6baed6",
    "RL": "#2171b5",
    "GRPO": "#2171b5",
    "SFT_real": "#6baed6",
}

ORDER: tuple[str, ...] = ("Real", "Base", "SFT", "RL")
EDGE = "#1f3a5f"


def set_paper_style() -> None:
    """Configure matplotlib rcParams to match the camera-ready style."""
    plt.rcParams.update({
        "font.family": "DejaVu Sans",
        "font.size": 11,
        "axes.titleweight": "bold",
        "axes.titlesize": 12,
        "axes.labelweight": "bold",
        "axes.labelsize": 11,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.edgecolor": "#222222",
        "axes.linewidth": 1.0,
        "axes.grid": True,
        "axes.axisbelow": True,
        "grid.color": "#cccccc",
        "grid.linestyle": "--",
        "grid.linewidth": 0.6,
        "grid.alpha": 0.7,
        "xtick.color": "#222222",
        "ytick.color": "#222222",
        "legend.frameon": True,
        "legend.framealpha": 0.95,
        "legend.edgecolor": "#cccccc",
        "savefig.dpi": 200,
        "savefig.bbox": "tight",
        "figure.facecolor": "white",
    })


def violin(ax, values_per_cell: list, cells: Iterable[str],
           reference_median: float | None = None,
           reference_label: str | None = None) -> None:
    """Draw a model-coloured violin plot with a quartile box overlay.

    ``values_per_cell`` is a list of 1-D arrays, one per cell name in
    ``cells``. Cells must appear in :data:`PALETTE`. When
    ``reference_median`` is not None, a dashed horizontal guideline is
    drawn at that y-value.
    """
    cells = list(cells)
    parts = ax.violinplot(values_per_cell, showmedians=False,
                          showextrema=False, widths=0.85)
    for body, name in zip(parts["bodies"], cells):
        body.set_facecolor(PALETTE[name])
        body.set_edgecolor(EDGE)
        body.set_alpha(0.85)
        body.set_linewidth(0.8)

    bp = ax.boxplot(values_per_cell, widths=0.18, showfliers=False,
                    patch_artist=True,
                    medianprops=dict(color="white", linewidth=1.5),
                    whiskerprops=dict(color=EDGE, linewidth=1.0),
                    capprops=dict(color=EDGE, linewidth=1.0))
    for box in bp["boxes"]:
        box.set_facecolor(EDGE)
        box.set_edgecolor(EDGE)

    ax.set_xticks(range(1, len(cells) + 1))
    ax.set_xticklabels(cells)

    if reference_median is not None:
        ax.axhline(reference_median, color="#888", linestyle="--",
                   linewidth=0.8, label=reference_label, zorder=0)


def model_bar(ax, cells: Iterable[str], values: Iterable[float], *,
              labels: Iterable[str] | None = None, alpha: float = 0.9) -> list:
    """Draw model-coloured bars with edge and an optional value label above each."""
    cells = list(cells)
    values = list(values)
    bars = ax.bar(cells, values,
                  color=[PALETTE[c] for c in cells],
                  edgecolor=EDGE, linewidth=1.0, alpha=alpha)
    if labels is not None:
        for bar, lbl in zip(bars, list(labels)):
            ax.text(bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + (max(values) * 0.01),
                    lbl, ha="center", va="bottom",
                    fontsize=9, fontweight="bold")
    return bars


__all__ = ["PALETTE", "ORDER", "EDGE", "set_paper_style", "violin", "model_bar"]
