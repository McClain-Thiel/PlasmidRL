"""Reward-component ablation table + heatmap.

Pass-rate numbers come from a uniform strict-QC re-run on g6-big across
all six cells (full_reward + 5 ablations) at two temperatures
(T=0.95, T=1.15). Each cell is 4000 sequences (500 × 8 prompts) generated
with vLLM and scored by the packaged PlasmidRL strict-QC pipeline.
"""
import marimo

__generated_with = "0.23.1"
app = marimo.App(width="medium")


@app.cell
def _():
    import os
    from pathlib import Path

    import marimo as mo
    import matplotlib.pyplot as plt
    import numpy as np
    import pandas as pd

    from plasmidrl import style

    style.set_paper_style()
    FIGURE_DIR = Path(os.environ.get(
        "PLASMIDRL_FIGURE_DIR",
        Path(__file__).resolve().parents[1] / "paper" / "figures"))
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    DATA_DIR = Path(__file__).resolve().parents[1] / "data"
    return DATA_DIR, FIGURE_DIR, mo, np, pd, plt, style


@app.cell
def _(mo):
    mo.md(
        r"""
    # Reward-component ablations (strict QC, apples-to-apples)

    Each ablation freezes everything except the reward function. The
    reward is a sum of `length_prior`, `cassette_bonus`, `repeat_penalty`,
    and `cds_density`. Ablations remove one component (`no_*`) or keep
    only one (`*_only`); `full_reward` is the headline RL run.

    Both temperatures (T=0.95, T=1.15) shown side-by-side. T=1.15 is the
    GRPO-optimal sampling temperature used for the headline RL number;
    T=0.95 was the original ablation-eval setpoint.
    """
    )
    return


@app.cell
def _(DATA_DIR, pd):
    table = pd.read_csv(DATA_DIR / "ablation_metrics.csv")
    df_095 = table[table["T"] == 0.95].copy()
    df_115 = table[table["T"] == 1.15].copy()
    pivot = (table.pivot(index="ablation", columns="T",
                         values="pass_rate_pct")
             .sort_values(1.15, ascending=False))
    return df_095, df_115, pivot, table


@app.cell
def _(pivot):
    pivot
    return


@app.cell
def _(FIGURE_DIR, np, pivot, plt, style):
    ablations = list(pivot.index)
    _x = np.arange(len(ablations))
    _w = 0.4

    _fig, _ax = plt.subplots(figsize=(9, 4.2))
    _bars1 = _ax.bar(_x - _w / 2, pivot[1.15].values, _w,
                     color=style.PALETTE["RL"], edgecolor=style.EDGE,
                     linewidth=1.0, alpha=0.92, label="T = 1.15")
    _bars2 = _ax.bar(_x + _w / 2, pivot[0.95].values, _w,
                     color=style.PALETTE["SFT"], edgecolor=style.EDGE,
                     linewidth=1.0, alpha=0.85, label="T = 0.95")
    for _b, _p in zip(_bars1, pivot[1.15].values):
        _ax.text(_b.get_x() + _b.get_width() / 2, _b.get_height() + 1.5,
                 f"{_p:.1f}", ha="center", fontsize=8.5, fontweight="bold")
    for _b, _p in zip(_bars2, pivot[0.95].values):
        _ax.text(_b.get_x() + _b.get_width() / 2, _b.get_height() + 1.5,
                 f"{_p:.1f}", ha="center", fontsize=8.5, fontweight="bold")

    _ax.set_xticks(_x)
    _ax.set_xticklabels(ablations, rotation=20, ha="right")
    _ax.set_ylabel("Strict-QC pass rate (%)")
    _ax.set_xlabel("Reward variant")
    _ax.set_ylim(0, 100)
    _ax.set_title("Reward-component ablations (n=4000 per cell, strict QC)",
                  loc="left")
    _ax.legend(title="Sampling T", loc="upper right")
    _fig.tight_layout()
    _fig.savefig(FIGURE_DIR / "fig1_ablation_pass_rate.pdf")
    _fig.savefig(FIGURE_DIR / "fig1_ablation_pass_rate.png", dpi=200)
    _fig
    return


@app.cell
def _(mo):
    mo.md(
        r"""
    ## Sequence Diversity

    Diversity is measured as `1 - mean pairwise 21-mer Jaccard similarity`
    over all sequences that passed the strict QC filter. The Addgene
    baseline represents the diversity of the training corpus.
    """
    )
    return


@app.cell
def _(table):
    div_df = table[["T", "ablation", "diversity_21mer_jaccard_dist"]].rename(
        columns={"diversity_21mer_jaccard_dist": "diversity"}
    )
    div_pivot = div_df.pivot(index="ablation", columns="T",
                             values="diversity")
    addgene_baseline = 0.9245
    return addgene_baseline, div_df, div_pivot


@app.cell
def _(div_pivot):
    div_pivot
    return


@app.cell
def _(FIGURE_DIR, addgene_baseline, div_pivot, np, plt, style):
    if div_pivot is not None:
        ablations_d = list(div_pivot.index)
        _x = np.arange(len(ablations_d))
        _w = 0.4
        _fig, _ax = plt.subplots(figsize=(9, 4.2))
        _ax.bar(_x - _w / 2, div_pivot[1.15].values, _w,
                color=style.PALETTE["RL"], edgecolor=style.EDGE,
                linewidth=1.0, alpha=0.92, label="T = 1.15")
        _ax.bar(_x + _w / 2, div_pivot[0.95].values, _w,
                color=style.PALETTE["SFT"], edgecolor=style.EDGE,
                linewidth=1.0, alpha=0.85, label="T = 0.95")
        if addgene_baseline is not None:
            _ax.axhline(addgene_baseline, color=style.EDGE, linestyle="--",
                        linewidth=1.0, label="Addgene baseline")
        _ax.set_xticks(_x)
        _ax.set_xticklabels(ablations_d, rotation=20, ha="right")
        _ax.set_ylabel("Diversity (1 − mean 21-mer Jaccard)")
        _ax.set_xlabel("Reward variant")
        _ax.set_ylim(0, 1.0)
        _ax.set_title("Ablation diversity (passing sequences)", loc="left")
        _ax.legend(loc="upper right")
        _fig.tight_layout()
        _fig.savefig(FIGURE_DIR / "fig_ablation_diversity.pdf")
        _fig.savefig(FIGURE_DIR / "fig_ablation_diversity.png", dpi=200)
        _fig_out = _fig
    else:
        _fig_out = None
    _fig_out
    return


if __name__ == "__main__":
    app.run()
