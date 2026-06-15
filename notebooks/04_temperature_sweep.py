"""Pass rate × diversity × folding stability vs sampling temperature."""
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

    from plasmidrl import data, style

    style.set_paper_style()
    FIGURE_DIR = Path(os.environ.get(
        "PLASMIDRL_FIGURE_DIR",
        Path(__file__).resolve().parents[1] / "paper" / "figures"))
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    return FIGURE_DIR, data, mo, np, pd, plt, style


@app.cell
def _(mo):
    mo.md(
        r"""
    # Temperature sweep — pass rate × diversity × folding

    Sampling temperature trades off pass rate against diversity. Per-cell
    summaries store both quantities at four temperatures
    (`evaluation/temperature_sweep/{Base,SFT,GRPO}_summary.csv`); diversity
    is `1 − mean pairwise 21-mer Jaccard` over passing sequences.
    Headline numbers in the paper use the per-cell optimum: Base 1.0,
    SFT 1.0, RL 1.15. The lower row shows folding free energy on a
    2-prompt subset for SFT and RL.
    """
    )
    return


@app.cell
def _(data, pd):
    base = data.load_csv("evaluation/temperature_sweep/Base_summary.csv")
    sft = data.load_csv("evaluation/temperature_sweep/SFT_summary.csv")
    grpo = data.load_csv("evaluation/temperature_sweep/GRPO_summary.csv")
    base["cell"] = "Base"
    sft["cell"] = "SFT"
    grpo["cell"] = "RL"
    sweep = pd.concat([base, sft, grpo], ignore_index=True)
    sweep = sweep.sort_values(["cell", "temperature"]).reset_index(drop=True)
    return sweep,


@app.cell
def _(sweep):
    sweep[["cell", "temperature", "n_total", "n_passed",
           "pass_rate_pct", "diversity_21mer_jaccard"]]
    return


@app.cell
def _(data, pd):
    sft_mfe = data.load_csv("mfe/SFT_temp_sweep/per_T_per_seq.csv")
    rl_mfe = data.load_csv("mfe/RL_temp_sweep_2prompt/per_T_per_seq.csv")
    sft_mfe["cell"] = "SFT"
    rl_mfe["cell"] = "RL"
    mfe_sweep = pd.concat(
        [sft_mfe[["cell", "T", "mfe_density_dna"]],
         rl_mfe[["cell", "T", "mfe_density_dna"]]],
        ignore_index=True,
    ).dropna(subset=["mfe_density_dna"])
    mfe_means = (mfe_sweep.groupby(["cell", "T"])
                 .agg(mean=("mfe_density_dna", "mean"),
                      std=("mfe_density_dna", "std"),
                      n=("mfe_density_dna", "size"))
                 .reset_index())
    return mfe_means, mfe_sweep


@app.cell
def _(FIGURE_DIR, mfe_means, plt, style, sweep):
    _fig, _axes = plt.subplots(1, 3, figsize=(13, 4))

    _ax = _axes[0]
    for _cell, _sub in sweep.groupby("cell"):
        _ax.plot(_sub["temperature"], _sub["pass_rate_pct"], "o-",
                 color=style.PALETTE[_cell], label=_cell, linewidth=2,
                 markersize=7, markeredgecolor=style.EDGE)
    _ax.set_xlabel("Sampling temperature")
    _ax.set_ylabel("QC Pass Rate (%)")
    _ax.set_ylim(0, 100)
    _ax.set_title("A) Pass rate vs T", loc="left")
    _ax.legend(title="Model")

    _ax = _axes[1]
    for _cell, _sub in sweep.groupby("cell"):
        _ax.plot(_sub["temperature"], _sub["diversity_21mer_jaccard"], "o-",
                 color=style.PALETTE[_cell], label=_cell, linewidth=2,
                 markersize=7, markeredgecolor=style.EDGE)
    _ax.set_xlabel("Sampling temperature")
    _ax.set_ylabel("Diversity (21-mer Jaccard)")
    _ax.set_title("B) Diversity vs T", loc="left")
    _ax.legend(title="Model")

    _ax = _axes[2]
    for _cell, _sub in mfe_means.groupby("cell"):
        _ax.errorbar(_sub["T"], _sub["mean"], yerr=_sub["std"], fmt="o-",
                     color=style.PALETTE[_cell], label=_cell, capsize=3,
                     linewidth=2, markersize=7, markeredgecolor=style.EDGE)
    _ax.axhline(-0.15, color="#888", linestyle="--", linewidth=0.8,
                label="Real centre")
    _ax.set_xlabel("Sampling temperature")
    _ax.set_ylabel("MFE density (kcal/mol/bp)")
    _ax.set_title("C) Folding stability vs T", loc="left")
    _ax.legend()

    _fig.tight_layout()
    _fig.savefig(FIGURE_DIR / "fig_temperature_sweep.pdf")
    _fig.savefig(FIGURE_DIR / "fig_temperature_sweep.png", dpi=200)
    _fig
    return


if __name__ == "__main__":
    app.run()
