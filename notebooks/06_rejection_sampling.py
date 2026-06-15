"""Rejection sampling — top-K success curve, diversity collapse, ORI/AMR usage."""
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
    # Rejection sampling — does test-time compute close the gap?

    A natural baseline for "RL gives me higher pass rate" is best-of-K
    rejection sampling at the same compute budget: draw K samples per
    prompt from the Base or SFT model, keep one that passes QC.

    `rejection_topK/` stores success rates and diversity at K ∈ {1, 4, 16,
    64}. Even at K=64 the Base model lags the RL pass rate, while RL
    diversity collapses far less than rejection-sampled SFT.
    """
    )
    return


@app.cell
def _(data):
    success = data.load_csv("rejection_topK/success_per_model_K.csv")
    diversity = data.load_csv("rejection_topK/diversity.csv")
    ori_usage = data.load_csv("rejection_topK/ori_usage.csv")
    amr_usage = data.load_csv("rejection_topK/amr_usage.csv")
    rs_summary = data.load_json("rejection_topK/summary.json")

    label_map = {"Base": "Base", "SFT_real": "SFT", "GRPO": "RL"}
    color_map = {"Base": "Base", "SFT_real": "SFT", "GRPO": "RL"}
    return amr_usage, color_map, diversity, label_map, ori_usage, rs_summary, success


@app.cell
def _(success):
    success
    return


@app.cell
def _(FIGURE_DIR, color_map, diversity, label_map, plt, style, success):
    _fig, _axes = plt.subplots(1, 2, figsize=(11, 4))

    _ax = _axes[0]
    for _model, _sub in success.groupby("model"):
        _ss = _sub.sort_values("K")
        _ax.plot(_ss["K"], _ss["success_rate_pct"], "o-",
                 color=style.PALETTE[color_map.get(_model, "Base")],
                 label=label_map.get(_model, _model), linewidth=2,
                 markersize=8, markeredgecolor=style.EDGE)
    _ax.set_xscale("log", base=2)
    _ax.set_xlabel("K (samples per prompt)")
    _ax.set_ylabel("Best-of-K success rate (%)")
    _ax.set_ylim(0, 105)
    _ax.set_title("A) Pass rate vs test-time compute", loc="left")
    _ax.legend(title="Model")

    _ax = _axes[1]
    for _model, _sub in diversity.groupby("model"):
        _ss = _sub.sort_values("K")
        _ax.plot(_ss["K"], _ss["mean_jaccard"], "o-",
                 color=style.PALETTE[color_map.get(_model, "Base")],
                 label=label_map.get(_model, _model), linewidth=2,
                 markersize=8, markeredgecolor=style.EDGE)
    _ax.set_xscale("log", base=2)
    _ax.set_xlabel("K (samples per prompt)")
    _ax.set_ylabel("Mean pairwise 21-mer Jaccard")
    _ax.set_title("B) Diversity collapse with K", loc="left")
    _ax.legend(title="Model")

    _fig.tight_layout()
    _fig.savefig(FIGURE_DIR / "fig_rejection_topK.pdf")
    _fig.savefig(FIGURE_DIR / "fig_rejection_topK.png", dpi=200)
    _fig
    return


@app.cell
def _(mo):
    mo.md(
        r"""
    ## ORI / AMR usage among kept passers

    Top-3 most frequent ORIs and AMR genes per model at K=1 and K=64. If
    RL collapses to a single ORI/AMR pair while rejection sampling
    spreads broadly across the curated database, this surfaces it.
    """
    )
    return


@app.cell
def _(amr_usage, label_map, ori_usage, pd, rs_summary):
    Ks = rs_summary["Ks"]
    K_max = max(Ks)

    def _top3(df, key_col, model, K):
        _s = df[(df["model"] == model) & (df["K"] == K)]\
            .sort_values("n", ascending=False).head(3)
        return list(zip(_s[key_col], _s["n"]))

    _rows = []
    for _K in [1, K_max]:
        for _model in ori_usage["model"].unique():
            _ori_top = _top3(ori_usage, "ori", _model, _K)
            _amr_top = _top3(amr_usage, "amr", _model, _K)
            _rows.append({
                "model": label_map.get(_model, _model), "K": _K,
                "top_oris": "; ".join(f"{n}×{o}" for o, n in _ori_top),
                "top_amrs": "; ".join(f"{n}×{a}" for a, n in _amr_top),
            })
    usage = pd.DataFrame(_rows)
    return Ks, K_max, usage


@app.cell
def _(usage):
    usage
    return


if __name__ == "__main__":
    app.run()
