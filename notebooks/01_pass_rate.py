"""QC pass rate across cells, ablations, and prompts."""
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
    # QC pass rate

    A generated sequence *passes QC* iff:

    - exactly one origin of replication is detected by BLAST against the
      curated oriDB at ≥99% identity and ≥99% subject coverage,
    - at least one antibiotic resistance gene is called by AMRFinderPlus in
      nucleotide mode at 100% identity and coverage,
    - no exact direct repeat ≥50 bp is present.

    Each cell is generated at the per-cell optimal temperature (Base 1.0,
    SFT 1.0, RL 1.15), 500 samples × 8 prompts = 4000 sequences.
    """
    )
    return


@app.cell
def _(data, pd):
    summaries = {label: data.load_json(f"evaluation/eight_prompt/{label}/summary.json")
                 for label in ("Base", "SFT", "RL")}
    headline = pd.DataFrame([
        {"Model": label,
         "n": s["n_sequences"],
         "passed": s["n_passed"],
         "pass_rate_pct": round(100 * s["n_passed"] / s["n_sequences"], 2)}
        for label, s in summaries.items()
    ])
    return headline, summaries


@app.cell
def _(headline):
    headline
    return


@app.cell
def _(FIGURE_DIR, headline, plt, style):
    _fig, _ax = plt.subplots(figsize=(5, 4))
    style.model_bar(_ax, headline["Model"], headline["pass_rate_pct"],
                    labels=[f"{p:.1f}%" for p in headline["pass_rate_pct"]])
    _ax.set_ylabel("QC Pass Rate (%)")
    _ax.set_xlabel("Model")
    _ax.set_ylim(0, 100)
    _ax.set_title("QC pass rate, n=4000 per model", loc="left")
    _fig.tight_layout()
    _fig.savefig(FIGURE_DIR / "fig_qc_pass_rate.pdf")
    _fig.savefig(FIGURE_DIR / "fig_qc_pass_rate.png", dpi=200)
    _fig
    return


@app.cell
def _(mo):
    mo.md(
        r"""
    ## Per-prompt breakdown across cells

    Pass rate varies materially across the eight evaluation prompts —
    structural prompts (pUC/p15A ORI seeds) are easiest, generic seeds
    (single ATG, short random) are hardest.
    """
    )
    return


@app.cell
def _(data, pd):
    cells = ["Base", "SFT", "RL"]
    _rows = []
    for _cell in cells:
        _outs = data.load_csv(f"evaluation/eight_prompt/{_cell}/outputs.csv")
        _passed = data.load_csv(
            f"evaluation/eight_prompt/{_cell}/qc/passed.csv")
        if "prompt_name" not in _outs.columns:
            _sft_out = data.load_csv("evaluation/eight_prompt/SFT/outputs.csv")
            _lbl = (_sft_out.drop_duplicates("prompt")
                    .set_index("prompt")["prompt_name"].to_dict())
            _outs["prompt_name"] = _outs["prompt"].map(_lbl)
        _passed_ids = set(_passed["Plasmid_ID"])
        _outs["passed"] = _outs["id"].isin(_passed_ids)
        _per_prompt = (_outs.groupby("prompt_name")
                       .agg(n=("id", "size"), passed=("passed", "sum"))
                       .reset_index())
        _per_prompt["pass_rate_pct"] = (100 * _per_prompt["passed"]
                                         / _per_prompt["n"]).round(2)
        _per_prompt["cell"] = _cell
        _rows.append(_per_prompt)
    per_prompt_long = pd.concat(_rows, ignore_index=True)
    pivot = per_prompt_long.pivot(index="prompt_name", columns="cell",
                                  values="pass_rate_pct")[cells]
    pivot = pivot.sort_values("RL", ascending=False)
    return per_prompt_long, pivot


@app.cell
def _(pivot):
    pivot
    return


@app.cell
def _(FIGURE_DIR, np, pivot, plt, style):
    _prompts = list(pivot.index)
    _x = np.arange(len(_prompts))
    _w = 0.27

    _fig, _ax = plt.subplots(figsize=(10, 4.5))
    for _i, _cell in enumerate(["Base", "SFT", "RL"]):
        _ax.bar(_x + (_i - 1) * _w, pivot[_cell].values, _w,
                color=style.PALETTE[_cell], edgecolor=style.EDGE,
                linewidth=1.0, alpha=0.9, label=_cell)
    _ax.set_xticks(_x)
    _ax.set_xticklabels(_prompts, rotation=20, ha="right")
    _ax.set_ylabel("QC Pass Rate (%)")
    _ax.set_xlabel("Prompt")
    _ax.set_ylim(0, 105)
    _ax.set_title("Pass rate by prompt", loc="left")
    _ax.legend(title="Model", loc="upper right")
    _fig.tight_layout()
    _fig.savefig(FIGURE_DIR / "fig_pass_rate_by_prompt.pdf")
    _fig.savefig(FIGURE_DIR / "fig_pass_rate_by_prompt.png", dpi=200)
    _fig
    return


if __name__ == "__main__":
    app.run()
