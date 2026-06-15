"""Distributional alignment grid — length / JS-3mer / MFE density / GC.

Reproduces paper Figure 2 (`fig_distribution_grid.png`): one figure with
four panels showing per-sequence summary statistics for Real plasmids,
Base, SFT, and RL.
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
    # Distributional alignment

    Per-sequence summary statistics for Real plasmids, Base, SFT, and RL,
    laid out as the four panels of paper Fig 2:

    - **A) Sequence length** — post-cleaning length in bp.
    - **B) 3-mer compositional divergence** — Jensen–Shannon divergence
      against the training-corpus 3-mer reference. Lower means more
      Addgene-like 3-mer composition.
    - **C) Thermodynamic stability** — folding free energy density
      (kcal/mol/bp), Mathews 2004 DNA params.
    - **D) GC content** — fraction of bases in {G, C}.

    Real plasmids come from `reference/addgene_500/` (with MFE from
    `mfe/SFT_real/`); each model cell has 4000 sequences from the
    8-prompt evaluation at the per-cell optimal temperature, with MFE
    pulled from `mfe/{cell}/mfe_results.csv`.
    """
    )
    return


@app.cell
def _(data, pd):
    cells = ["Base", "SFT", "RL"]
    per_seq = {}
    for _cell in cells:
        _dist = data.load_csv(f"analysis/distribution/per_seq_{_cell}.csv")
        _mfe = data.load_csv(
            f"mfe/{_cell}/mfe_results.csv")[["id", "mfe_density_dna"]]
        per_seq[_cell] = _dist.merge(_mfe, on="id", how="left")

    addgene_metrics = data.load_csv("reference/addgene_500/metrics.csv")
    addgene_mfe = data.load_csv("mfe/SFT_real/mfe_results.csv")
    real = pd.DataFrame({
        "length": addgene_metrics["length"].values,
        "gc": addgene_metrics["gc"].values,
        "mfe_density_dna": addgene_metrics["mfe_density_dna"].values,
    })
    real_mfe_extended = pd.concat(
        [real["mfe_density_dna"], addgene_mfe["mfe_density_dna"]],
        ignore_index=True).dropna()
    return addgene_metrics, addgene_mfe, cells, per_seq, real, real_mfe_extended


@app.cell
def _(per_seq, real, real_mfe_extended):
    _rows = []
    for _cell, _df in per_seq.items():
        _rows.append({
            "cell": _cell,
            "n": len(_df),
            "length_median": _df["length"].median(),
            "gc_median": _df["gc"].median(),
            "jsd_3mer_median": _df["jsd_3mer"].median(),
            "mfe_density_median": _df["mfe_density_dna"].median(),
        })
    _rows.append({
        "cell": "Real",
        "n": len(real),
        "length_median": real["length"].median(),
        "gc_median": real["gc"].median(),
        "jsd_3mer_median": float("nan"),
        "mfe_density_median": real_mfe_extended.median(),
    })
    distribution_summary = (
        __import__("pandas").DataFrame(_rows)
        .set_index("cell").round(4))
    distribution_summary
    return distribution_summary,


@app.cell
def _(FIGURE_DIR, np, per_seq, plt, real, real_mfe_extended, style):
    order = ["Real", "Base", "SFT", "RL"]
    length_data = [real["length"].dropna().values] + [
        per_seq[c]["length"].dropna().values for c in ["Base", "SFT", "RL"]]
    gc_data = [real["gc"].dropna().values] + [
        per_seq[c]["gc"].dropna().values for c in ["Base", "SFT", "RL"]]
    mfe_data = [real_mfe_extended.values] + [
        per_seq[c]["mfe_density_dna"].dropna().values
        for c in ["Base", "SFT", "RL"]]
    jsd_cells = ["Base", "SFT", "RL"]
    jsd_data = [per_seq[c]["jsd_3mer"].dropna().values for c in jsd_cells]

    real_length_med = float(np.median(length_data[0]))
    real_gc_med = float(np.median(gc_data[0]))
    real_mfe_med = float(np.median(mfe_data[0]))

    _fig, _axes = plt.subplots(2, 2, figsize=(10, 8))

    _ax = _axes[0, 0]
    style.violin(_ax, length_data, order, reference_median=real_length_med,
                 reference_label="Real median")
    _ax.set_yscale("linear")
    _ax.set_ylabel("Length (bp)")
    _ax.set_xlabel("Model")
    _ax.set_title("A) Sequence Length", loc="left")

    _ax = _axes[0, 1]
    style.violin(_ax, jsd_data, jsd_cells)
    _ax.set_ylabel("JS Divergence (3-mer)")
    _ax.set_xlabel("Model")
    _ax.set_title("B) 3-mer Compositional Divergence", loc="left")

    _ax = _axes[1, 0]
    style.violin(_ax, mfe_data, order, reference_median=real_mfe_med,
                 reference_label="Real median")
    _ax.set_ylabel("MFE Density (kcal/mol/bp)")
    _ax.set_xlabel("Model")
    _ax.set_title("C) Thermodynamic Stability", loc="left")

    _ax = _axes[1, 1]
    style.violin(_ax, gc_data, order, reference_median=real_gc_med,
                 reference_label="Real median")
    _ax.set_ylabel("GC Content")
    _ax.set_xlabel("Model")
    _ax.set_title("D) GC Content", loc="left")

    _fig.tight_layout()
    _fig.savefig(FIGURE_DIR / "fig_distribution_grid.pdf")
    _fig.savefig(FIGURE_DIR / "fig_distribution_grid.png", dpi=200)
    _fig
    return


if __name__ == "__main__":
    app.run()
