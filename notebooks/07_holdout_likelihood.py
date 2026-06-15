"""Held-out continuation + surprisal benchmarks (paper Figs 3 + 4)."""
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
    # Held-out continuation + surprisal

    Two metrics on the 29-plasmid `holdout30_non_addgene` subset, hand-curated
    to be structurally distinct from the Addgene-style training corpus:

    - **Continuation log-probability** — average per-token log-prob over
      100 bp sliding windows after a 400 bp prefix.
    - **Surprisal** — mean log-prob over windows starting at annotated
      promoter / CDS junctions.

    The Base → RL comparison is the relevant generalisation test. RL must
    not lose held-out coverage relative to Base. SFT is shown for context;
    SFT is trained directly on the Addgene corpus and so sets a strong
    predictive bar that RL does not exceed.
    """
    )
    return


@app.cell
def _(data):
    holdout = data.load_csv(
        "continuation_benchmark/holdout30_non_addgene/holdout30_non_addgene.csv")
    return holdout,


@app.cell
def _(holdout):
    cells = ["Base", "SFT", "RL"]
    headline_table = [
        {"Model": c,
         "n": len(holdout),
         "comp_mean": float(holdout[f"comp_{c}"].mean()),
         "comp_std": float(holdout[f"comp_{c}"].std()),
         "comp_sem": float(holdout[f"comp_{c}"].sem()),
         "surp_mean": float(holdout[f"surp_{c}"].mean()),
         "surp_std": float(holdout[f"surp_{c}"].std()),
         "surp_sem": float(holdout[f"surp_{c}"].sem())}
        for c in cells
    ]
    return cells, headline_table


@app.cell
def _(headline_table):
    headline_table
    return


@app.cell
def _(FIGURE_DIR, cells, headline_table, plt, style):
    comp_means = [r["comp_mean"] for r in headline_table]
    surp_means = [r["surp_mean"] for r in headline_table]
    comp_sem = [r["comp_sem"] for r in headline_table]
    surp_sem = [r["surp_sem"] for r in headline_table]

    _fig, _ax = plt.subplots(figsize=(5.5, 4.2))
    _bars = _ax.bar(cells, comp_means, yerr=comp_sem,
                    color=[style.PALETTE[c] for c in cells],
                    edgecolor=style.EDGE, linewidth=1.0, alpha=0.9,
                    capsize=4, ecolor=style.EDGE)
    for _b, _v in zip(_bars, comp_means):
        _ax.text(_b.get_x() + _b.get_width() / 2, _v + 0.1,
                 f"{_v:.2f}", ha="center", va="bottom",
                 fontsize=9, fontweight="bold")
    _ax.set_ylabel("Mean Log-Probability")
    _ax.set_xlabel("Model")
    _ax.set_title("Held-out continuation (29 non-Addgene plasmids)", loc="left")
    _fig.tight_layout()
    _fig.savefig(FIGURE_DIR / "fig_completion_benchmark.pdf")
    _fig.savefig(FIGURE_DIR / "fig_completion_benchmark.png", dpi=200)
    _fig
    return cells, comp_means, comp_sem, surp_means, surp_sem


@app.cell
def _(FIGURE_DIR, cells, plt, style, surp_means, surp_sem):
    _fig, _ax = plt.subplots(figsize=(5.5, 4.2))
    _bars = _ax.bar(cells, surp_means, yerr=surp_sem,
                    color=[style.PALETTE[c] for c in cells],
                    edgecolor=style.EDGE, linewidth=1.0, alpha=0.9,
                    capsize=4, ecolor=style.EDGE)
    for _b, _v in zip(_bars, surp_means):
        _ax.text(_b.get_x() + _b.get_width() / 2, _v + 0.15,
                 f"{_v:.2f}", ha="center", va="bottom",
                 fontsize=9, fontweight="bold")
    _ax.set_ylabel("Mean Log-Probability")
    _ax.set_xlabel("Model")
    _ax.set_title("Held-out CDS-junction surprisal", loc="left")
    _fig.tight_layout()
    _fig.savefig(FIGURE_DIR / "fig_surprisal_benchmark.pdf")
    _fig.savefig(FIGURE_DIR / "fig_surprisal_benchmark.png", dpi=200)
    _fig
    return


@app.cell
def _(mo):
    mo.md(
        r"""
    ## Significance tests — RL vs Base, RL vs SFT

    Paired tests across the 29 plasmids: paired t-test (parametric),
    Wilcoxon signed-rank (non-parametric), Cohen's d (paired effect
    size), and the win count (number of plasmids where RL beats the
    comparison cell). Reported for both metrics (continuation,
    surprisal). Higher log-prob = better.
    """
    )
    return


@app.cell
def _(holdout, np, pd):
    from scipy import stats

    def _paired_test(a, b):
        diff = a - b
        n = len(diff)
        mean = float(diff.mean())
        std = float(diff.std(ddof=1))
        cohens_d = mean / std if std > 0 else float("nan")
        ci95 = 1.96 * std / np.sqrt(n)
        t_stat, t_p = stats.ttest_rel(a, b)
        try:
            w_stat, w_p = stats.wilcoxon(a, b, zero_method="wilcox")
        except ValueError:
            w_stat, w_p = float("nan"), float("nan")
        wins = int((diff > 0).sum())
        return {
            "n": n, "mean_diff": round(mean, 4),
            "ci95": round(float(ci95), 4),
            "t_stat": round(float(t_stat), 4),
            "t_pvalue": float(t_p),
            "wilcoxon_stat": float(w_stat) if w_stat == w_stat else None,
            "wilcoxon_pvalue": float(w_p) if w_p == w_p else None,
            "cohens_d": round(float(cohens_d), 4),
            "wins": wins, "ties": int((diff == 0).sum()),
            "losses": int((diff < 0).sum()),
        }

    rows = []
    for metric, prefix in [("continuation", "comp"), ("surprisal", "surp")]:
        for cmp_a, cmp_b in [("RL", "Base"), ("RL", "SFT")]:
            r = _paired_test(holdout[f"{prefix}_{cmp_a}"].values,
                             holdout[f"{prefix}_{cmp_b}"].values)
            rows.append({"metric": metric, "comparison": f"{cmp_a} − {cmp_b}",
                         **r})
    sig = pd.DataFrame(rows)
    return sig, stats


@app.cell
def _(sig):
    sig
    return


@app.cell
def _(FIGURE_DIR, holdout, np, plt, sig, style):
    sig_lookup = {(r["metric"], r["comparison"]): r for _, r in sig.iterrows()}

    sorted_h = holdout.sort_values("comp_RL_Base").reset_index(drop=True)
    _x = np.arange(len(sorted_h))

    _fig, _axes = plt.subplots(2, 1, figsize=(11, 7), sharex=True)

    _ax = _axes[0]
    _ax.bar(_x - 0.2, sorted_h["comp_RL_Base"], 0.4,
            color=style.PALETTE["RL"], edgecolor=style.EDGE,
            linewidth=0.8, alpha=0.9, label="RL − Base")
    _ax.bar(_x + 0.2, sorted_h["comp_RL_SFT"], 0.4,
            color=style.PALETTE["SFT"], edgecolor=style.EDGE,
            linewidth=0.8, alpha=0.9, label="RL − SFT")
    _ax.axhline(0, color=style.EDGE, linewidth=0.8)
    _ax.set_ylabel("Δ continuation log-prob")
    _r1 = sig_lookup[("continuation", "RL − Base")]
    _r2 = sig_lookup[("continuation", "RL − SFT")]
    _ax.set_title(
        f"Continuation: RL−Base mean={_r1['mean_diff']:+.2f} "
        f"(t={_r1['t_stat']:.2f}, p={_r1['t_pvalue']:.1e}, d={_r1['cohens_d']:+.2f}, "
        f"wins {_r1['wins']}/29)   |   "
        f"RL−SFT mean={_r2['mean_diff']:+.2f} "
        f"(p={_r2['t_pvalue']:.1e}, d={_r2['cohens_d']:+.2f})",
        loc="left", fontsize=9)
    _ax.legend(loc="upper left")

    _surp_diff_rl_base = (holdout["surp_RL"] - holdout["surp_Base"]).values
    _surp_diff_rl_sft = (holdout["surp_RL"] - holdout["surp_SFT"]).values
    _order = np.argsort(_surp_diff_rl_base)
    _ax = _axes[1]
    _ax.bar(_x - 0.2, _surp_diff_rl_base[_order], 0.4,
            color=style.PALETTE["RL"], edgecolor=style.EDGE,
            linewidth=0.8, alpha=0.9, label="RL − Base")
    _ax.bar(_x + 0.2, _surp_diff_rl_sft[_order], 0.4,
            color=style.PALETTE["SFT"], edgecolor=style.EDGE,
            linewidth=0.8, alpha=0.9, label="RL − SFT")
    _ax.axhline(0, color=style.EDGE, linewidth=0.8)
    _ax.set_xticks(_x)
    _ax.set_xticklabels(sorted_h["Plasmid"].values[_order], rotation=80,
                        ha="right", fontsize=7)
    _ax.set_ylabel("Δ surprisal log-prob")
    _r1 = sig_lookup[("surprisal", "RL − Base")]
    _r2 = sig_lookup[("surprisal", "RL − SFT")]
    _ax.set_title(
        f"Surprisal: RL−Base mean={_r1['mean_diff']:+.2f} "
        f"(t={_r1['t_stat']:.2f}, p={_r1['t_pvalue']:.1e}, d={_r1['cohens_d']:+.2f}, "
        f"wins {_r1['wins']}/29)   |   "
        f"RL−SFT mean={_r2['mean_diff']:+.2f} "
        f"(p={_r2['t_pvalue']:.1e}, d={_r2['cohens_d']:+.2f})",
        loc="left", fontsize=9)
    _ax.legend(loc="upper left")

    _fig.tight_layout()
    _fig.savefig(FIGURE_DIR / "fig_holdout29.pdf")
    _fig.savefig(FIGURE_DIR / "fig_holdout29.png", dpi=200)
    _fig
    return


if __name__ == "__main__":
    app.run()
