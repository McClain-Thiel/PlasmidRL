"""Render every paper figure to ``PLASMIDRL_FIGURE_DIR`` in one pass."""
import marimo

__generated_with = "0.23.1"
app = marimo.App(width="medium")


@app.cell
def _():
    import importlib.util
    import os
    from pathlib import Path

    import marimo as mo

    NOTEBOOK_DIR = Path(__file__).resolve().parent
    FIGURE_DIR = Path(os.environ.get(
        "PLASMIDRL_FIGURE_DIR", NOTEBOOK_DIR.parent / "paper" / "figures"))
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    return FIGURE_DIR, NOTEBOOK_DIR, importlib, mo, os


@app.cell
def _(mo):
    mo.md(
        r"""
    # Render all paper figures

    Runs notebooks 01–07 in order. Each notebook saves its figures to
    `PLASMIDRL_FIGURE_DIR` (default `paper/figures/`). Override with the env
    var to write elsewhere:

    ```
    PLASMIDRL_FIGURE_DIR=/tmp/figs uv run marimo run notebooks/09_figures.py
    ```

    Notebook 08 (conditional fidelity / 3×3 logprob matrix) is intentionally
    not part of the public release — its rare-ORI prompt artefacts are not
    yet mirrored to `UCL-CSSB/PlasmidRL-ICML`.
    """
    )
    return


@app.cell
def _(FIGURE_DIR, NOTEBOOK_DIR, importlib):
    targets = [
        "01_pass_rate",
        "02_distributions",
        "04_temperature_sweep",
        "05_ablations",
        "06_rejection_sampling",
        "07_holdout_likelihood",
    ]
    rendered = []
    for _nb_name in targets:
        _path = NOTEBOOK_DIR / f"{_nb_name}.py"
        _spec = importlib.util.spec_from_file_location(_nb_name, _path)
        _module = importlib.util.module_from_spec(_spec)
        _spec.loader.exec_module(_module)
        _module.app.run()
        rendered.append(_nb_name)
    return rendered, targets


@app.cell
def _(FIGURE_DIR, rendered):
    written = sorted(p.name for p in FIGURE_DIR.iterdir())
    print(f"Rendered to {FIGURE_DIR} ({len(rendered)} notebooks, {len(written)} files):")
    for _f in written:
        print(" ", _f)
    return written,


if __name__ == "__main__":
    app.run()
