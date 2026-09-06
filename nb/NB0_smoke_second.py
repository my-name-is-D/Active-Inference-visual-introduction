import marimo

__generated_with = "0.10.0"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo
    return (mo,)


@app.cell
def _(mo):
    mo.md(
        """
        # Second page, for load-time measurement

        This page exists only to answer one question: having loaded the first
        page, how long does a second page take? If the Python runtime and the
        wheels are shared across pages, this should be noticeably faster. If
        it takes as long as the first, every notebook pays the full download
        and the tutorial needs fewer, longer pages.

        Open this page immediately after the smoke test page and time it.
        """
    )
    return


@app.cell
def _():
    import numpy as np

    from smoke.hello import transition_matrix

    B = transition_matrix(3, 0.5)
    B
    return B, np, transition_matrix


@app.cell
def _(B, mo, np):
    mo.md(
        f"""
        Local module imported again on a separate page. Column sums
        `{B.sum(axis=0)}`, numpy `{np.__version__}`.

        **Second-page local import: {"PASS" if abs(B.sum(axis=0) - 1.0).max() < 1e-12 else "FAIL"}**
        """
    )
    return


if __name__ == "__main__":
    app.run()
