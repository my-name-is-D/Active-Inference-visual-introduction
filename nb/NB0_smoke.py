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
        # Deployment smoke test

        This page exists to prove the build works, not to teach anything. It
        checks four things that the rest of the tutorial depends on, and each
        section says plainly whether it passed.

        If you are a reader who arrived here by accident, the tutorial itself
        is linked from the [landing page](../index.html).
        """
    )
    return


@app.cell
def _(mo):
    mo.md(
        """
        ## 1. Does a local module load in the browser?

        Everything in the tutorial lives in a local package rather than inside
        the notebooks. That package has to be bundled into a wheel and
        installed into the browser's Python before this cell can run.
        """
    )
    return


@app.cell
def _():
    import numpy as np

    from smoke.hello import column_sums, transition_matrix

    B = transition_matrix(4, 0.7)
    B
    return B, column_sums, np, transition_matrix


@app.cell
def _(B, column_sums, mo):
    _sums = column_sums(B)
    _ok = bool(abs(_sums - 1.0).max() < 1e-12)
    mo.md(
        f"""
        Column sums: `{_sums}`

        **Local module import: {"PASS" if _ok else "FAIL"}**
        """
    )
    return


@app.cell
def _(mo):
    mo.md(
        """
        ## 2. Does matplotlib draw?

        Every argument in this tutorial is carried by a figure, so a
        matplotlib failure in the browser would be fatal.
        """
    )
    return


@app.cell
def _(B):
    import matplotlib.pyplot as plt

    _fig, _ax = plt.subplots(figsize=(3.2, 3.0))
    _im = _ax.imshow(B, cmap="viridis", vmin=0.0, vmax=1.0)
    for _i in range(B.shape[0]):
        for _j in range(B.shape[1]):
            _ax.text(
                _j,
                _i,
                f"{B[_i, _j]:.2f}",
                ha="center",
                va="center",
                color="white" if B[_i, _j] < 0.5 else "black",
                fontsize=8,
            )
    _ax.set_xlabel("from state")
    _ax.set_ylabel("to state")
    _ax.set_title("B, with numbers printed")
    _fig.colorbar(_im, ax=_ax, shrink=0.8)
    _fig
    return (plt,)


@app.cell
def _(mo):
    mo.md(
        """
        The numbers are printed on the heatmap on purpose. No claim in this
        tutorial should rest on colour alone, because colour fails for a
        colourblind reader and in print.

        ## 3. Do the interaction idioms survive the export?

        Three are tested. The tutorial's design depends on knowing which work.
        """
    )
    return


@app.cell
def _(mo):
    stay = mo.ui.slider(
        start=0.1,
        stop=0.95,
        step=0.05,
        value=0.7,
        label="stay probability",
        show_value=True,
    )
    stay
    return (stay,)


@app.cell
def _(np, plt, stay, transition_matrix):
    _B = transition_matrix(4, stay.value)
    _fig2, _ax2 = plt.subplots(figsize=(3.2, 3.0))
    _ax2.imshow(_B, cmap="viridis", vmin=0.0, vmax=1.0)
    _ax2.set_title(f"stay probability = {stay.value:.2f}")
    _ax2.set_xlabel("from state")
    _ax2.set_ylabel("to state")
    _fig2
    return


@app.cell
def _(mo):
    mo.md(
        """
        **Idiom 1, slider:** if the figure above changed when you moved the
        slider, reactive widgets work.

        Next, whether the reader can pick a tile on the grid itself rather
        than only move a slider. Drag a box over one of the squares below.
        """
    )
    return


@app.cell
def _(mo, np, plt):
    # A plain figure renders as a static image and cannot be clicked. Wrapping
    # the Axes in mo.ui.matplotlib renders it to a canvas and returns the
    # dragged region in data coordinates, which is how the reader moves the
    # goal in the transfer test later on.
    _grid = np.zeros((5, 5))
    _grid[3, 1] = 1.0

    _fig3, _ax3 = plt.subplots(figsize=(3.5, 3.5))
    _ax3.imshow(
        _grid, cmap="viridis", vmin=0, vmax=1, extent=(-0.5, 4.5, 4.5, -0.5)
    )
    _ax3.set_xticks(range(5))
    _ax3.set_yticks(range(5))
    _ax3.set_title("drag a box over a tile")
    picker = mo.ui.matplotlib(_ax3)
    picker
    return (picker,)


@app.cell
def _(mo, picker):
    _selection = picker.value
    if _selection is None:
        _msg = "Nothing picked yet. Drag a box over a square above."
    else:
        _col = round((_selection.x_min + _selection.x_max) / 2)
        _row = round((_selection.y_min + _selection.y_max) / 2)
        _msg = f"You picked row {_row}, column {_col}."

    mo.md(
        f"""
        **Idiom 2, picking a tile:** {_msg}

        The selection is a dragged box rather than a single point, so the tile
        is recovered by rounding the midpoint. That is enough to move a goal
        or place a lamp.
        """
    )
    return


@app.cell
def _(mo):
    depth = mo.accordion(
        {
            "Why this is behind a reveal (click to open)": mo.md(
                """
                This is the third idiom, and the one the tutorial's structure
                most depends on. Depth that a stronger reader wants lives in
                sections like this one, so that the main path stays readable
                for someone meeting the material for the first time.

                If you can read this sentence, `mo.accordion` survives the
                export and the plan's approach to handling depth is sound.
                """
            )
        }
    )
    depth
    return (depth,)


@app.cell
def _(mo):
    mo.md(
        """
        ## 4. How long did this take to load?

        The first visit to any page here downloads a Python runtime, numpy and
        matplotlib. That is tens of megabytes. Time it from a cold cache and
        record the number, because it is the reader's actual first experience.
        """
    )
    return


@app.cell
def _(mo, np):
    import sys

    mo.md(
        f"""
        Running Python `{sys.version.split()[0]}`, numpy `{np.__version__}`.

        Platform is `{sys.platform}`, which reads `emscripten` in the browser
        and something else when run locally.
        """
    )
    return (sys,)


if __name__ == "__main__":
    app.run()
