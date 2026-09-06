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
        than only move a slider. Click a square below.
        """
    )
    return


@app.cell
def _(mo, np, plt):
    # A plain figure renders as a static image and cannot be clicked. Wrapping
    # the Axes in mo.ui.matplotlib returns the clicked region in data
    # coordinates, which is how the reader moves the goal later on.
    #
    # debounce=True sends the selection on mouse-up only. Without it the value
    # streams during the interaction and passes through the empty selection,
    # which makes the output below flicker away mid-click.
    def tile_grid(row, col):
        """Draw a 5x5 grid with one tile lit, and return the pickable Axes.

        Tile (r, c) covers x in [c, c+1] and y in [r, r+1], so a coordinate
        maps to a tile by taking its floor. Ticks sit at the tile centres,
        because a tick on the boundary labels the gap between two tiles
        rather than either of them.
        """
        grid = np.zeros((5, 5))
        grid[row, col] = 1.0

        fig, ax = plt.subplots(figsize=(3.5, 3.5))
        ax.imshow(grid, cmap="viridis", vmin=0, vmax=1, extent=(0, 5, 5, 0))
        ax.set_xticks([c + 0.5 for c in range(5)], labels=range(5))
        ax.set_yticks([r + 0.5 for r in range(5)], labels=range(5))
        ax.set_xticks(range(6), minor=True)
        ax.set_yticks(range(6), minor=True)
        ax.grid(which="minor", color="white", linewidth=0.8)
        ax.set_xlabel("column")
        ax.set_ylabel("row")
        return fig, ax

    _fig3, _ax3 = tile_grid(3, 1)
    _ax3.set_title("click a tile")
    picker = mo.ui.matplotlib(_ax3, debounce=True)
    picker
    return picker, tile_grid


@app.cell
def _(mo, picker, plt, tile_grid):
    def picked_tile(selection):
        """Return (row, col) for a selection, or None if there is none.

        An empty selection is falsy rather than None, so it is tested with a
        plain truth check. A click gives a box of zero or near-zero area, so
        the midpoint is the clicked point; a shift-drag gives a lasso instead,
        whose vertices are averaged. Taking the floor maps the point to the
        tile containing it, clamped in case the click lands on the outer edge.
        """
        if not selection:
            return None
        if hasattr(selection, "vertices"):
            xs = [v[0] for v in selection.vertices]
            ys = [v[1] for v in selection.vertices]
            x, y = sum(xs) / len(xs), sum(ys) / len(ys)
        else:
            x = (selection.x_min + selection.x_max) / 2
            y = (selection.y_min + selection.y_max) / 2
        col = min(4, max(0, int(x)))
        row = min(4, max(0, int(y)))
        return row, col

    _tile = picked_tile(picker.value)

    if _tile is None:
        _out = mo.md("**Idiom 2, picking a tile:** nothing picked yet.")
    else:
        _row, _col = _tile
        _fig4, _ax4 = tile_grid(_row, _col)
        _ax4.set_title(f"you picked row {_row}, column {_col}")
        _out = mo.vstack(
            [
                mo.md(
                    f"**Idiom 2, picking a tile:** row {_row}, column {_col}. "
                    "The lit tile below follows your click."
                ),
                _fig4,
            ]
        )
    _out
    return (picked_tile,)


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
