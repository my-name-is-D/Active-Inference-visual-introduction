"""Drawing the world and the agent's state.

This is pedagogical apparatus rather than a utility module: the figures are
the argument, not decoration for it. Two rules hold throughout.

Numbers are printed on anything that carries a claim, so that no argument
depends on colour alone. Colour fails for a colourblind reader and in print.

Colourmaps are perceptually uniform (`viridis`, `cividis`), never `jet`.
"""

import matplotlib.pyplot as plt
import numpy as np

# Tile (r, c) covers x in [c, c+1] and y in [r, r+1]. Ticks sit at tile
# centres, because a tick on a boundary labels the gap between two tiles
# rather than either of them.
DARK = "#2a2a35"
LAMP = "#f2c14e"
AGENT = "#e8563f"
GOAL = "#3fb27f"
TRAIL = "#8f8fa3"


def _grid_axes(rows, cols, title=None, figsize=None):
    """An axes with tile (r, c) occupying the unit square at (c, r)."""
    figsize = figsize or (0.7 * cols + 1.0, 0.7 * rows + 1.0)
    fig, ax = plt.subplots(figsize=figsize)
    ax.set_xlim(0, cols)
    ax.set_ylim(rows, 0)
    ax.set_xticks([c + 0.5 for c in range(cols)], labels=range(cols))
    ax.set_yticks([r + 0.5 for r in range(rows)], labels=range(rows))
    ax.set_xticks(range(cols + 1), minor=True)
    ax.set_yticks(range(rows + 1), minor=True)
    ax.grid(which="minor", color="white", linewidth=1.2)
    ax.set_aspect("equal")
    ax.set_xlabel("column")
    ax.set_ylabel("row")
    if title:
        ax.set_title(title)
    return fig, ax


def number_tiles(ax, rows, cols, colour="white", fontsize=7):
    """Write the state index in the corner of every tile.

    This is what lets a bar chart over states be read against the grid: the
    reader traces bar 7 back to the tile labelled 7, so the one-dimensional
    plot and the two-dimensional world are visibly the same thing.
    """
    for row in range(rows):
        for col in range(cols):
            ax.text(
                col + 0.08, row + 0.22, str(row * cols + col),
                fontsize=fontsize, color=colour, alpha=0.85, zorder=5,
            )


def draw_world(
    world, position=None, trajectory=None, title=None, show_dark=True, numbered=False
):
    """Draw the grid, the goal, the agent and optionally where it has been.

    Args:
        world: a GridWorld.
        position: (row, col) to draw the agent at, defaulting to where it is.
            Passing this explicitly is how a lesson scrubs through a run.
        trajectory: list of positions to draw as a trail. Pass
            `world.trajectory[:step]` to show the run so far.
        show_dark: shade unlit tiles. Off for a fully lit world, where the
            shading would carry no information.

    Returns the figure and axes.
    """
    position = world.position if position is None else position
    fig, ax = _grid_axes(world.rows, world.cols, title)

    # Unlit tiles first, so everything else sits on top of them.
    if show_dark and not world.lit.all():
        for row in range(world.rows):
            for col in range(world.cols):
                colour = LAMP if world.lit[row, col] else DARK
                alpha = 0.35 if world.lit[row, col] else 0.85
                ax.add_patch(
                    plt.Rectangle((col, row), 1, 1, facecolor=colour, alpha=alpha)
                )

    if trajectory:
        # Offset to tile centres so the line runs through the squares.
        xs = [c + 0.5 for _, c in trajectory]
        ys = [r + 0.5 for r, _ in trajectory]
        ax.plot(xs, ys, color=TRAIL, linewidth=2, alpha=0.8, zorder=2)

    goal_row, goal_col = world.goal
    ax.add_patch(
        plt.Rectangle(
            (goal_col, goal_row), 1, 1,
            facecolor=GOAL, alpha=0.75, zorder=1,
        )
    )
    ax.text(
        goal_col + 0.5, goal_row + 0.5, "goal",
        ha="center", va="center", fontsize=8, zorder=3,
    )

    if numbered:
        number_tiles(ax, world.rows, world.cols, colour="#555560")

    row, col = position
    ax.plot(
        col + 0.5, row + 0.5,
        marker="o", markersize=14, color=AGENT, zorder=4,
    )

    return fig, ax


def draw_belief_pair(belief, rows, cols, position=None, title=None, highlight=None):
    """The same belief twice: over the grid, and as a bar per tile.

    The grid says where the mass is; the bars say how much, and are easier to
    compare by eye than shades of colour. Tiles are numbered so the two
    pictures can be read against each other, since bar `n` is the tile
    labelled `n`.

    `position` marks where the agent actually is, which is not something the
    agent knows. `highlight` marks one state to draw attention to.
    """
    belief = np.asarray(belief, dtype=float)
    grid = belief.reshape(rows, cols)

    fig, (ax_grid, ax_bars) = plt.subplots(
        1, 2, figsize=(0.62 * cols + 6.2, 0.62 * rows + 1.4),
        gridspec_kw={"width_ratios": [1, 1.6]},
    )

    image = ax_grid.imshow(
        grid, cmap="viridis", vmin=0.0, vmax=max(belief.max(), 1e-12),
        extent=(0, cols, rows, 0),
    )
    ax_grid.set_xlim(0, cols)
    ax_grid.set_ylim(rows, 0)
    ax_grid.set_xticks([c + 0.5 for c in range(cols)], labels=range(cols))
    ax_grid.set_yticks([r + 0.5 for r in range(rows)], labels=range(rows))
    ax_grid.set_xticks(range(cols + 1), minor=True)
    ax_grid.set_yticks(range(rows + 1), minor=True)
    ax_grid.grid(which="minor", color="white", linewidth=1.0)
    ax_grid.set_aspect("equal")
    ax_grid.set_xlabel("column")
    ax_grid.set_ylabel("row")
    ax_grid.set_title("belief on the grid")
    number_tiles(ax_grid, rows, cols)

    if position is not None:
        row, col = position
        ax_grid.plot(
            col + 0.5, row + 0.5,
            marker="o", markersize=9, markerfacecolor="none",
            markeredgecolor=AGENT, markeredgewidth=2.5, zorder=6,
        )

    colours = ["#4c72b0"] * len(belief)
    if highlight is not None:
        colours[highlight] = AGENT
    ax_bars.bar(range(len(belief)), belief, color=colours)
    ax_bars.set_xlabel("tile number")
    ax_bars.set_ylabel("probability")
    ax_bars.set_ylim(0, 1.0)
    ax_bars.set_xticks(range(0, len(belief), max(1, len(belief) // 12)))
    ax_bars.set_title("the same belief, as a distribution")
    # The probabilities sum to one, and saying so on the figure saves the
    # reader wondering.
    ax_bars.text(
        0.98, 0.92, f"sums to {belief.sum():.2f}",
        transform=ax_bars.transAxes, ha="right", fontsize=8, color="#555560",
    )

    if title:
        fig.suptitle(title)
    fig.tight_layout()
    return fig, (ax_grid, ax_bars)


def draw_belief(belief, rows, cols, title=None, position=None, annotate=True):
    """Draw a distribution over positions as a heatmap with its numbers on it.

    `belief` is a flat array of length rows*cols, indexed the way
    `GridWorld.state_index` indexes it.
    """
    grid = np.asarray(belief, dtype=float).reshape(rows, cols)
    fig, ax = _grid_axes(rows, cols, title)

    image = ax.imshow(
        grid, cmap="viridis", vmin=0.0, vmax=max(grid.max(), 1e-12),
        extent=(0, cols, rows, 0),
    )

    if annotate:
        # Printed numbers mean the claim never rests on colour alone.
        for row in range(rows):
            for col in range(cols):
                value = grid[row, col]
                ax.text(
                    col + 0.5, row + 0.5, f"{value:.2f}",
                    ha="center", va="center", fontsize=7,
                    color="white" if value < 0.5 * grid.max() else "black",
                )

    if position is not None:
        row, col = position
        ax.plot(
            col + 0.5, row + 0.5,
            marker="o", markersize=10, markerfacecolor="none",
            markeredgecolor=AGENT, markeredgewidth=2.5,
        )

    fig.colorbar(image, ax=ax, shrink=0.8)
    return fig, ax


def demo():
    import sys

    sys.path.insert(0, ".")
    from worlds.gridworld import GridWorld, dark_with_lamps

    world = GridWorld(size=(4, 5), goal=(3, 4))
    fig, ax = draw_world(world, trajectory=[(0, 0), (0, 1), (1, 1)])
    assert ax.get_xlim() == (0, 5)
    # y is inverted so row 0 is at the top, as in a matrix.
    assert ax.get_ylim() == (4, 0)
    plt.close(fig)

    dark = dark_with_lamps(size=(4, 4), lamps=[(1, 1)], goal=(3, 3))
    fig, ax = draw_world(dark)
    plt.close(fig)

    belief = np.zeros(20)
    belief[7] = 1.0
    fig, ax = draw_belief(belief, 4, 5, position=(1, 2))
    plt.close(fig)

    # The paired view: grid and bars, with the tiles numbered so the two can
    # be read against one another.
    spread = np.full(20, 1.0 / 20)
    fig, (ax_grid, ax_bars) = draw_belief_pair(
        spread, 4, 5, position=(1, 2), highlight=7
    )
    assert ax_bars.get_ylim() == (0.0, 1.0)
    assert len(ax_bars.patches) == 20
    plt.close(fig)

    fig, ax = draw_world(GridWorld(size=(3, 3)), numbered=True)
    plt.close(fig)

    print("rendering.plots ok")


if __name__ == "__main__":
    demo()
