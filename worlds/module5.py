"""Module 5's world: the single Python source for its layout, A and B.

The layout (the true kind of every cell, night and day) is read from the
generated `site-src/module-5-data.js`, the same data module 5's widgets draw,
so it is never retyped. Dependency: this parser expects the JSON object that
follows `export const A_FIXED_B_DATA=` in that file, with
`worlds.night` and `worlds.day`, each holding `truth` (25 kind indices) and
`kinds` (their names). Changing that export's name or format breaks Python
verification; `demo()` asserts the expected keys so the break is loud.

Parameters are those of the generator named in the data file's header and
stated in module 5's text: the sensor reports a cell's true kind with 0.8 and
shares the remaining 0.2 equally among the other kinds; a move reaches its
target with 0.8 and otherwise stays; stay is certain; two walls block the
works corner.
"""

import json
import re
from pathlib import Path

import numpy as np

DATA = Path(__file__).resolve().parents[1] / "site-src" / "module-5-data.js"
ROWS = COLS = 5
N = ROWS * COLS
ACTIONS = ["north", "south", "west", "east", "stay"]
MOVES = [(-1, 0), (1, 0), (0, -1), (0, 1), (0, 0)]
ACCURACY = 0.8
RELIABILITY = 0.8
START = 3 * COLS + 1                     # (4, 2) in the text's 1-indexed cells
WALLS = [(0, 5), (6, 11)]                # (1,1)-(2,1) and (2,2)-(3,2)


def load_worlds():
    """{'night': {'truth', 'kinds'}, 'day': {'truth', 'kinds'}}."""
    text = DATA.read_text()
    match = re.search(r"export const A_FIXED_B_DATA\s*=\s*", text)
    if match is None:
        raise ValueError(f"{DATA} no longer exports A_FIXED_B_DATA")
    data, _ = json.JSONDecoder().raw_decode(text[match.end():])
    return data["worlds"]


def observation_model(mode, accuracy=ACCURACY):
    """A[o, s] for 'night' or 'day'."""
    world = load_worlds()[mode]
    truth = np.array(world["truth"])
    K = len(world["kinds"])
    A = np.full((K, N), (1 - accuracy) / (K - 1))
    A[truth, np.arange(N)] = accuracy
    return A


def transition_model(reliability=RELIABILITY):
    """B[s', s, a]: target with `reliability`, otherwise stay; a wall means stay."""
    blocked = {edge for a, b in WALLS for edge in ((a, b), (b, a))}
    B = np.zeros((N, N, len(ACTIONS)))
    for state in range(N):
        row, col = divmod(state, COLS)
        for action, (d_row, d_col) in enumerate(MOVES):
            target = ((row + d_row) % ROWS) * COLS + (col + d_col) % COLS
            if (state, target) in blocked:
                target = state
            B[target, state, action] += reliability
            B[state, state, action] += 1 - reliability
    return B


def demo():
    worlds = load_worlds()
    for mode in ("night", "day"):
        assert set(worlds[mode]) >= {"truth", "kinds"}, worlds[mode].keys()
        assert len(worlds[mode]["truth"]) == N
        A = observation_model(mode)
        assert np.allclose(A.sum(axis=0), 1.0)
    assert worlds["night"]["kinds"] == ["dark", "lamp"]
    B = transition_model()
    assert np.allclose(B.sum(axis=0), 1.0)
    stay, south = ACTIONS.index("stay"), ACTIONS.index("south")
    assert all(B[s, s, stay] == 1.0 for s in range(N))
    assert B[0, 0, south] == 1.0 and B[5, 0, south] == 0.0       # the wall holds
    print("worlds.module5 ok")


if __name__ == "__main__":
    demo()
