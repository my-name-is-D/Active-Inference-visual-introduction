"""The four arrays an agent's model of the world is made of.

    A[observation, state]          what it expects to see in each state
    B[next_state, state, action]   how states change when it acts
    C[observation]                 preference probabilities (logged when scoring)
    D[state]                       where it believes it starts

That is the whole generative model. It is not learned and it is not deep: on
this grid it is written down by hand, and a lesson can print every entry.

Both `A` and `B` are column-stochastic, meaning each column is a distribution
and therefore sums to one. `A[:, s]` is "given I am in state s, what might I
see", and `B[:, s, a]` is "given I am in state s and do a, where do I end up".
The validators here exist because a column that does not sum to one is the
commonest way to get silently wrong answers later.
"""

import numpy as np


def check_columns_sum_to_one(array, name, tolerance=1e-10):
    """Raise unless every column of `array` is a distribution.

    Works for `A` of shape (observations, states) and `B` of shape
    (next_states, states, actions), summing over the first axis in both cases.
    """
    sums = array.sum(axis=0)
    if not np.allclose(sums, 1.0, atol=tolerance):
        worst = np.unravel_index(np.argmax(np.abs(sums - 1.0)), sums.shape)
        raise ValueError(
            f"{name} has a column that does not sum to one: "
            f"column {worst} sums to {sums[worst]:.6f}"
        )
    if (array < 0.0).any():
        raise ValueError(f"{name} contains a negative probability")


def observation_model(world):
    """Build `A` for a gridworld.

    On a lit cell the agent sees its own position, with `sensor_noise` of the
    probability spread over the cells it could be confused with. Unlit cells
    all produce the same observation, the last index, which carries no
    information about position: that single shared "dark" outcome is what
    makes the dark world hard.

    Returns an array of shape (n_states + 1, n_states).
    """
    n_states = world.n_states
    dark = n_states
    A = np.zeros((n_states + 1, n_states))

    for state in range(n_states):
        position = world.state_position(state)

        if not world.lit[position]:
            A[dark, state] = 1.0
            continue

        neighbours = []
        for action in range(5):
            landing = world.move(position, action)
            if landing != position:
                neighbours.append(world.state_index(landing))

        # Noise is per cell: a cell can be dim rather than merely lit or dark.
        noise = float(world.sensor_noise[position])
        if neighbours and noise > 0.0:
            A[state, state] = 1.0 - noise
            for neighbour in neighbours:
                A[neighbour, state] += noise / len(neighbours)
        else:
            A[state, state] = 1.0

    check_columns_sum_to_one(A, "A")
    return A


def kind_observation_model(world, p_correct=0.9, mark_goal=False):
    """Build `A` where the agent senses the *kind* of cell it stands on.

    The other model here, `observation_model`, gives a lit cell its own
    observation; with a perfect sensor it resolves position outright. This one reports
    only a kind, so cells of the same kind are indistinguishable. Two lamps
    look alike: a lit reading favours lamp cells without distinguishing them.
    With sensor noise, that reading is not conclusive about being at a lamp.

    The kinds, and therefore the number of rows:

        mark_goal=False   2 rows: DARK=0, LIT=1
        mark_goal=True    3 rows: DARK=0, LIT=1, MARKED=2

    `mark_goal` gives the goal cell a kind of its own, which is what a lesson
    needs when a preference over *observations* has to be able to single the
    goal out: a goal that reads as plain dark cannot be preferred, because
    every dark cell would satisfy the preference equally.

    `p_correct` is how often the sensor reports the true kind; the rest is
    split evenly over the other kinds. At `p_correct=1` the sensor is perfect
    and a reading is conclusive about the kind. It identifies a cell only
    when that kind belongs to a unique cell, such as the marked goal.

    Returns shape (2, n_states) or (3, n_states).
    """
    DARK, LIT, MARKED = 0, 1, 2
    n_kinds = 3 if mark_goal else 2
    n_states = world.n_states
    A = np.zeros((n_kinds, n_states))

    for state in range(n_states):
        position = world.state_position(state)
        if mark_goal and position == world.goal:
            true_kind = MARKED
        elif world.lit[position]:
            true_kind = LIT
        else:
            true_kind = DARK
        for kind in range(n_kinds):
            if kind == true_kind:
                A[kind, state] = p_correct
            else:
                A[kind, state] = (1.0 - p_correct) / (n_kinds - 1)

    check_columns_sum_to_one(A, "A")
    return A


def transition_model(world):
    """Build `B` for a gridworld.

    `B[:, s, a]` is where the agent ends up doing `a` from `s`. With `slip`
    greater than zero some of that mass goes where a failed step could leave
    the agent instead: the cell it started in, for a step that did not carry,
    and the cells beside the intended one, for a step that drifted. See
    `GridWorld.slip_destinations`, which the simulator uses for the same
    purpose, so the agent's model and the world agree on how a step fails.

    Returns an array of shape (n_states, n_states, n_actions).
    """
    n_actions = 5
    n_states = world.n_states
    B = np.zeros((n_states, n_states, n_actions))

    for state in range(n_states):
        position = world.state_position(state)

        for action in range(n_actions):
            intended = world.state_index(world.move(position, action))

            # Accumulate rather than assign: the destinations collide. At a
            # wall the intended cell is the origin, so one cell can take the
            # reliability AND a share of the spread, and it must keep both.
            B[intended, state, action] += 1.0 - world.slip

            if world.slip > 0.0:
                destinations = world.slip_destinations(position, action)
                share = world.slip / len(destinations)
                for landing in destinations:
                    B[world.state_index(landing), state, action] += share

    check_columns_sum_to_one(B, "B")
    return B


def starting_belief(world, known=True):
    """Build `D`, where the agent believes it begins.

    `known=True` means it starts certain; `known=False` means it has no idea,
    which is the interesting case once the world goes dark.
    """
    D = np.zeros(world.n_states)
    if known:
        D[world.state_index(world.start)] = 1.0
    else:
        D[:] = 1.0 / world.n_states
    return D


def demo():
    import sys

    sys.path.insert(0, ".")
    from worlds.gridworld import GridWorld, dark_with_lamps

    world = GridWorld(size=(3, 3), goal=(2, 2))
    A = observation_model(world)
    B = transition_model(world)

    # A perfect sensor on a lit grid means seeing state s exactly in state s.
    assert A.shape == (10, 9)
    assert np.allclose(np.diag(A[:9, :9]), 1.0)
    check_columns_sum_to_one(A, "A")

    # Deterministic movement: one destination per (state, action).
    assert B.shape == (9, 9, 5)
    assert np.allclose(B.sum(axis=0), 1.0)
    assert B[1, 0, 3] == 1.0     # right from cell 0 lands on cell 1
    assert B[6, 0, 0] == 1.0     # up from cell 0 wraps to the bottom row

    # B is doubly stochastic: columns sum to one because the agent ends up
    # somewhere, and ROWS sum to one because the grid wraps, so every cell is
    # reached by exactly as much as it sends out. The row property is what
    # makes a prediction step unable to concentrate a belief, which is the
    # claim lesson 2 makes when it says predicting can only ever widen one.
    assert np.allclose(B.sum(axis=1), 1.0)

    # Every dark cell produces the same observation, so seeing it tells the
    # agent nothing about which cell it is on.
    dark = dark_with_lamps(size=(3, 3), lamps=[(1, 1)], goal=(2, 2))
    A_dark = observation_model(dark)
    dark_index = dark.n_states
    dark_columns = [s for s in range(dark.n_states) if A_dark[dark_index, s] == 1.0]
    assert len(dark_columns) == 8
    assert A_dark[4, 4] == 1.0   # the lit cell still reports itself

    # kind_observation_model: the agent senses a KIND, so cells of the same
    # kind are indistinguishable. Two rows without a marked goal, three with.
    lamps2 = dark_with_lamps(size=(5, 5), lamps=[(1, 4), (2, 3)], goal=(0, 4))
    A_kind2 = kind_observation_model(lamps2)
    assert A_kind2.shape == (2, 25)
    check_columns_sum_to_one(A_kind2, "A_kind2")

    A_kind3 = kind_observation_model(lamps2, mark_goal=True)
    assert A_kind3.shape == (3, 25)
    check_columns_sum_to_one(A_kind3, "A_kind3")

    # The goal reads as MARKED only when asked for; otherwise it is just dark.
    goal_state = lamps2.state_index(lamps2.goal)
    assert np.isclose(A_kind3[2, goal_state], 0.9)
    assert np.isclose(A_kind2[0, goal_state], 0.9)

    # The two lamps are genuinely identical: same column, so a reading cannot
    # tell them apart. This is what lesson 4's ambiguity rests on.
    lamp_states = [lamps2.state_index(p) for p in [(1, 4), (2, 3)]]
    assert np.allclose(A_kind3[:, lamp_states[0]], A_kind3[:, lamp_states[1]])
    assert np.isclose(A_kind3[1, lamp_states[0]], 0.9)

    # A perfect sensor is conclusive about kind but not which of two lamps.
    A_sharp = kind_observation_model(lamps2, p_correct=1.0, mark_goal=True)
    assert np.allclose(A_sharp[:, lamp_states[0]], [0.0, 1.0, 0.0])
    assert np.allclose(A_sharp[:, lamp_states[0]], A_sharp[:, lamp_states[1]])

    # A noisy sensor keeps its columns proper distributions.
    noisy = GridWorld(size=(3, 3), sensor_noise=0.3)
    A_noisy = observation_model(noisy)
    check_columns_sum_to_one(A_noisy, "A")
    assert np.isclose(A_noisy[4, 4], 0.7)

    # Slip spreads the transition without breaking it.
    icy = GridWorld(size=(3, 3), slip=0.2)
    B_icy = transition_model(icy)
    check_columns_sum_to_one(B_icy, "B")
    assert np.isclose(B_icy[1, 0, 3], 0.8)
    assert np.allclose(B_icy.sum(axis=1), 1.0)   # still doubly stochastic with slip

    # The property those two sums buy: prediction never lowers entropy.
    from aif.beliefs import predict

    def _entropy(v):
        nz = v[v > 0]
        return float(-(nz * np.log(nz)).sum())

    belief = np.zeros(icy.n_states)
    belief[4] = 1.0
    previous = _entropy(belief)
    for action in [3, 1, 2, 0, 3, 3, 4, 1]:
        belief = predict(belief, B_icy, action)
        current = _entropy(belief)
        assert current >= previous - 1e-12, "prediction narrowed a belief"
        previous = current

    # The validator actually catches a bad column.
    broken = np.ones((2, 2))
    try:
        check_columns_sum_to_one(broken, "broken")
    except ValueError:
        pass
    else:
        raise AssertionError("validator missed a column summing to 2")

    assert np.allclose(starting_belief(world, known=False), 1.0 / 9)
    assert starting_belief(world)[0] == 1.0

    print("aif.generative_model ok")


if __name__ == "__main__":
    demo()
