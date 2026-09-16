"""A grid the agent moves on, and what it can see from each cell.

The whole tutorial runs on this one world so that a change in behaviour is
visibly a change in the objective, not a change in the problem. Four
configurations are used:

    lesson              configuration                what it forces
    perception          lit grid, noisy sensor       beliefs
    epistemic value     dark grid with lamps         looking before moving
    novelty             dark grid, icy               an uncertain model
    preferences         preference over a region     preferences as distributions

The world is deliberately not an active inference object. It holds no beliefs
and no model. It knows where the agent is, moves it when asked, and reports an
observation. Everything else is the agent's problem.
"""

import numpy as np

# Actions are indices into this list, so that a policy is an array of ints.
ACTIONS = ["up", "down", "left", "right", "stay"]

_MOVES = {
    "up": (-1, 0),
    "down": (1, 0),
    "left": (0, -1),
    "right": (0, 1),
    "stay": (0, 0),
}


class GridWorld:
    """A rectangular grid with an agent on it.

    Every quantity is a public attribute holding a plain value or array, so a
    lesson can plot the world's state without calling anything.

    Args:
        size: (rows, cols).
        goal: (row, col) the agent is trying to reach.
        start: (row, col) the agent begins at. Defaults to the top left.
        lit: array of shape `size`, True where the agent can see its position.
            Defaults to every cell lit, the perception-lesson configuration.
        sensor_noise: probability that a lit cell reports a neighbouring
            position instead of the true one. 0 means a perfect sensor. A
            scalar applies everywhere; an array of shape `size` sets it per
            cell, which is how a cell can be dim rather than merely dark. The
            difference between a dim cell and a dark one is the difference
            between an uncertain reading and no reading at all, and the
            epistemic-value lesson turns on it.
        slip: probability that a move goes somewhere other than intended.
            0 means deterministic movement. Used from the novelty lesson on.
        seed: seeds the world's own generator, so a lesson that quotes a
            number in prose gets the same number every time it runs.
    """

    def __init__(
        self,
        size=(5, 5),
        goal=(4, 4),
        start=(0, 0),
        lit=None,
        sensor_noise=0.0,
        slip=0.0,
        seed=0,
    ):
        self.rows, self.cols = size
        self.n_states = self.rows * self.cols
        self.goal = goal
        self.start = start
        self.position = start
        # Stored per cell so that noise can vary by location. A scalar
        # broadcasts, so the simple worlds read exactly as before.
        self.sensor_noise = np.broadcast_to(
            np.asarray(sensor_noise, dtype=float), size
        ).copy()
        self.slip = slip
        self.rng = np.random.default_rng(seed)

        # Every cell lit is the simplest world: the agent always sees where it
        # is. The epistemic-value lesson switches most of these off, leaving lamps.
        self.lit = np.ones(size, dtype=bool) if lit is None else np.asarray(lit, dtype=bool)

        # The history is kept so a lesson can scrub back through a run.
        self.trajectory = [start]
        self.observations = [self.observe()]

    def state_index(self, position):
        """Turn (row, col) into a single integer, the way A and B index it."""
        row, col = position
        return row * self.cols + col

    def state_position(self, index):
        """Turn a state index back into (row, col)."""
        return divmod(index, self.cols)

    def in_bounds(self, position):
        row, col = position
        return 0 <= row < self.rows and 0 <= col < self.cols

    def move(self, position, action):
        """Where `action` leads from `position`, ignoring slip.

        The grid wraps: stepping off one side arrives on the other. There are
        no walls, so no cell is special and every column of B is a shifted
        copy of every other.

        That is not only tidiness. A wall would stop the agent, and being
        stopped is information: an agent that walks west and does not move
        learns it is against the west edge, without observing anything. The
        lessons are about what looking buys, so the world must not leak
        position through the act of moving. Wrapping removes the leak, and it
        makes B doubly stochastic, which is what licenses lesson 2's claim
        that a prediction step can only ever widen a belief.
        """
        d_row, d_col = _MOVES[ACTIONS[action] if isinstance(action, (int, np.integer)) else action]
        return (
            (position[0] + d_row) % self.rows,
            (position[1] + d_col) % self.cols,
        )

    def slip_destinations(self, position, action):
        """Where a step that went wrong can leave the agent.

        A step fails in one of two ways: it does not carry the agent out of
        the cell it started in, or it drifts to a cell beside the one it
        aimed for. Those are the destinations this returns, and nothing
        else, so a step east can never leave the agent west of where it
        started. That is what odometry error and a step across a dark room
        actually look like, and it is the model lesson 2 describes.

        Returns a list of positions, possibly with repeats where the
        candidates coincide: under `stay` the intended cell IS the origin, so
        the origin appears twice. The caller shares the mass out over the list
        as it stands, so a repeated cell correctly collects more than one
        share.

        The grid wraps, so every cell has exactly four neighbours and this
        list is always the same length. No cell is special.
        """
        intended = self.move(position, action)
        destinations = [position]
        for neighbour_action in range(len(ACTIONS)):
            if ACTIONS[neighbour_action] == "stay":
                continue
            destinations.append(self.move(intended, neighbour_action))
        return destinations

    def observe(self, position=None):
        """What the agent sees from `position`, defaulting to where it is.

        Returns a state index on a lit cell, or None in the dark. `None` is
        deliberately not a state index: the agent in the dark receives no
        information about position, rather than receiving a particular
        position it should distrust.
        """
        position = self.position if position is None else position

        if not self.lit[position]:
            return None

        noise = self.sensor_noise[position]
        if noise > 0.0 and self.rng.random() < noise:
            neighbours = [
                self.move(position, a)
                for a in range(len(ACTIONS))
                if self.move(position, a) != position
            ]
            if neighbours:
                position = neighbours[self.rng.integers(len(neighbours))]

        return self.state_index(position)

    def step(self, action):
        """Take one action. Returns the observation after moving.

        The action is the agent's choice; where it ends up is the world's, and
        with slip greater than zero those differ.
        """
        intended = self.move(self.position, action)

        if self.slip > 0.0 and self.rng.random() < self.slip:
            others = self.slip_destinations(self.position, action)
            self.position = others[self.rng.integers(len(others))] if others else intended
        else:
            self.position = intended

        observation = self.observe()
        self.trajectory.append(self.position)
        self.observations.append(observation)
        return observation

    def at_goal(self):
        return self.position == self.goal

    def reset(self):
        """Return to the start, forgetting the run but keeping the layout."""
        self.position = self.start
        self.trajectory = [self.start]
        self.observations = [self.observe()]


def dark_with_lamps(size=(5, 5), lamps=(), **kwargs):
    """A world where only the cells in `lamps` reveal the agent's position.

    This is the epistemic-value configuration. The agent must reach a lamp to
    find out where it is, and the detour is the behaviour the tutorial exists
    to explain.
    """
    lit = np.zeros(size, dtype=bool)
    for lamp in lamps:
        lit[lamp] = True
    return GridWorld(size=size, lit=lit, **kwargs)


def demo():
    world = GridWorld(size=(5, 5), goal=(4, 4), seed=0)
    assert world.position == (0, 0)
    assert world.observe() == 0
    assert world.state_index((1, 2)) == 7
    assert world.state_position(7) == (1, 2)

    # Walking into the top wall is a no-op.
    # The grid wraps: up from the top row arrives on the bottom row.
    assert world.move((0, 0), ACTIONS.index("up")) == (world.rows - 1, 0)
    assert world.move((0, 0), ACTIONS.index("left")) == (0, world.cols - 1)
    assert world.move((0, 0), ACTIONS.index("right")) == (0, 1)

    world.step(ACTIONS.index("right"))
    assert world.position == (0, 1)
    assert world.trajectory == [(0, 0), (0, 1)]

    # In the dark the observation is None, not a state index.
    dark = dark_with_lamps(size=(5, 5), lamps=[(2, 2)], goal=(4, 4))
    assert dark.observe((0, 0)) is None
    assert dark.observe((2, 2)) == dark.state_index((2, 2))

    # A noisy sensor still reports somewhere adjacent, never off the grid.
    noisy = GridWorld(size=(3, 3), sensor_noise=1.0, seed=1)
    for _ in range(20):
        assert noisy.observe((1, 1)) in range(9)

    # Slip moves the agent somewhere other than intended, but still on-grid.
    slippery = GridWorld(size=(3, 3), slip=1.0, seed=2)
    for _ in range(20):
        slippery.reset()
        slippery.step(ACTIONS.index("right"))
        assert slippery.in_bounds(slippery.position)

    print("worlds.gridworld ok")


if __name__ == "__main__":
    demo()
