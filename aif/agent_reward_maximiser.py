"""Finite-horizon expected-reward planning over fixed action sequences.

    score(pi) = sum_tau q(s_tau | pi) . r

The belief rollout matches agent_active_inference.py. Future actions in each
candidate sequence cannot depend on future observations. Executing its first
action, updating by Bayes externally, and replanning gives receding-horizon
control. This is NOT QMDP: QMDP uses a fully observed MDP value function for
continuation after the first action; this implementation never does that.

The score averages over the full belief; it does not assume a sharp belief.
It omits the instrumental value of future observation-contingent decisions.
A full POMDP reward planner can value information without an information bonus.

Rewards are collected on arrival at every step, with no discount or terminal
state. The goal remains traversable and repeated visits earn repeated reward.
For an objective-matched comparison, use r = A.T @ log(C): its expected reward
is the negative pragmatic term. The demo's goal-only r is NOT that mapping.
With matched rewards and a policy-independent information term, the two
planners have the same ordering of equal-length action sequences.
"""

import itertools

import numpy as np


def goal_reward(world, value=10.0):
    """`r`: a scalar on a state, zero everywhere but the goal.

    Note what this object is not. It has one entry per *state*, where a
    preference `C` has one entry per *observation*; and nothing constrains
    these numbers, where the entries of `C` are a distribution and must sum to
    one. A reward can be any size, negative, or a hundred times another.
    """
    r = np.zeros(world.n_states)
    r[world.state_index(world.goal)] = value
    return r


def expected_reward(belief, B, r, policy):
    """Score one policy by the reward it expects to collect.

    The rollout is identical to the active inference agent's: carry the belief
    through B, once per action. Only the scoring differs.

    Returns `(total, trace)`.
    """
    predicted = np.asarray(belief, dtype=float)
    total = 0.0
    trace = []
    for action in policy:
        predicted = B[:, :, action] @ predicted
        total += float(predicted @ r)
        trace.append(predicted.copy())
    return total, trace


def score_all_policies(belief, B, r, horizon, n_actions=None):
    """Every action sequence of length `horizon`, best first.

    Sorted descending: this agent *maximises*, where expected free energy is
    minimised. A lesson putting the two side by side has to say so, or the
    comparison reads backwards.
    """
    if horizon < 0:
        raise ValueError("horizon must be non-negative")
    if n_actions is None:
        n_actions = B.shape[2]
    if not 1 <= n_actions <= B.shape[2]:
        raise ValueError("n_actions must be between 1 and B.shape[2]")
    scored = []
    for policy in itertools.product(range(n_actions), repeat=horizon):
        total, _ = expected_reward(belief, B, r, policy)
        scored.append((total, policy))
    scored.sort(key=lambda row: -row[0])
    return scored


def choose_action(belief, B, r, horizon=3):
    """The first action of the best-scoring policy. Argmax, no precision."""
    if horizon < 1:
        raise ValueError("choosing an action requires horizon >= 1")
    return score_all_policies(belief, B, r, horizon)[0][1][0]


def demo():
    import sys

    sys.path.insert(0, ".")
    from worlds.gridworld import dark_with_lamps
    from aif.generative_model import kind_observation_model, transition_model

    world = dark_with_lamps(
        size=(5, 5), lamps=[(1, 4), (2, 3)], goal=(0, 4), start=(3, 1), slip=0.3
    )
    A = kind_observation_model(world, mark_goal=True)
    B = transition_model(world)
    r = goal_reward(world)

    belief = np.zeros(world.n_states)
    for cell in [(1, 3), (2, 2), (3, 1)]:
        belief[world.state_index(cell)] = 1.0 / 3.0

    # r is indexed by state, and nothing makes it a distribution.
    assert r.shape == (world.n_states,)
    assert not np.isclose(r.sum(), 1.0)
    assert np.isclose(r.sum(), 10.0)

    # The per-action expected reward from docs/learning-records/0009.
    # Actions are [up, down, left, right, stay]; up and right both score 0.2.
    expected = {0: 0.2, 1: 0.0, 2: 0.0, 3: 0.2, 4: 0.0}
    for action, want in expected.items():
        total, _ = expected_reward(belief, B, r, [action])
        assert np.isclose(total, want, atol=5e-4), (action, total, want)

    # It maximises: the best policy is the highest-scoring one.
    scored = score_all_policies(belief, B, r, horizon=2)
    assert scored[0][0] >= scored[-1][0]
    assert np.isclose(scored[0][0], max(s[0] for s in scored))

    # The rollout matches the active inference agent's exactly. Same belief,
    # same B, same policy, so the predicted states must agree: the two agents
    # differ in scoring alone, which is the claim the comparison rests on.
    from aif.agent_active_inference import expected_free_energy

    policy = [3, 0]
    _, reward_trace = expected_reward(belief, B, r, policy)
    _, _, _, aif_trace = expected_free_energy(belief, B, A, np.array([0.05, 0.10, 0.85]), policy)
    for mine, theirs in zip(reward_trace, aif_trace):
        assert np.allclose(mine, theirs), "the two agents must share a rollout"

    # Down and stay have zero one-step reward for this particular belief.
    # Up/right can reach home through slip, as checked above.
    down, _ = expected_reward(belief, B, r, [1])
    stay, _ = expected_reward(belief, B, r, [4])
    assert np.isclose(down, stay, atol=1e-12), (down, stay)

    print("aif.agent_reward_maximiser ok")


if __name__ == "__main__":
    demo()
