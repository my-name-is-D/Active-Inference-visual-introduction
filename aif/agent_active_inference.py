"""The active inference agent: it chooses by minimising expected free energy.

This agent scores predicted observations, combining a preference cost and
expected information gain. The reward agent also scores future outcomes.
For a comparison isolating information gain, its state reward must be matched
to these preferences: r = A.T @ log(C). See learning record 0009.

    G(pi) = sum_tau  epistemic(tau) + pragmatic(tau)

with, at each step of the policy,

    epistemic = - sum_o q(o|pi) KL[ q(s|o,pi) || q(s|pi) ]
    pragmatic = - sum_o q(o|pi) log C[o]

Both terms carry a leading minus. More information means a more negative
epistemic contribution. The epistemic term is a negated
expected KL and so is never positive: a policy that would teach the agent
something can only ever lower G, never raise it.

See `agent_reward_maximiser.py` for the same planner scoring reward instead,
and `agent_qlearning.py` for learned values over the same Bayesian beliefs.
"""

import itertools

import numpy as np


def kl_divergence(q, p):
    """KL[q || p] in nats, skipping the zeros where q says the term vanishes."""
    mask = q > 1e-300
    return float((q[mask] * np.log(q[mask] / p[mask])).sum())


def entropy(p):
    mask = p > 1e-300
    return float(-(p[mask] * np.log(p[mask])).sum())


def expected_free_energy(belief, B, A, C, policy):
    """Score one policy: a fixed sequence of actions.

    A policy here is a sequence settled in advance, not a plan that reacts to
    what it sees. That is the standard discrete formulation, and it means the
    rollout never folds an observation back in: `q(s_tau|pi)` is B applied
    repeatedly. On the tutorial's doubly stochastic toroidal transitions,
    prediction entropy cannot decrease; this is not true of arbitrary B. The epistemic term
    scores the information each step would yield; it does not model the agent
    later *using* that information.

    Returns `(G, epistemic, pragmatic, trace)` with `trace` the predicted
    belief at every step, so a lesson can plot the spreading.
    """
    predicted = np.asarray(belief, dtype=float)
    total = epistemic_total = pragmatic_total = 0.0
    trace = []

    for action in policy:
        predicted = B[:, :, action] @ predicted

        # The joint q(o,s|pi), and the two marginals read off it.
        joint = A * predicted[None, :]
        expected_obs = joint.sum(axis=1)
        posteriors = np.divide(
            joint, expected_obs[:, None],
            out=np.zeros_like(joint), where=expected_obs[:, None] > 1e-300,
        )

        epistemic = -sum(
            expected_obs[o] * kl_divergence(posteriors[o], predicted)
            for o in range(A.shape[0])
            if expected_obs[o] > 1e-300
        )
        pragmatic = -float((expected_obs * np.log(C)).sum())

        total += epistemic + pragmatic
        epistemic_total += epistemic
        pragmatic_total += pragmatic
        trace.append(predicted.copy())

    return total, epistemic_total, pragmatic_total, trace


def score_all_policies(belief, B, A, C, horizon, n_actions=5):
    """Every action sequence of length `horizon`, scored and sorted by G."""
    scored = []
    for policy in itertools.product(range(n_actions), repeat=horizon):
        G, epistemic, pragmatic, _ = expected_free_energy(belief, B, A, C, policy)
        scored.append((G, policy, epistemic, pragmatic))
    scored.sort()
    return scored


def softmax_policies(scores, gamma=4.0):
    """Turn G into a distribution over policies: sigma(-gamma G).

    Lower G becomes higher probability. `gamma` is the precision: near zero the
    distribution flattens towards uniform and the agent chooses almost at
    random however different the scores are; as it grows the probability
    concentrates on the lowest-scoring policy.
    """
    G = np.asarray(scores, dtype=float)
    shifted = -gamma * (G - G.min())   # subtract the min for numerical safety
    weights = np.exp(shifted)
    return weights / weights.sum()


def choose_action(belief, B, A, C, horizon=3, gamma=None, rng=None):
    """The agent's next action: the first of the best-scoring policy.

    With `gamma` set, the policy is sampled from sigma(-gamma G) instead of
    taken outright, which is what the lesson means by the agent trusting its
    own scores more or less.
    """
    scored = score_all_policies(belief, B, A, C, horizon)
    if gamma is None:
        return scored[0][1][0]
    probabilities = softmax_policies([s[0] for s in scored], gamma)
    rng = np.random.default_rng() if rng is None else rng
    return scored[int(rng.choice(len(scored), p=probabilities))][1][0]


def demo():
    import sys

    sys.path.insert(0, ".")
    from worlds.gridworld import dark_with_lamps
    from aif.generative_model import kind_observation_model, transition_model

    # Lesson 4's world. The lesson is 1-indexed and this is 0-indexed:
    # lamps (2,5),(3,4) -> (1,4),(2,3); home (1,5) -> (0,4); true (4,2) -> (3,1).
    world = dark_with_lamps(
        size=(5, 5), lamps=[(1, 4), (2, 3)], goal=(0, 4), start=(3, 1), slip=0.3
    )
    A = kind_observation_model(world, mark_goal=True)
    B = transition_model(world)
    C = np.array([0.05, 0.10, 0.85])

    belief = np.zeros(world.n_states)
    for cell in [(1, 3), (2, 2), (3, 1)]:
        belief[world.state_index(cell)] = 1.0 / 3.0

    assert A.shape == (3, 25)
    assert np.isclose(entropy(belief), np.log(3))

    # The worked single step from docs/learning-records/0009. Action 3 is east.
    G, epistemic, pragmatic, trace = expected_free_energy(belief, B, A, C, [3])
    assert np.isclose(G, 1.9990, atol=5e-4), G
    assert np.isclose(epistemic, -0.4973, atol=5e-4), epistemic
    assert np.isclose(pragmatic, 2.4963, atol=5e-4), pragmatic

    predicted = trace[0]
    assert np.isclose(entropy(predicted), 1.9773, atol=5e-4)
    lamps = [world.state_index(c) for c in [(1, 4), (2, 3)]]
    assert np.isclose(sum(predicted[s] for s in lamps), 0.4667, atol=5e-4)

    # G reached through ambiguity + risk: an independent implementation of
    # the same identity, sharing the model and predicted distribution.
    expected_obs = A @ predicted
    ambiguity = float(sum(predicted[s] * entropy(A[:, s]) for s in range(world.n_states)))
    risk = kl_divergence(expected_obs, C)
    assert np.isclose(ambiguity + risk, G, atol=1e-12), (ambiguity + risk, G)

    # Epistemic value is a negated expected KL, so it is never positive.
    rng = np.random.default_rng(0)
    for _ in range(200):
        b = rng.random(world.n_states) + 1e-3
        b /= b.sum()
        policy = [int(rng.integers(5)) for _ in range(2)]
        _, epistemic, _, _ = expected_free_energy(b, B, A, C, policy)
        assert epistemic <= 1e-12, epistemic

    # A uniform belief is a fixed point of every action, because B is doubly
    # stochastic on a torus (see docs/learning-records/0001). Every policy then
    # scores identically, which looks like a broken objective and is not.
    uniform = np.full(world.n_states, 1.0 / world.n_states)
    scores = [G for G, _, _, _ in score_all_policies(uniform, B, A, C, horizon=1)]
    assert np.allclose(scores, scores[0]), "uniform belief should tie every policy"

    # Softmax: gamma near zero is near-uniform, large gamma concentrates.
    sample = [1.0, 2.0, 3.0]
    flat = softmax_policies(sample, gamma=0.01)
    sharp = softmax_policies(sample, gamma=50.0)
    assert np.allclose(flat, flat[0], atol=0.01)
    assert sharp[0] > 0.99
    assert np.isclose(flat.sum(), 1.0) and np.isclose(sharp.sum(), 1.0)

    # The lamp route beats staying dark, at every horizon (there is no
    # crossover in this world; see docs/learning-records/0008).
    for horizon in [1, 2, 3]:
        best = score_all_policies(belief, B, A, C, horizon)[0]
        _, policy, _, _ = best
        _, _, _, trace = expected_free_energy(belief, B, A, C, policy)
        lamp_mass = max(sum(step[s] for s in lamps) for step in trace)
        assert lamp_mass > 0.25, (horizon, policy, lamp_mass)

    print("aif.agent_active_inference ok")


if __name__ == "__main__":
    demo()
