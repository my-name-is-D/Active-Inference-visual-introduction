"""What the agent thinks about where it is, and how that changes.

A belief here is a plain numpy array of length `n_states` that sums to one:
the agent's probability for each tile. It is never a single position. The
whole of this module is two operations on that array, and every intermediate
quantity is returned rather than hidden, so a lesson can plot the parts.
"""

import numpy as np


def uniform_belief(n_states):
    """Knowing nothing: every state equally likely.

    This is `D`, the prior over where the agent starts, in the case where it
    has no idea. A different `D` is just a different array.
    """
    return np.full(n_states, 1.0 / n_states)


def predict(belief, B, action):
    """Where the agent expects to be after acting, before it looks.

    This is the transition step: each state's probability is spread over the
    states it could lead to, weighted by how likely each was to begin with.

        predicted(s') = sum over s of B[s', s, action] * belief(s)

    Probability mass moves but is never created, so the result still sums to
    one. Written as an explicit loop rather than a matrix product, because the
    loop shows what is being summed over.
    """
    n_states = len(belief)
    predicted = np.zeros(n_states)

    for next_state in range(n_states):
        for state in range(n_states):
            predicted[next_state] += B[next_state, state, action] * belief[state]

    return predicted


def update(prior, A, observation):
    """Fold an observation into a belief. Bayes, then normalise.

    Returns three arrays rather than only the answer, because each one is
    worth looking at:

        likelihood: A[observation], how likely that observation is from each
            state. This is a row of A, not a distribution over states, and it
            does not sum to one.
        unnormalised: likelihood * prior, elementwise. The shape of the
            answer is already here; only the scale is wrong.
        posterior: unnormalised / unnormalised.sum(), which sums to one.

    The denominator is the probability of the observation under the current
    belief, which reappears in the next function as surprise.
    """
    likelihood = A[observation]
    unnormalised = likelihood * prior
    evidence = unnormalised.sum()

    if evidence == 0.0:
        raise ValueError(
            f"observation {observation} is impossible under this belief: "
            "every state that could produce it has probability zero"
        )

    posterior = unnormalised / evidence
    return posterior, likelihood, unnormalised


def surprise(prior, A, observation):
    """How unexpected an observation was, in nats.

        surprise = -log p(observation)
                 = -log sum over s of A[observation, s] * prior(s)

    Zero when the observation was certain, large when the agent did not see it
    coming. This is the quantity free energy bounds, and it is worth watching
    on its own first.
    """
    probability = (A[observation] * prior).sum()
    if probability == 0.0:
        return np.inf
    return -np.log(probability)


def demo():
    import sys

    # Two states, a perfectly informative observation.
    A = np.array([[0.9, 0.1], [0.1, 0.9]])
    prior = np.array([0.5, 0.5])

    posterior, likelihood, unnormalised = update(prior, A, 0)
    assert np.allclose(likelihood, [0.9, 0.1])
    assert np.allclose(unnormalised, [0.45, 0.05])
    assert np.allclose(posterior, [0.9, 0.1]), posterior
    assert np.isclose(posterior.sum(), 1.0)

    # The exercise from the plan: prior [0.5, 0.5], likelihood row [0.9, 0.1],
    # answer [0.9, 0.1]. Three multiplications and no logarithms.
    assert np.allclose(update(np.array([0.5, 0.5]), A, 0)[0], [0.9, 0.1])

    # An uninformative observation leaves the belief alone.
    flat = np.array([[0.5, 0.5], [0.5, 0.5]])
    unchanged, _, _ = update(prior, flat, 0)
    assert np.allclose(unchanged, prior)

    # Surprise is zero when the observation was certain.
    certain = np.array([[1.0, 0.0], [0.0, 1.0]])
    assert np.isclose(surprise(np.array([1.0, 0.0]), certain, 0), 0.0)
    # and ln 2 when the agent thought it was a coin flip.
    assert np.isclose(surprise(prior, certain, 0), np.log(2))

    # Prediction moves mass without creating any.
    B = np.zeros((2, 2, 1))
    B[1, 0, 0] = 1.0    # state 0 leads to state 1
    B[1, 1, 0] = 1.0    # state 1 stays at state 1
    predicted = predict(np.array([1.0, 0.0]), B, 0)
    assert np.allclose(predicted, [0.0, 1.0])
    assert np.isclose(predicted.sum(), 1.0)

    assert np.allclose(uniform_belief(4), [0.25] * 4)

    # Cross-check fixture for the JavaScript twin (site-src/aif.js). Every case
    # is (prior, A, observation) with the posterior update produces. The JS
    # demo() asserts it reproduces each posterior.
    import json
    import pathlib

    flat2 = np.array([[0.5, 0.5], [0.5, 0.5]])

    sys.path.insert(0, ".")
    from aif.generative_model import observation_model
    from worlds.gridworld import GridWorld

    noisy = GridWorld(size=(5, 5), sensor_noise=0.3)
    A_noisy = observation_model(noisy)
    prior_noisy = uniform_belief(noisy.n_states)

    cases = [
        {"prior": [0.5, 0.5], "A": A.tolist(), "obs": 0},
        {"prior": [0.5, 0.5], "A": flat2.tolist(), "obs": 0},
        {"prior": prior_noisy.tolist(), "A": A_noisy.tolist(), "obs": 7},
        {"prior": prior_noisy.tolist(), "A": A_noisy.tolist(), "obs": 12},
    ]
    for case in cases:
        posterior, _, _ = update(
            np.array(case["prior"]), np.array(case["A"]), case["obs"]
        )
        case["posterior"] = posterior.tolist()

    # The transition model is the other half of the twin. B is spread (slip >
    # 0), so this pins down the exact shape of a failed step: without it the
    # JS could drift to a different-but-plausible spread and nothing would
    # catch it, which is precisely what happened once.
    from aif.generative_model import transition_model

    slippery = GridWorld(size=(5, 5), slip=0.3)
    B_slip = transition_model(slippery)

    # A prediction from a known cell, so the JS checks predict() against this B
    # rather than only comparing matrices.
    start = np.zeros(slippery.n_states)
    start[7] = 1.0
    RIGHT = 3
    predicted_once = predict(start, B_slip, RIGHT)

    transitions = {
        "rows": 5,
        "cols": 5,
        "slip": 0.3,
        "B": B_slip.tolist(),
        "predict_from": 7,
        "predict_action": RIGHT,
        "predicted": predicted_once.tolist(),
    }

    out = pathlib.Path(__file__).with_name("belief_cases.json")
    out.write_text(
        json.dumps({"updates": cases, "transitions": transitions}, indent=2) + "\n"
    )

    reloaded = json.loads(out.read_text())
    assert len(reloaded["updates"]) == 4
    assert np.isclose(sum(reloaded["updates"][0]["posterior"]), 1.0)
    assert np.isclose(sum(reloaded["transitions"]["predicted"]), 1.0)

    print("aif.beliefs ok")


if __name__ == "__main__":
    demo()
