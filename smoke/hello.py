"""A stand-in for the numerical code the tutorial will hold.

Deliberately trivial, but it uses numpy and returns an array rather than a
scalar, because that is what every real module here will do.
"""

import numpy as np


def transition_matrix(n_states, stay_probability):
    """Build a square transition matrix whose columns sum to one.

    Each column is the distribution over next states given the current state:
    the agent stays put with `stay_probability` and otherwise spreads the
    remaining mass uniformly over the other states.

    Returns an array of shape (n_states, n_states).
    """
    if not 0.0 <= stay_probability <= 1.0:
        raise ValueError("stay_probability must lie in [0, 1]")
    if n_states < 2:
        raise ValueError("n_states must be at least 2")

    spread = (1.0 - stay_probability) / (n_states - 1)
    B = np.full((n_states, n_states), spread)
    np.fill_diagonal(B, stay_probability)
    return B


def column_sums(B):
    """Return the sum of each column, which should be all ones."""
    return B.sum(axis=0)


def demo():
    B = transition_matrix(4, 0.7)
    assert B.shape == (4, 4)
    assert np.allclose(column_sums(B), 1.0)
    assert np.isclose(B[0, 0], 0.7)
    assert np.isclose(B[1, 0], 0.1)
    print("smoke.hello ok")
    print(B)


if __name__ == "__main__":
    demo()
