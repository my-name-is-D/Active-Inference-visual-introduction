"""What module 6 hands back to the agent: sampling, habits, precision, starting prior.

    policy_posterior     q(pi) = softmax(ln E - gamma G)
    action_marginals     P(u): q(pi) added up over policies starting with u
    sharpen              P(u)**alpha, normalised (pymdp sample_action, SPM spm_MDP_VB_X)
    update_habits        e <- e + q(pi)
    policy_free_energy   accumulated F(pi) for trial-spanning policies
    update_precision     beta from its prior beta0 and the disagreement (pi - pi0).G
    smoothed_start       q(s_0 | o_1:T, u_1:T), the start in hindsight

Our G is minimised. SPM and Smith et al. (2022) write the same update with the
negated G, so every sign here has been converted; `verification/module6_check.py` checks the direction.

Precision learning requires policies that span the trial. If policies are
created afresh on every step, they share the same past free energy and the
precision stays at its prior.
"""

import numpy as np

from aif.fixed_lag import smooth_records


def softmax(x):
    x = np.asarray(x, dtype=float)
    weights = np.exp(x - x.max())
    return weights / weights.sum()


def policy_posterior(G, gamma, E=None):
    """q(pi) = softmax(ln E - gamma G). A flat or missing E changes nothing."""
    G = np.asarray(G, dtype=float)
    log_E = np.zeros_like(G) if E is None else np.log(E)
    return softmax(log_E - gamma * G)


def action_marginals(q, policies, n_actions):
    """P(u): the probability of every policy whose first action is u, added up."""
    P = np.zeros(n_actions)
    for q_pi, policy in zip(q, policies):
        P[policy[0]] += q_pi
    return P


def sharpen(P, alpha):
    """softmax(alpha ln P), which is P**alpha normalised: pymdp's and SPM's action precision."""
    return softmax(alpha * np.log(np.maximum(P, 1e-300)))


def update_habits(e, q):
    """Add this trial's posterior over policies to the habit counts, once per trial. E is e normalised."""
    return e + q


def policy_free_energy(initial_belief, B, A, observations, policies):
    """Accumulated ``F(pi) = -log p(o_1:t | pi)`` for fixed trial policies.

    Each candidate policy supplies the action at every observed trial step.
    Exact Bayesian filtering gives the observation evidence; at its optimum,
    variational free energy equals negative log evidence.
    """
    observations = tuple(observations)
    if any(len(policy) < len(observations) for policy in policies):
        raise ValueError('each policy must cover every observed trial step')
    F = np.zeros(len(policies))
    for i, policy in enumerate(policies):
        belief = np.asarray(initial_belief, dtype=float)
        for t, observation in enumerate(observations):
            predicted = B[:, :, policy[t]] @ belief
            evidence = float(A[observation] @ predicted)
            F[i] -= np.log(max(evidence, 1e-300))
            belief = A[observation] * predicted / max(evidence, 1e-300)
    return F


def update_precision(beta0, G, F, E=None, iterations=16):
    """Posterior beta after one observation, starting from the prior beta0.

        pi0 = softmax(ln E - gamma G)          scores alone
        pi  = softmax(ln E - gamma G - F)      scores and the evidence
        beta = beta0 + (pi - pi0) . G          at the fixed point, gamma = 1 / beta

    Solved by the damped iteration used in SPM and Smith et al. (2022). The
    update restarts from beta0 at every observation; it does not accumulate.
    Evidence favouring low-G policies makes (pi - pi0) . G negative, so beta
    falls and gamma rises.
    """
    G = np.asarray(G, dtype=float)
    beta = beta0
    for _ in range(iterations):
        pi0 = policy_posterior(G, 1.0 / beta, E)
        pi = softmax(np.log(pi0) - F)
        beta -= (beta - beta0 - (pi - pi0) @ G) / 2
        if beta <= 0:
            raise ValueError('precision update produced a non-positive Gamma rate')
    return beta


def smoothed_start(D, B, A, actions, observations):
    """q(s_0 | o_1:T, u_1:T): where the whole run says the agent started.

    The first observation arrives after the first step, so s_0 is never
    observed directly. A forward pass stores each step, then the backward pass
    shared with module 5 (`fixed_lag.smooth_records`) runs over the whole run.
    """
    belief = np.asarray(D, dtype=float)
    records = []
    for u, o in zip(actions, observations):
        transition = B[:, :, u]
        records.append(dict(previous=belief, transition=transition, likelihood=A[o]))
        belief = A[o] * (transition @ belief)
        belief = belief / belief.sum()
    _, first_joint = smooth_records(records)[0]
    return first_joint.sum(axis=0)
