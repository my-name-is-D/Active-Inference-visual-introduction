"""Observation-indexed tabular Q-learning: the no-belief controller.

Tabular Q-learning with the latest observation used as the table index.

    Q[o,a] <- Q[o,a] + alpha * (r + discount * max_a' Q[o',a'] - Q[o,a])

The update is standard, but this observation representation is generally not
Markov. In the demo, 22 dark cells share an identical observation likelihood,
not a deterministic observation: every cell can emit all three noisy readings.
The greedy policy can select only one action per reading and stores neither
observation history nor a posterior over positions. It therefore cannot
represent general belief-dependent policies. This does not prove it cannot
reach home or learn reward-producing reactive behaviour.

A missing belief representation is not a point-mass belief or certainty about
position. There is no explicit information-gain bonus here, but informative
readings can still affect subsequent actions and thus learned returns.

A and B are used ONLY by the simulator to generate experience. The update uses
(o, action, reward, next_o), not model-based planning. Rewards are received on
arrival; home is not terminal. `steps` truncates continuing-task samples and
bootstrapping continues at the last step. This is not a finite-horizon episodic
return. Constant alpha and observation aliasing mean the usual tabular MDP
almost-sure convergence theorem does not apply to this setup.

It shares the objective (reward `log C[o]`), the world and the common prior
with the belief agents, so it runs in `aif.comparison` as kind `tabular`.
It does NOT share their position representation, and cannot: it holds no
belief. That is the difference under study for this agent, and it is a
different difference from the one separating AIF from matched reward.
Any performance gap therefore confounds objective with representation.
The discount and continuing-task update also differ from the belief agents'
undiscounted receding-horizon planning; this is not an ablation of one term.
"""

import numpy as np


def state_aliasing(world, A):
    """Group cells by their most probable observation (not its full support).

    Returns a list, one entry per observation, of the cells reporting it most
    often. Equal most-probable labels alone do not prove equal likelihood
    columns; the demo's same-kind columns are identical by construction.
    """
    groups = [[] for _ in range(A.shape[0])]
    for state in range(world.n_states):
        groups[int(np.argmax(A[:, state]))].append(world.state_position(state))
    return groups


def train(world, A, B, C, initial_belief, episodes=2000, steps=40, alpha=0.1,
          discount=0.95, epsilon=0.1, seed=0, over='observations', world_B=None):
    """Learn `Q[observation][action]` by acting, with no model of the world.

    Reward is `log C[o]` on the reading received, the same objective the
    belief agents optimise. The initial hidden state is drawn from the common
    prior, so repeated episodes reveal no initial condition the others lack.

    Exploration is epsilon-greedy: with probability epsilon, draw uniformly
    from ALL actions (including the greedy action). Otherwise use the shared
    `greedy_action` tie rule. This is one exploration rule, not a statement
    that reward maximisation cannot value information.
    """
    from aif.agent_qlearning import preference_reward, greedy_action
    reward = preference_reward(A, C, over)   # also rejects a non-distribution C
    C = np.asarray(C, dtype=float)
    b = np.asarray(initial_belief, dtype=float)
    if (b.shape != (world.n_states,) or np.any(b < 0)
            or not np.isclose(b.sum(), 1., rtol=0, atol=1e-12)):
        raise ValueError('initial belief must be a distribution over states')
    if episodes < 1 or steps < 1 or not 0 < alpha <= 1 or not 0 <= epsilon <= 1:
        raise ValueError('invalid training budget, alpha or epsilon')
    rng = np.random.default_rng(seed)
    world_B = B if world_B is None else np.asarray(world_B, dtype=float)
    if world_B.shape != B.shape:
        raise ValueError('world_B must match the agent transition matrix shape')
    n_actions = B.shape[2]
    Q = np.zeros((A.shape[0], n_actions))

    for _ in range(episodes):
        state = int(rng.choice(world.n_states, p=b))
        obs = int(rng.choice(A.shape[0], p=A[:, state]))

        for _ in range(steps):
            if rng.random() < epsilon:
                action = int(rng.integers(n_actions))
            else:
                action = greedy_action(Q[obs])

            next_state = int(rng.choice(world.n_states, p=world_B[:, state, action]))
            next_obs = int(rng.choice(A.shape[0], p=A[:, next_state]))
            r = float(np.log(C[next_obs]) if over == 'observations'
                      else reward[next_state])

            target = r + discount * Q[next_obs].max()
            Q[obs, action] += alpha * (target - Q[obs, action])

            state, obs = next_state, next_obs

    return Q


def choose_action(Q, obs):
    """Greedy in the learned table. The belief is not an input: there isn't one."""
    from aif.agent_qlearning import greedy_action
    return greedy_action(Q[obs])


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
    C = np.array([0.05, 0.10, 0.85])
    belief = np.zeros(world.n_states)
    belief[[8, 12, 16]] = 1.0 / 3.0

    # The structural claim, which needs no training to check: 22 of 25 cells
    # have identical observation likelihoods.
    groups = state_aliasing(world, A)
    assert len(groups) == 3
    assert len(groups[0]) == 22, len(groups[0])   # dark
    assert len(groups[1]) == 2                    # the two lamps
    assert len(groups[2]) == 1                    # the marked goal
    assert (3, 1) in groups[0] and (0, 0) in groups[0]   # true position aliases

    # 25 hidden cells, but only 3 rows in this restricted representation.
    Q = train(world, A, B, C, belief, episodes=400, steps=20)
    assert Q.shape == (3, 5)
    assert Q.shape[0] < world.n_states

    # On receiving a dark reading it chooses one action, independent of the
    # actual cell or observation history. Noisy readings can select other rows
    # even at a dark cell; this is not a deterministic cell-to-action mapping.
    dark_action = choose_action(Q, 0)
    assert 0 <= dark_action < 5

    # The Bayesian agents explicitly represent positional uncertainty.
    # The Q table has no comparable belief variable.
    from aif.agent_active_inference import entropy

    assert entropy(belief) > 1.0, "the modelling agents hold real uncertainty"
    # One action-value vector per observed label; this is not a belief.
    assert Q[0].shape == (5,)

    # It runs in the shared harness, and acts on the reading, not the belief.
    from aif.comparison import BeliefAgent
    agent = BeliefAgent("tabular", A, B, C, belief, horizon=3, Q=Q)
    agent.reset(belief, observation=0)
    assert agent.act() == choose_action(Q, 0)
    # Same belief, different reading -> the decision can change. The converse
    # (same reading, different belief) is what it structurally cannot do.
    agent.reset(belief, observation=1)
    assert np.array_equal(agent.action_values(), Q[1])
    spread = agent.belief.copy()
    agent.observe(0, 2)
    assert not np.allclose(agent.belief, spread), "belief is still filtered for diagnostics"
    assert np.array_equal(agent.action_values(), Q[2]), "but decisions use the reading alone"

    print("aif.agent_qlearning_observation ok (no-belief controller, shared objective)")


if __name__ == "__main__":
    demo()
