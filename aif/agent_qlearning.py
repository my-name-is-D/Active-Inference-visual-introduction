"""Finite-horizon Q-learning on Bayesian position beliefs.

The observation-only historical baseline is agent_qlearning_observation.py.
This learner uses the same A, B and Bayesian filter as the planning agents.
Only the simulator sees the actual cell. Reward is log C[o], a function of an
already shared reading: it reveals no additional information about position.
Its expected immediate reward is q(next_state) @ (A.T @ log C).

Q_h(b,a) approximates the undiscounted return for h remaining actions:
    target = log C[o] + max_a' Q_(h-1)(Bayes(b,a,o),a'),  Q_0 = 0.
The controller uses Q_H at each real step, matching receding-horizon planning.

The representation contains every belief entry and every quadratic product.
It preserves the complete belief input but restricts the value function;
it is NOT exact tabular Q-learning and has no tabular convergence guarantee.
Training is off-policy from uniformly random actions, with replay. Horizon
heads are fitted in ascending order with their predecessor frozen. Updates
are normalised semi-gradient TD steps, with explicit learning rate and budget.
Q-learning can value future observation-contingent actions; the original
fixed-sequence planners cannot. The comparison reports this distinction and
checks the learner against exact adaptive reward planning at small horizons.
"""
import hashlib
import json
import numpy as np

from aif.beliefs import filter_step


def model_signature(A, B, C, prior):
    """Reject silent reuse of trained weights after a model/preference change."""
    digest = hashlib.sha256()
    for array in (A, B, C, prior):
        value = np.ascontiguousarray(array, dtype=np.float64)
        digest.update(str(value.shape).encode())
        digest.update(value.tobytes())
    return digest.hexdigest()


def preference_reward(A, C, over='observations'):
    """State reward whose expected value equals negative pragmatic cost.

    `over='observations'`: C is a distribution over readings and the state
    utility is A.T @ log C, so E[reward] = (A @ q) . log C.
    `over='states'`: C is a distribution over cells and the state utility is
    log C directly, so E[reward] = q . log C. The agent then scores its
    preference on the belief without needing an observation to express it,
    which is the "home is unlit at night" world.
    """
    if over not in ('observations', 'states'):
        raise ValueError("over must be 'observations' or 'states'")
    C = np.asarray(C, dtype=float)
    n = A.shape[0] if over == 'observations' else A.shape[1]
    if C.shape != (n,) or not np.all(np.isfinite(C)) or np.any(C <= 0):
        raise ValueError(f'C must contain one finite, strictly positive probability per {over[:-1]}')
    if not np.isclose(C.sum(), 1., rtol=0, atol=1e-12):
        raise ValueError('C must sum to one')
    return A.T @ np.log(C) if over == 'observations' else np.log(C)


def greedy_action(values, tolerance=1e-12):
    """First action among numerical ties; shared with comparison planners."""
    values = np.asarray(values, dtype=float)
    if values.ndim != 1 or not len(values) or not np.all(np.isfinite(values)):
        raise ValueError('action values must be a nonempty finite vector')
    return int(np.flatnonzero(values >= values.max() - tolerance)[0])


class BeliefQ:
    def __init__(self, n_states, n_actions, horizon=3):
        if any(not isinstance(v, (int, np.integer)) or v < 1
               for v in (n_states, n_actions, horizon)):
            raise ValueError('state/action counts and horizon must be positive integers')
        self.n_states, self.n_actions, self.horizon = n_states, n_actions, horizon
        self.pairs = np.triu_indices(n_states)
        self.weights = np.zeros((horizon + 1, n_actions,
                                 n_states + len(self.pairs[0])))
        self.training = {}
        self.signature = None

    def features(self, belief):
        b = np.asarray(belief, dtype=float)
        if b.shape != (self.n_states,) or not np.all(np.isfinite(b)) or np.any(b < 0):
            raise ValueError('belief must be a finite nonnegative vector over states')
        if not np.isclose(b.sum(), 1., rtol=0, atol=1e-12):
            raise ValueError('belief must sum to one')
        return np.concatenate((b, b[self.pairs[0]] * b[self.pairs[1]]))

    def values(self, belief, horizon=None):
        h = self.horizon if horizon is None else horizon
        if not isinstance(h, (int, np.integer)) or not 0 <= h <= self.horizon:
            raise ValueError('horizon outside trained range')
        return self.weights[h] @ self.features(belief)

    def update(self, belief, action, reward, next_belief, horizon, alpha=.05):
        """One normalised semi-gradient update; no bootstrap beyond h=1."""
        if not 1 <= horizon <= self.horizon or not 0 <= action < self.n_actions:
            raise ValueError('invalid horizon or action')
        if not 0 < alpha <= 1 or not np.isfinite(reward):
            raise ValueError('alpha must be in (0,1] and reward finite')
        phi = self.features(belief)
        target = reward + self.values(next_belief, horizon - 1).max()
        error = float(target - self.weights[horizon, action] @ phi)
        self.weights[horizon, action] += alpha * error * phi / (phi @ phi)
        return error

    def save(self, path):
        """Store weights and training/model metadata without pickle objects."""
        metadata = dict(n_states=self.n_states, n_actions=self.n_actions,
                        horizon=self.horizon, signature=self.signature,
                        training=self.training)
        np.savez_compressed(path, weights=self.weights, metadata=json.dumps(metadata))

    @classmethod
    def load(cls, path):
        with np.load(path, allow_pickle=False) as saved:
            metadata = json.loads(str(saved['metadata']))
            Q = cls(metadata['n_states'], metadata['n_actions'], metadata['horizon'])
            weights = saved['weights']
            if (weights.shape != Q.weights.shape or not np.isfinite(weights).all()
                    or np.any(weights[0] != 0)):
                raise ValueError('invalid Q weights or nonzero terminal head')
            Q.weights = weights.copy()
            Q.signature, Q.training = metadata['signature'], metadata['training']
        return Q


def choose_action(Q, belief, horizon=None):
    return greedy_action(Q.values(belief, horizon))


def collect_experience(A, B, C, initial_belief, *, episodes=3000, steps=9, seed=0,
                       over='observations', world_B=None):
    """Uniform off-policy exploration; simulator state is never an input feature.

    The initial hidden state is sampled from the common prior, rather than
    exposing a fixed true cell through repeated training episodes.
    """
    reward = preference_reward(A, C, over)
    probe = BeliefQ(B.shape[0], B.shape[2], 1)
    probe.features(initial_belief)
    if episodes < 1 or steps < 1:
        raise ValueError('episodes and steps must be positive')
    rng = np.random.default_rng(seed)
    world_B = B if world_B is None else np.asarray(world_B, dtype=float)
    if world_B.shape != B.shape:
        raise ValueError('world_B must match the agent transition matrix shape')
    beliefs, actions, rewards, next_beliefs = [], [], [], []
    for _ in range(episodes):
        b = np.asarray(initial_belief, dtype=float).copy()
        state = int(rng.choice(len(b), p=b))
        for _ in range(steps):
            action = int(rng.integers(B.shape[2]))
            state = int(rng.choice(len(b), p=world_B[:, state, action]))
            observation = int(rng.choice(A.shape[0], p=A[:, state]))
            next_b = filter_step(b, B, A, action, observation)
            beliefs.append(b); actions.append(action)
            # Over observations the reward is log C[o]; over states the agent
            # is paid for the cell it actually occupies, which the simulator
            # knows and the learner does not observe directly.
            rewards.append(float(np.log(C[observation]) if over == 'observations'
                                 else reward[state]))
            next_beliefs.append(next_b)
            b = next_b
    return (np.asarray(beliefs), np.asarray(actions), np.asarray(rewards),
            np.asarray(next_beliefs))


def train(world, A, B, C, initial_belief, *, horizon=3, episodes=3000,
          steps=9, passes=4, alpha=.05, seed=0, over='observations', world_B=None):
    """Fit Q_h from sampled transitions; no exact planner supplies targets."""
    if passes < 1 or not 0 < alpha <= 1:
        raise ValueError('passes must be positive and alpha in (0,1]')
    Q = BeliefQ(world.n_states, B.shape[2], horizon)
    beliefs, actions, rewards, next_beliefs = collect_experience(
        A, B, C, initial_belief, episodes=episodes, steps=steps, seed=seed,
        over=over, world_B=world_B)
    phi = np.array([Q.features(b) for b in beliefs])
    next_phi = np.array([Q.features(b) for b in next_beliefs])
    norm = np.einsum('ij,ij->i', phi, phi)
    rng = np.random.default_rng(np.random.SeedSequence([seed, 1]))
    losses = []
    for h in range(1, horizon + 1):
        # Lower-horizon head is frozen throughout this regression stage.
        targets = rewards + np.einsum('nf,af->na', next_phi, Q.weights[h-1]).max(axis=1)
        head_losses = []
        for pass_index in range(passes):
            averaged = np.zeros_like(Q.weights[h])
            average_count = 0
            for step, i in enumerate(rng.permutation(len(actions))):
                a = actions[i]
                error = targets[i] - Q.weights[h,a] @ phi[i]
                Q.weights[h,a] += alpha * error * phi[i] / norm[i]
                # Average the final replay pass to reduce constant-step noise.
                # No evaluation observations, scores or optimal actions enter.
                if passes > 1 and pass_index == passes - 1 and (step + 1) % 50 == 0:
                    averaged += Q.weights[h]
                    average_count += 1
            if average_count:
                Q.weights[h] = averaged / average_count
            residual = targets - np.einsum('nf,nf->n', phi, Q.weights[h,actions])
            head_losses.append(float(np.mean(residual**2)))
        losses.append(head_losses)
    Q.signature = model_signature(A, B, C, initial_belief)
    Q.training = dict(model_signature=Q.signature, seed=seed, episodes=episodes, steps=steps, passes=passes,
                      alpha=alpha, horizon=horizon, discount=1.,
                      samples=len(actions), action_counts=np.bincount(actions, minlength=B.shape[2]).tolist(),
                      training_mse_by_head_and_pass=losses,
                      averaging='every 50 updates in final pass when passes > 1',
                      exploration='uniform random actions; no evaluation exploration',
                      world_transition='same as agent B' if world_B is None else 'separate supplied world_B',
                      features='full belief and upper-triangular quadratic products')
    return Q


def demo():
    from worlds.gridworld import dark_with_lamps
    from aif.generative_model import kind_observation_model, transition_model
    w = dark_with_lamps(size=(5,5), lamps=[(1,4),(2,3)],
                        goal=(0,4), start=(3,1), slip=.3)
    A = kind_observation_model(w, mark_goal=True)
    B = transition_model(w)
    C = np.array([.05,.10,.85])
    b = np.zeros(w.n_states)
    b[[8,12,16]] = 1/3
    Q = train(w,A,B,C,b,episodes=40,steps=6,passes=2)
    assert Q.values(b).shape == (5,)
    assert np.isfinite(Q.weights).all()
    assert not np.any(Q.weights[0])
    print('aif.agent_qlearning ok (belief input; approximation, not convergence proof)')


if __name__ == '__main__':
    demo()
