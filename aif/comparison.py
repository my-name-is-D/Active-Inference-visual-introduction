"""Shared filtering and objective-matched controllers for lesson 4.

AIF and reward enumerate fixed sequences. Belief Q-learning approximates
observation-contingent returns. `adaptive_reward_values` is an exact small-H
reference for that latter problem, not a training teacher or a fourth learner.

Kind `sophisticated` optimises the same expected free energy but over policy
trees, re-deciding at every posterior instead of committing to a sequence
(Friston et al. 2021). It completes the 2x2 of {reward, reward+epistemic} x
{fixed sequence, branching}, whose fourth cell is `adaptive_reward_values`.
Cost is exponential in the horizon.

Kind `tabular` is the exception to the shared-belief protocol: it acts on the
latest reading alone. Its `belief` is maintained for the runner's diagnostics
only and is never an input to its decisions. Comparisons against it confound
objective with position representation; see agent_qlearning_observation.
"""
import itertools
import numpy as np

from aif.beliefs import filter_step
from aif.agent_qlearning import preference_reward, greedy_action


def adaptive_reward_values(belief, B, A, C, horizon, over='observations'):
    """Exact undiscounted belief-MDP Bellman recursion.

    Reward is log C[o] under `over='observations'`, or the state utility
    log C[s] under `over='states'`; both reduce to q @ r.
    """
    if not isinstance(horizon, (int, np.integer)) or horizon < 0:
        raise ValueError('horizon must be a nonnegative integer')
    r = preference_reward(A, C, over)

    def recurse(b, h):
        values = np.zeros(B.shape[2])
        if h == 0:
            return values
        for a in range(B.shape[2]):
            q = B[:,:,a] @ b
            p = A @ q
            values[a] = q @ r
            if h > 1:
                for o in range(A.shape[0]):
                    if p[o] > 0:
                        post = filter_step(b, B, A, a, o)
                        values[a] += p[o] * recurse(post, h-1).max()
        return values

    return recurse(np.asarray(belief, dtype=float), horizon)


# One place naming every controller, so sweeps, reports and plots agree on
# which agents exist and how they are drawn. `trains` marks kinds needing a
# fitted Q; `branching` marks kinds whose cost is exponential in the horizon.
AGENTS = {
    'aif':           dict(label='AIF',                  colour='#2563eb', trains=False, branching=False),
    'sophisticated': dict(label='Sophisticated AIF',    colour='#7c3aed', trains=False, branching=True),
    'reward':        dict(label='Reward planner',       colour='#c2410c', trains=False, branching=False),
    'qlearning':     dict(label='Belief Q',             colour='#15803d', trains=True,  branching=False),
    'tabular':       dict(label='Tabular Q (no belief)',colour='#a16207', trains=True,  branching=False),
}


def sophisticated_values(belief, B, A, C, horizon, over='observations'):
    """Expected free energy per first action, re-optimising at every posterior.

    The branching counterpart of the fixed-sequence AIF score, and the
    epistemic counterpart of `adaptive_reward_values`: same tree, same
    branching on readings, but each step costs pragmatic + epistemic rather
    than pragmatic alone. Checked against `lesson4_audit.branching`, whose
    recursion this follows.
    """
    if not isinstance(horizon, (int, np.integer)) or horizon < 1:
        raise ValueError('horizon must be a positive integer')
    reward = preference_reward(A, C, over)      # per-state utility, both modes
    logA = np.zeros_like(A)
    np.log(A, out=logA, where=A > 0)
    ambiguity = -(A*logA).sum(axis=0)

    def step_cost(q, p):
        logp = np.zeros_like(p)
        np.log(p, out=logp, where=p > 0)
        # epistemic = ambiguity - entropy(p) = q@ambiguity + p@logp.
        # Pragmatic is -q@reward in both modes: over observations that equals
        # -p@log C, and over states it equals -q@log C.
        return float(q @ ambiguity + p @ logp - q @ reward)

    def recurse(b, h):
        values = np.empty(B.shape[2])
        for a in range(B.shape[2]):
            q = B[:,:,a] @ b
            p = A @ q
            cost = step_cost(q, p)
            if h > 1:
                for o in range(len(p)):
                    if p[o] > 0:
                        cost += p[o] * recurse(A[o]*q/p[o], h-1).min()
            values[a] = cost
        return values

    return recurse(np.asarray(belief, dtype=float), horizon)


def train_agent(kind, world, A, B, C, prior, *, horizon=3, episodes=3000,
                steps=9, passes=4, seed=0, over='observations', world_B=None):
    """Fit whatever `kind` needs, or return None. Keeps callers kind-agnostic."""
    if kind not in AGENTS:
        raise ValueError(f'unknown agent {kind!r}')
    if not AGENTS[kind]['trains']:
        return None
    if kind == 'qlearning':
        from aif.agent_qlearning import train
        return train(world, A, B, C, prior, horizon=horizon, episodes=episodes,
                     steps=steps, passes=passes, seed=seed, over=over, world_B=world_B)
    from aif.agent_qlearning_observation import train
    return train(world, A, B, C, prior, episodes=episodes, steps=steps,
                 seed=seed, over=over, world_B=world_B)


class BeliefAgent:
    """Identical position representation and filter for all three controllers.

    The controller sees only its current belief. True-state diagnostics are
    kept in the experiment runner and never supplied here. Evaluation is
    greedy with common deterministic tie handling and no online Q updates.
    """
    def __init__(self, kind, A, B, C, initial_belief, horizon=3, Q=None,
                 over='observations'):
        if kind not in AGENTS:
            raise ValueError(f'unknown agent {kind!r}; expected one of {sorted(AGENTS)}')
        if not isinstance(horizon, (int, np.integer)) or horizon < 1:
            raise ValueError('horizon must be a positive integer')
        self.kind, self.A, self.B = kind, np.asarray(A), np.asarray(B)
        self.C = np.asarray(C, dtype=float)
        self.over = over
        self.reward = preference_reward(self.A, self.C, over)
        self.belief = np.asarray(initial_belief, dtype=float).copy()
        if (self.belief.shape != (B.shape[0],) or np.any(self.belief < 0)
                or not np.isfinite(self.belief).all()
                or not np.isclose(self.belief.sum(), 1., rtol=0, atol=1e-12)):
            raise ValueError('invalid initial belief')
        self.horizon, self.Q = horizon, Q
        if kind == 'tabular':
            # Table is indexed by reading, not belief. `observation` is the
            # only decision input; the filtered belief is carried for the
            # runner's entropy diagnostics and deliberately unused in act().
            if Q is None or np.asarray(Q).shape != (self.A.shape[0], B.shape[2]):
                raise ValueError('tabular Q must be one row per observation')
            self.Q = np.asarray(Q, dtype=float)
            self.observation = None
            return
        if kind == 'sophisticated':
            return
        if kind == 'qlearning':
            if Q is None or (Q.n_states, Q.n_actions, Q.horizon) != (len(self.belief), B.shape[2], horizon):
                raise ValueError('Q must match model dimensions and planning horizon')
            if Q.signature is not None:
                from aif.agent_qlearning import model_signature
                if Q.signature != model_signature(A, B, C, initial_belief):
                    raise ValueError('Q was trained on different A, B, C or prior; retrain it')
            return
        self.policies = np.array(list(itertools.product(range(B.shape[2]), repeat=horizon)))
        matrices = np.broadcast_to(np.eye(len(self.belief)),
                                   (len(self.policies),len(self.belief),len(self.belief))).copy()
        transforms = []
        for t in range(horizon):
            matrices = np.matmul(B[:,:,self.policies[:,t]].transpose(2,0,1),matrices)
            transforms.append(matrices.copy())
        transforms = np.stack(transforms,axis=1)
        self.reward_vectors = np.einsum('ptsj,s->pj', transforms,self.reward)
        self.observation_transforms = np.einsum('os,ptsj->ptoj',A,transforms)
        logA = np.zeros_like(A)
        np.log(A,out=logA,where=A>0)
        ambiguity = -(A*logA).sum(axis=0)
        self.ambiguity_vectors = np.einsum('ptsj,s->pj',transforms,ambiguity)

    def action_values(self):
        if self.kind == 'sophisticated':
            # Negated: sophisticated_values returns expected free energy, a
            # cost, while every other kind returns a value to maximise.
            return -sophisticated_values(self.belief,self.B,self.A,self.C,self.horizon,self.over)
        if self.kind == 'tabular':
            if self.observation is None:
                raise ValueError('tabular agent has not observed yet; seed it with reset()')
            return self.Q[self.observation]
        if self.kind == 'qlearning':
            return self.Q.values(self.belief)
        if self.kind == 'reward':
            scores = self.reward_vectors @ self.belief
        else:
            p = self.observation_transforms @ self.belief
            logp = np.zeros_like(p)
            np.log(p,out=logp,where=p>0)
            # `scores` is NEGATED expected free energy, a value to maximise.
            # Both modes are entropy(p) minus the pragmatic cost; the pragmatic
            # cost is -sum_t p_t @ log C over observations, and the summed
            # state utility in reward_vectors over states.
            if self.over == 'observations':
                scores = -(p*(logp-np.log(self.C))).sum(axis=(1,2))
            else:
                scores = -(p*logp).sum(axis=(1,2)) + self.reward_vectors @ self.belief
            scores = scores - self.ambiguity_vectors @ self.belief
        return np.array([scores[self.policies[:,0]==a].max() for a in range(self.B.shape[2])])

    def act(self):
        return greedy_action(self.action_values())

    def observe(self, action, observation):
        if self.kind == 'tabular':
            self.observation = int(observation)
        self.belief = filter_step(self.belief,self.B,self.A,action,observation)
        return self.belief.copy()

    def reset(self, belief, observation=None):
        """Start an episode. `observation` seeds the tabular agent's only input."""
        self.belief = np.asarray(belief, dtype=float).copy()
        if self.kind == 'tabular':
            if observation is None:
                raise ValueError('tabular agent needs an initial reading')
            self.observation = int(observation)
