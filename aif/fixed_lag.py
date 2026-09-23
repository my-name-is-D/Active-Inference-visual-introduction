"""Online expected-count learning with a bounded backward-smoothing window.

Each reading is used once. Recent expected counts are replaced after smoothing;
older contributions are frozen. Per-step likelihood and transition snapshots
are retained, so this is exact HMM smoothing for that frozen sequence, embedded
in approximate online parameter learning, not exact joint Bayesian learning.
"""
from collections import deque
import numpy as np


def normalise_columns(counts):
    return counts / counts.sum(axis=0, keepdims=True)


def smooth_records(records):
    """Return destination marginals and pair joints using only received readings."""
    if not records:
        return []
    backward = np.ones_like(records[-1]['previous'])
    results = []
    for record in reversed(records):
        message = record['likelihood'] * backward
        joint = record['transition'] * record['previous'][None, :] * message[:, None]
        joint /= joint.sum()
        results.append((joint.sum(axis=1), joint))
        backward = record['transition'].T @ message
        backward /= backward.max()
    return results[::-1]


class FixedLagLearner:
    """window=1 is the ordinary post-reading joint-transition update.

    window=5 revises contributions for the last five transitions/readings,
    including the latest; no count is revised after it leaves that window.
    The state used for planning remains the current filtered state posterior.
    """
    def __init__(self, a, b, q, window=1, learn_a=True, learn_b=True):
        if window < 1 or int(window) != window:
            raise ValueError('window must be a positive integer')
        self.a = np.array(a, dtype=float, copy=True)
        self.b = np.array(b, dtype=float, copy=True)
        self.frozen_a = self.a.copy()
        self.frozen_b = self.b.copy()
        self.q = np.array(q, dtype=float, copy=True)
        self.window = int(window)
        self.learn_a = learn_a
        self.learn_b = learn_b
        self.records = deque()
        self.pending = None

    def predict(self, action):
        if self.pending is not None:
            raise RuntimeError('Receive the pending reading before predicting again')
        A, B = normalise_columns(self.a), normalise_columns(self.b)
        transition = B[:, :, action].copy()
        previous = self.q.copy()
        predicted = transition @ previous
        self.pending = dict(action=action, transition=transition,
                            previous=previous, observation_model=A.copy())
        return predicted, A @ predicted

    def see_and_update(self, observation):
        if self.pending is None:
            raise RuntimeError('Predict before receiving a reading')
        record = self.pending
        self.pending = None
        record['observation'] = int(observation)
        record['likelihood'] = record.pop('observation_model')[observation].copy()
        joint = record['transition'] * record['previous'][None, :]
        joint *= record['likelihood'][:, None]
        joint /= joint.sum()
        self.q = joint.sum(axis=1)
        if len(self.records) == self.window:
            old = self.records.popleft()
            if self.learn_a:
                self.frozen_a[old['observation']] += old['state_counts']
            if self.learn_b:
                self.frozen_b[:, :, old['action']] += old['transition_counts']
        self.records.append(record)
        revised = smooth_records(list(self.records))
        self.a = self.frozen_a.copy()
        self.b = self.frozen_b.copy()
        for item, (states, transitions) in zip(self.records, revised):
            item['state_counts'] = states
            item['transition_counts'] = transitions
            if self.learn_a:
                self.a[item['observation']] += states
            if self.learn_b:
                self.b[:, :, item['action']] += transitions
        return self.q.copy()
