"""
Tabular Q-Learning agent (off-policy TD control).

Update rule (Watkins, 1989):

    Q(s, a) ← Q(s, a) + α · [ r + γ · max_a' Q(s', a')  −  Q(s, a) ]

The agent bootstraps its TD target using the *greedy* action at the next
state (max over a'), so it learns the optimal action-value function
regardless of the behaviour policy used to collect data.
"""
from collections import defaultdict
import numpy as np


class QLearningAgent:
    """Tabular Q-Learning with linear ε-decay."""

    def __init__(self, n_actions=5, alpha=0.30, gamma=0.95,
                 eps=1.0, eps_min=0.06, eps_decay_eps=200):
        self.n_actions = n_actions
        self.Q       = defaultdict(lambda: np.zeros(n_actions))
        self.alpha   = alpha
        self.gamma   = gamma
        self.epsilon = eps
        self.eps_min = eps_min
        self.eps_dec = (eps - eps_min) / max(eps_decay_eps, 1)
        self.episode = 0
        self.best    = 0

    def act(self, state):
        """ε-greedy action selection over Q[state]."""
        if np.random.random() < self.epsilon:
            return int(np.random.randint(self.n_actions))
        return int(np.argmax(self.Q[state]))

    def learn(self, s, a, r, ns, done):
        """Off-policy TD(0) update with the greedy max bootstrap."""
        td_target = r + (0.0 if done else self.gamma * np.max(self.Q[ns]))
        self.Q[s][a] += self.alpha * (td_target - self.Q[s][a])

    def on_done(self, raw_score):
        """End-of-episode bookkeeping: anneal ε, update best score."""
        self.epsilon = max(self.eps_min, self.epsilon - self.eps_dec)
        self.best    = max(self.best, raw_score)
        self.episode += 1
