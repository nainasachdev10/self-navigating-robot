"""
Tabular SARSA agent (on-policy TD control).

Update rule (Rummery & Niranjan, 1994):

    Q(s, a) ← Q(s, a) + α · [ r + γ · Q(s', a')  −  Q(s, a) ]

where a' is the action *actually taken* at the next state under the same
ε-greedy behaviour policy. This is "on-policy" — the agent learns the
value of the policy it follows (including its exploration), so it tends
to be more conservative around penalty regions than off-policy Q-Learning.
"""
from collections import defaultdict
import numpy as np


class SARSAAgent:
    """Tabular SARSA with linear ε-decay.

    Stores the next ε-greedy action between calls so that `learn()` and
    the subsequent `act()` use the *same* sampled a'."""

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
        self._next_a = None    # a' chosen during learn(), reused on next act()

    def _eps_greedy(self, state):
        if np.random.random() < self.epsilon:
            return int(np.random.randint(self.n_actions))
        return int(np.argmax(self.Q[state]))

    def act(self, state):
        """If learn() already sampled a' for this state, reuse it; else
        sample a fresh ε-greedy action."""
        if self._next_a is not None:
            a = self._next_a
            self._next_a = None
            return a
        return self._eps_greedy(state)

    def learn(self, s, a, r, ns, done):
        """On-policy TD(0) update using the sampled next action a'."""
        if done:
            td_target = r
            self._next_a = None
        else:
            a_next       = self._eps_greedy(ns)
            self._next_a = a_next
            td_target    = r + self.gamma * self.Q[ns][a_next]
        self.Q[s][a] += self.alpha * (td_target - self.Q[s][a])

    def on_done(self, raw_score):
        """End-of-episode: anneal ε, update best score, drop lookahead."""
        self.epsilon  = max(self.eps_min, self.epsilon - self.eps_dec)
        self.best     = max(self.best, raw_score)
        self.episode += 1
        self._next_a  = None
