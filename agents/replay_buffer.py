import numpy as np
from typing import Tuple, Dict
import torch


class ReplayBuffer:
    """Uniform experience replay buffer."""

    def __init__(self, capacity: int):
        self.capacity = capacity
        self._buf: list = []
        self._pos = 0

    def push(self, transition: tuple):
        if len(self._buf) < self.capacity:
            self._buf.append(transition)
        else:
            self._buf[self._pos] = transition
        self._pos = (self._pos + 1) % self.capacity

    def sample(self, batch_size: int) -> list:
        indices = np.random.randint(0, len(self._buf), size=batch_size)
        return [self._buf[i] for i in indices], None, indices

    def update_priorities(self, indices, priorities):
        pass   # no-op for uniform buffer

    def __len__(self):
        return len(self._buf)


class SumTree:
    """Binary sum tree for O(log n) priority sampling."""

    def __init__(self, capacity: int):
        self.capacity = capacity
        self.tree = np.zeros(2 * capacity, dtype=np.float64)
        self.data: list = [None] * capacity
        self._pos = 0
        self._size = 0

    def add(self, priority: float, data):
        idx = self._pos + self.capacity
        self.data[self._pos] = data
        self.update(idx, priority)
        self._pos = (self._pos + 1) % self.capacity
        self._size = min(self._size + 1, self.capacity)

    def update(self, idx: int, priority: float):
        delta = priority - self.tree[idx]
        self.tree[idx] = priority
        while idx > 1:
            idx //= 2
            self.tree[idx] += delta

    def get(self, value: float) -> Tuple[int, float, object]:
        idx = 1
        while idx < self.capacity:
            left = 2 * idx
            if value <= self.tree[left]:
                idx = left
            else:
                value -= self.tree[left]
                idx = left + 1
        data_idx = idx - self.capacity
        return idx, self.tree[idx], self.data[data_idx]

    @property
    def total(self) -> float:
        return self.tree[1]

    @property
    def max_priority(self) -> float:
        return self.tree[self.capacity: self.capacity + self._size].max()

    def __len__(self):
        return self._size


class PrioritizedReplayBuffer:
    """
    Prioritized Experience Replay (PER).
    Samples transitions with probability proportional to |TD error|^alpha.
    Importance-sampling weights correct for the bias.
    """

    def __init__(self, capacity: int, alpha: float = 0.6,
                 beta: float = 0.4, beta_end: float = 1.0,
                 beta_steps: int = 100_000, epsilon: float = 1e-6):
        self.tree = SumTree(capacity)
        self.alpha = alpha
        self.beta = beta
        self.beta_end = beta_end
        self.beta_increment = (beta_end - beta) / beta_steps
        self.epsilon = epsilon
        self.max_priority = 1.0

    def push(self, transition: tuple):
        self.tree.add(self.max_priority, transition)

    def sample(self, batch_size: int) -> Tuple[list, np.ndarray, np.ndarray]:
        self.beta = min(self.beta_end, self.beta + self.beta_increment)

        segment = self.tree.total / batch_size
        indices = np.zeros(batch_size, dtype=np.int64)
        priorities = np.zeros(batch_size, dtype=np.float64)
        samples = []

        for i in range(batch_size):
            lo, hi = segment * i, segment * (i + 1)
            val = np.random.uniform(lo, hi)
            idx, priority, data = self.tree.get(val)
            indices[i] = idx
            priorities[i] = priority
            samples.append(data)

        # Importance-sampling weights
        prob = priorities / self.tree.total
        weights = (len(self.tree) * prob) ** (-self.beta)
        weights /= weights.max()
        return samples, weights.astype(np.float32), indices

    def update_priorities(self, indices: np.ndarray, td_errors: np.ndarray):
        priorities = (np.abs(td_errors) + self.epsilon) ** self.alpha
        self.max_priority = max(self.max_priority, priorities.max())
        for idx, p in zip(indices, priorities):
            self.tree.update(int(idx), float(p))

    def __len__(self):
        return len(self.tree)
