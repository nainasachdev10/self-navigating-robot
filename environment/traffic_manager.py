import numpy as np
from typing import List, Tuple
from utils.config import EnvConfig


class TrafficManager:
    """
    Manages dynamic traffic zones using a Markov chain model.
    Each zone independently transitions between congestion levels [0..3].
    Rush hours spike transition probabilities toward higher congestion.
    """
    LEVELS = 4  # 0=clear, 1=light, 2=heavy, 3=gridlock

    def __init__(self, cfg: EnvConfig, rng: np.random.Generator):
        self.cfg = cfg
        self.rng = rng
        self.grid_h, self.grid_w = cfg.grid_size

        # Fixed zone positions and their congestion levels
        self.zone_positions: List[Tuple[int, int]] = []
        self.zone_levels: np.ndarray = np.zeros(cfg.num_traffic_zones, dtype=np.int32)
        self._congestion_grid = np.zeros((self.grid_h, self.grid_w), dtype=np.int32)

        # Very stable base transition — traffic barely changes during an episode
        self._base_transition = np.array([
            [0.97, 0.03, 0.00, 0.00],
            [0.05, 0.92, 0.03, 0.00],
            [0.00, 0.05, 0.92, 0.03],
            [0.00, 0.00, 0.05, 0.95],
        ])
        self._rush_transition = np.array([
            [0.80, 0.15, 0.05, 0.00],
            [0.05, 0.75, 0.15, 0.05],
            [0.00, 0.10, 0.75, 0.15],
            [0.00, 0.00, 0.10, 0.90],
        ])

    def reset(self, blocked_cells: set):
        """Place zones randomly, avoiding blocked cells and borders."""
        self.zone_positions = []
        attempts = 0
        while len(self.zone_positions) < self.cfg.num_traffic_zones and attempts < 1000:
            r = self.rng.integers(1, self.grid_h - 1)
            c = self.rng.integers(1, self.grid_w - 1)
            if (r, c) not in blocked_cells and (r, c) not in self.zone_positions:
                self.zone_positions.append((r, c))
            attempts += 1

        # Start zones at random congestion levels (skewed toward lower)
        self.zone_levels = self.rng.choice(
            self.LEVELS, size=len(self.zone_positions), p=[0.5, 0.3, 0.15, 0.05]
        )
        self._rebuild_grid()

    def step(self, current_step: int):
        """Transition traffic only every 8 steps so the map stays readable."""
        if current_step % 8 != 0:
            return

        in_rush = any(
            rh <= current_step <= rh + self.cfg.rush_hour_duration
            for rh in self.cfg.rush_hour_steps
        )
        matrix = self._rush_transition if in_rush else self._base_transition

        for i, lvl in enumerate(self.zone_levels):
            self.zone_levels[i] = self.rng.choice(self.LEVELS, p=matrix[lvl])

        self._rebuild_grid()

    def _rebuild_grid(self):
        self._congestion_grid[:] = 0
        for (r, c), lvl in zip(self.zone_positions, self.zone_levels):
            r0 = max(0, r-1); r1 = min(self.grid_h, r+2)
            c0 = max(0, c-1); c1 = min(self.grid_w, c+2)
            bleed = max(0, lvl - 1)
            self._congestion_grid[r0:r1, c0:c1] = np.maximum(
                self._congestion_grid[r0:r1, c0:c1], bleed)
            if 0 <= r < self.grid_h and 0 <= c < self.grid_w:
                self._congestion_grid[r, c] = max(self._congestion_grid[r, c], lvl)

    def congestion_at(self, r: int, c: int) -> int:
        return int(self._congestion_grid[r, c])

    def get_grid(self) -> np.ndarray:
        return self._congestion_grid.copy()

    def penalty_at(self, r: int, c: int) -> float:
        lvl = self.congestion_at(r, c)
        return [0.0, -4.0, -10.0, -20.0][lvl]

    def battery_cost_at(self, r: int, c: int) -> int:
        lvl = self.congestion_at(r, c)
        return [1, 2, 3, 5][lvl]
