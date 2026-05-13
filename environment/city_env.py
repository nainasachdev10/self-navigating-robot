import numpy as np
import gymnasium as gym
from gymnasium import spaces
from typing import Optional, Dict, Any, List, Tuple
from dataclasses import dataclass

from utils.config import EnvConfig
from environment.traffic_manager import TrafficManager


@dataclass
class DeliveryTask:
    destination: Tuple[int, int]
    priority: int        # 1=normal, 2=express, 3=urgent
    time_limit: int      # steps remaining before this delivery expires
    reward_multiplier: float


class CityDeliveryEnv(gym.Env):
    """
    Advanced city delivery environment.

    State (observation):
        - Local FOV grid patch (fov_radius*2+1 x fov_radius*2+1 x channels)
          Channels: [wall, congestion_level, delivery_targets, battery_stations, agent]
        - Global context vector:
          [agent_r/H, agent_c/W, battery/max, steps_left/max,
           next_dest_r/H, next_dest_c/W, next_dest_priority/3,
           deliveries_remaining/num_deliveries, in_rush_hour]

    Actions: 0=up, 1=down, 2=left, 3=right, 4=stay

    Difficulty levels (curriculum):
        0: 10x10 grid, 1 delivery, no rush hour, fewer traffic zones
        1: 12x12 grid, 2 deliveries, 1 rush hour
        2: 15x15 grid, 3 deliveries, 2 rush hours
        3: 15x15 grid, 3 deliveries, dynamic incidents, tighter time
    """

    metadata = {"render_modes": ["human", "rgb_array"]}
    FOV_CHANNELS = 5
    GLOBAL_DIM = 11   # added: relative row & col direction to goal
    ACTIONS = 5

    def __init__(self, cfg: Optional[EnvConfig] = None, difficulty: int = 2,
                 render_mode: Optional[str] = None, seed: int = 42):
        super().__init__()
        self.base_cfg = cfg or EnvConfig()
        self.difficulty = difficulty
        self.render_mode = render_mode
        self._rng = np.random.default_rng(seed)

        self.cfg = self._apply_difficulty(self.base_cfg, difficulty)
        self.grid_h, self.grid_w = self.cfg.grid_size
        self.traffic = TrafficManager(self.cfg, self._rng)

        fov = self.cfg.fov_radius * 2 + 1
        self.observation_space = spaces.Dict({
            "local": spaces.Box(
                low=0.0, high=1.0,
                shape=(self.FOV_CHANNELS, fov, fov),
                dtype=np.float32
            ),
            "global": spaces.Box(
                low=0.0, high=1.0,
                shape=(self.GLOBAL_DIM,),
                dtype=np.float32
            ),
        })
        self.action_space = spaces.Discrete(self.ACTIONS)

        # State variables (initialized in reset)
        self.agent_pos: Tuple[int, int] = (0, 0)
        self.battery: int = 0
        self.step_count: int = 0
        self.delivery_queue: List[DeliveryTask] = []
        self.completed_deliveries: int = 0
        self.battery_stations: List[Tuple[int, int]] = []
        self._walls: set = set()
        self._renderer = None
        self._last_action = None
        self._last_reward = None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def reset(self, *, seed=None, options=None):
        if seed is not None:
            self._rng = np.random.default_rng(seed)

        self.cfg = self._apply_difficulty(self.base_cfg, self.difficulty)
        self.grid_h, self.grid_w = self.cfg.grid_size
        self.traffic = TrafficManager(self.cfg, self._rng)

        self._walls = self._generate_walls()
        self._wall_grid = np.zeros((self.grid_h, self.grid_w), dtype=np.float32)
        for (wr, wc) in self._walls:
            if 0 <= wr < self.grid_h and 0 <= wc < self.grid_w:
                self._wall_grid[wr, wc] = 1.0
        self.battery_stations = self._place_items(2, self._walls)
        self.agent_pos = self._random_free_cell(self._walls | set(self.battery_stations))
        self.battery = self.cfg.max_battery
        self.step_count = 0
        self.completed_deliveries = 0

        blocked = self._walls | set(self.battery_stations) | {self.agent_pos}
        self.traffic.reset(blocked)

        self.delivery_queue = self._generate_deliveries(blocked)
        return self._get_obs(), self._get_info()

    def step(self, action: int):
        assert self.action_space.contains(action)
        self._last_action = action
        self.step_count += 1
        self.traffic.step(self.step_count)

        # Move agent
        new_pos = self._apply_action(action)
        reward = self.cfg.penalty_step

        # Traffic penalty + battery drain
        r, c = new_pos
        traffic_penalty = self.traffic.penalty_at(r, c)
        battery_cost = self.traffic.battery_cost_at(r, c)
        reward += traffic_penalty
        self.battery = max(0, self.battery - battery_cost)

        self.agent_pos = new_pos

        # Battery station pickup
        if self.agent_pos in self.battery_stations:
            recharge = min(self.cfg.max_battery - self.battery,
                           self.cfg.max_battery // 2)
            self.battery += recharge
            if recharge > 0:
                reward += self.cfg.reward_battery_pickup

        # Delivery check
        if self.delivery_queue:
            task = self.delivery_queue[0]
            if self.agent_pos == task.destination:
                base = self.cfg.reward_delivery * task.reward_multiplier
                reward += base
                if task.time_limit > 0:
                    reward += self.cfg.reward_in_time * task.reward_multiplier
                self.delivery_queue.pop(0)
                self.completed_deliveries += 1

        # Tick delivery time limits
        for task in self.delivery_queue:
            task.time_limit -= 1

        # Termination conditions
        terminated = len(self.delivery_queue) == 0
        truncated = (
            self.step_count >= self.cfg.max_steps
            or self.battery <= 0
        )

        if truncated and self.battery <= 0:
            reward += self.cfg.penalty_dead_battery
        elif truncated and self.delivery_queue:
            reward += self.cfg.penalty_timeout * len(self.delivery_queue)

        self._last_reward = reward
        obs = self._get_obs()
        info = self._get_info()

        if self.render_mode == "human":
            self.render()
            if (terminated or truncated) and self._renderer is not None:
                if hasattr(self._renderer, "on_episode_end"):
                    self._renderer.on_episode_end(self)

        return obs, reward, terminated, truncated, info

    def render(self):
        if self._renderer is None:
            try:
                import pygame
                from environment.renderer import CityRenderer
                self._renderer = CityRenderer(self.cfg)
            except Exception:
                from environment.renderer_mpl import MatplotlibRenderer
                self._renderer = MatplotlibRenderer(self.cfg)
        return self._renderer.render(self,
                                     action=self._last_action,
                                     reward=self._last_reward)

    def close(self):
        if self._renderer is not None:
            self._renderer.close()

    def set_difficulty(self, difficulty: int):
        self.difficulty = difficulty

    # ------------------------------------------------------------------
    # Observation construction
    # ------------------------------------------------------------------

    def _get_obs(self) -> Dict[str, np.ndarray]:
        local = self._build_local_obs()
        global_vec = self._build_global_obs()
        return {"local": local, "global": global_vec}

    def _build_local_obs(self) -> np.ndarray:
        fov  = self.cfg.fov_radius
        size = fov * 2 + 1
        obs  = np.zeros((self.FOV_CHANNELS, size, size), dtype=np.float32)
        ar, ac = self.agent_pos

        # Row/col indices into the full grid for every FOV cell
        rows = np.arange(ar - fov, ar + fov + 1)
        cols = np.arange(ac - fov, ac + fov + 1)
        rr, cc = np.meshgrid(rows, cols, indexing="ij")  # (size, size)

        # Mask valid (in-bounds) cells
        valid = (rr >= 0) & (rr < self.grid_h) & (cc >= 0) & (cc < self.grid_w)
        obs[0][~valid] = 1.0                              # out-of-bounds = wall

        vr, vc = rr[valid], cc[valid]                     # flat valid indices
        pi, pj = np.where(valid)

        # Channel 0: walls
        obs[0, pi, pj] = self._wall_grid[vr, vc]

        # Channel 1: traffic congestion
        obs[1, pi, pj] = self.traffic.get_grid()[vr, vc] / 3.0

        # Channel 2: delivery targets
        for i, task in enumerate(self.delivery_queue[:3]):
            dr, dc = task.destination
            mask = (vr == dr) & (vc == dc)
            obs[2, pi[mask], pj[mask]] = (i + 1) / 3.0

        # Channel 3: battery stations
        for (sr, sc) in self.battery_stations:
            mask = (vr == sr) & (vc == sc)
            obs[3, pi[mask], pj[mask]] = 1.0

        # Channel 4: agent (center)
        obs[4, fov, fov] = 1.0
        return obs

    def _build_global_obs(self) -> np.ndarray:
        ar, ac = self.agent_pos
        vec = np.zeros(self.GLOBAL_DIM, dtype=np.float32)
        vec[0] = ar / (self.grid_h - 1)
        vec[1] = ac / (self.grid_w - 1)
        vec[2] = self.battery / self.cfg.max_battery
        vec[3] = 1.0 - self.step_count / self.cfg.max_steps

        if self.delivery_queue:
            gr, gc = self.delivery_queue[0].destination
            vec[4] = gr / (self.grid_h - 1)
            vec[5] = gc / (self.grid_w - 1)
            vec[6] = self.delivery_queue[0].priority / 3.0
            # Relative direction to goal — key for navigation around walls
            vec[9]  = (gr - ar) / (self.grid_h - 1)   # +ve = goal is below
            vec[10] = (gc - ac) / (self.grid_w - 1)   # +ve = goal is right
        vec[7] = len(self.delivery_queue) / self.cfg.num_deliveries

        in_rush = any(
            rh <= self.step_count <= rh + self.cfg.rush_hour_duration
            for rh in self.cfg.rush_hour_steps
        )
        vec[8] = 1.0 if in_rush else 0.0
        return vec

    def _get_info(self) -> Dict[str, Any]:
        return {
            "step": self.step_count,
            "battery": self.battery,
            "deliveries_done": self.completed_deliveries,
            "deliveries_remaining": len(self.delivery_queue),
            "agent_pos": self.agent_pos,
        }

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _apply_action(self, action: int) -> Tuple[int, int]:
        r, c = self.agent_pos
        deltas = [(-1, 0), (1, 0), (0, -1), (0, 1), (0, 0)]
        dr, dc = deltas[action]
        nr, nc = r + dr, c + dc
        if (0 <= nr < self.grid_h and 0 <= nc < self.grid_w
                and (nr, nc) not in self._walls):
            return (nr, nc)
        return self.agent_pos

    def _generate_walls(self) -> set:
        walls = set()
        # Border walls
        for r in range(self.grid_h):
            walls.add((r, 0)); walls.add((r, self.grid_w - 1))
        for c in range(self.grid_w):
            walls.add((0, c)); walls.add((self.grid_h - 1, c))

        if self.cfg.wall_density <= 0:
            return walls

        # Place blockers — skip any cell that would create a dead-end corridor
        num_blocks = int(self.grid_h * self.grid_w * self.cfg.wall_density)
        interior = [(r, c)
                    for r in range(2, self.grid_h - 2)
                    for c in range(2, self.grid_w - 2)]
        self._rng.shuffle(interior)

        placed = 0
        for (r, c) in interior:
            if placed >= num_blocks:
                break
            # Fast heuristic: don't place if it would leave a neighbour with only 1 free exit
            walls.add((r, c))
            if not self._creates_bottleneck(r, c, walls):
                placed += 1
            else:
                walls.discard((r, c))
        return walls

    def _creates_bottleneck(self, r: int, c: int, walls: set) -> bool:
        """Return True if placing a wall at (r,c) traps any neighbour cell."""
        for dr, dc in [(-1,0),(1,0),(0,-1),(0,1)]:
            nr, nc = r+dr, c+dc
            if (nr, nc) in walls:
                continue
            if not (0 < nr < self.grid_h-1 and 0 < nc < self.grid_w-1):
                continue
            free_exits = sum(
                1 for er, ec in [(-1,0),(1,0),(0,-1),(0,1)]
                if (nr+er, nc+ec) not in walls
                and 0 <= nr+er < self.grid_h
                and 0 <= nc+ec < self.grid_w
            )
            if free_exits <= 1:
                return True
        return False

    def _place_items(self, n: int, avoid: set) -> List[Tuple[int, int]]:
        items = []
        for _ in range(n * 100):
            if len(items) >= n:
                break
            pos = self._random_free_cell(avoid | set(items))
            items.append(pos)
        return items

    def _random_free_cell(self, avoid: set) -> Tuple[int, int]:
        for _ in range(10000):
            r = int(self._rng.integers(1, self.grid_h - 1))
            c = int(self._rng.integers(1, self.grid_w - 1))
            if (r, c) not in avoid:
                return (r, c)
        raise RuntimeError("Could not find free cell")

    def _generate_deliveries(self, blocked: set) -> List[DeliveryTask]:
        tasks = []
        for i in range(self.cfg.num_deliveries):
            dest = self._random_free_cell(blocked | {t.destination for t in tasks})
            priority = int(self._rng.integers(1, 4))
            time_limit = self.cfg.max_steps - self.step_count - i * 20
            tasks.append(DeliveryTask(
                destination=dest,
                priority=priority,
                time_limit=time_limit,
                reward_multiplier=1.0 + (priority - 1) * 0.5,
            ))
        return tasks

    @staticmethod
    def _apply_difficulty(cfg: EnvConfig, difficulty: int) -> EnvConfig:
        import copy
        c = copy.deepcopy(cfg)
        if difficulty == 0:
            c.grid_size = (10, 10)
            c.num_deliveries = 1
            c.num_traffic_zones = 3
            c.max_steps = 150
            c.rush_hour_steps = []
        elif difficulty == 1:
            c.grid_size = (12, 12)
            c.num_deliveries = 2
            c.num_traffic_zones = 5
            c.max_steps = 200
            c.rush_hour_steps = [80]
        elif difficulty == 2:
            pass   # default config
        elif difficulty == 3:
            c.max_steps = 250
            c.traffic_transition_prob = 0.10
            c.num_traffic_zones = 12
        return c
