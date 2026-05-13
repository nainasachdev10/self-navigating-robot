import typing
import yaml
from dataclasses import dataclass, field
from typing import Tuple, List, Any, Dict, get_origin, get_args


@dataclass
class EnvConfig:
    grid_size: Tuple[int, int] = (10, 10)
    num_deliveries: int = 1
    max_steps: int = 100
    fov_radius: int = 2              # partial observability field of view
    max_battery: int = 200
    battery_per_step: int = 1
    battery_per_traffic: int = 3
    num_traffic_zones: int = 3
    traffic_transition_prob: float = 0.05
    rush_hour_steps: List[int] = field(default_factory=list)
    rush_hour_duration: int = 30
    wall_density: float = 0.08   # fraction of interior cells that are walls

    # Rewards
    reward_delivery: float = 50.0
    reward_in_time: float = 30.0
    penalty_traffic: float = -10.0
    penalty_step: float = -1.0
    penalty_timeout: float = -100.0
    penalty_dead_battery: float = -80.0
    reward_battery_pickup: float = 15.0


@dataclass
class DQNConfig:
    hidden_dims: List[int] = field(default_factory=lambda: [128, 128])
    lr: float = 1e-3
    gamma: float = 0.99
    epsilon_start: float = 1.0
    epsilon_end: float = 0.05
    epsilon_decay_steps: int = 10_000
    batch_size: int = 32
    buffer_size: int = 10_000
    target_update_freq: int = 200       # hard update every N steps
    tau: float = 0.005
    use_soft_update: bool = False
    grad_clip: float = 10.0
    use_per: bool = False
    per_alpha: float = 0.6
    per_beta_start: float = 0.4
    per_beta_end: float = 1.0
    per_beta_steps: int = 100_000
    per_epsilon: float = 1e-6


@dataclass
class TrainingConfig:
    total_episodes: int = 500
    eval_every: int = 50
    eval_episodes: int = 10
    save_every: int = 100
    log_dir: str = "logs"
    checkpoint_dir: str = "checkpoints"
    curriculum: bool = False
    curriculum_thresholds: List[float] = field(default_factory=lambda: [0.4, 0.65, 0.80])
    seed: int = 42


@dataclass
class ProjectConfig:
    env: EnvConfig = field(default_factory=EnvConfig)
    dqn: DQNConfig = field(default_factory=DQNConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    agent_type: str = "dqn"

    @staticmethod
    def _coerce_section(dc_cls: type, raw: Dict[str, Any]) -> Dict[str, Any]:
        """PyYAML 1.1 can parse scientific notation (e.g. 3e-4) as strings; fix types."""
        if not raw:
            return {}
        hints = typing.get_type_hints(dc_cls)
        out: Dict[str, Any] = {}
        for k, v in raw.items():
            if k not in hints:
                out[k] = v
                continue
            t = hints[k]
            origin = get_origin(t)
            if t is float and isinstance(v, str):
                v = float(v)
            elif t is int and isinstance(v, str):
                v = int(float(v))
            elif origin is tuple and isinstance(v, (list, tuple)):
                args = get_args(t)
                if args and all(a is int for a in args):
                    v = tuple(int(float(x)) if isinstance(x, str) else int(x) for x in v)
            elif origin is list and isinstance(v, list):
                args = get_args(t)
                if args and args[0] is float:
                    v = [float(x) if isinstance(x, str) else x for x in v]
                elif args and args[0] is int:
                    v = [int(float(x)) if isinstance(x, str) else int(x) for x in v]
            out[k] = v
        return out

    @classmethod
    def from_yaml(cls, path: str) -> "ProjectConfig":
        with open(path) as f:
            data = yaml.safe_load(f)
        cfg = cls()
        if "env" in data:
            cfg.env = EnvConfig(**cls._coerce_section(EnvConfig, data["env"]))
        if "dqn" in data:
            cfg.dqn = DQNConfig(**cls._coerce_section(DQNConfig, data["dqn"]))
        if "training" in data:
            cfg.training = TrainingConfig(
                **cls._coerce_section(TrainingConfig, data["training"])
            )
        if "agent_type" in data:
            cfg.agent_type = data["agent_type"]
        return cfg
