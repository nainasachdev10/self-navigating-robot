import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from typing import Dict, Optional, Tuple
import copy

from utils.config import DQNConfig, EnvConfig
from models.networks import build_network
from agents.replay_buffer import PrioritizedReplayBuffer, ReplayBuffer


def obs_to_tensors(obs: Dict, device: torch.device) -> Tuple[torch.Tensor, torch.Tensor]:
    local_t = torch.FloatTensor(obs["local"]).unsqueeze(0).to(device)
    global_t = torch.FloatTensor(obs["global"]).unsqueeze(0).to(device)
    return local_t, global_t


def batch_to_tensors(batch, device):
    """Unpack a list of (obs, action, reward, next_obs, done) transitions."""
    local_obs   = torch.FloatTensor(np.stack([b[0]["local"] for b in batch])).to(device)
    global_obs  = torch.FloatTensor(np.stack([b[0]["global"] for b in batch])).to(device)
    actions     = torch.LongTensor([b[1] for b in batch]).to(device)
    rewards     = torch.FloatTensor([b[2] for b in batch]).to(device)
    local_next  = torch.FloatTensor(np.stack([b[3]["local"] for b in batch])).to(device)
    global_next = torch.FloatTensor(np.stack([b[3]["global"] for b in batch])).to(device)
    dones       = torch.FloatTensor([b[4] for b in batch]).to(device)
    return local_obs, global_obs, actions, rewards, local_next, global_next, dones


class DQNAgent:
    """
    Unified DQN / Double DQN / Dueling DQN agent.
    Supports Prioritized Experience Replay and soft target updates.
    """

    def __init__(self, env_cfg: EnvConfig, dqn_cfg: DQNConfig,
                 agent_type: str = "dueling_ddqn",
                 device: Optional[torch.device] = None):
        self.cfg = dqn_cfg
        self.agent_type = agent_type
        self.device = device or (
            torch.device("cuda") if torch.cuda.is_available()
            else torch.device("mps") if torch.backends.mps.is_available()
            else torch.device("cpu")
        )

        from environment.city_env import CityDeliveryEnv
        fov = env_cfg.fov_radius
        fov_size = fov * 2 + 1
        local_ch   = CityDeliveryEnv.FOV_CHANNELS
        global_dim = CityDeliveryEnv.GLOBAL_DIM
        n_actions  = CityDeliveryEnv.ACTIONS

        self.online_net = build_network(
            agent_type, local_ch, fov_size, global_dim,
            n_actions, dqn_cfg.hidden_dims
        ).to(self.device)

        self.target_net = copy.deepcopy(self.online_net).to(self.device)
        self.target_net.eval()

        self.optimizer = optim.Adam(self.online_net.parameters(), lr=dqn_cfg.lr)
        self.loss_fn = nn.SmoothL1Loss(reduction="none")  # Huber loss

        if dqn_cfg.use_per:
            self.buffer = PrioritizedReplayBuffer(
                dqn_cfg.buffer_size, dqn_cfg.per_alpha,
                dqn_cfg.per_beta_start, dqn_cfg.per_beta_end,
                dqn_cfg.per_beta_steps, dqn_cfg.per_epsilon,
            )
        else:
            self.buffer = ReplayBuffer(dqn_cfg.buffer_size)

        self.steps_done = 0
        self.updates_done = 0
        self._epsilon = dqn_cfg.epsilon_start
        self._eps_decay = (
            (dqn_cfg.epsilon_start - dqn_cfg.epsilon_end)
            / dqn_cfg.epsilon_decay_steps
        )

    # ------------------------------------------------------------------
    # Action selection
    # ------------------------------------------------------------------

    @property
    def epsilon(self) -> float:
        return self._epsilon

    def select_action(self, obs: Dict, greedy: bool = False) -> int:
        self.steps_done += 1
        self._epsilon = max(
            self.cfg.epsilon_end,
            self._epsilon - self._eps_decay,
        )

        if not greedy and np.random.random() < self._epsilon:
            return np.random.randint(5)

        local_t, global_t = obs_to_tensors(obs, self.device)
        with torch.no_grad():
            q = self.online_net(local_t, global_t)
        return int(q.argmax(dim=1).item())

    # ------------------------------------------------------------------
    # Learning
    # ------------------------------------------------------------------

    def store(self, obs, action, reward, next_obs, done):
        self.buffer.push((obs, action, reward, next_obs, done))

    def learn(self) -> Optional[float]:
        if len(self.buffer) < self.cfg.batch_size:
            return None

        batch, weights, indices = self.buffer.sample(self.cfg.batch_size)
        (local_obs, global_obs, actions, rewards,
         local_next, global_next, dones) = batch_to_tensors(batch, self.device)

        # Compute current Q(s,a)
        q_values = self.online_net(local_obs, global_obs)
        q_sa = q_values.gather(1, actions.unsqueeze(1)).squeeze(1)

        # Compute target Q
        with torch.no_grad():
            if self.agent_type in ("ddqn", "dueling_ddqn"):
                # Double DQN: online selects action, target evaluates
                next_actions = self.online_net(local_next, global_next).argmax(1)
                next_q = self.target_net(local_next, global_next)
                next_q_sa = next_q.gather(1, next_actions.unsqueeze(1)).squeeze(1)
            else:
                next_q_sa = self.target_net(local_next, global_next).max(1)[0]

            targets = rewards + self.cfg.gamma * next_q_sa * (1 - dones)

        td_errors = (targets - q_sa).detach().cpu().numpy()
        element_loss = self.loss_fn(q_sa, targets)

        if weights is not None:
            w = torch.FloatTensor(weights).to(self.device)
            loss = (element_loss * w).mean()
        else:
            loss = element_loss.mean()

        self.optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.online_net.parameters(), self.cfg.grad_clip)
        self.optimizer.step()

        self.buffer.update_priorities(indices, td_errors)

        self.updates_done += 1
        self._update_target()

        return loss.item()

    def _update_target(self):
        if self.cfg.use_soft_update:
            tau = self.cfg.tau
            for online_p, target_p in zip(
                self.online_net.parameters(), self.target_net.parameters()
            ):
                target_p.data.copy_(tau * online_p.data + (1 - tau) * target_p.data)
        else:
            if self.updates_done % self.cfg.target_update_freq == 0:
                self.target_net.load_state_dict(self.online_net.state_dict())

    # ------------------------------------------------------------------
    # Save / load
    # ------------------------------------------------------------------

    def save(self, path: str):
        torch.save({
            "online_net": self.online_net.state_dict(),
            "target_net": self.target_net.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "steps_done": self.steps_done,
            "updates_done": self.updates_done,
            "epsilon": self._epsilon,
        }, path)

    def load(self, path: str):
        ckpt = torch.load(path, map_location=self.device)
        self.online_net.load_state_dict(ckpt["online_net"])
        self.target_net.load_state_dict(ckpt["target_net"])
        self.optimizer.load_state_dict(ckpt["optimizer"])
        self.steps_done = ckpt["steps_done"]
        self.updates_done = ckpt["updates_done"]
        self._epsilon = ckpt["epsilon"]
