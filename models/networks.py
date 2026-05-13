import torch
import torch.nn as nn
from typing import List


class FlatDQN(nn.Module):
    """
    Fast flat MLP: concatenate flattened FOV + global vector → Q-values.
    No CNN — 8-10x faster training than a convolutional encoder.
    """

    def __init__(self, local_channels: int, fov_size: int,
                 global_dim: int, n_actions: int,
                 hidden_dims: List[int]):
        super().__init__()
        flat_dim = local_channels * fov_size * fov_size + global_dim

        layers: list = [nn.Flatten()]
        # local flatten happens outside; we cat then pass through MLP
        # Build MLP
        mlp = []
        in_dim = flat_dim
        for h in hidden_dims:
            mlp += [nn.Linear(in_dim, h), nn.ReLU()]
            in_dim = h
        mlp.append(nn.Linear(in_dim, n_actions))
        self.net = nn.Sequential(*mlp)
        self._flat_local = nn.Flatten()

    def forward(self, local_obs: torch.Tensor, global_obs: torch.Tensor) -> torch.Tensor:
        flat_local = self._flat_local(local_obs)
        x = torch.cat([flat_local, global_obs], dim=1)
        return self.net(x)


def build_network(agent_type: str, local_channels: int, fov_size: int,
                  global_dim: int, n_actions: int,
                  hidden_dims: List[int]) -> nn.Module:
    # All variants use the same fast flat MLP for now
    return FlatDQN(local_channels, fov_size, global_dim, n_actions, hidden_dims)
