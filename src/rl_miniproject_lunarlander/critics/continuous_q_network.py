import torch
import torch.nn as nn
from torch import Tensor
from rl_mind.nn import build_mlp


class ContinuousQNetwork(nn.Module):
    """The Q-network $Q(s, a)$ for continuous actions"""

    def __init__(self, obs_dim: int, hidden: tuple[int, ...], action_dim: int):
        super().__init__()
        self.model = build_mlp([obs_dim + action_dim, *hidden, 1])

    def forward(self, obs: Tensor, action: Tensor) -> Tensor:
        """Compute $Q(s, a)$ for a batch: `[B, obs_dim] x [B, action_dim] -> [B]`"""
        output: Tensor = self.model(torch.cat([obs, action], dim=1))
        return output.squeeze(-1)
