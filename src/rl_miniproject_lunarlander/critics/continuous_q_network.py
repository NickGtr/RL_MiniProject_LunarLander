import torch
import torch.nn as nn
from torch import Tensor
from rl_mind.nn import build_mlp


class ContinuousQNetwork(nn.Module):
    """The Q-network $Q(s, a)$ for continuous actions"""

    def __init__(self, obs_dim: int, hidden: tuple[int, ...], action_dim: int, layer_norm: bool = False):
        """:param layer_norm: Add a LayerNorm after each hidden linear layer (before the activation)"""
        super().__init__()
        sizes = [obs_dim + action_dim, *hidden, 1]
        if not layer_norm:
            self.model = build_mlp(sizes)
        else:
            layers: list[nn.Module] = []
            for n_in, n_out in zip(sizes[:-2], sizes[1:-1]):
                layers += [nn.Linear(n_in, n_out), nn.LayerNorm(n_out), nn.ReLU()]
            layers.append(nn.Linear(sizes[-2], sizes[-1]))
            self.model = nn.Sequential(*layers)

    def forward(self, obs: Tensor, action: Tensor) -> Tensor:
        """Compute $Q(s, a)$ for a batch: `[B, obs_dim] x [B, action_dim] -> [B]`"""
        output: Tensor = self.model(torch.cat([obs, action], dim=1))
        return output.squeeze(-1)
