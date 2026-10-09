import torch
import torch.nn as nn
from torch import Tensor
from rl_mind.core import Action, Actor
from rl_mind.nn import build_mlp

class ContinuousDeterministicActor(Actor[Action]):
    def __init__(self, obs_dim: int, hidden: tuple[int, ...], action_dim: int):
        """Creates a new actor with continuous actions in `action_dim` dimensions.

        :param obs_dim: The dimension of the observation space
        :param hidden: A tuple of integers specifying the sizes of the hidden layers
        :param action_dim: The dimension of the action space
        """
        super().__init__()
        self.model = build_mlp(
            [obs_dim, *hidden, action_dim], output_activation=nn.Tanh()
        )

    def forward(self, obs: Tensor) -> Action:
        return Action(value=self.model(obs))

class GaussianNoise(Actor[Action]):
    """Adds Gaussian noise to the actions of another actor (at training time)"""

    def __init__(self, actor: Actor[Action], sigma: float):
        super().__init__()
        self.actor = actor
        self.sigma = sigma

    def forward(self, obs: Tensor) -> Action:
        action = self.actor(obs).value
        noisy_action = action + self.sigma * torch.randn_like(action)
        noisy_action = noisy_action.clamp(-1, 1)
        return Action(value=noisy_action)

    def act(self, obs: Tensor) -> Tensor:
        return self.actor.act(obs)

