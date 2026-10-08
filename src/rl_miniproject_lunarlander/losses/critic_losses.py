import torch
from torch import Tensor
import torch.nn.functional as F
from rl_mind.data import Transitions
from rl_mind.core import Action
import torch.nn.functional as F

def compute_critic_loss(
    gamma: float, batch: Transitions[Action], q_values: Tensor, next_q_values: Tensor
) -> Tensor:
    """Compute the DDPG critic loss from a batch of transitions

    :param gamma: The discount factor
    :param batch: The batch of transitions
    :param q_values: $Q(s_t, a_t)$ from the critic (shape `[B]`)
    :param next_q_values: $Q'(s_{t+1}, \\pi(s_{t+1}))$ from the target critic
        (shape `[B]`)
    :return: The critic loss (a scalar)
    """
    # Compute the target (do not bootstrap when `batch.terminated`), then the MSE loss
    with torch.no_grad():
        target = batch.reward + gamma * ~batch.terminated * next_q_values
    return F.mse_loss(q_values, target)