import torch
from torch import Tensor

def compute_actor_loss(q_values: Tensor) -> Tensor:
    """Return the actor loss given $Q(s_t, \\pi(s_t))$ (shape `[B]`)"""
    # Compute the actor loss

    return -q_values.mean()