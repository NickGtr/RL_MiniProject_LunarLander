import copy
from pathlib import Path
import torch
import torch.nn as nn
import torch.nn.functional as F

from torch import Tensor
from torch.utils.tensorboard import SummaryWriter
from tqdm.auto import tqdm
from .config import TD3Config

import rl_mind.envs
from rl_mind.core import Action, Actor
from rl_mind.nn import soft_update
from rl_mind.env import VecEnv
from rl_mind.data import ReplayBuffer, Transitions
from rl_mind.collectors import TransitionCollector
from rl_mind.evaluation import Evaluator
from rl_mind.notebook import run_directory

from ..actors.continuous_actor import ContinuousDeterministicActor, GaussianNoise
from ..critics.continuous_q_network import ContinuousQNetwork
from ..losses.actor_losses import compute_actor_loss
from ..losses.critic_losses import compute_critic_loss

def run_td3(cfg: TD3Config, run_dir: Path | None = None) -> Evaluator:
    torch.manual_seed(cfg.seed)
    env = VecEnv(cfg.env_name, cfg.n_envs, seed=cfg.seed, **cfg.env_kwargs)

    # Create the actor, the two critics, their three targets and the optimizers
    actor = ContinuousDeterministicActor(env.observation_dim, cfg.actor_hidden, env.action_dim)

    target_actor = copy.deepcopy(actor)
    critic_1 = ContinuousQNetwork(env.observation_dim, cfg.critic_hidden, env.action_dim)
    critic_2 = copy.deepcopy(critic_1)
    target_critic_1 = copy.deepcopy(critic_1)
    target_critic_2 = copy.deepcopy(critic_1)
    # one optimizer for the actor, one for both critics (give it the
    # parameters of both)
    actor_optimizer = torch.optim.Adam(actor.parameters(), lr=cfg.lr_actor)
    critic_optimizer = torch.optim.Adam(
                        list(critic_1.parameters()) + list(critic_2.parameters()),
                        lr=cfg.lr_critic
                                        )

    # Data collection (with exploration noise), replay buffer and evaluation
    collector = TransitionCollector(env, GaussianNoise(actor, cfg.action_noise))
    buffer = ReplayBuffer(cfg.buffer_size)
    if run_dir is None:
        run_dir = run_directory(f"td3-{cfg.env_name}-S{cfg.seed}")
    evaluator = Evaluator(
        VecEnv(cfg.env_name, cfg.n_eval_envs, seed=cfg.seed + 100, **cfg.env_kwargs),
        every=cfg.eval_interval,
        run_dir=run_dir,
        writer=SummaryWriter(run_dir),
    )

    updates = 0  # number of gradient steps (for the policy delay)
    pbar = tqdm(total=cfg.max_steps)
    while collector.steps < cfg.max_steps:
        buffer.add(collector.collect(cfg.steps_per_update))
        pbar.update(collector.steps - pbar.n)
        if len(buffer) < cfg.learning_starts:
            continue

        batch = buffer.sample(cfg.batch_size)

        with torch.no_grad():
            # Target policy smoothing: target actor + clipped Gaussian noise, clipped to [-1, 1]

            next_actions = target_actor(batch.next_obs).value
            next_actions += (cfg.target_noise * torch.randn_like(next_actions)).clamp(-cfg.target_noise_clip, cfg.target_noise_clip)

            # Clipped double-Q: a single target, from the min of the two target critics
            next_q_values_1 = target_critic_1(batch.next_obs, next_actions)
            next_q_values_2 = target_critic_2(batch.next_obs, next_actions)

            target = batch.reward + cfg.gamma * ~batch.terminated * torch.minimum(next_q_values_1, next_q_values_2)

        # Critic update: both critics are regressed towards the same target
        q_values_1 = critic_1(batch.obs, batch.action.value)
        q_values_2 = critic_2(batch.obs, batch.action.value)

        critic_loss_1 = F.mse_loss(q_values_1, target)
        critic_loss_2 = F.mse_loss(q_values_2, target)

        critic_optimizer.zero_grad()
        critic_loss = critic_loss_1 + critic_loss_2
        critic_loss.backward()
        critic_optimizer.step()

        evaluator.writer.add_scalar("loss/critic", critic_loss.item(), collector.steps)

        updates += 1
        if updates % cfg.policy_delay == 0:
            # Delayed policy update: update the actor (with `critic_1`)

            actor_optimizer.zero_grad()
            actor_loss = compute_actor_loss(critic_1(batch.obs, actor(batch.obs).value))
            actor_loss.backward()
            actor_optimizer.step()

            evaluator.writer.add_scalar(
                "loss/actor", actor_loss.item(), collector.steps
            )

            # Soft update of the three target networks

        soft_update(source=critic_1, target=target_critic_1, tau=cfg.tau)
        soft_update(source=critic_2, target=target_critic_2, tau=cfg.tau)
        soft_update(source=actor, target=target_actor, tau=cfg.tau)

        if result := evaluator.run_if_needed(collector.steps, actor):
            pbar.set_description(
                f"eval={result.mean:7.1f} best={evaluator.best_reward:7.1f}"
            )

    pbar.close()
    return evaluator