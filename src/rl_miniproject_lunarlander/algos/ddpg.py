import copy
import torch
import torch.nn as nn
import torch.nn.functional as F

from torch import Tensor
from torch.utils.tensorboard import SummaryWriter
from tqdm.auto import tqdm
from .config import DDPGConfig

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


def run_ddpg(cfg: DDPGConfig) -> Evaluator:
    torch.manual_seed(cfg.seed)
    env = VecEnv(cfg.env_name, cfg.n_envs, seed=cfg.seed, **cfg.env_kwargs)
    # Create the actor, the critic and its target, and the optimizers

    actor = ContinuousDeterministicActor(env.observation_dim,
                                     cfg.actor_hidden,
                                     env.action_dim
                                     )  
    critic = ContinuousQNetwork(env.observation_dim, cfg.critic_hidden, env.action_dim)
    # the target critic is a copy of the critic (`copy.deepcopy`)
    target_critic = copy.deepcopy(critic)
    # one optimizer per network (`cfg.lr_actor`, `cfg.lr_critic`)
    actor_optimizer = torch.optim.Adam(actor.parameters(), lr=cfg.lr_actor)
    critic_optimizer = torch.optim.Adam(critic.parameters(), lr=cfg.lr_critic)

    # Data collection (with exploration noise), replay buffer and evaluation
    collector = TransitionCollector(env, GaussianNoise(actor, cfg.action_noise))
    buffer = ReplayBuffer(cfg.buffer_size)
    run_dir = run_directory(f"ddpg-{cfg.env_name}-S{cfg.seed}")
    evaluator = Evaluator(
        VecEnv(cfg.env_name, cfg.n_eval_envs, seed=cfg.seed + 100),
        every=cfg.eval_interval,
        run_dir=run_dir,
        writer=SummaryWriter(run_dir),
    )

    pbar = tqdm(total=cfg.max_steps)
    while collector.steps < cfg.max_steps:
        buffer.add(collector.collect(cfg.steps_per_update))
        pbar.update(collector.steps - pbar.n)
        if len(buffer) < cfg.learning_starts:
            continue

        batch = buffer.sample(cfg.batch_size)

        # Q-values of the actions that were played (this is where gradients flow)

        q_values = critic(batch.obs, batch.action.value)

        with torch.no_grad():
            # Q-values of the *current* actor's actions in the next states, from the target critic

            # as in Double DQN: the actor selects the action, the target
            # critic evaluates it
            next_q_values = target_critic(batch.next_obs, actor(batch.next_obs).value)

        # Critic update: compute the loss, then a gradient step
        critic_optimizer.zero_grad()
        critic_loss = compute_critic_loss(cfg.gamma, batch, q_values, next_q_values)
        critic_loss.backward()
        critic_optimizer.step()

        # Actor update: maximize Q(s, pi(s)), with a gradient step on the actor only
        actor_optimizer.zero_grad()
        actor_loss = compute_actor_loss(critic(batch.obs, actor(batch.obs).value))
        actor_loss.backward()
        actor_optimizer.step()

        # Soft update of the target critic (`soft_update`)

        soft_update(source=critic, target=target_critic, tau=cfg.tau)

        evaluator.writer.add_scalar("loss/critic", critic_loss.item(), collector.steps)
        evaluator.writer.add_scalar("loss/actor", actor_loss.item(), collector.steps)
        if result := evaluator.run_if_needed(collector.steps, actor):
            pbar.set_description(
                f"eval={result.mean:7.1f} best={evaluator.best_reward:7.1f}"
            )

    pbar.close()
    return evaluator