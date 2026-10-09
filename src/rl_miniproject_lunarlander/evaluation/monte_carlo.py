import numpy as np
import gymnasium as gym
from gymnasium.envs.box2d.lunar_lander import (
    VIEWPORT_W,
    VIEWPORT_H,
    SCALE,
    FPS,
    LEG_DOWN,
)
import torch
from torch import Tensor

from typing import Any, Callable

from rl_mind.core import Action, Actor
from rl_mind.nn import build_mlp, soft_update
from rl_mind.env import VecEnv


from ..algos.config import DDPGConfig

def monte_carlo_Q(
        forced_inital_env : VecEnv,
        initial_state : Tensor,
        initial_action : Tensor,
        actor : Actor,
        n_episodes : int,
        gamma : float,
        ):

    """Collect the Monte Carlo discounted reward of n full episodes by doing an action and then following the actor
    Used for MonteCarlo. The reset() method of the environment should reset to a given state.
    """
    env = forced_inital_env

    # VecEnv doesn't expose set_initial_state directly; delegate to each
    # sub-environment that has been wrapped with ForceInitialStateWrapper.
    for sub_env in env.gym_env.envs:
        sub_env.set_initial_state(initial_state)

    steps = 0
    obs = env.reset()
    just_reset = torch.zeros(env.num_envs, dtype=torch.bool)
    is_first_step = torch.ones(env.num_envs, dtype=torch.bool)
    first_action =  initial_action.unsqueeze(0).expand(env.num_envs, -1)

    reward_lists: list[list[float]] = [[] for _ in range(env.num_envs)]
    final_reward_list = []
    with torch.no_grad():
        while len(final_reward_list) < n_episodes:
            action = actor(obs)
            action_to_apply = torch.where(is_first_step[:, None], first_action, action.value)
            step = env.step(action_to_apply)

            for i in range(env.num_envs):
                if just_reset[i]:
                    # This environment was auto-resetting: nothing to record
                    continue
                reward_lists[i].append(float(step.reward[i]))
                steps += 1
                is_first_step[i] = False

                if step.done[i]:
                    q_sa = 0
                    for t, reward in enumerate(reward_lists[i]):
                        q_sa += gamma ** t * reward
                    final_reward_list.append(q_sa)
                    reward_lists[i] = []
                    is_first_step[i] = True

            obs = step.obs
            if not env.same_step_reset:
                just_reset = step.done

    return torch.tensor(final_reward_list).float().mean()

class MonteCarloQLogger:
    """Every `cfg.mc_interval` steps, compares the critics' Q(s, a) with a Monte Carlo
    estimate on the first `cfg.mc_n_samples` transitions of a (randomly sampled) batch,
    and logs both to tensorboard. Does nothing if `cfg.mc_interval` is 0.
    """
    def __init__(self, cfg : DDPGConfig, writer):
        self.cfg = cfg
        self.writer = writer
        self.next_step = cfg.mc_interval
        self.env = None
        if cfg.mc_interval > 0:
            self.env = VecEnv(
                cfg.env_name,
                cfg.monte_carlo_n_envs,
                seed=cfg.monte_carlo_seed if cfg.monte_carlo_seed is not None else cfg.seed + 1000,
                wrappers=[ForceInitialStateWrapper],
                **cfg.env_kwargs,
            )

    def run_if_needed(self, steps : int, batch, actor : Actor, critics : dict[str, Callable[[Tensor, Tensor], Tensor]]):
        if self.env is None or steps < self.next_step:
            return
        while self.next_step <= steps:
            self.next_step += self.cfg.mc_interval

        obs = batch.obs[:self.cfg.mc_n_samples]
        actions = batch.action.value[:self.cfg.mc_n_samples]
        q_mc = torch.stack([
            monte_carlo_Q(self.env, o, a, actor, self.cfg.mc_n_episodes, self.cfg.gamma)
            for o, a in zip(obs, actions)
        ])
        self._log("q_mc", q_mc, steps)

        with torch.no_grad():
            for name, critic in critics.items():
                q = critic(obs, actions)
                self._log(f"q_{name}", q, steps)
                # > 0 means the critic overestimates
                self._log(f"bias_{name}", q - q_mc, steps)

    def _log(self, tag : str, values : Tensor, steps : int):
        """Logs the mean over the sampled (s, a) pairs, and its standard error as `<tag>_se`"""
        self.writer.add_scalar(f"monte_carlo/{tag}", values.mean().item(), steps)
        self.writer.add_scalar(f"monte_carlo/{tag}_se", (values.std() / len(values) ** 0.5).item(), steps)

class ForceInitialStateWrapper(gym.Wrapper):
    """One must set_initial_state of this environment so that
        reset() method forces into initial_state when called.
    """
    def __init__(self, env : gym.Env):
        super().__init__(env)

    def set_initial_state(self, initial_state : Any):
        self.initial_state = np.asarray(initial_state, dtype=np.float32)

    def reset(self, *, seed: int | None = None, options: dict[str, Any] | None = None) -> tuple[Any, dict[str, Any]]:
        _, info = super().reset(seed=seed, options=options)
        set_state(self.env.unwrapped, self.initial_state)

        obs = get_state(self.env.unwrapped) #should be the same as initial state, but to get rid of small discrepencies
        return obs, info

def set_state(env, state):
    x, y, vx, vy, theta, omega, leg1, leg2 = map(float, state)

    W = VIEWPORT_W / SCALE
    H = VIEWPORT_H / SCALE

    world_x = x * (W / 2) + W / 2
    world_y = y * (H / 2) + (env.helipad_y + LEG_DOWN / SCALE)

    world_vx = vx * FPS / (W / 2)
    world_vy = vy * FPS / (H / 2)

    world_omega = omega * FPS / 20.0

    env.lander.position = (world_x, world_y)
    env.lander.linearVelocity = (world_vx, world_vy)

    env.lander.angle = theta
    env.lander.angularVelocity = world_omega

    env.lander.awake = True

def get_state(env):
    pos = env.lander.position
    vel = env.lander.linearVelocity

    return np.array([
        (pos.x - VIEWPORT_W / SCALE / 2) / (VIEWPORT_W / SCALE / 2),
        (pos.y - (env.helipad_y + LEG_DOWN / SCALE))
            / (VIEWPORT_H / SCALE / 2),
        vel.x * (VIEWPORT_W / SCALE / 2) / FPS,
        vel.y * (VIEWPORT_H / SCALE / 2) / FPS,
        env.lander.angle,
        20.0 * env.lander.angularVelocity / FPS,
        float(env.legs[0].ground_contact),
        float(env.legs[1].ground_contact),
    ], dtype=np.float32)