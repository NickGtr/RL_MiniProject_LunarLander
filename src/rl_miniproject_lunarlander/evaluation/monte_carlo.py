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

from typing import Any

from rl_mind.core import Action, Actor
from rl_mind.nn import build_mlp, soft_update
from rl_mind.env import VecEnv


from ..algos.config import DDPGConfig

def monte_carlo_Q(
        cfg : DDPGConfig,
        initial_state : Tensor,
        initial_action : Tensor,
        actor : Actor,
        n_episodes : int
        ):
    
    env = VecEnv(
        env_name=cfg.env_name,
        num_envs=cfg.monte_carlo_n_envs,
        seed=cfg.monte_carlo_seed,
        wrappers=[lambda env: ForceInitialStateWrapper(env, initial_state),]
    )

    """Collect the Monte Carlo discounted reward of n full episodes by doing an action and then following the actor
    Used for MonteCarlo. The reset() method of the environment should reset to a given state.
    """

    steps = 0
    obs = env.reset()
    just_reset = torch.zeros(cfg.monte_carlo_n_envs, dtype=torch.bool)
    is_first_step = torch.ones(cfg.monte_carlo_n_envs, dtype=torch.bool)
    first_action =  initial_action.unsqueeze(0).expand(cfg.monte_carlo_n_envs, -1)

    reward_lists: list[list[float]] = [[] for _ in range(cfg.monte_carlo_n_envs)]
    final_reward_list = []
    with torch.no_grad():
        while len(final_reward_list) < n_episodes:
            action = actor(obs)
            action_to_apply = torch.where(is_first_step[:, None], first_action, action.value)
            step = env.step(action_to_apply)

            for i in range(cfg.monte_carlo_n_envs):
                if just_reset[i]:
                    # This environment was auto-resetting: nothing to record
                    continue
                reward_lists[i].append(float(step.reward[i]))
                steps += 1
                is_first_step[i] = False

                if step.done[i]:
                    q_sa = 0
                    for t, reward in enumerate(reward_lists[i]):
                        q_sa += cfg.gamma ** t * reward
                    final_reward_list.append(q_sa)
                    reward_lists[i] = []
                    is_first_step[i] = True

            obs = step.obs
            if not env.same_step_reset:
                just_reset = step.done

    return torch.tensor(final_reward_list).float().mean()

class ForceInitialStateWrapper(gym.Wrapper):
    def __init__(self, env : gym.Env, initial_state : Any):
        super().__init__(env)
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