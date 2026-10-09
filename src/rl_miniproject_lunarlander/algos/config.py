from dataclasses import dataclass, field

@dataclass(frozen=True)
class DDPGConfig:
    env_name: str = 'LunarLanderContinuous-v3'
    env_kwargs: dict = field(default_factory=lambda: {})
    seed: int = 1
    monte_carlo_seed : int = 2
    monte_carlo_n_envs : int = 1
    mc_n_episodes : int = 100
    #: Steps between two Monte Carlo checks of the critic (0 = disabled)
    mc_interval : int = 0
    #: Number of (s, a) pairs of the batch evaluated by Monte Carlo at each check
    mc_n_samples : int = 5

    #: Total number of environment steps
    max_steps: int = 30_000
    #: Number of parallel training environments
    n_envs: int = 1
    #: Environment steps between two gradient updates
    steps_per_update: int = 1
    #: Steps before learning starts
    learning_starts: int = 1_000

    #: Replay buffer capacity
    buffer_size: int = 200_000
    batch_size: int = 64

    #: Discount factor
    gamma: float = 0.98
    #: Target network update coefficient
    tau: float = 0.05
    #: Exploration noise
    action_noise: float = 0.1

    actor_hidden: tuple[int, ...] = (64, 64)
    critic_hidden: tuple[int, ...] = (64, 64)
    lr_actor: float = 1e-3
    lr_critic: float = 1e-3

    #: Steps between two evaluations, and number of evaluation episodes
    eval_interval: int = 2_000
    n_eval_envs: int = 10

@dataclass(frozen=True)
class TD3Config(DDPGConfig):
    #: Number of critic updates between two policy updates
    policy_delay: int = 2
    #: Std of the noise added to the target policy actions
    target_noise: float = 0.2
    #: Clipping of the target policy noise
    target_noise_clip: float = 0.5