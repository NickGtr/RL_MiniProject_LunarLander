from .algos.ddpg import run_ddpg
from .algos.td3 import run_td3
from .algos.config import DDPGConfig, TD3Config
from .evaluation.monte_carlo import monte_carlo_Q

__all__ = ["run_ddpg", "run_td3", "DDPGConfig", "TD3Config", "monte_carlo_Q"]

def main() -> None:
    print("Hello from rl-miniproject-lunarlander!")