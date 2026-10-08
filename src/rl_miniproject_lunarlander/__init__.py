from .algos.ddpg import run_ddpg
from .algos.td3 import run_td3
from .algos.config import DDPGConfig, TD3Config

__all__ = ["run_ddpg", "run_td3", "DDPGConfig", "TD3Config"]

def main() -> None:
    print("Hello from rl-miniproject-lunarlander!")