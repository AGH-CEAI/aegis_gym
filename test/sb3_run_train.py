import time

import gymnasium as gym

import aegis_gym  # noqa: F401
import aegis_gym.sim.genesis as sim_genesis  # noqa: F401
from aegis_gym.sim.utils import TorchToNumpyWrapper

try:
    from stable_baselines3 import PPO
except ImportError:
    print(
        ">>> To run this SB3 integration test, you need SB3: pip install stable-baselines3"
    )
    PPO = None


def main():
    # 0 JOINTS: move joints to target, waits until reached
    # 1 JOINTS_SERVO: continuous joint commands
    # 2 CARTESIAN_POSITION: move TCP to target, waits until reached
    # 3 CARTESIAN_POSITION_SERVO: continuous TCP commands

    # ROS
    env_name = aegis_gym.ENV_IDS[0]

    # GENESIS
    # env_name = sim_genesis.ENV_IDS[0]

    print(f"Training on environment: {env_name}")

    env = gym.make(env_name)
    env = TorchToNumpyWrapper(env)

    # Sleep 5 seconds and print countdown
    for i in range(5, 0, -1):
        print(f"Starting in {i} seconds...", end="\r")
        time.sleep(1)
    print("Starting now!            ")

    model = PPO("MlpPolicy", env, verbose=1)
    model.learn(total_timesteps=10_000)


if __name__ == "__main__":
    if PPO:
        main()
