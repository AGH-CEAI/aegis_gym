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
    CONTROL_TYPES = {
        "JOINTS": 0,  # Move joints to target, waits until reached
        "JOINTS_SERVO": 1,  # Continuous joint commands
        "CARTESIAN_POSITION": 2,  # Move TCP to target, waits until reached
        "CARTESIAN_POSITION_SERVO": 3,  # Continuous TCP commands
    }

    # ROS
    # env_name = aegis_gym.ENV_IDS[CONTROL_TYPES["JOINTS"]]
    # env_name = aegis_gym.ENV_IDS[CONTROL_TYPES["JOINTS_SERVO"]]

    # GENESIS

    env_name = sim_genesis.ENV_IDS[CONTROL_TYPES["JOINTS_SERVO"]]

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
