import time

import gymnasium as gym
import aegis_gym.sim.genesis as sim_genesis

try:
    from stable_baselines3 import PPO
except ImportError:
    print(
        ">>> To run this SB3 integration test, you need SB3: pip install stable-baselines3"
    )
    raise ImportError


def main():
    env_name = sim_genesis.ENV_IDS[0]
    print(f"Training on environment: {env_name}")

    env = gym.make(env_name, render_mode="rgb_array")

    # Sleep 5 seconds and print countdown
    for i in range(5, 0, -1):
        print(f"Starting in {i} seconds...", end="\r")
        time.sleep(1)
    print("Starting now!            ")

    model = PPO("MlpPolicy", env, verbose=1)
    model.learn(total_timesteps=10_000)


if __name__ == "__main__":
    main()
