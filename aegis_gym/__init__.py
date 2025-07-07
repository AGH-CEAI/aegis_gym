from importlib.metadata import version, PackageNotFoundError

try:
    __version__ = version("aegis_gym")
except PackageNotFoundError:
    __version__ = "unknown"

from gymnasium.envs.registration import register

ENV_IDS = []

for task in ["Reacher"]:
    control_suffix = "Joints"
    reward_suffix = "Dense"
    env_id = f"Aegis{task}{control_suffix}{reward_suffix}-v1"

    register(
        id=env_id,
        entry_point=f"aegis_gym.envs:Aegis{task}Env",
        kwargs={
            "reward_type": reward_suffix.lower(),
            "control_type": control_suffix.lower(),
        },
        max_episode_steps=50,
    )
    ENV_IDS.append(env_id)
