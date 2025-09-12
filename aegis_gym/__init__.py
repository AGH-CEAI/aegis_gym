from importlib.metadata import version, PackageNotFoundError

try:
    __version__ = version("aegis_gym")
except PackageNotFoundError:
    __version__ = "unknown"

from gymnasium.envs.registration import register
from .scene.scene_director_factory import SceneDirectorType
from .envs.env_types import (
    EnvControlType,
    EnvObservationType,
    EnvRewardType,
    EnvRenderMode,
)

ENV_IDS = []

tasks = ["Reacher", "Pusher"]
obs_type = EnvObservationType.STATE.name.capitalize()
reward_type = EnvRewardType.DENSE.name.capitalize()
control_type = EnvControlType.JOINTS.name.capitalize()

for task in tasks:
    env_id = f"Aegis{task}{obs_type}{control_type}{reward_type}-v1"
    register(
        id=env_id,
        entry_point=f"aegis_gym.envs:Aegis{task}Env",
        kwargs={
            "render_mode": EnvRenderMode.NONE.name,
            "observation_type": obs_type.upper(),
            "control_type": control_type.upper(),
            "reward_type": reward_type.upper(),
            "scene_type": SceneDirectorType.ROS,
            "device": "cuda",
        },
        max_episode_steps=50,
    )
    ENV_IDS.append(env_id)
