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
obs_types = [EnvObservationType.STATE]
control_types = [
    EnvControlType.JOINTS,
    EnvControlType.JOINTS_SERVO,
    EnvControlType.CARTESIAN_POSITION,
    EnvControlType.CARTESIAN_POSITION_SERVO,
]
reward_types = [EnvRewardType.DENSE]

for task in tasks:
    for obs_type in obs_types:
        for reward_type in reward_types:
            for control_type in control_types:
                env_id = (
                    f"Aegis{task}"
                    f"{obs_type.value.capitalize()}"
                    f"{control_type.value.capitalize()}"
                    f"{reward_type.value.capitalize()}"
                    f"-v1"
                )
                register(
                    id=env_id,
                    entry_point=f"aegis_gym.envs:Aegis{task}Env",
                    kwargs={
                        "render_mode": EnvRenderMode.NONE.value,
                        "observation_type": obs_type.value.lower(),
                        "control_type": control_type.value.lower(),
                        "reward_type": reward_type.value.lower(),
                        "scene_type": SceneDirectorType.ROS,
                        "device": "cuda",
                    },
                    max_episode_steps=50,
                )
                ENV_IDS.append(env_id)
