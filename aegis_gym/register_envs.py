from importlib import import_module

from gymnasium.envs.registration import register
from .scene.scene_director_factory import SceneDirectorType
from .envs.env_types import (
    EnvControlType,
    EnvObservationType,
    EnvRewardType,
    EnvRenderMode,
)

tasks = ["Reacher", "Pusher"]
obs_types = [EnvObservationType.STATE]
control_types = [
    EnvControlType.JOINTS,
    EnvControlType.JOINTS_SERVO,
    EnvControlType.CARTESIAN_POSITION,
    EnvControlType.CARTESIAN_POSITION_SERVO,
]
reward_types = [EnvRewardType.DENSE]


def register_envs(scene_type: SceneDirectorType) -> list[str]:
    env_ids = []
    for task in tasks:
        for obs_type in obs_types:
            for reward_type in reward_types:
                for control_type in control_types:
                    env_id = (
                        f"Aegis{scene_type.value}{task}"
                        f"{obs_type.value.capitalize()}"
                        f"{control_type.value.capitalize()}"
                        f"{reward_type.value.capitalize()}"
                        f"-v1"
                    )

                    env_file_path = f"aegis_gym.envs.aegis_{task.lower()}"
                    env_module = import_module(env_file_path)
                    env_cfg = getattr(env_module, "ENV_CFG")

                    register(
                        id=env_id,
                        entry_point=f"aegis_gym.envs:Aegis{task}Env",
                        kwargs={
                            "render_mode": EnvRenderMode.NONE.value,
                            "observation_type": obs_type.value.lower(),
                            "control_type": control_type.value.lower(),
                            "reward_type": reward_type.value.lower(),
                            "scene_type": scene_type,
                            "device": "cuda",
                        },
                        max_episode_steps=env_cfg["max_episode_length"],
                    )
                    env_ids.append(env_id)
    return env_ids
