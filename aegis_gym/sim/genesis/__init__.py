from importlib.metadata import version, PackageNotFoundError

try:
    __version__ = version("aegis_gym")
except PackageNotFoundError:
    __version__ = "unknown"

from gymnasium.envs.registration import register
from ...scene.scene_director_factory import SceneDirectorType
from ...envs.env_types import (
    EnvControlType,
    EnvObservationType,
    EnvRewardType,
    EnvRenderMode,
)

ENV_IDS = []

tasks = ["Reacher", "Pusher"]
sim_name = EnvRewardType.SIM_GENESIS.value
obs_type = EnvObservationType.STATE.value.capitalize()
reward_type = EnvRewardType.DENSE.value.capitalize()
control_type = EnvControlType.JOINTS.value.capitalize()

for task in tasks:
    env_id = f"Aegis{sim_name}{obs_type}{task}{control_type}{reward_type}-v1"
    register(
        id=env_id,
        entry_point=f"aegis_gym.envs:Aegis{task}Env",
        kwargs={
            "render_mode": EnvRenderMode.NONE.value,
            "observation_type": obs_type.upper(),
            "control_type": control_type.upper(),
            "reward_type": reward_type.lower(),
            "scene_type": SceneDirectorType.SIM_GENESIS,
            "device": "cuda",
        },
        max_episode_steps=50,
    )
    ENV_IDS.append(env_id)
