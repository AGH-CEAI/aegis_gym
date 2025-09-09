from importlib.metadata import version, PackageNotFoundError

try:
    __version__ = version("aegis_gym")
except PackageNotFoundError:
    __version__ = "unknown"

from gymnasium.envs.registration import register
from ...robot.robot_commander_factory import RobotCommanderType

ENV_IDS = []

sim_name = RobotCommanderType.SIM_GENESIS.name
tasks = ["Reacher", "Pusher"]
control_suffix = "Joints"
reward_suffix = "Dense"

for task in tasks:
    env_id = f"Aegis{sim_name}{task}{control_suffix}{reward_suffix}-v1"
    register(
        id=env_id,
        entry_point=f"aegis_gym.envs:Aegis{task}Env",
        kwargs={
            "reward_type": reward_suffix.lower(),
            "control_type": control_suffix.lower(),
            "device": "cuda",
            "robot_interface": RobotCommanderType.SIM_GENESIS,
        },
        max_episode_steps=50,
    )
    ENV_IDS.append(env_id)
