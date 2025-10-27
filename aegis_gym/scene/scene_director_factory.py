import os
import warnings
from enum import auto

# Python 3.11+: Change to builtin StrEnum
from strenum import StrEnum

from .scene_director_interface import SceneDirectorInterface
from .scene_director_mock import SceneDirectorMock


def is_mock_needed() -> bool:
    if os.environ.get("PYTEST_CURRENT_TEST") is not None:
        return True
    return False


class SceneDirectorType(StrEnum):
    MOCK = auto()
    ROS = auto()
    SIM_GENESIS = "SimGenesis"


def get_scene_director(
    mode: SceneDirectorType = SceneDirectorType.ROS,
    visual_obs: bool = False,
) -> SceneDirectorInterface:
    if is_mock_needed():
        print("\n> Deteceted pytest env, using MOCK env implementations.")
        mode = SceneDirectorType.MOCK

    match mode:
        case SceneDirectorType.MOCK:
            return SceneDirectorMock()

        case SceneDirectorType.ROS:
            try:
                from ..ros.scene_director_ros import SceneDirectorROS
            except ImportError:
                SceneDirectorROS = None

            if SceneDirectorROS:
                return SceneDirectorROS()
            warnings.warn(
                "\n[IMPORT ERROR] Failed to import SceneDirectorROS. Double check if the ROS is sourced properly via `source /opt/ros/ROS_DISTRO/setup.sh`."
            )
            raise ImportError

        case SceneDirectorType.SIM_GENESIS:
            try:
                from ..sim.genesis.scene_director_genesis import SceneDirectorSimGenesis
            except ImportError:
                SceneDirectorSimGenesis = None

            if SceneDirectorSimGenesis:
                return SceneDirectorSimGenesis(visual_obs=visual_obs)
            warnings.warn(
                "\n[IMPORT ERROR] Failed to import SceneDirectorSimGenesis. Double check if the 'aegis_gym' is instatalled with optional dependencies: 'pip3 install ./aegis_gym.whl[sim-genesis]'."
            )
            raise ImportError

        case _:
            print(f"[VALUE ERROR] Not defined RobotCommanderInterface '{mode.name}'.")
            raise ValueError
