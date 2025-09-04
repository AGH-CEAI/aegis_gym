import warnings
from enum import auto
from typing import Literal

# TODO for Python 3.11: change to buildin StrEnum
from strenum import StrEnum

from .base_ros_interface import BaseROSInterface
from .ros_interface_mock import ROSInterfaceMock

try:
    from .ros_interface import ROSInterface
except ImportError:
    ROSInterface = None

class ROSInterfaceType(StrEnum):
    MOCK = auto()
    REAL = auto()

def get_ros_interface(mode: ROSInterfaceType = ROSInterfaceType.REAL) -> BaseROSInterface:
    match mode:
        case ROSInterfaceType.MOCK:
                return ROSInterfaceMock()
        case ROSInterfaceType.REAL:
            if ROSInterface:
                return ROSInterface()
            warnings.warn("\nFailed to import ROSInterface. Double check if the ROS is sourced properly via `source /opt/ros/ROS_DISTRO/setup.sh`.")
            raise ImportError
        case _:
            print(f"Not defined ROSInterfaceType '{mode.name}'.")
            raise ValueError