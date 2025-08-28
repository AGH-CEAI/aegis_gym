from typing import Literal

from .base_ros_interface import BaseROSInterface
from .ros_interface import ROSInterface
from .ros_interface_mock import ROSInterfaceMock


def get_ros_interface(mode: Literal["real", "mock"] = "real") -> BaseROSInterface:
    if mode == "mock":
        return ROSInterfaceMock()
    else:
        return ROSInterface()
