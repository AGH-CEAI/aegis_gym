from dataclasses import dataclass
from pathlib import Path
from enum import auto

from strenum import StrEnum


class Algorithm(StrEnum):
    RL = "rl"
    BC = "bc"


class Control(StrEnum):
    SIM = "sim"
    ROS = "ros"


class CamerasSetup(StrEnum):
    DEFAULT = "default"
    SCENE_DUAL = "scene_dual"


class CameraName(StrEnum):
    CAMERA_SCENE = "cam_scene"
    CAMERA_SCENE_LEFT = "cam_scene_left"
    CAMERA_SCENE_RIGHT = "cam_scene_right"
    CAMERA_TOOL_LEFT = "cam_tool_left"
    CAMERA_TOOL_RIGHT = "cam_right_cam"


class CameraLink(StrEnum):
    CAMERA_SCENE = "cam_scene_rgb_camera_frame"
    CAMERA_TOOL_LEFT = "cam_tool_left"
    CAMERA_TOOL_RIGHT = "cam_tool_right"


class CameraModality(StrEnum):
    RGB = auto()
    RGBD = auto()
    DEPTH = auto()


@dataclass(frozen=True, order=True, slots=True)
class Checkpoint:
    step: int
    path: Path
