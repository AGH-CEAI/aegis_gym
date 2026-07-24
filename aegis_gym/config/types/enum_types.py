from dataclasses import dataclass
from enum import auto
from pathlib import Path

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
    CAMERA_TOOL_RIGHT = "cam_tool_right"


class CameraLink(StrEnum):
    CAMERA_SCENE = "cam_scene_rgb_camera_frame"
    CAMERA_TOOL_LEFT = "cam_tool_left"
    CAMERA_TOOL_RIGHT = "cam_tool_right"


class CameraModality(StrEnum):
    RGB = auto()
    RGBD = auto()
    DEPTH = auto()


class Modality(StrEnum):
    TCP_POSE = auto()
    TCP_VELOCITY = auto()
    TCP_WRENCH = auto()
    CAMERA_SCENE_RGB = CameraName.CAMERA_SCENE
    CAMERA_TOOL_LEFT_RGB = CameraName.CAMERA_SCENE_LEFT
    CAMERA_TOOL_RIGHT_RGB = CameraName.CAMERA_SCENE_RIGHT
    OBJECT_POSE = auto()


IMAGE_MODALITIES: tuple[Modality, ...] = (
    Modality.CAMERA_SCENE_RGB,
    Modality.CAMERA_TOOL_LEFT_RGB,
    Modality.CAMERA_TOOL_RIGHT_RGB,
)

CAMERAS_LINKS: dict[CameraName, CameraLink] = {
    CameraName.CAMERA_SCENE: CameraLink.CAMERA_SCENE,
    CameraName.CAMERA_TOOL_LEFT: CameraLink.CAMERA_TOOL_LEFT,
    CameraName.CAMERA_TOOL_RIGHT: CameraLink.CAMERA_TOOL_RIGHT,
}


@dataclass(frozen=True, order=True, slots=True)
class Checkpoint:
    step: int
    path: Path
