from strenum import StrEnum
from dataclasses import dataclass
from pathlib import Path


class Algorithm(StrEnum):
    RL = "rl"
    BC = "bc"


class Control(StrEnum):
    SIM = "sim"
    ROS = "ros"

class CamerasSetup(StrEnum):
    DEFAULT = "default"
    SCENE_DUAL = "scene_dual"


@dataclass(frozen=True, order=True, slots=True)
class Checkpoint:
    step: int
    path: Path
