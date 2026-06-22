from strenum import StrEnum
from dataclasses import dataclass
from pathlib import Path


class Stage(StrEnum):
    RL = "rl"
    BC = "bc"


class Control(StrEnum):
    SIM = "sim"
    ROS = "ros"


@dataclass(frozen=True, order=True, slots=True)
class Checkpoint:
    step: int
    path: Path
