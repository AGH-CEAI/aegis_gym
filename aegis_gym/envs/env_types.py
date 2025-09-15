from enum import auto
from strenum import LowercaseStrEnum


class EnvRenderMode(LowercaseStrEnum):
    NONE = auto()
    HUMAN = auto()
    RGB_ARRAY = auto()


class EnvRewardType(LowercaseStrEnum):
    DENSE = auto()
    SPARSE = auto()


class EnvObservationType(LowercaseStrEnum):
    STATE = auto()
    VISION = auto()


class EnvControlType(LowercaseStrEnum):
    JOINTS = auto()
    CARTESIAN_POSITION = auto()
    CARTESIAN_POSE = auto()
