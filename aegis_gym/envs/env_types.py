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
    MULTIMODAL = auto()


class EnvControlType(LowercaseStrEnum):
    JOINTS = auto()
    JOINTS_SERVO = auto()
    CARTESIAN_POSITION = auto()
    CARTESIAN_POSITION_SERVO = auto()
    CARTESIAN_POSE = auto()
    CARTESIAN_POSE_SERVO = auto()
