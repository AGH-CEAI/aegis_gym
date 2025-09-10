from enum import auto
from strenum import StrEnum


class LowerStrEnum(StrEnum):
    def _generate_next_value_(name, start, count, last_values):
        return name.lower()


class EnvRenderMode(LowerStrEnum):
    NONE = auto()
    HUMAN = auto()
    RGB_ARRAY = auto()


class EnvRewardType(LowerStrEnum):
    DENSE = auto()
    SPARSE = auto()


class EnvObservationType(LowerStrEnum):
    STATE = auto()
    VISION = auto()


class EnvControlType(LowerStrEnum):
    JOINTS = auto()
    CARTESIAN_POSITION = auto()
    CARTESIAN_POSE = auto()
