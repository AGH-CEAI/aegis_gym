from dataclass import dataclass
from typing import ClassVar, Any

import torch as th

from .domain_randomization import DomainRandomizationCfg
from .debug import DebugCfg
from .rl import RLCfg
from .env import EnvCfg
from .bc import BCCfg
from .logger import LoggerCfg
from .robot import RobotCfg


@dataclass(slots=True, frozen=True)
class ExpConfig:
    logger_cfg: LoggerCfg
    rl_cfg: RLCfg
    bc_cfg: BCCfg
    env_cfg: EnvCfg
    robot_cfg: RobotCfg
    dr_cfg: DomainRandomizationCfg
    debug_cfg: DebugCfg

    args: Any  # TODO(issue#111) fix the cycle import
    _device: ClassVar["th.device"]

    @classmethod
    def set_device(cls, device: "th.device") -> None:
        cls._device = device

    @classmethod
    def get_device(cls) -> "th.device":
        return cls._device
