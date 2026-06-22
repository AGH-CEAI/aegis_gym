import pickle
from dataclasses import dataclass, fields, asdict, is_dataclass
from typing import ClassVar, Any
from pathlib import Path

import torch as th
from clearml import Task
from .domain_randomization import DomainRandomizationCfg
from .debug import DebugCfg
from .rl import RLCfg
from .env import EnvCfg
from .bc import BCCfg
from .logger import LoggerCfg


@dataclass(slots=True, frozen=True)
class GraspConfig:
    logger_cfg: LoggerCfg
    rl_cfg: RLCfg
    bc_cfg: BCCfg
    env_cfg: EnvCfg
    robot_cfg: RobotCfg
    dr_cfg: DomainRandomizationCfg
    debug_cfg: DebugCfg

    _device: ClassVar["th.device"] = None
    _instance: ClassVar["GraspConfig | None"] = None

    @classmethod
    def set_device(cls, device: "th.device") -> None:
        cls._device = device

    @classmethod
    def get_device(cls) -> "th.device":
        return cls._device

    @classmethod
    def get_instance(cls) -> "GraspConfig":
        if cls._instance is None:
            raise RuntimeError("GraspConfig has not been created.")
        return cls._instance

    @classmethod
    def create(cls) -> "GraspConfig":
        # TODO: initialize cfg from defaults in YAML files
        cls._instance = cls(
            logger_cfg=get_logger_cfg(),
            rl_cfg=get_rl_cfg(),
            bc_cfg=get_bc_cfg(),
            env_cfg=get_env_cfg(),
            robot_cfg=get_robot_cfg(),
            dr_cfg=DomainRandomizationCfg.from_dict(get_dr_cfg()),
            debug_cfg=DebugCfg.from_dict({}),
        )
        return cls._instance

    @classmethod
    def create_with_clearml(cls, task: Task) -> "GraspConfig":
        # TODO create with defaults from YAML files
        instance = cls.create()

        values: dict[str, Any] = {}

        for field in fields(instance):
            value = getattr(instance, field.name)

            if is_dataclass(value):
                connected = task.connect_configuration(
                    asdict(value),
                    name=field.name,
                )

                value = type(value).from_dict(connected)

            else:
                value = task.connect_configuration(
                    value,
                    name=field.name,
                )

            values[field.name] = value

        cls._instance = cls(**values)
        return cls._instance

    def to_pickle(self, path: Path) -> None:
        data = {field.name: getattr(self, field.name) for field in fields(self)}
        with path.open("wb") as f:
            pickle.dump(data, f)
