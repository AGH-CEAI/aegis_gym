import yaml
from typing import Optional, Callable
from pathlib import Path

import torch as th
from clearml import Task

from .args_parser import LaunchArgs, parse_arguments
from .types import (
    GraspConfig,
    LoggerCfg,
    RLCfg,
    BCCfg,
    EnvCfg,
    RobotCfg,
    DomainRandomizationCfg,
    DebugCfg,
)


class ConfigManager:
    _global_cfg: Optional[GraspConfig] = None

    @classmethod
    def get_config(cls) -> GraspConfig:
        if cls._global_cfg is None:
            raise AttributeError("Global configuration is not initialized!")
        return cls._global_cfg

    @classmethod
    def setup_config(
        cls,
        argv: list[str] | LaunchArgs,
        extra_argparser: Optional[Callable] = None,
        device: Optional[th.device] = None,
        task: Optional[Task] = None,
    ) -> None:
        """
        Initializes the global config based on the launch arguments.
        Launch arguments can be parsed by providing raw `argv` with optional `extra_argparser` options.
        Allows to setup the global `device` argument.
        Allows to connect config to the ClearML `task`.
        """
        if cls._global_cfg is not None:
            raise AttributeError("Tried to reinitialize global config!")
        cls._global_cfg = cls._initalize_config(
            argv=argv, extra_argparser=extra_argparser, device=device, task=task
        )

    @classmethod
    def _initalize_config(
        cls,
        argv: list[str] | LaunchArgs,
        extra_argparser: Optional[Callable],
        device: Optional[th.device],
        task: Optional[Task],
    ) -> GraspConfig:
        if not isinstance(argv, LaunchArgs):
            args: LaunchArgs = parse_arguments(
                argv=argv, extra_argparser=extra_argparser
            )
        else:
            args = argv

        cfg_dict = cls._get_default_config_dict()
        if args.config_path is not None:
            print(f"Patching default config with file: {args.config_path}")
            cfg_file_dict = cls._load_yaml(args.config_path)
            cfg_dict.update(cfg_file_dict)

        if not args.enforce_current_config and task is not None:
            print(f"Connecting config to the ClearML task id {task.task_id}")
            for cfg_sec_name, cfg_sec_dict in cfg_dict.items():
                connected = task.connect_configuration(
                    cfg_sec_dict,
                    name=cfg_sec_name,
                )
                cfg_dict[cfg_sec_name] = connected

        cfg = GraspConfig(
            logger_cfg=LoggerCfg.from_dict(cfg_dict.get("logger", None)),
            rl_cfg=RLCfg.from_dict(cfg_dict.get("rl", None)),
            bc_cfg=BCCfg.from_dict(cfg_dict.get("bc", None)),
            env_cfg=EnvCfg.from_dict(cfg_dict.get("env", None)),
            robot_cfg=RobotCfg.from_dict(cfg_dict.get("robot", None)),
            dr_cfg=DomainRandomizationCfg.from_dict(cfg_dict.get("dr", None)),
            debug_cfg=DebugCfg.from_dict(cfg_dict.get("debug", None)),
            args=args,
        )
        device = device or th.device("cpu")
        cfg.set_device(device=device)
        return cfg

    @classmethod
    def _get_default_config_dict(cls) -> dict:
        default_files = {
            "logger": Path(__file__).parent / Path("defaults/logger.yaml"),
            "rl": Path(__file__).parent / Path("defaults/rl.yaml"),
            "bc": Path(__file__).parent / Path("defaults/bc.yaml"),
            "env": Path(__file__).parent / Path("defaults/env.yaml"),
            "robot": Path(__file__).parent / Path("defaults/robot.yaml"),
            "dr": Path(__file__).parent / Path("defaults/domain_randomization.yaml"),
            "debug": Path(__file__).parent / Path("defaults/debug.yaml"),
        }

        result = {}
        for name, path in default_files.items():
            result[name] = cls._load_yaml(path)

        return result

    @classmethod
    def _load_yaml(cls, path: Path) -> dict:
        """
        Loads YAML file from given `path` as a dict.
        """
        with path.open("r", encoding="utf-8") as f:
            data = yaml.safe_load(f)
        return data
