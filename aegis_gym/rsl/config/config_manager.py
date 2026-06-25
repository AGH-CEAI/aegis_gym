import yaml
from typing import Optional, Callable
from pathlib import Path

import torch as th
from clearml import Task

from .args_parser import LaunchArgs, parse_arguments
from .types import (
    Control,
    ExpConfig,
    LoggerCfg,
    RLCfg,
    BCCfg,
    EnvCfg,
    RobotCfg,
    DomainRandomizationCfg,
    DebugCfg,
)


class ConfigManager:
    _global_cfg: Optional[ExpConfig] = None

    @classmethod
    def get_config(cls) -> ExpConfig:
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
    ) -> ExpConfig:
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

        cls._patch_config(args=args, cfg_dict=cfg_dict)
        cfg = ExpConfig(
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

    @classmethod
    def _patch_config(cls, args: LaunchArgs, cfg_dict: dict) -> None:
        # You can control only 1 real instance
        if args.control_type == Control.ROS:
            cfg_dict["env"]["num_envs"] = 1

        # Add project_suffix to the logger outputs
        # TODO rename the args.learning_method to args.train_type
        project_suffix = f"_{str(args.learning_method)}-{str(args.control_type)}"
        cfg_dict["logger"]["wandb_project"] += project_suffix
        cfg_dict["logger"]["clearml_project"] += project_suffix
        cfg_dict["logger"]["neptune_project"] += project_suffix

        # Define the local_log_dir if not given
        if not cfg_dict["logger"]["local_log_dir"]:
            train_type = str(args.learning_method)
            log_dir = (
                Path("/tmp/aegis_gym_logs") / f"{args.experiment_name}_{train_type}"
            )
            log_dir.mkdir(parents=True, exist_ok=True)
            cfg_dict["logger"]["local_log_dir"] = str(log_dir)

        # Apply the debug flags
        if args.debug_enable:
            cfg_dict["debug"]["enabled"] = args.debug_enable
            cfg_dict["debug"]["swap_tool_cameras"] = args.debug_swap_tool_cameras
            cfg_dict["debug"]["enable_vis_preview"] = args.debug_preview_vis_obs
            cfg_dict["debug"]["enable_receord_obs"] = args.debug_record_vis_obs
            cfg_dict["debug"]["record_dir"] = args.debug_record_dir

        # Apply launch arguments if given
        if args.experiment_name:
            cfg_dict["rl"]["experiment_name"] = args.experiment_name
        if args.max_iterations:
            cfg_dict["rl"]["max_iterations"] = args.max_iterations
        if args.num_envs:
            cfg_dict["env"]["num_envs"] = args.num_envs
        if args.visualize_camera:
            cfg_dict["env"]["visualize_camera"] = args.visualize_camera

        # Confirm types of data
        cfg_dict["env"]["image_resolution"] = tuple(cfg_dict["env"]["image_resolution"])
