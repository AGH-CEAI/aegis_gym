import ast
from argparse import ArgumentParser
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional
from ..utils import Stage, Control


@dataclass(slots=True, frozen=True)
class LaunchArgs:
    experiment_name: str = "dummy_experiment"
    disable_headless: bool = False  # TODO rename it??? from --vis
    num_envs: Optional[int] = None
    episode_length_s: Optional[float] = None
    project_name: Optional[str] = None

    enable_plotjuggler: bool = False
    max_iterations: Optional[int] = False
    algorithm: Stage = Stage.RL

    # TODO think how to generalize it (multiple models for an experiments)
    load_rl_task_id: Optional[str] = None
    load_rl_model_id: Optional[str] = None
    load_bc_task_id: Optional[str] = None
    load_bc_model_id: Optional[str] = None

    enforce_current_config: bool = False
    control_type: Control = Control.SIM

    calibration_move: Optional[list] = None
    calibration_move_cartesian: Optional[list] = None
    calibration_steps: Optional[int] = None

    visualize_camera: bool = False
    disable_vision: bool = False

    debug_enable: bool = False
    debug_swap_tool_cameras: bool = False
    debug_preview_vis_obs: bool = False
    debug_record_vis_obs: bool = False
    debug_record_dir: Optional[Path] = None

    enable_recording: bool = False
    video_path: Optional[Path] = None
    load_from_pickle: bool = False
    bc_all_checkpoints: bool = False
    bc_eval_every: Optional[int] = None
    seed: int = 44

    _args_raw: Any = None

    def get_raw_args(self) -> Any:
        return self._args_raw

    def as_dict(self) -> dict:
        return {slot: getattr(self, slot) for slot in self.__slots__}


def parse_arguments(argv, extra_argparser: Optional[callable] = None) -> LaunchArgs:
    def str_to_list(arg: Optional[str]) -> Optional[list[float]]:
        if arg is None:
            return None
        return ast.literal_eval(arg)

    p = ArgumentParser(argv, add_help=True)

    p.add_argument("-e", "--exp-name", type=str, default="dummy_run")
    p.add_argument("-v", "--vis", action="store_true", default=False)
    p.add_argument("-B", "--num-envs", type=int, default=4096)
    p.add_argument("--episode-length-s", type=float, default=None)
    p.add_argument(
        "--episode-length",
        type=float,
        default=None,
        help="Overwrite the default episode length during evaluation (in seconds).",
    )
    p.add_argument("--project-name", type=str, default="dummy_experiment")
    p.add_argument("--plotjuggler", action="store_true", default=False)
    p.add_argument("--max-iterations", type=int, default=300)
    p.add_argument("--stage", type=Stage, choices=list(Stage), default=Stage.RL)
    p.add_argument("--load-rl-task-id", type=str, default=None)
    p.add_argument("--load-rl-model-id", type=str, default=None)
    p.add_argument(
        "--enforce-current-config",
        action="store_true",
        help="Do not load config from RL/BC checkpoint",
    )
    # changed --stage to --control
    p.add_argument(
        "--control", type=Control, choices=list(Control), default=Control.SIM
    )
    p.add_argument("--calibration-move", type=str_to_list, default=None)
    p.add_argument("--calibration-move-cart", type=str_to_list, default=None)
    p.add_argument("--calibration-steps", type=int, default=500)
    p.add_argument("--visualize-camera", action="store_true", default=False)
    p.add_argument("--disable-vision", action="store_true", default=False)

    p.add_argument(
        "--debug-enable",
        action="store_true",
        default=False,
        help="Enable debugging tools",
    )
    p.add_argument(
        "--debug-swap-tool-cameras",
        action="store_true",
        default=False,
        help="Swap the sides of the tool cameras (i.e. left<->right).",
    )
    p.add_argument(
        "--debug-enable-vis-preview",
        action="store_true",
        default=False,
        help="Show a windows with preview of the visual observations.",
    )
    p.add_argument(
        "--debug-record-vis-obs",
        action="store_true",
        default=False,
        help="Record visual observations to a given directory in '--debug-record-dir'.",
    )
    p.add_argument("--debug-record-dir", type=Path, default=Path("/tmp/aegis_vis_obs"))

    p.add_argument(
        "--record",
        action="store_true",
        help="Record stereo images as video during evaluation",
    )
    p.add_argument(
        "--video-path",
        type=str,
        default=None,
        help="Path to save the video file (default: auto-generated)",
    )
    p.add_argument(
        "--load-from-pickle",
        action="store_true",
        help="Load configs from saved pickle instead of generating them from code",
    )
    p.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Seed for box poses across checkpoints",
    )

    if extra_argparser is not None:
        extra_argparser(p)

    args = p.parse_args(argv)

    checkpoint_path = args.checkpoint if args.checkpoint else None
    train_dirs = frozenset(args.train_dirs) if args.train_dirs else None
    val_dirs = frozenset(args.test_dirs) if args.val_dirs else None
    output_dir = args.output_dir if args.output_dir else None

    return LaunchArgs(
        config_file=args.config_file,
        checkpoint_path=checkpoint_path,
        train_dirs=train_dirs,
        val_dirs=val_dirs,
        output_dir=output_dir,
        verbosity=args.verbose,
        image_format=args.image_format,
        train_dataset_caching=args.train_caching,
        val_dataset_caching=args.val_caching,
        tracking_enabled=args.track,
        _args_raw=args,
    )


def setup_config_train(args: Namespace, task: Task) -> GraspConfig:
    cfg = GraspConfig.create_with_clearml(task)

    cfg.rl_cfg["experiment_name"] = args.exp_name or cfg.rl_cfg["experiment_name"]
    cfg.rl_cfg["max_iterations"] = args.max_iterations or cfg.rl_cfg["max_iterations"]
    cfg.env_cfg["num_envs"] = args.num_envs or cfg.env_cfg["num_envs"]
    cfg.env_cfg["visualize_camera"] = (
        args.visualize_camera or cfg.env_cfg["visualize_camera"]
    )

    # TODO(issue#111) simplify config structure
    project_suffix = f"_{str(args.stage)}-{str(args.control)}"
    cfg.logger_cfg["wandb_project"] += project_suffix
    cfg.logger_cfg["clearml_project"] += project_suffix
    cfg.logger_cfg["neptune_project"] += project_suffix
    cfg.rl_cfg.update(cfg.logger_cfg)
    cfg.bc_cfg.update(cfg.logger_cfg)

    train_type = str(args.stage)
    log_dir = Path("logs") / f"{args.exp_name}_{train_type}"
    log_dir.mkdir(parents=True, exist_ok=True)
    cfg.logger_cfg["local_log_dir"] = str(log_dir)

    cfg.debug_cfg.enabled = args.debug_enable
    if args.debug_enable:
        cfg.debug_cfg.swap_tool_cameras = args.debug_swap_tool_cameras
        cfg.debug_cfg.enable_vis_preview = args.debug_enable_vis_preview
        cfg.debug_cfg.enable_record_obs = args.debug_record_vis_obs
        cfg.debug_cfg.record_dir = args.debug_record_dir

    return cfg


def setup_config_eval(args: Namespace, task: Task) -> GraspConfig:
    if args.load_from_pickle:
        raise NotImplementedError(
            "There is no mapping for loading configs from pickle. Try loading it from ClearML."
        )

    cfg = GraspConfig.create_with_clearml(task)

    # TODO dynamic config variables should be grouped into other dict/object
    stage_type = str(args.stage)
    log_dir = Path("logs") / f"{args.exp_name}_{stage_type}_eval"
    log_dir.mkdir(parents=True, exist_ok=True)
    cfg.logger_cfg["local_log_dir"] = str(log_dir)

    cfg.env_cfg["episode_length_s"] = (
        args.episode_length_s or cfg.env_cfg["episode_length_s"]
    )
    episode_len_s = cfg.env_cfg["episode_length_s"]
    cfg.env_cfg["max_steps"] = int(episode_len_s / cfg.env_cfg["policy_dt"])

    cfg.debug_cfg.enabled = args.debug_enable
    if args.debug_enable:
        cfg.debug_cfg.swap_tool_cameras = args.debug_swap_tool_cameras
        cfg.debug_cfg.enable_vis_preview = args.debug_enable_vis_preview
        cfg.debug_cfg.enable_record_obs = args.debug_record_vis_obs
        cfg.debug_cfg.record_dir = args.debug_record_dir

    return cfg
