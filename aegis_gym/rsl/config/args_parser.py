import sys
import ast
from argparse import ArgumentParser
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional, Callable

from .types import Stage, Control


@dataclass(slots=True, frozen=True)
class LaunchArgs:
    config_path: Optional[Path]

    experiment_name: Optional[str]
    disable_headless: Optional[bool]
    num_envs: Optional[int]
    episode_length_s: Optional[float]
    project_name: Optional[str]

    enable_plotjuggler: Optional[bool]
    max_iterations: Optional[int]
    learning_method: Optional[Stage]

    # TODO(issue#111) think how to generalize it (multiple models for an experiments)
    # IDEA: connect model_id to somekind of architecture info/store architecture config in the model.
    # IDEA: move rl/bc model passing to a configuration file:
    load_rl_task_id: Optional[str]
    load_rl_model_id: Optional[str]
    load_bc_task_id: Optional[str]
    load_bc_model_id: Optional[str]

    enforce_current_config: Optional[bool]
    control_type: Optional[Control]

    calibration_move: Optional[list]
    calibration_move_cartesian: Optional[list]
    calibration_steps: Optional[int]

    visualize_camera: Optional[bool]
    disable_vision: Optional[bool]

    debug_enable: Optional[bool]
    debug_swap_tool_cameras: Optional[bool]
    debug_preview_vis_obs: Optional[bool]
    debug_record_vis_obs: Optional[bool]
    debug_record_dir: Optional[Path]

    enable_recording: Optional[bool]
    video_path: Optional[Path]
    load_from_pickle: Optional[bool]
    bc_all_checkpoints: Optional[bool]
    bc_eval_every: Optional[int]
    seed: Optional[int]

    _args_raw: Any

    def get_raw_args(self) -> Any:
        return self._args_raw

    def as_dict(self) -> dict:
        return {slot: getattr(self, slot) for slot in self.__slots__}


def parse_arguments(
    argv: Optional[list[str]] = None, extra_argparser: Optional[Callable] = None
) -> LaunchArgs:
    def str_to_list(arg: Optional[str]) -> Optional[list[float]]:
        if arg is None:
            return None
        return ast.literal_eval(arg)

    if argv is None:
        argv = sys.argv

    p = ArgumentParser(
        prog="DeepRL Experiment Runner",
        description="Run a parametrized training or evaluation of a RL model.",
        add_help=True,
    )

    p.add_argument("-c", "--config", type=Path, default=None)

    p.add_argument("-e", "--exp-name", type=str, default=None)
    p.add_argument("-v", "--vis", action="store_true", default=None)
    p.add_argument("-B", "--num-envs", type=int, default=None)
    p.add_argument("--episode-length-s", type=float, default=None)
    p.add_argument(
        "--project-name", type=str, default="TEST_PLAYGROUND/aegis_grasp"
    )  # TODO(issue#111) take it from the file if none is given
    p.add_argument("--plotjuggler", action="store_true", default=False)
    p.add_argument("--max-iterations", type=int, default=None)
    p.add_argument("--stage", type=Stage, choices=list(Stage), default=Stage.RL)
    p.add_argument("--load-rl-task-id", type=str, default=None)
    p.add_argument("--load-rl-model-id", type=str, default=None)
    p.add_argument("--load-bc-task-id", type=str, default=None)
    p.add_argument("--load-bc-model-id", type=str, default=None)
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
    p.add_argument("--calibration-steps", type=int, default=None)
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
        type=Path,
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
        default=None,
        help="Seed for box poses across checkpoints",
    )

    p.add_argument(
        "--bc-all-checkpoints",
        action="store_true",
        default=False,
        help="Sweep over all BC training checkpoints",
    )
    p.add_argument(
        "--bc-eval-every",
        type=int,
        default=None,
        help="Sweep every N-th BC checkpoint",
    )

    if extra_argparser is not None:
        extra_argparser(p)

    args = p.parse_args(argv[1:])

    return LaunchArgs(
        config_path=args.config,
        experiment_name=args.exp_name,
        disable_headless=args.vis,
        num_envs=args.num_envs,
        episode_length_s=args.episode_length_s,
        project_name=args.project_name,
        enable_plotjuggler=args.plotjuggler,
        max_iterations=args.max_iterations,
        learning_method=args.stage,
        load_rl_task_id=args.load_rl_task_id,
        load_rl_model_id=args.load_rl_model_id,
        load_bc_task_id=args.load_bc_task_id,
        load_bc_model_id=args.load_bc_model_id,
        enforce_current_config=args.enforce_current_config,
        control_type=args.control,
        calibration_move=args.calibration_move,
        calibration_move_cartesian=args.calibration_move_cart,
        calibration_steps=args.calibration_steps,
        visualize_camera=args.visualize_camera,
        disable_vision=args.disable_vision,
        debug_enable=args.debug_enable,
        debug_swap_tool_cameras=args.debug_swap_tool_cameras,
        debug_preview_vis_obs=args.debug_enable_vis_preview,
        debug_record_vis_obs=args.debug_record_vis_obs,
        debug_record_dir=args.debug_record_dir,
        enable_recording=args.record,
        video_path=args.video_path,
        load_from_pickle=args.load_from_pickle,
        bc_all_checkpoints=args.bc_all_checkpoints,
        bc_eval_every=args.bc_eval_every,
        seed=args.seed,
        _args_raw=args,
    )
