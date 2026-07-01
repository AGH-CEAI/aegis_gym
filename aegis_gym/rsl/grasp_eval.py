import time
from pathlib import Path
from typing import Any, Callable

import torch as th
from clearml import Task
from tqdm import tqdm


from behavior_cloning import BehaviorCloning
from utils import load_rl_policy, load_bc_policy, get_bc_checkpoints

from config import ConfigManager, LaunchArgs, parse_arguments
from config.types import ExpConfig, DebugCfg, Algorithm, Control, CamerasSetup
from envs import BaseEnv

from grasp_train import init_clearml_task, create_env


def main():
    # Set PyTorch default dtype to float32 for better performance
    th.set_default_dtype(th.float32)

    args: LaunchArgs = parse_arguments()
    # The ClearML task must exists for connecting configuration
    task = init_clearml_task(
        # TODO setup the ClearML task in the Configmanager to avoid the problem with project_name
        project_name=args.project_name,
        algorithm=args.algorithm,
        control=args.control_type,
        exp_name=args.experiment_name,
    )
    device = th.device("cuda" if th.cuda.is_available() else "cpu")
    ConfigManager.setup_config(argv=args, device=device, task=task)
    cfg: ExpConfig = ConfigManager.get_config()
    sweep = is_checkpoints_sweep_required(args)

    env = create_env(cfg)
    print(
        f"[GraspEval] The episode length is defined as {cfg.env_cfg.episode_length_s} s, which corresponds to {cfg.env_cfg.max_steps}"
    )
    print("[GraspEval] Setup done")

    with th.no_grad():
        start_cameras_recording(env=env, cfg=cfg)

        if sweep:
            eval_policy_sweep(env=env, cfg=cfg, task=task)
        else:
            eval_policy_single(env=env, cfg=cfg, task=task)

        stop_cameras_recording(env=env, cfg=cfg)

    print("[GraspEval] Finished evaluation script")


def is_checkpoints_sweep_required(args: LaunchArgs) -> bool:
    sweep = (args.algorithm == Algorithm.BC) and (
        args.bc_all_checkpoints or args.bc_eval_every is not None
    )
    if args.algorithm == Algorithm.RL and (
        args.bc_all_checkpoints or args.bc_eval_every is not None
    ):
        print(
            "[GraspEval] WARNING: multi-checkpoint sweep are only supported for BC; ignoring for RL"
        )
    if sweep and args.enable_recording:
        print("[GraspEval] WARNING: record is ignored during multi-checkpoint sweep")
        # args.enable_recording = False # TODO ensure to disable the recording
    return sweep


def load_policy(env: BaseEnv, cfg: ExpConfig) -> Callable:
    args: LaunchArgs = cfg.args
    device = cfg.get_device()

    algorithm = args.algorithm
    if algorithm == Algorithm.RL:
        return load_rl_policy(
            env=env,
            rl_cfg=cfg.rl_cfg,
            logger_cfg=cfg.logger_cfg,
            device=device,
            load_cfg_from_clearml=not args.enforce_current_config,
            exp_name=args.experiment_name,
            clearml_task_id=args.load_rl_task_id,
            clearml_model_id=args.load_rl_model_id,
            clearml_artifact_name="model",
            enable_logging=False,
        )
    if algorithm == Algorithm.BC:
        policy = load_bc_policy(
            env=env,
            bc_cfg=cfg.bc_cfg,
            logger_cfg=cfg.logger_cfg,
            debug_cfg=cfg.debug_cfg,
            device=device,
            load_cfg_from_clearml=not args.enforce_current_config,
            exp_name=args.experiment_name,
            clearml_task_id=args.load_bc_task_id,
            clearml_model_id=args.load_bc_model_id,
            enable_logging=False,
        )
        policy.eval()
        return policy
    raise ValueError("Unknown learning method")


def start_cameras_recording(env: BaseEnv, cfg: ExpConfig) -> None:
    args = cfg.args
    if not args.control_type == Control.SIM:
        print(
            f"[GraspEval] Skipping camera setup for control type: {str(args.control_type)}"
        )
    if not args.enable_recording:
        return
    cameras_setup = cfg.env_cfg.cameras_setup

    # TODO(issue#41): Refactor camera handling to use a unified camera registry instead of dynamic attributes
    match cameras_setup:
        case CamerasSetup.DEFAULT:
            env.record_cam.start_recording()
            env._cameras["scene_cam"].start_recording()
            env._cameras["tool_left_cam"].start_recording()
            env._cameras["tool_right_cam"].start_recording()
        case CamerasSetup.SCENE_DUAL:
            env.record_cam.start_recording()
            env._cameras["scene_left_cam"].start_recording()
            env._cameras["scene_right_cam"].start_recording()
    print(f"[GraspEval] Recording video (camera setup: {cameras_setup})...")


def stop_cameras_recording(env: BaseEnv, cfg: ExpConfig) -> None:
    args = cfg.args
    if not args.control_type == Control.SIM:
        return

    if not args.enable_recording:
        return

    print("[GraspEval] Stopping video recording...")
    cameras_setup = cfg.env_cfg.cameras_setup
    fps = int(1 / cfg.env_cfg.policy_dt)

    match cameras_setup:
        case CamerasSetup.DEFAULT:
            env.record_cam.stop_recording(
                save_to_filename=str(args.video_path),
                fps=fps,
            )
            env._cameras["scene_cam"].stop_recording(
                save_to_filename="scene_cam.mp4",
                fps=fps,
            )
            env._cameras["tool_left_cam"].stop_recording(
                save_to_filename="tool_left_cam.mp4",
                fps=fps,
            )
            env._cameras["tool_right_cam"].stop_recording(
                save_to_filename="tool_right_cam.mp4",
                fps=fps,
            )
        case CamerasSetup.SCENE_DUAL:
            env.record_cam.stop_recording(
                save_to_filename=str(args.video_path),
                fps=fps,
            )
            env._cameras["scene_right_cam"].stop_recording(
                save_to_filename="scene_left_cam.mp4",
                fps=fps,
            )
            env._cameras["scene_left_cam"].stop_recording(
                save_to_filename="scene_right_cam.mp4",
                fps=fps,
            )


def eval_policy_single(
    env: Any,
    cfg: ExpConfig,
    task: Task,
) -> None:
    args = cfg.args
    record_render = args.control_type == Control.SIM and args.enable_recording
    device = cfg.get_device()
    max_steps = cfg.env_cfg.max_steps

    # TODO(issue#101): Design arguments and config manager for policy loading
    policy = load_policy(env, cfg)
    obs, _ = env.reset()
    metrics = run_eval(
        env,
        policy,
        args.algorithm,
        max_steps,
        obs,
        device,
        record_render=record_render,
        debug_cfg=cfg.debug_cfg,
    )
    log_metrics(task, metrics)


def eval_policy_sweep(
    env: Any,
    cfg: ExpConfig,
    task: Task,
) -> None:
    args = cfg.args
    log_dir = Path(cfg.logger_cfg.local_log_dir)
    device = cfg.get_device()
    max_steps = cfg.env_cfg.max_steps

    checkpoints = get_bc_checkpoints(
        log_dir=log_dir,
        clearml_task_id=args.load_bc_task_id,
        clearml_model_id=args.load_bc_model_id,
    )
    if args.bc_eval_every is not None:
        checkpoints = [
            ckpt for ckpt in checkpoints if ckpt.step % args.bc_eval_every == 0
        ]
        if not checkpoints:
            raise ValueError(
                f"[GraspEval] No checkpoints match every {args.bc_eval_every}"
            )
    print(
        f"[GraspEval] Evaluating {len(checkpoints)} BC checkpoint(s): {[ckpt.step for ckpt in checkpoints]}"
    )

    bc_runner = BehaviorCloning(
        env,
        bc_cfg=cfg.bc_cfg,
        logger_cfg=cfg.logger_cfg,
        debug_cfg=cfg.debug_cfg,
        teacher=None,
        device=device,
    )
    object_pose = env.generate_object_poses(seed=args.seed)

    for ckpt in checkpoints:
        print(f"\n[GraspEval] === Checkpoint iter {ckpt.step:04d} ===")
        bc_runner.load(str(ckpt.path))
        policy = bc_runner._policy
        policy.eval()

        obs, _ = env.reset()
        env.apply_object_poses(object_pose)
        env.scene.step()
        obs = env.get_observations()

        metrics = run_eval(
            env,
            policy,
            args.algorithm,
            max_steps,
            obs,
            device,
            record_render=False,
            debug_cfg=cfg.debug_cfg,
        )
        log_metrics(task, metrics, step=ckpt.step)


# TODO(issue#100): Unify policy model types under a common base class or type alias
def run_eval(
    env: Any,
    policy: Callable,
    stage: Algorithm,
    max_steps: int,
    obs: Any,
    device: th.device,
    debug_cfg: DebugCfg,
    record_render: bool = False,
) -> dict[str, float]:
    total_rewards = th.zeros(env.num_envs, device=device)
    episode_lengths = th.zeros(env.num_envs, device=device)

    start_time = time.perf_counter()
    total_inference_time = 0.0

    def get_obs_vis() -> th.Tensor:
        if not debug_cfg.enabled:
            return env.get_observations_vis()
        return env.get_observations_vis(
            swap_tool_cameras=debug_cfg.swap_tool_cameras,
            enable_vis_preview=debug_cfg.enable_vis_preview,
            enable_record_obs=debug_cfg.enable_record_obs,
            record_dir=debug_cfg.record_dir,
        )

    vis_debug_params = {}
    if debug_cfg.enabled:
        vis_debug_params = debug_cfg.as_dict()
        vis_debug_params.pop("enabled")

    for _ in tqdm(range(max_steps), desc="Evaluation", unit="step"):
        match stage:
            case Algorithm.RL:
                actions = policy(obs)
            case Algorithm.BC:
                rgb_obs = get_obs_vis()
                tcp_pose = env.robot.get_tcp_pose()
                actions = policy(rgb_obs, tcp_pose)
                if record_render:
                    env.record_cam.render()

        obs, rews, dones, infos = env.step(actions)

        total_rewards += rews
        episode_lengths += 1

    print("[GraspEval] Finished model inference, proceeding to procedural grasp demo")

    end_time = time.perf_counter()
    total_inference_time += end_time - start_time

    mean_reward = total_rewards.mean().item()
    mean_episode_length = episode_lengths.mean().item()
    mean_inference_time = total_inference_time / max_steps
    fps = 1.0 / mean_inference_time

    success_rate = env.grasp_and_lift_demo()

    return {
        "success_rate": success_rate,
        "mean_reward": mean_reward,
        "mean_episode_length": mean_episode_length,
        "mean_inference_time_s": mean_inference_time,
        "policy_fps": fps,
    }


def log_metrics(task: Task, metrics: dict[str, float], step: int = 0) -> None:
    info_str = (
        f"Success rate: {metrics['success_rate']:.2f}\n"
        f"Mean reward: {metrics['mean_reward']:.6f}\n"
        f"Mean episode length: {metrics['mean_episode_length']:.0f}\n"
        f"Mean inference time: {metrics['mean_inference_time_s']:.6f}\n"
        f"FPS: {metrics['policy_fps']:.2f}"
    )
    print(info_str)

    logger = task.get_logger()
    logger.report_scalar("Eval/success_rate", "series", metrics["success_rate"], step)
    logger.report_scalar("Eval/mean_reward", "series", metrics["mean_reward"], step)
    logger.report_scalar(
        "Eval/mean_episode_length", "series", metrics["mean_episode_length"], step
    )
    logger.report_scalar(
        "Perf/mean_inference_time_s", "series", metrics["mean_inference_time_s"], step
    )
    logger.report_scalar("Perf/fps", "series", metrics["policy_fps"], step)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n\n[GraspEval] > Exiting (invoked by user)")
