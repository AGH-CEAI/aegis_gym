import time
from argparse import ArgumentParser, Namespace
from pathlib import Path
from typing import Any, Callable

import torch as th
from clearml import Task
from tqdm import tqdm


from behavior_cloning import BehaviorCloning
from grasp_cfgs import GraspConfig, get_logger_cfg
from utils import load_rl_policy, load_bc_policy, get_bc_checkpoints, Stage, Control

from envs.grasp_env import GraspEnv

try:
    from envs.grasp_env_ros import GraspEnvROS
except ImportError:
    GraspEnvROS = None

GraspEnvironment = GraspEnv | GraspEnvROS


def main():
    # Set PyTorch default dtype to float32 for better performance
    th.set_default_dtype(th.float32)
    args = parse_arguments()

    task = Task.init(
        project_name=f"{args.project_name}_eval-{str(args.stage)}-{str(args.control)}",
        task_name=f"{args.exp_name}_{str(args.stage)}_eval",
        # Probably there will bo no way to control parameters from ClearML UI without reusing task
        reuse_last_task_id=False,
    )
    cfg = setup_config(args, task)
    cfg.set_device(th.device("cuda" if th.cuda.is_available() else "cpu"))
    sweep = is_checkpoints_sweep_required(args)

    env = create_env(args, cfg)
    if env is None:
        print("[GraspEval] > Env is not configured. Exiting...")
        return

    episode_len_s = cfg.env_cfg["episode_length_s"]
    max_steps = cfg.env_cfg["max_steps"]
    print(
        f"[GraspEval] The episode length is defined as {episode_len_s} s, which corresponds to {max_steps} steps"
    )
    print("[GraspEval] Setup done")

    with th.no_grad():
        start_cameras_recording(env=env, args=args, cfg=cfg)

        if sweep:
            eval_policy_sweep(env=env, args=args, cfg=cfg, task=task)
        else:
            eval_policy_single(env=env, args=args, cfg=cfg, task=task)

        stop_cameras_recording(env=env, args=args, cfg=cfg)

    print("[GraspEval] Finished evaluation script")


def parse_arguments() -> Namespace:
    # TODO(issue#101) resolve the precedence of default values
    default_project_name = get_logger_cfg()["clearml_project"]

    p = ArgumentParser()
    p.add_argument("-e", "--exp-name", type=str, default="grasp")
    p.add_argument("-v", "--vis", action="store_true", default=False)
    p.add_argument("-B", "--num-envs", type=int, default=100)
    p.add_argument("--project-name", type=str, default=default_project_name)
    p.add_argument("--plotjuggler", action="store_true", default=False)
    p.add_argument(
        "--episode-length",
        type=float,
        default=None,
        help="Overwrite the default episode length during evaluation (in seconds).",
    )
    p.add_argument(
        "--stage",
        type=Stage,
        default=Stage.RL,
        choices=list(Stage),
        help=f"Model type: '{str(Stage.RL)}' for reinforcement learning, '{str(Stage.BC)}' for behavior cloning",
    )
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
        "--control", type=Control, choices=list(Control), default=Control.SIM
    )
    p.add_argument(
        "--load-from-pickle",
        action="store_true",
        help="Load configs from saved pickle instead of generating them from code",
    )
    p.add_argument("--load-rl-task-id", type=str, default=None)
    p.add_argument("--load-rl-model-id", type=str, default=None)
    p.add_argument("--load-bc-task-id", type=str, default=None)
    p.add_argument("--load-bc-model-id", type=str, default=None)
    p.add_argument(
        "--enforce-current-config",
        action="store_true",
        help="Do not load config from RL/BC checkpoint",
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
    p.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Seed for box poses across checkpoints",
    )

    p.add_argument(
        "--debug-swap-tool-cameras",
        action="store_true",
        default=False,
        help="Swap the sides of the tool cameras (i.e. left<->right).",
    )

    return p.parse_args()


def setup_config(args: Namespace, task: Task) -> GraspConfig:
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

    return cfg


def is_checkpoints_sweep_required(args: Namespace) -> bool:
    sweep = (args.stage == Stage.BC) and (
        args.bc_all_checkpoints or args.bc_eval_every is not None
    )
    if args.stage == Stage.RL and (
        args.bc_all_checkpoints or args.bc_eval_every is not None
    ):
        print(
            "[GraspEval] WARNING: multi-checkpoint sweep are only supported for BC; ignoring for RL"
        )
    if sweep and args.record:
        print("[GraspEval] WARNING: record is ignored during multi-checkpoint sweep")
        args.record = False
    return sweep


def create_env(args: Namespace, cfg: GraspConfig) -> GraspEnvironment | None:
    device = cfg.get_device()
    env = None
    if args.control == Control.SIM:
        cfg.env_cfg["max_visualize_FPS"] = 60
        cfg.env_cfg["num_envs"] = args.num_envs
        cfg.env_cfg["box_collision"] = True
        cfg.env_cfg["box_fixed"] = False
        cfg.env_cfg["visualize_camera"] = args.record

        import genesis as gs
        from envs.grasp_env import GraspEnv

        gs.init(logging_level="info", precision="32")
        env = GraspEnv(
            env_cfg=cfg.env_cfg,
            robot_cfg=cfg.robot_cfg,
            show_viewer=args.vis,
            enable_plot_juggler=args.plotjuggler,
        )
    if args.control == Control.ROS:
        cfg.env_cfg["max_visualize_FPS"] = int(1 / cfg.env_cfg["policy_dt"])
        cfg.env_cfg["num_envs"] = 1

        if GraspEnvROS is None:
            print("[GraspTrain] >>>> ERROR: Can not import GraspEnvROS. \n>>>> Exiting")
            exit()

        env = GraspEnvROS(
            env_cfg=cfg.env_cfg,
            robot_cfg=cfg.robot_cfg,
            device=device,
        )
    return env


def load_policy(
    env: GraspEnvironment, args: Namespace, cfg: GraspConfig
) -> Callable | None:
    device = cfg.get_device()
    log_dir = Path(cfg.logger_cfg["local_log_dir"])
    policy = None

    if args.stage == Stage.RL:
        policy = load_rl_policy(
            env=env,
            rl_cfg=cfg.rl_cfg,
            device=device,
            load_cfg_from_clearml=not args.enforce_current_config,
            log_dir=log_dir,
            clearml_task_id=args.load_rl_task_id,
            clearml_model_id=args.load_rl_model_id,
        )
    if args.stage == Stage.BC:
        policy = load_bc_policy(
            env=env,
            bc_cfg=cfg.bc_cfg,
            device=device,
            load_cfg_from_clearml=not args.enforce_current_config,
            log_dir=log_dir,
            clearml_task_id=args.load_bc_task_id,
            clearml_model_id=args.load_bc_model_id,
        )
        policy.eval()
    return policy


def start_cameras_recording(
    env: GraspEnvironment, args: Namespace, cfg: GraspConfig
) -> None:
    if not args.control == Control.SIM:
        print(
            f"[GraspEval] Skipping camera setup for control type: {str(args.control)}"
        )
    if not args.record:
        return
    camera_setup = cfg.env_cfg["camera_setup"]

    # TODO(issue#41): Refactor camera handling to use a unified camera registry instead of dynamic attributes
    match camera_setup:
        case "default":
            env.record_cam.start_recording()
            env._cameras["scene_cam"].start_recording()
            env._cameras["tool_left_cam"].start_recording()
            env._cameras["tool_right_cam"].start_recording()
        case "scene_dual":
            env.record_cam.start_recording()
            env._cameras["scene_left_cam"].start_recording()
            env._cameras["scene_right_cam"].start_recording()
        case _:
            raise RuntimeError(f"Unknown camera_setup: {camera_setup}")
    print(f"[GraspEval] Recording video (camera setup: {camera_setup})...")


def stop_cameras_recording(
    env: GraspEnvironment, args: Namespace, cfg: GraspConfig
) -> None:
    if not args.control == Control.SIM:
        return
    if not args.record:
        return

    print("[GraspEval] Stopping video recording...")
    cameras_setup = cfg.env_cfg["camera_setup"]
    fps = cfg.env_cfg["max_visualize_FPS"]

    match cameras_setup:
        case "default":
            env.record_cam.stop_recording(
                save_to_filename=args.video_path,
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
        case "scene_dual":
            env.record_cam.stop_recording(
                save_to_filename=args.video_path,
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
    cfg: GraspConfig,
    args: Namespace,
    task: Task,
) -> None:
    record_render = args.control == Control.SIM and args.record
    device = cfg.get_device()
    max_steps = cfg.env_cfg["max_steps"]

    # TODO(issue#101): Design arguments and config manager for policy loading
    policy = load_policy(env, args, cfg)
    obs, _ = env.reset()
    metrics = run_eval(
        env,
        policy,
        args.stage,
        max_steps,
        obs,
        device,
        record_render=record_render,
        swap_tool_cameras=args.debug_swap_tool_cameras,
    )
    log_metrics(task, metrics)


def eval_policy_sweep(
    env: Any,
    args: Namespace,
    cfg: GraspConfig,
    task: Task,
) -> None:
    log_dir = Path(cfg.logger_cfg["local_log_dir"])
    device = cfg.get_device()
    max_steps = cfg.env_cfg["max_steps"]

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
        env, cfg=cfg.bc_cfg, teacher=None, log_dir=log_dir, device=device
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
            args.stage,
            max_steps,
            obs,
            device,
            record_render=False,
            swap_tool_cameras=args.debug_swap_tool_cameras,
        )
        log_metrics(task, metrics, step=ckpt.step)


# TODO(issue#100): Unify policy model types under a common base class or type alias
def run_eval(
    env: Any,
    policy: Callable,
    stage: Stage,
    max_steps: int,
    obs: Any,
    device: th.device,
    record_render: bool = False,
    swap_tool_cameras: bool = False,
) -> dict[str, float]:
    total_rewards = th.zeros(env.num_envs, device=device)
    episode_lengths = th.zeros(env.num_envs, device=device)

    start_time = time.perf_counter()
    total_inference_time = 0.0

    for _ in tqdm(range(max_steps), desc="Evaluation", unit="step"):
        match stage:
            case Stage.RL:
                actions = policy(obs)
            case Stage.BC:
                rgb_obs = env.get_observations_vis(
                    normalize=True, swap_tool_cameras=swap_tool_cameras
                ).float()
                ee_pose = env.robot.ee_pose.float()
                actions = policy(rgb_obs, ee_pose)
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
