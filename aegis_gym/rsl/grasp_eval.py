import argparse
from pathlib import Path
from typing import Any, Callable
import time

import torch as th
from clearml import Task

from behavior_cloning import BehaviorCloning
from grasp_cfgs import get_task_cfgs, get_rl_cfg, get_bc_cfg, get_logger_cfg
from utils import load_rl_policy, load_bc_policy, get_bc_checkpoints, Stage


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-e", "--exp_name", type=str, default="grasp")
    parser.add_argument("-B", "--num_envs", type=int, default=100)
    parser.add_argument("-v", "--vis", action="store_true", default=False)
    parser.add_argument("--plotjuggler", action="store_true", default=False)
    parser.add_argument(
        "--stage",
        type=Stage,
        default=Stage.RL,
        choices=list(Stage),
        help="Model type: 'rl' for reinforcement learning, 'bc' for behavior cloning",
    )
    parser.add_argument(
        "--record",
        action="store_true",
        help="Record stereo images as video during evaluation",
    )
    parser.add_argument(
        "--video_path",
        type=str,
        default=None,
        help="Path to save the video file (default: auto-generated)",
    )
    parser.add_argument("--control", type=str, choices=["sim", "ros"], default="sim")
    parser.add_argument(
        "--load-from-pickle",
        action="store_true",
        help="Load configs from saved pickle instead of generating them from code",
    )
    parser.add_argument("--load-rl-task-id", type=str, default=None)
    parser.add_argument("--load-rl-model-id", type=str, default=None)
    parser.add_argument("--load-bc-task-id", type=str, default=None)
    parser.add_argument("--load-bc-model-id", type=str, default=None)
    parser.add_argument(
        "--all-checkpoints",
        action="store_true",
        default=False,
        help="Sweep over all BC training checkpoints",
    )
    parser.add_argument(
        "--eval-every",
        type=int,
        default=None,
        help="Sweep every N-th BC checkpoint",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Seed for box poses across checkpoints",
    )
    args = parser.parse_args()

    sweep = (args.stage == Stage.BC) and (
        args.all_checkpoints or args.eval_every is not None
    )
    if args.stage == Stage.RL and (args.all_checkpoints or args.eval_every is not None):
        print(
            "[GraspEval] WARNING: multi-checkpoint sweep are only supported for BC; ignoring for RL"
        )
    if sweep and args.record:
        print("[GraspEval] WARNING: record is ignored during multi-checkpoint sweep")
        args.record = False

    logger_cfg = get_logger_cfg()

    task = Task.init(
        project_name=f"{logger_cfg['clearml_project']}_eval-{args.stage}-{args.control}",
        task_name=f"{args.exp_name}_{args.stage}_eval",
        reuse_last_task_id=False,
    )

    # Set PyTorch default dtype to float32 for better performance
    th.set_default_dtype(th.float32)

    log_dir = Path("logs") / f"{args.exp_name + '_' + args.stage}"

    # Load configurations
    if args.load_from_pickle:
        import pickle

        env_cfg, robot_cfg, rl_train_cfg, bc_train_cfg = pickle.load(
            open(log_dir / "cfgs.pkl", "rb")
        )
        print("[GraspEval] Loaded configs from pickle")
    else:
        env_cfg, robot_cfg = get_task_cfgs()
        if args.stage == Stage.RL:
            rl_train_cfg = get_rl_cfg()
        else:
            bc_train_cfg = get_bc_cfg()
        print("[GraspEval] Using configs generated from code")

    device = th.device("cuda" if th.cuda.is_available() else "cpu")
    env = None
    if args.control == "sim":
        env_cfg["max_visualize_FPS"] = 60
        env_cfg["num_envs"] = args.num_envs
        env_cfg["box_collision"] = True
        env_cfg["box_fixed"] = False
        env_cfg["visualize_camera"] = args.record

        import genesis as gs
        from envs.grasp_env import GraspEnv

        gs.init(logging_level="info", precision="32")
        env = GraspEnv(
            env_cfg=env_cfg,
            robot_cfg=robot_cfg,
            show_viewer=args.vis,
            enable_plot_juggler=args.plotjuggler,
        )
    if args.control == "ros":
        env_cfg["max_visualize_FPS"] = int(1 / env_cfg["policy_dt"])
        env_cfg["num_envs"] = 1

        try:
            from envs.grasp_env_ros import GraspEnvROS
        except ImportError as e:
            print(
                f"[GraspEval] >>>> ERROR: Can not import GraspEnvROS. Error:\n{e}\n>>>> Exiting"
            )
            return
        env = GraspEnvROS(
            env_cfg=env_cfg,
            robot_cfg=robot_cfg,
            device=device,
        )

    episode_len_s = env_cfg["episode_length_s"]
    max_steps = int(episode_len_s / env_cfg["policy_dt"])
    print(
        f"[GraspEval] The episode length is defined as {episode_len_s} s, which corresponds to {max_steps} steps"
    )

    # TODO(issue#41): Refactor camera handling to use a unified camera registry instead of dynamic attributes
    with th.no_grad():
        if args.control == "sim":
            if args.record:
                print("[GraspEval] Recording video...")
                if env_cfg["camera_setup"] == "default":
                    env.record_cam.start_recording()
                    env._cameras["scene_cam"].start_recording()
                    env._cameras["tool_left_cam"].start_recording()
                    env._cameras["tool_right_cam"].start_recording()
                elif env_cfg["camera_setup"] == "scene_dual":
                    env.record_cam.start_recording()
                    env._cameras["scene_left_cam"].start_recording()
                    env._cameras["scene_right_cam"].start_recording()
                else:
                    raise RuntimeError(
                        f"Unknown camera_setup: {env_cfg['camera_setup']}"
                    )
        else:
            print(f"[GraspEval] Skipping camera setup for control type: {args.control}")

        record_render = args.control == "sim" and args.record
        train_cfg = rl_train_cfg if args.stage == Stage.RL else bc_train_cfg

        if not sweep:
            eval_policy_single(
                env, train_cfg, args, log_dir, task, max_steps, record_render, device
            )
        else:
            eval_policy_sweep(env, train_cfg, args, log_dir, task, max_steps, device)

        if args.control == "sim":
            if args.record:
                print("[GraspEval] Stopping video recording...")
                if env_cfg["camera_setup"] == "default":
                    env.record_cam.stop_recording(
                        save_to_filename=args.video_path,
                        fps=env_cfg["max_visualize_FPS"],
                    )
                    env._cameras["scene_cam"].stop_recording(
                        save_to_filename="scene_cam.mp4",
                        fps=env_cfg["max_visualize_FPS"],
                    )
                    env._cameras["tool_left_cam"].stop_recording(
                        save_to_filename="tool_left_cam.mp4",
                        fps=env_cfg["max_visualize_FPS"],
                    )
                    env._cameras["tool_right_cam"].stop_recording(
                        save_to_filename="tool_right_cam.mp4",
                        fps=env_cfg["max_visualize_FPS"],
                    )
                elif env_cfg["camera_setup"] == "scene_dual":
                    env.record_cam.stop_recording(
                        save_to_filename=args.video_path,
                        fps=env_cfg["max_visualize_FPS"],
                    )
                    env._cameras["scene_right_cam"].stop_recording(
                        save_to_filename="scene_left_cam.mp4",
                        fps=env_cfg["max_visualize_FPS"],
                    )
                    env._cameras["scene_left_cam"].stop_recording(
                        save_to_filename="scene_right_cam.mp4",
                        fps=env_cfg["max_visualize_FPS"],
                    )


def eval_policy_single(
    env: Any,
    train_cfg: dict,
    args: argparse.Namespace,
    log_dir: Path,
    task: Task,
    max_steps: int,
    record_render: bool,
    device: th.device,
) -> None:
    # TODO(issue#101): Design arguments and config manager for policy loading
    if args.stage == Stage.RL:
        policy = load_rl_policy(
            env=env,
            rl_cfg=train_cfg,
            device=device,
            log_dir=log_dir,
            clearml_task_id=args.load_rl_task_id,
            clearml_model_id=args.load_rl_model_id,
        )
    else:
        policy = load_bc_policy(
            env=env,
            bc_cfg=train_cfg,
            device=device,
            log_dir=log_dir,
            clearml_task_id=args.load_bc_task_id,
            clearml_model_id=args.load_bc_model_id,
        )
        policy.eval()
    obs, _ = env.reset()
    metrics = run_eval(
        env, policy, args.stage, max_steps, obs, device, record_render=record_render
    )
    log_metrics(task, metrics)


def eval_policy_sweep(
    env: Any,
    train_cfg: dict,
    args: argparse.Namespace,
    log_dir: Path,
    task: Task,
    max_steps: int,
    device: th.device,
) -> None:
    checkpoints = get_bc_checkpoints(
        log_dir=log_dir,
        clearml_task_id=args.load_bc_task_id,
        clearml_model_id=args.load_bc_model_id,
    )
    if args.eval_every is not None:
        checkpoints = [ckpt for ckpt in checkpoints if ckpt.step % args.eval_every == 0]
        if not checkpoints:
            raise ValueError(
                f"[GraspEval] No checkpoints match every {args.eval_every}"
            )
    print(f"[GraspEval] Evaluating {len(checkpoints)} BC checkpoint(s)")

    bc_runner = BehaviorCloning(
        env, cfg=train_cfg, teacher=None, log_dir=log_dir, device=device
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
            env, policy, args.stage, max_steps, obs, device, record_render=False
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
) -> dict[str, float]:
    total_rewards = th.zeros(env.num_envs, device=device)
    episode_lengths = th.zeros(env.num_envs, device=device)

    start_time = time.perf_counter()
    total_inference_time = 0.0

    for _ in range(max_steps):
        match stage:
            case Stage.RL:
                actions = policy(obs)
            case Stage.BC:
                rgb_obs = env.get_observations_vis(normalize=True).float()
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
    main()
