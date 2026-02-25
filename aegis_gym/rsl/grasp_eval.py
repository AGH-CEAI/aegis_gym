import argparse
from pathlib import Path
import time

import torch as th

from grasp_cfgs import get_task_cfgs, get_rl_cfg, get_bc_cfg
from utils import load_rl_policy, load_bc_policy

from clearml import Task


def log_metrics(task, metrics):
    info_str = (
        f"Success rate: {metrics['success_rate']:.2f}\n"
        f"Mean reward: {metrics['mean_reward']:.6f}\n"
        f"Mean episode length:{metrics['mean_episode_length']:.2f}\n"
        f"Mean inference time:{metrics['mean_inference_time_s']:.6f}\n"
        f"FPS: {metrics['policy_fps']:.2f}"
    )
    print(info_str)

    logger = task.get_logger()
    logger.report_scalar("Evaluation", "success_rate", metrics["success_rate"], 0)
    logger.report_scalar("Evaluation", "mean_reward", metrics["mean_reward"], 0)
    logger.report_scalar(
        "Evaluation", "mean_episode_length", metrics["mean_episode_length"], 0
    )
    logger.report_scalar(
        "Performance", "mean_inference_time_s", metrics["mean_inference_time_s"], 0
    )
    logger.report_scalar("Performance", "policy_fps", metrics["policy_fps"], 0)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-e", "--exp_name", type=str, default="grasp")
    parser.add_argument("-B", "--num_envs", type=int, default=100)
    parser.add_argument(
        "--stage",
        type=str,
        default="rl",
        choices=["rl", "bc"],
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
    parser.add_argument("-nv", "--no_vis", action="store_true", default=False)
    parser.add_argument("-ss", "--sim-substeps", type=int, default=2)
    parser.add_argument(
        "-lp",
        "--load-from-pickle",
        action="store_true",
        help="Load configs from saved pickle instead of generating them from code",
    )
    args = parser.parse_args()

    task = Task.init(
        project_name="Grasp",
        task_name=f"grasp_eval_{args.exp_name}_{args.stage}",
    )

    task.connect(vars(args))

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
        if args.stage == "rl":
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
        env_cfg["sim_substeps"] = args.sim_substeps

        import genesis as gs
        from envs.grasp_env import GraspEnv

        gs.init(logging_level="warning", precision="32")
        env = GraspEnv(
            env_cfg=env_cfg,
            robot_cfg=robot_cfg,
            show_viewer=not args.no_vis,
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

    # Load the appropriate policy based on model type
    if args.stage == "rl":
        policy = load_rl_policy(env, rl_train_cfg, log_dir, device)
    else:
        policy = load_bc_policy(env, bc_train_cfg, log_dir, device)
        policy.eval()

    obs, _ = env.reset()
    max_steps = int(env_cfg["episode_length_s"] / env_cfg["policy_dt"])

    # TODO(issue#41): Refactor camera handling to use a unified camera registry instead of dynamic attributes
    with th.no_grad():
        if args.control == "sim":
            if args.record:
                print("[GraspEval] Recording video...")
                if env_cfg["camera_setup"] == "default":
                    env.record_cam.start_recording()
                    env.scene_cam.start_recording()
                    env.tool_left_cam.start_recording()
                    env.tool_right_cam.start_recording()
                elif env_cfg["camera_setup"] == "scene_dual":
                    env.record_cam.start_recording()
                    env.scene_left_cam.start_recording()
                    env.scene_right_cam.start_recording()
                else:
                    raise RuntimeError(
                        f"Unknown camera_setup: {env_cfg['camera_setup']}"
                    )
        else:
            print(f"[GraspEval] Skipping camera setup for control type: {args.control}")

        total_rewards = th.zeros(env_cfg["num_envs"], device=device)
        episode_lengths = th.zeros(env_cfg["num_envs"], device=device)

        start_time = time.perf_counter()
        total_inference_time = 0.0

        for _ in range(max_steps):
            if args.stage == "rl":
                actions = policy(obs)
            else:
                rgb_obs = env.get_observations_vis(normalize=True).float()
                ee_pose = env.robot.ee_pose.float()

                actions = policy(rgb_obs, ee_pose)

                # Collect frame for video recording
                if args.control == "sim" and args.record:
                    env.record_cam.render()  # render the visualization camera

            obs, rews, dones, infos = env.step(actions)

            total_rewards += rews
            episode_lengths += 1

        end_time = time.perf_counter()
        total_inference_time += end_time - start_time

        mean_reward = total_rewards.mean().item()
        mean_episode_length = episode_lengths.mean().item()
        mean_inference_time = total_inference_time / max_steps
        fps = 1.0 / mean_inference_time

        success_rate = env.grasp_and_lift_demo()

        metrics = {
            "success_rate": success_rate,
            "mean_reward": mean_reward,
            "mean_episode_length": mean_episode_length,
            "mean_inference_time_s": mean_inference_time,
            "policy_fps": fps,
        }

        log_metrics(task, metrics)

        if args.control == "sim":
            if args.record:
                print("[GraspEval] Stopping video recording...")
                if env_cfg["camera_setup"] == "default":
                    env.record_cam.stop_recording(
                        save_to_filename=args.video_path,
                        fps=env_cfg["max_visualize_FPS"],
                    )
                    env.scene_cam.stop_recording(
                        save_to_filename="scene_cam.mp4",
                        fps=env_cfg["max_visualize_FPS"],
                    )
                    env.tool_left_cam.stop_recording(
                        save_to_filename="tool_left_cam.mp4",
                        fps=env_cfg["max_visualize_FPS"],
                    )
                    env.tool_right_cam.stop_recording(
                        save_to_filename="tool_right_cam.mp4",
                        fps=env_cfg["max_visualize_FPS"],
                    )
                elif env_cfg["camera_setup"] == "scene_dual":
                    env.record_cam.stop_recording(
                        save_to_filename=args.video_path,
                        fps=env_cfg["max_visualize_FPS"],
                    )
                    env.scene_left_cam.stop_recording(
                        save_to_filename="scene_left_cam.mp4",
                        fps=env_cfg["max_visualize_FPS"],
                    )
                    env.scene_right_cam.stop_recording(
                        save_to_filename="scene_right_cam.mp4",
                        fps=env_cfg["max_visualize_FPS"],
                    )


if __name__ == "__main__":
    main()
