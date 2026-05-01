from argparse import ArgumentParser, Namespace
from pathlib import Path
import time

import torch as th
from clearml import Task

from grasp_cfgs import GraspConfig, get_logger_cfg
from utils import load_rl_policy, load_bc_policy

from envs.grasp_env import GraspEnv

try:
    from envs.grasp_env_ros import GraspEnvROS
except ImportError:
    GraspEnvROS = None

GraspEnvironemnt = GraspEnv | GraspEnvROS


def main():
    # Set PyTorch default dtype to float32 for better performance
    th.set_default_dtype(th.float32)
    args = parse_arguments()

    task = Task.init(
        project_name=f"{args.project_name}_eval-{args.stage}-{args.control}",
        task_name=f"{args.exp_name}_{args.stage}_eval",
        # Probably there will bo no way to control parameters from ClearML UI without reusing task
        reuse_last_task_id=False,
    )
    cfg = setup_config(args, task)
    cfg.set_device(th.device("cuda" if th.cuda.is_available() else "cpu"))
    device = cfg.get_device()

    env = create_env(args, cfg)
    if env is None:
        print("[GraspEval] > Env is not configured. Exiting...")
        return

    policy = load_policy(env, args, cfg)
    if env is None:
        print("[GraspEval] > Failed to load policy. Exiting...")
        return

    obs, _ = env.reset()
    episode_len_s = cfg.env_cfg["episode_length_s"]
    max_steps = int(episode_len_s / cfg.env_cfg["policy_dt"])
    print(
        f"[GraspEval] The episode length is defined as {episode_len_s} s, which corresponds to {max_steps} steps"
    )
    print("[GraspEval] Setup done")

    # TODO(issue#41): Refactor camera handling to use a unified camera registry instead of dynamic attributes
    with th.no_grad():
        start_cameras_recording(env, args, cfg)

        total_rewards = th.zeros(cfg.env_cfg["num_envs"], device=device)
        episode_lengths = th.zeros(cfg.env_cfg["num_envs"], device=device)

        start_time = time.perf_counter()
        total_inference_time = 0.0

        print("[GraspEval] Starting evaluation")
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
        print(
            "[GraspEval] Finished model inference, proceeding to procedural grasp demo"
        )

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
        stop_cameras_recording(env, args, cfg)

    print("[GraspEval] Finished evaluation script")


def parse_arguments() -> Namespace:
    # TODO resolve the precedence of default values
    default_project_name = get_logger_cfg()["clearml_project"]

    p = ArgumentParser()
    p.add_argument("-e", "--exp_name", type=str, default="grasp")
    p.add_argument("-v", "--vis", action="store_true", default=False)
    p.add_argument("-B", "--num_envs", type=int, default=100)
    p.add_argument("--project-name", type=str, default=default_project_name)
    p.add_argument("--plotjuggler", action="store_true", default=False)
    p.add_argument(
        "--stage",
        type=str,
        default="rl",
        choices=["rl", "bc"],
        help="Model type: 'rl' for reinforcement learning, 'bc' for behavior cloning",
    )
    p.add_argument(
        "--record",
        action="store_true",
        help="Record stereo images as video during evaluation",
    )
    p.add_argument(
        "--video_path",
        type=str,
        default=None,
        help="Path to save the video file (default: auto-generated)",
    )
    p.add_argument("--control", type=str, choices=["sim", "ros"], default="sim")
    p.add_argument(
        "--load-from-pickle",
        action="store_true",
        help="Load configs from saved pickle instead of generating them from code",
    )
    p.add_argument("--load-rl-task-id", type=str, default=None)
    p.add_argument("--load-rl-model-id", type=str, default=None)
    p.add_argument("--load-bc-task-id", type=str, default=None)
    p.add_argument("--load-bc-model-id", type=str, default=None)
    return p.parse_args()


def setup_config(args: Namespace, task: Task) -> GraspConfig:
    if args.load_from_pickle:
        raise NotImplementedError(
            "There is no mapping for loading configs from pickle. Try loading it from ClearML."
        )

    return GraspConfig.create_with_clearml(task)


def create_env(args: Namespace, cfg: GraspConfig) -> GraspEnvironemnt | None:
    device = cfg.get_device()
    env = None
    if args.control == "sim":
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
    if args.control == "ros":
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
    env: GraspEnvironemnt, args: Namespace, cfg: GraspConfig
) -> callable | None:
    device = cfg.get_device()
    log_dir = Path(cfg.logger_cfg["local_log_dir"])
    policy = None

    if args.stage == "rl":
        policy = load_rl_policy(
            env=env,
            rl_cfg=cfg.rl_cfg,
            device=device,
            log_dir=log_dir,
            clearml_task_id=args.load_rl_task_id,
            clearml_model_id=args.load_rl_model_id,
        )
    if args.stage == "bc":
        policy = load_bc_policy(
            env=env,
            bc_cfg=cfg.bc_cfg,
            device=device,
            log_dir=log_dir,
            clearml_task_id=args.load_bc_task_id,
            clearml_model_id=args.load_bc_model_id,
        )
        policy.eval()
    return policy


def start_cameras_recording(
    env: GraspEnvironemnt, args: Namespace, cfg: GraspConfig
) -> None:
    if not args.control == "sim":
        print(f"[GraspEval] Skipping camera setup for control type: {args.control}")
    if not args.record:
        return
    camera_setup = cfg.env_cfg["camera_setup"]

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
    print(f"[GraspEval] Recording video (camera setup: {args.control})...")


def stop_cameras_recording(
    env: GraspEnvironemnt, args: Namespace, cfg: GraspConfig
) -> None:
    if not args.control == "sim":
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


def log_metrics(task: Task, metrics: dict) -> None:
    print(
        f"Success rate: {metrics['success_rate']:.2f}\n"
        f"Mean reward: {metrics['mean_reward']:.6f}\n"
        f"Mean episode length: {metrics['mean_episode_length']:.0f}\n"
        f"Mean inference time: {metrics['mean_inference_time_s']:.6f}\n"
        f"FPS: {metrics['policy_fps']:.2f}"
    )

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


if __name__ == "__main__":
    main()
