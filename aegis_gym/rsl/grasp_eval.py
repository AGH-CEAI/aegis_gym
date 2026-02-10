import argparse
import pickle
from pathlib import Path

import torch as th
import genesis as gs
from utils import check_rsl_rl_version, load_rl_policy, load_bc_policy


def main():
    check_rsl_rl_version()
    parser = argparse.ArgumentParser()
    parser.add_argument("-e", "--exp_name", type=str, default="grasp")
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
    args = parser.parse_args()

    # Set PyTorch default dtype to float32 for better performance
    th.set_default_dtype(th.float32)

    log_dir = Path("logs") / f"{args.exp_name + '_' + args.stage}"

    # Load configurations
    if args.stage == "rl":
        # For RL, load the standard configs
        env_cfg, reward_cfg, robot_cfg, rl_train_cfg, bc_train_cfg = pickle.load(
            open(log_dir / "cfgs.pkl", "rb")
        )
    else:
        # For BC, we need to load the configs and create BC config
        env_cfg, reward_cfg, robot_cfg, rl_train_cfg, bc_train_cfg = pickle.load(
            open(log_dir / "cfgs.pkl", "rb")
        )

    device = th.device("cuda" if th.cuda.is_available() else "cpu")
    env = None
    if args.control == "sim":
        env_cfg["max_visualize_FPS"] = 60
        env_cfg["num_envs"] = 100
        env_cfg["box_collision"] = True
        env_cfg["box_fixed"] = False
        env_cfg["visualize_camera"] = args.record
        env_cfg["sim_substeps"] = args.sim_substeps

        from envs.grasp_env import GraspEnv

        gs.init()
        env = GraspEnv(
            env_cfg=env_cfg,
            reward_cfg=reward_cfg,
            robot_cfg=robot_cfg,
            show_viewer=not args.no_vis,
        )
    if args.control == "ros":
        env_cfg["max_visualize_FPS"] = int(1 / env_cfg["policy_dt"])
        env_cfg["num_envs"] = 1

        try:
            from envs.grasp_env_ros import GraspEnvROS
        except ImportError as e:
            print(f">>>> ERROR: Can not import GraspEnvROS. Error:\n{e}\n>>>> Exiting")
            return
        env = GraspEnvROS(
            env_cfg=env_cfg,
            reward_cfg=reward_cfg,
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
    # max_sim_step = int(env_cfg["episode_length_s"] * env_cfg["max_visualize_FPS"])
    max_sim_step = int(env_cfg["episode_length_s"] / env_cfg["policy_dt"])

    # TODO(issue#41): Refactor camera handling to use a unified camera registry instead of dynamic attributes
    with th.no_grad():
        if args.control == "sim":
            if args.record:
                print("Recording video...")
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
            print(f"Skipping camera setup for control type: {args.control}")

        for step in range(max_sim_step):
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
        env.grasp_and_lift_demo()

        if args.control == "sim":
            if args.record:
                print("Stopping video recording...")
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
