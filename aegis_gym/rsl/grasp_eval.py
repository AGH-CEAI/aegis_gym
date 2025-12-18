import argparse
import pickle
from pathlib import Path

import torch
import genesis as gs

from utils import check_rsl_rl_version, load_rl_policy, load_bc_policy
from grasp_env import GraspEnv


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
    parser.add_argument("-nv", "--no_vis", action="store_true", default=False)
    args = parser.parse_args()

    # Set PyTorch default dtype to float32 for better performance
    torch.set_default_dtype(torch.float32)

    gs.init()

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

    # set the max FPS for visualization
    env_cfg["max_visualize_FPS"] = 60
    # set the box collision
    env_cfg["box_collision"] = True
    # set the box fixed
    env_cfg["box_fixed"] = False
    # set the number of envs for evaluation
    env_cfg["num_envs"] = 10
    # for video recording
    env_cfg["visualize_camera"] = args.record

    env = GraspEnv(
        env_cfg=env_cfg,
        reward_cfg=reward_cfg,
        robot_cfg=robot_cfg,
        show_viewer=not args.no_vis,
    )

    # Load the appropriate policy based on model type
    if args.stage == "rl":
        policy = load_rl_policy(env, rl_train_cfg, log_dir)
    else:
        policy = load_bc_policy(env, bc_train_cfg, log_dir)
        policy.eval()

    obs, _ = env.reset()

    max_sim_step = int(env_cfg["episode_length_s"] * env_cfg["max_visualize_FPS"])

    with torch.no_grad():
        if args.record:
            print("Recording video...")
            env.vis_cam.start_recording()
            env.left_cam.start_recording()
            env.right_cam.start_recording()
        for step in range(max_sim_step):
            if args.stage == "rl":
                actions = policy(obs)
            else:
                # Get stereo grayscale images and ensure float32
                rgb_obs = env.get_stereo_rgb_images(normalize=True).float()
                ee_pose = env.robot.ee_pose.float()

                actions = policy(rgb_obs, ee_pose)

                # Collect frame for video recording
                if args.record:
                    env.vis_cam.render()  # render the visualization camera

            obs, rews, dones, infos = env.step(actions)
        env.grasp_and_lift_demo()
        if args.record:
            print("Stopping video recording...")
            env.vis_cam.stop_recording(
                save_to_filename="video.mp4", fps=env_cfg["max_visualize_FPS"]
            )
            env.left_cam.stop_recording(
                save_to_filename="left_cam.mp4", fps=env_cfg["max_visualize_FPS"]
            )
            env.right_cam.stop_recording(
                save_to_filename="right_cam.mp4", fps=env_cfg["max_visualize_FPS"]
            )


if __name__ == "__main__":
    main()
