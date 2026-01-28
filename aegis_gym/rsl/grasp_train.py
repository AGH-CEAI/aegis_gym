import argparse
import pickle
from pathlib import Path

import genesis as gs
import torch as th
from rsl_rl.runners import OnPolicyRunner

from behavior_cloning import BehaviorCloning
from grasp_cfgs import get_task_cfgs, get_rl_cfg, get_bc_cfg
from utils import check_rsl_rl_version, load_teacher_policy


def main():
    check_rsl_rl_version()
    parser = argparse.ArgumentParser()
    parser.add_argument("-e", "--exp_name", type=str, default="grasp")
    parser.add_argument("-v", "--vis", action="store_true", default=False)
    parser.add_argument("-B", "--num_envs", type=int, default=4096)
    parser.add_argument("--max_iterations", type=int, default=300)
    parser.add_argument("--stage", type=str, choices=["rl", "bc"], default="rl")
    parser.add_argument("--control", type=str, choices=["sim", "ros"], default="sim")
    args = parser.parse_args()

    # === task cfgs and training algos cfgs ===
    env_cfg, reward_scales, robot_cfg = get_task_cfgs()
    rl_train_cfg = get_rl_cfg(args.exp_name, args.max_iterations)
    bc_train_cfg = get_bc_cfg()

    project_suffix = f"_{args.stage}-{args.control}"
    rl_train_cfg["neptune_project"] += project_suffix
    rl_train_cfg["wandb_project"] += project_suffix
    rl_train_cfg["clearml_project"] += project_suffix

    # === log dir ===
    log_dir = Path("logs") / f"{args.exp_name + '_' + args.stage}"
    log_dir.mkdir(parents=True, exist_ok=True)

    with open(log_dir / "cfgs.pkl", "wb") as f:
        pickle.dump((env_cfg, reward_scales, robot_cfg, rl_train_cfg, bc_train_cfg), f)

    # === env ===
    # BC only needs a small number of envs
    env_cfg["num_envs"] = args.num_envs if args.stage == "rl" else 10

    device = th.device("cuda" if th.cuda.is_available() else "cpu")
    env = None
    if args.control == "sim":
        from envs.grasp_env import GraspEnv

        gs.init(logging_level="warning", precision="32")
        env = GraspEnv(
            env_cfg=env_cfg,
            reward_cfg=reward_scales,
            robot_cfg=robot_cfg,
            show_viewer=args.vis,
        )
    if args.control == "ros":
        try:
            from envs.grasp_env_ros import GraspEnvROS
        except ImportError as e:
            print(f">>>> ERROR: Can not import GraspEnvROS. Error:\n{e}\n>>>> Exiting")
            return
        env = GraspEnvROS(
            env_cfg=env_cfg,
            reward_cfg=reward_scales,
            robot_cfg=robot_cfg,
            device=device,
        )

    if env is None:
        print("> Env is not configured. Exiting...")
        return

    # === runner ===
    if args.stage == "bc":
        teacher_policy = load_teacher_policy(env, rl_train_cfg, args.exp_name)
        bc_train_cfg["teacher_policy"] = teacher_policy
        runner = BehaviorCloning(
            env, bc_train_cfg, teacher_policy, log_dir=log_dir, device=device
        )
        runner.learn(num_learning_iterations=args.max_iterations)
    else:
        runner = OnPolicyRunner(env, rl_train_cfg, log_dir, device=device)
        runner.learn(
            num_learning_iterations=args.max_iterations, init_at_random_ep_len=True
        )


if __name__ == "__main__":
    main()
