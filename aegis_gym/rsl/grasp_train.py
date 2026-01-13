import argparse
import pickle
from pathlib import Path

import genesis as gs
from rsl_rl.runners import OnPolicyRunner

from behavior_cloning import BehaviorCloning
from utils import check_rsl_rl_version, load_teacher_policy
from grasp_env import GraspEnv
from grasp_cfgs import get_task_cfgs, get_rl_cfg, get_bc_cfg


def main():
    check_rsl_rl_version()
    parser = argparse.ArgumentParser()
    parser.add_argument("-e", "--exp_name", type=str, default="grasp")
    parser.add_argument("-v", "--vis", action="store_true", default=False)
    parser.add_argument("-B", "--num_envs", type=int, default=4096)
    parser.add_argument("--max_iterations", type=int, default=300)
    parser.add_argument("--stage", type=str, default="rl")
    args = parser.parse_args()

    # === init ===
    gs.init(logging_level="warning", precision="32")

    # === task cfgs and training algos cfgs ===
    env_cfg, reward_scales, robot_cfg = get_task_cfgs()
    rl_train_cfg = get_rl_cfg(args.exp_name, args.max_iterations)
    bc_train_cfg = get_bc_cfg()

    # === log dir ===
    log_dir = Path("logs") / f"{args.exp_name + '_' + args.stage}"
    log_dir.mkdir(parents=True, exist_ok=True)

    with open(log_dir / "cfgs.pkl", "wb") as f:
        pickle.dump((env_cfg, reward_scales, robot_cfg, rl_train_cfg, bc_train_cfg), f)

    # === env ===
    # BC only needs a small number of envs
    env_cfg["num_envs"] = args.num_envs if args.stage == "rl" else 10
    env = GraspEnv(
        env_cfg=env_cfg,
        reward_cfg=reward_scales,
        robot_cfg=robot_cfg,
        show_viewer=args.vis,
    )

    # === runner ===
    if args.stage == "bc":
        teacher_policy = load_teacher_policy(env, rl_train_cfg, args.exp_name)
        bc_train_cfg["teacher_policy"] = teacher_policy
        runner = BehaviorCloning(env, bc_train_cfg, teacher_policy, device=gs.device)
        runner.learn(num_learning_iterations=args.max_iterations, log_dir=log_dir)
    else:
        runner = OnPolicyRunner(env, rl_train_cfg, log_dir, device=gs.device)
        runner.learn(
            num_learning_iterations=args.max_iterations, init_at_random_ep_len=True
        )


if __name__ == "__main__":
    main()
