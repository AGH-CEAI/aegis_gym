import ast
import argparse
import pickle
from pathlib import Path
from typing import Optional

import genesis as gs
import torch as th
from rsl_rl.runners import OnPolicyRunner

from behavior_cloning import BehaviorCloning
from grasp_cfgs import get_task_cfgs, get_rl_cfg, get_bc_cfg, get_logger_cfg
from utils import load_teacher_policy


def str_to_list(arg: Optional[str]) -> list[float]:
    if arg is None:
        return None
    return ast.literal_eval(arg)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-e", "--exp_name", type=str, default="grasp")
    parser.add_argument("-v", "--vis", action="store_true", default=False)
    parser.add_argument("-B", "--num_envs", type=int, default=4096)
    parser.add_argument("--plotjuggler", action="store_true", default=False)
    parser.add_argument("--max_iterations", type=int, default=300)
    parser.add_argument("--stage", type=str, choices=["rl", "bc"], default="rl")
    parser.add_argument("--control", type=str, choices=["sim", "ros"], default="sim")
    parser.add_argument("--calibration-move", type=str_to_list, default=None)
    parser.add_argument("--calibration-move-cart", type=str_to_list, default=None)
    parser.add_argument("--calibration-steps", type=int, default=500)
    args = parser.parse_args()

    # Set PyTorch default dtype to float32 for better performance
    th.set_default_dtype(th.float32)

    # === task cfgs and training algos cfgs ===
    env_cfg, robot_cfg = get_task_cfgs()
    rl_train_cfg = get_rl_cfg()
    rl_train_cfg["experiment_name"] = args.exp_name
    rl_train_cfg["max_iterations"] = args.max_iterations
    bc_train_cfg = get_bc_cfg()
    logger_cfg = get_logger_cfg()

    project_suffix = f"_{args.stage}-{args.control}"
    logger_cfg["wandb_project"] += project_suffix
    logger_cfg["clearml_project"] += project_suffix
    logger_cfg["neptune_project"] += project_suffix
    rl_train_cfg.update(logger_cfg)
    bc_train_cfg.update(logger_cfg)

    # === log dir ===
    log_dir = Path("logs") / f"{args.exp_name + '_' + args.stage}"
    log_dir.mkdir(parents=True, exist_ok=True)

    with open(log_dir / "cfgs.pkl", "wb") as f:
        pickle.dump((env_cfg, robot_cfg, rl_train_cfg, bc_train_cfg), f)

    # === env ===
    # BC only needs a small number of envs
    env_cfg["num_envs"] = args.num_envs if args.stage != "bc" else 10

    device = th.device("cuda" if th.cuda.is_available() else "cpu")
    env = None
    if args.control == "sim":
        from envs.grasp_env import GraspEnv

        gs.init(logging_level="warning", precision="32")
        env = GraspEnv(
            env_cfg=env_cfg,
            robot_cfg=robot_cfg,
            show_viewer=args.vis,
            enable_plot_juggler=args.plotjuggler,
        )
    if args.control == "ros":
        try:
            from envs.grasp_env_ros import GraspEnvROS
        except ImportError as e:
            print(
                f"[GraspTrain] >>>> ERROR: Can not import GraspEnvROS. Error:\n{e}\n>>>> Exiting"
            )
            return
        env = GraspEnvROS(
            env_cfg=env_cfg,
            robot_cfg=robot_cfg,
            device=device,
        )

    if env is None:
        print("[GraspTrain] > Env is not configured. Exiting...")
        return

    # === calibration movement ===
    if args.calibration_move or args.calibration_move_cart:
        cart_diff = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
        joints_diff = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
        steps = args.calibration_steps

        if args.calibration_move:
            n_j = len(args.calibration_move)
            joints_diff[:n_j] = args.calibration_move
            print(
                f"[GraspTrain] >>> Starting relative joints movement of {joints_diff}"
            )
            joints_diff = th.tensor(joints_diff, device=device)
            joints_diff[:6] *= th.pi / 180.0
            joints_diff.unsqueeze(dim=0)
            env.calib_run(joints_diff=joints_diff, steps=steps)

        if args.calibration_move_cart:
            n_j = len(args.calibration_move_cart)
            cart_diff[:n_j] = args.calibration_move_cart
            print(
                f"[GraspTrain] >>> Starting relative cartesian movement of {cart_diff}"
            )
            cart_diff = th.tensor([cart_diff], device=device)
            cart_diff.unsqueeze(dim=0)
            env.calib_run(cart_diff=cart_diff, steps=steps)

        print("[GraspTrain] >>> Finished relative joints movement.")
        exit()

    # === runner ===
    match args.stage:
        case "bc":
            teacher_policy = load_teacher_policy(
                env, rl_train_cfg, args.exp_name, device
            )
            bc_train_cfg["teacher_policy"] = teacher_policy
            runner = BehaviorCloning(
                env, bc_train_cfg, teacher_policy, log_dir=log_dir, device=device
            )
            runner.learn(num_learning_iterations=args.max_iterations)
        case "rl":
            runner = OnPolicyRunner(env, rl_train_cfg, log_dir, device=device)
            runner.learn(
                num_learning_iterations=args.max_iterations, init_at_random_ep_len=True
            )


if __name__ == "__main__":
    main()
