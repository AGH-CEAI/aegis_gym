import ast
from argparse import ArgumentParser, Namespace
from pathlib import Path
from typing import Optional

import genesis as gs
import torch as th
from rsl_rl.runners import OnPolicyRunner
from clearml import Task

from behavior_cloning import BehaviorCloning
from grasp_cfgs import GraspConfig, get_logger_cfg
from utils import load_rl_policy, Stage, Control

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
    # The ClearML task must exists for connecting configuration
    task = Task.init(
        project_name=f"{args.project_name}_{str(args.stage)}-{str(args.control)}",
        task_name=f"{args.exp_name}_{str(args.stage)}",
        reuse_last_task_id=True,
    )
    cfg = setup_config(args, task)
    cfg.set_device(th.device("cuda" if th.cuda.is_available() else "cpu"))

    env = create_env(args, cfg)
    if env is None:
        print("[GraspTrain] > Env is not configured. Exiting...")
        return
    print("[GraspTrain] > Setup done")

    if args.calibration_move or args.calibration_move_cart:
        print("[GraspTrain] > Proceeding to calibration movement")
        calibration_movment(env, args, cfg)
        return

    print("[GraspTrain] > Proceeding training")
    train_runner(env, args, cfg)


def parse_arguments() -> Namespace:
    def str_to_list(arg: Optional[str]) -> list[float]:
        if arg is None:
            return None
        return ast.literal_eval(arg)

    # TODO(issue#110) resolve the precedence of default values
    default_project_name = get_logger_cfg()["clearml_project"]

    p = ArgumentParser()
    p.add_argument("-e", "--exp-name", type=str, default="grasp")
    p.add_argument("-v", "--vis", action="store_true", default=False)
    p.add_argument("-B", "--num-envs", type=int, default=4096)
    p.add_argument("--project-name", type=str, default=default_project_name)
    p.add_argument("--plotjuggler", action="store_true", default=False)
    p.add_argument("--max-iterations", type=int, default=300)
    p.add_argument("--stage", type=Stage, choices=list(Stage), default=Stage.RL)
    p.add_argument("--load-rl-task-id", type=str, default=None)
    p.add_argument("--load-rl-model-id", type=str, default=None)
    p.add_argument(
        "--enforce-current-config",
        action="store_true",
        help="Do not load config from RL/BC checkpoint",
    )
    p.add_argument(
        "--control", type=Control, choices=list(Control), default=Control.SIM
    )
    p.add_argument("--calibration-move", type=str_to_list, default=None)
    p.add_argument("--calibration-move-cart", type=str_to_list, default=None)
    p.add_argument("--calibration-steps", type=int, default=500)
    p.add_argument("--visualize-camera", action="store_true", default=False)
    p.add_argument("--disable-vision", action="store_true", default=False)
    return p.parse_args()


def setup_config(args: Namespace, task: Task) -> GraspConfig:
    cfg = GraspConfig.create_with_clearml(task)

    cfg.rl_cfg["experiment_name"] = args.exp_name or cfg.rl_cfg["experiment_name"]
    cfg.rl_cfg["max_iterations"] = args.max_iterations or cfg.rl_cfg["max_iterations"]
    cfg.env_cfg["num_envs"] = args.num_envs or cfg.env_cfg["num_envs"]
    cfg.env_cfg["visualize_camera"] = (
        args.visualize_camera or cfg.env_cfg["visualize_camera"]
    )

    # TODO(issue#111) simplify config structure
    project_suffix = f"_{str(args.stage)}-{str(args.control)}"
    cfg.logger_cfg["wandb_project"] += project_suffix
    cfg.logger_cfg["clearml_project"] += project_suffix
    cfg.logger_cfg["neptune_project"] += project_suffix
    cfg.rl_cfg.update(cfg.logger_cfg)
    cfg.bc_cfg.update(cfg.logger_cfg)

    train_type = str(args.stage)
    log_dir = Path("logs") / f"{args.exp_name}_{train_type}"
    log_dir.mkdir(parents=True, exist_ok=True)
    cfg.logger_cfg["local_log_dir"] = str(log_dir)

    return cfg


def create_env(args: Namespace, cfg: GraspConfig) -> GraspEnvironemnt | None:
    device = cfg.get_device()
    env = None
    if args.control == Control.SIM:
        gs.init(logging_level="info", precision="32")
        env = GraspEnv(
            env_cfg=cfg.env_cfg,
            robot_cfg=cfg.robot_cfg,
            show_viewer=args.vis,
            enable_plot_juggler=args.plotjuggler,
        )
    if args.control == Control.ROS:
        if GraspEnvROS is None:
            print("[GraspTrain] >>>> ERROR: Can not import GraspEnvROS. \n>>>> Exiting")
            exit()
        cfg.env_cfg["num_envs"] = 1
        env = GraspEnvROS(
            env_cfg=cfg.env_cfg,
            robot_cfg=cfg.robot_cfg,
            disable_vision=args.disable_vision,
            device=device,
        )
    return env


def calibration_movment(
    env: GraspEnvironemnt, args: Namespace, cfg: GraspConfig
) -> None:
    device = cfg.get_device()

    cart_diff = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    joints_diff = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    steps = args.calibration_steps

    if args.calibration_move:
        n_j = len(args.calibration_move)
        joints_diff[:n_j] = args.calibration_move
        print(f"[GraspTrain] >>> Starting relative joints movement of {joints_diff}")
        joints_diff = th.tensor(joints_diff, device=device)
        joints_diff[:6] *= th.pi / 180.0
        joints_diff.unsqueeze(dim=0)
        env.calib_run(joints_diff=joints_diff, steps=steps)

    if args.calibration_move_cart:
        n_j = len(args.calibration_move_cart)
        cart_diff[:n_j] = args.calibration_move_cart
        print(f"[GraspTrain] >>> Starting relative cartesian movement of {cart_diff}")
        cart_diff = th.tensor([cart_diff], device=device)
        cart_diff.unsqueeze(dim=0)
        env.calib_run(cart_diff=cart_diff, steps=steps)

    print("[GraspTrain] >>> Finished relative joints movement.")


def train_runner(env: GraspEnvironemnt, args: Namespace, cfg: GraspConfig) -> None:
    device = cfg.get_device()
    log_dir = Path(cfg.logger_cfg["local_log_dir"])
    cfg_pickle_path = Path(cfg.logger_cfg["local_log_dir"]) / "cfgs.pkl"
    match args.stage:
        case Stage.BC:
            print("[GraspTrain] >>> Starting training: Behavioral Cloning (BC)")
            teacher_policy = load_rl_policy(
                env=env,
                rl_cfg=cfg.rl_cfg,
                device=device,
                load_cfg_from_clearml=not args.enforce_current_config,
                exp_name=args.exp_name,
                log_dir=log_dir,
                clearml_task_id=args.load_rl_task_id,
                clearml_model_id=args.load_rl_model_id,
                clearml_artifact_name="model",
                enable_logging=False,
            )
            cfg.to_pickle(cfg_pickle_path)
            print("[GraspTrain] > Saved config as a pickle.")

            runner = BehaviorCloning(
                env, cfg.bc_cfg, teacher_policy, log_dir=log_dir, device=device
            )
            runner.learn(num_learning_iterations=args.max_iterations)
        case Stage.RL:
            print("[GraspTrain] >>> Starting training: Reinforcement Learning (RL)")
            cfg.to_pickle(cfg_pickle_path)
            print("[GraspTrain] > Saved config as a pickle.")

            runner = OnPolicyRunner(env, cfg.rl_cfg, log_dir, device=device)
            runner.learn(
                num_learning_iterations=args.max_iterations, init_at_random_ep_len=True
            )
    print("[GraspTrain] > Training finished.")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n\n[GraspTrain] > Exiting (invoked by user)")
