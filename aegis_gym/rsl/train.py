import genesis as gs
import torch as th
from clearml import Task

from envs import BaseEnv
from envs.wrappers import VisionAugEnvWrapper, ObsPreviewEnvWrapper

from envs import ReacherEnv
from runners import OnPolicyRunner, BehaviorCloningRunner
from config import ConfigManager, LaunchArgs, parse_arguments
from config.types import ExpConfig, Algorithm, Control
from aux import load_rl_policy


from envs.scene import GenesisScene, RosGrcpScene


def init_clearml_task(
    project_name: str | None,
    algorithm: Algorithm | None,
    control: Control | None,
    exp_name: str | None,
) -> Task:
    assert None not in (project_name, algorithm, control, exp_name)
    return Task.init(
        project_name=f"{project_name}_{str(algorithm)}-{str(control)}",
        task_name=f"{exp_name}_{str(algorithm)}",
        reuse_last_task_id=True,
    )


# TODO(issue#130) Real training with BC doesn't work, mark this down
def main():
    # Set PyTorch default dtype to float32 for better performance
    th.set_default_dtype(th.float32)

    args: LaunchArgs = parse_arguments()
    # The ClearML task must exists for connecting configuration
    task = init_clearml_task(
        # TODO(issue#120) setup the ClearML task in the Configmanager to avoid the problem with project_name
        project_name=args.project_name,
        algorithm=args.algorithm,
        control=args.control_type,
        exp_name=args.experiment_name,
    )
    device = th.device("cuda" if th.cuda.is_available() else "cpu")
    ConfigManager.setup_config(argv=args, device=device, task=task)
    cfg: ExpConfig = ConfigManager.get_config()

    env = create_env(cfg)
    if env is None:
        print("[GraspTrain] > Env is not configured. Exiting...")
        return
    print("[GraspTrain] > Setup done")

    if args.calibration_move or args.calibration_move_cartesian:
        print("[GraspTrain] > Proceeding to calibration movement")
        calibration_movment(env, cfg)
        return

    print("[GraspTrain] > Proceeding training")
    train_runner(env=env, cfg=cfg)


def create_env(cfg: ExpConfig) -> BaseEnv:
    args: LaunchArgs = cfg.args
    control_type = args.control_type

    scene = None
    if control_type == Control.SIM:
        gs.init(logging_level="info", precision="32")
        scene = GenesisScene(cfg=cfg, device=cfg.get_device())
    if control_type == Control.ROS:
        if RosGrcpScene is None:
            print("[GraspTrain] >>>> ERROR: Can not import GraspEnvROS. \n>>>> Exiting")
            exit()
        scene = RosGrcpScene(cfg=cfg, device=cfg.get_device())

    if scene is None:
        raise ValueError("Scene is None")

    return ReacherEnv(scene=scene, cfg=cfg)


def calibration_movment(env: BaseEnv, cfg: ExpConfig) -> None:
    args = cfg.args
    device = cfg.get_device()

    cart_diff = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    joints_diff = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    steps = args.calibration_steps

    if args.calibration_movment:
        n_j = len(args.calibration_move)
        joints_diff[:n_j] = args.calibration_move
        print(f"[GraspTrain] >>> Starting relative joints movement of {joints_diff}")
        joints_diff = th.tensor(joints_diff, device=device)
        joints_diff[:6] *= th.pi / 180.0
        joints_diff.unsqueeze(dim=0)
        # TODO(issue#128) introduce a calibration feature for the BaseEnv
        env.calib_run(joints_diff=joints_diff, steps=steps)

    if args.calibration_move_cartesian:
        n_j = len(args.calibration_move_cartesian)
        cart_diff[:n_j] = args.calibration_move_cart
        print(f"[GraspTrain] >>> Starting relative cartesian movement of {cart_diff}")
        cart_diff = th.tensor([cart_diff], device=device)
        cart_diff.unsqueeze(dim=0)
        # TODO(issue#128) introduce a calibration feature for the BaseEnv
        env.calib_run(cart_diff=cart_diff, steps=steps)

    print("[GraspTrain] >>> Finished relative joints movement.")


def train_runner(env: BaseEnv, cfg: ExpConfig) -> None:
    args = cfg.args

    rsl_rl_cfg = cfg.rl_cfg.as_dict()
    rsl_rl_cfg.update(cfg.logger_cfg.as_dict())

    # TODO(issue#120) consider saving the whole config before starting training
    match args.algorithm:
        case Algorithm.BC:
            print("[GraspTrain] >>> Starting training: Behavioral Cloning (BC)")

            print("[GraspTrain] >>> (BC) Loading RL policy")
            teacher_policy = load_rl_policy(
                env=env,
                cfg=cfg,
                load_cfg_from_clearml=not args.enforce_current_config,
                exp_name=args.experiment_name,
                clearml_task_id=args.load_rl_task_id,
                clearml_model_id=args.load_rl_model_id,
                clearml_artifact_name="model",
            )

            if cfg.dr_cfg.enabled:
                print("[GraspTrain] >>> (BC) Wrapping env with VisionAugWrapper")
                env = VisionAugEnvWrapper(
                    env=env, cfg_image_aug=cfg.dr_cfg.image_aug, cfg_env=cfg.env_cfg
                )

            if cfg.debug_cfg.enabled and (
                cfg.debug_cfg.enable_vis_preview or cfg.debug_cfg.enable_record_obs
            ):
                print("[GraspTrain] >>> (BC) Wrapping env with ObsPreviewEnvWrapper")
                env = ObsPreviewEnvWrapper(env=env, cfg_debug=cfg.debug_cfg)

            print("[GraspTrain] >>> (BC) Preparing policy runner")
            runner = BehaviorCloningRunner(
                env=env,
                cfg=cfg,
                teacher=teacher_policy,
            )
            print("[GraspTrain] >>> (BC) Starting runner")
            runner.learn(num_learning_iterations=args.max_iterations)

        case Algorithm.RL:
            print("[GraspTrain] >>> Starting training: Reinforcement Learning (RL)")

            if cfg.debug_cfg.enabled and (
                cfg.debug_cfg.enable_vis_preview or cfg.debug_cfg.enable_record_obs
            ):
                print("[GraspTrain] >>> (RL) Wrapping env with ObsPreviewEnvWrapper")
                env = ObsPreviewEnvWrapper(env=env, cfg_debug=cfg.debug_cfg)

            print("[GraspTrain] >>> (RL) Preparing policy runner")
            runner = OnPolicyRunner(env=env, cfg=cfg)
            print("[GraspTrain] >>> (RL) Starting runner")
            runner.learn(
                num_learning_iterations=cfg.rl_cfg.max_iterations,
                init_at_random_ep_len=True,
            )
            # TODO(issue#120) debug why RL model in CleaRML gets model configuration as BC config
    print("[GraspTrain] > Training finished.")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n\n[GraspTrain] > Exiting (invoked by user)")
