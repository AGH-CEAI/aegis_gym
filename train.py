import sys

import genesis as gs
import torch as th
from clearml import Task

from aegis_gym.aux.logging import get_logger, setup_logger
from aegis_gym.aux.utils import load_policy
from aegis_gym.config import (
    ConfigManager,
    LaunchArgs,
    parse_arguments,
)
from aegis_gym.config.types import Algorithm, Control, ExpConfig
from aegis_gym.envs import BaseEnv, ReacherEnv
from aegis_gym.envs.scene import GenesisScene, RosGrcpScene
from aegis_gym.envs.wrappers import ObsPreviewEnvWrapper, VisionAugEnvWrapper
from aegis_gym.runners import BehaviorCloningRunner, OnPolicyRunner


def init_clearml_task(
    project_name: str | None,
    algorithm: Algorithm | None,
    control: Control | None,
    exp_name: str | None,
) -> Task:
    assert None not in (project_name, algorithm, control, exp_name)
    return Task.init(
        project_name=f"{project_name}_{algorithm!s}-{control!s}",
        task_name=f"{exp_name}_{algorithm!s}",
        reuse_last_task_id=True,
    )


# TODO(issue#130) Real training with BC doesn't work, mark this down
def main():
    logger = get_logger("Train")

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
    logger.info("Setup done")

    if args.calibration_move or args.calibration_move_cartesian:
        logger.info("Proceeding to calibration movement")
        calibration_movment(env, cfg)
        return

    logger.info("Proceeding training")
    train_runner(env=env, cfg=cfg)


def create_env(cfg: ExpConfig) -> BaseEnv:
    logger = get_logger("Train")
    args: LaunchArgs = cfg.args
    control_type = args.control_type

    scene = None
    if control_type == Control.SIM:
        gs.init(logging_level="info", precision="32")
        scene = GenesisScene(cfg=cfg, device=cfg.get_device())
    if control_type == Control.ROS:
        if RosGrcpScene is None:
            logger.error("Can not import RosGrcpScene. Exiting")
            sys.exit()
        scene = RosGrcpScene(cfg=cfg, device=cfg.get_device())

    if scene is None:
        raise ValueError("Scene is None")

    return ReacherEnv(scene=scene, cfg=cfg)


def calibration_movment(env: BaseEnv, cfg: ExpConfig) -> None:
    logger = get_logger("Train")
    args = cfg.args
    device = cfg.get_device()

    cart_diff = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    joints_diff = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    steps = args.calibration_steps

    if args.calibration_movment:
        n_j = len(args.calibration_move)
        joints_diff[:n_j] = args.calibration_move
        logger.info(f">>> Starting relative joints movement of {joints_diff}")
        joints_diff = th.tensor(joints_diff, device=device)
        joints_diff[:6] *= th.pi / 180.0
        joints_diff.unsqueeze(dim=0)
        # TODO(issue#128) introduce a calibration feature for the BaseEnv
        env.calib_run(joints_diff=joints_diff, steps=steps)

    if args.calibration_move_cartesian:
        n_j = len(args.calibration_move_cartesian)
        cart_diff[:n_j] = args.calibration_move_cart
        logger.info(f">>> Starting relative cartesian movement of {cart_diff}")
        cart_diff = th.tensor([cart_diff], device=device)
        cart_diff.unsqueeze(dim=0)
        # TODO(issue#128) introduce a calibration feature for the BaseEnv
        env.calib_run(cart_diff=cart_diff, steps=steps)

    logger.info(">>> Finished relative joints movement.")


def train_runner(env: BaseEnv, cfg: ExpConfig) -> None:
    logger = get_logger("Train")
    args = cfg.args

    # TODO(issue#120) consider saving the whole config before starting training
    match args.algorithm:
        case Algorithm.BC:
            logger.info(">>> Starting training: Behavioral Cloning (BC)")

            logger.info(">>> (BC) Loading RL policy")
            teacher_policy = load_policy(env=env, cfg=cfg, alg=Algorithm.RL)

            if cfg.dr_cfg.enabled:
                logger.info(">>> (BC) Wrapping env with VisionAugWrapper")
                env = VisionAugEnvWrapper(
                    env=env, cfg_image_aug=cfg.dr_cfg.image_aug, cfg_env=cfg.env_cfg
                )

            if cfg.debug_cfg.enabled and (
                cfg.debug_cfg.enable_vis_preview or cfg.debug_cfg.enable_record_obs
            ):
                logger.info(">>> (BC) Wrapping env with ObsPreviewEnvWrapper")
                env = ObsPreviewEnvWrapper(env=env, cfg_debug=cfg.debug_cfg)

            logger.info(">>> (BC) Preparing policy runner")
            runner = BehaviorCloningRunner(
                env=env,
                cfg=cfg,
                teacher=teacher_policy,
            )
            logger.info(">>> (BC) Starting runner")
            runner.learn(num_learning_iterations=args.max_iterations)

        case Algorithm.RL:
            logger.info(">>> Starting training: Reinforcement Learning (RL)")

            if cfg.debug_cfg.enabled and (
                cfg.debug_cfg.enable_vis_preview or cfg.debug_cfg.enable_record_obs
            ):
                logger.info(">>> (RL) Wrapping env with ObsPreviewEnvWrapper")
                env = ObsPreviewEnvWrapper(env=env, cfg_debug=cfg.debug_cfg)

            logger.info(">>> (RL) Preparing policy runner")
            runner = OnPolicyRunner(env=env, cfg=cfg)
            logger.info(">>> (RL) Starting runner")
            runner.learn(
                num_learning_iterations=cfg.rl_cfg.max_iterations,
                init_at_random_ep_len=True,
            )
            # TODO(issue#120) debug why RL model in CleaRML gets model configuration as BC config
    logger.info("Training finished.")


if __name__ == "__main__":
    setup_logger("INFO")
    logger = get_logger("Train")

    try:
        main()
    except KeyboardInterrupt:
        logger.info("\n\n\nExiting (invoked by user)")
