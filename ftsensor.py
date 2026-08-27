import sys

import genesis as gs
import torch as th
from clearml import Task

from aegis_gym.aux import load_policy
from aegis_gym.config import ConfigManager, LaunchArgs, parse_arguments
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
    print("[Train] > Setup done")

    if args.calibration_move or args.calibration_move_cartesian:
        print("[Train] > Proceeding to calibration movement")
        calibration_movment(env, cfg)
        return

    print("[Train] > Starting F/T sensor")
    ft_sensor_playground(env=env, cfg=cfg)
    # ft_sensor_playground_weld(env=env, cfg=cfg)
    # ft_sensor_playground_weld_hit_table(env=env, cfg=cfg)
    # ft_sensor_playground_weld_axes_joint(env=env, cfg=cfg)


def create_env(cfg: ExpConfig) -> BaseEnv:
    args: LaunchArgs = cfg.args
    control_type = args.control_type

    scene = None
    if control_type == Control.SIM:
        gs.init(logging_level="info", precision="32")
        scene = GenesisScene(cfg=cfg, device=cfg.get_device())
    if control_type == Control.ROS:
        if RosGrcpScene is None:
            print("[Train] >>>> ERROR: Can not import RosGrcpScene. \n>>>> Exiting")
            sys.exit()
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
        print(f"[Train] >>> Starting relative joints movement of {joints_diff}")
        joints_diff = th.tensor(joints_diff, device=device)
        joints_diff[:6] *= th.pi / 180.0
        joints_diff.unsqueeze(dim=0)
        # TODO(issue#128) introduce a calibration feature for the BaseEnv
        env.calib_run(joints_diff=joints_diff, steps=steps)

    if args.calibration_move_cartesian:
        n_j = len(args.calibration_move_cartesian)
        cart_diff[:n_j] = args.calibration_move_cart
        print(f"[Train] >>> Starting relative cartesian movement of {cart_diff}")
        cart_diff = th.tensor([cart_diff], device=device)
        cart_diff.unsqueeze(dim=0)
        # TODO(issue#128) introduce a calibration feature for the BaseEnv
        env.calib_run(cart_diff=cart_diff, steps=steps)

    print("[Train] >>> Finished relative joints movement.")


def train_runner(env: BaseEnv, cfg: ExpConfig) -> None:
    args = cfg.args

    # TODO(issue#120) consider saving the whole config before starting training
    match args.algorithm:
        case Algorithm.BC:
            print("[Train] >>> Starting training: Behavioral Cloning (BC)")

            print("[Train] >>> (BC) Loading RL policy")
            teacher_policy = load_policy(env=env, cfg=cfg, alg=Algorithm.RL)

            if cfg.dr_cfg.enabled:
                print("[Train] >>> (BC) Wrapping env with VisionAugWrapper")
                env = VisionAugEnvWrapper(
                    env=env, cfg_image_aug=cfg.dr_cfg.image_aug, cfg_env=cfg.env_cfg
                )

            if cfg.debug_cfg.enabled and (
                cfg.debug_cfg.enable_vis_preview or cfg.debug_cfg.enable_record_obs
            ):
                print("[Train] >>> (BC) Wrapping env with ObsPreviewEnvWrapper")
                env = ObsPreviewEnvWrapper(env=env, cfg_debug=cfg.debug_cfg)

            print("[Train] >>> (BC) Preparing policy runner")
            runner = BehaviorCloningRunner(
                env=env,
                cfg=cfg,
                teacher=teacher_policy,
            )
            print("[Train] >>> (BC) Starting runner")
            runner.learn(num_learning_iterations=args.max_iterations)

        case Algorithm.RL:
            print("[Train] >>> Starting training: Reinforcement Learning (RL)")

            if cfg.debug_cfg.enabled and (
                cfg.debug_cfg.enable_vis_preview or cfg.debug_cfg.enable_record_obs
            ):
                print("[Train] >>> (RL) Wrapping env with ObsPreviewEnvWrapper")
                env = ObsPreviewEnvWrapper(env=env, cfg_debug=cfg.debug_cfg)

            print("[Train] >>> (RL) Preparing policy runner")
            runner = OnPolicyRunner(env=env, cfg=cfg)
            print("[Train] >>> (RL) Starting runner")
            runner.learn(
                num_learning_iterations=cfg.rl_cfg.max_iterations,
                init_at_random_ep_len=True,
            )
            # TODO(issue#120) debug why RL model in CleaRML gets model configuration as BC config
    print("[Train] > Training finished.")


def ft_sensor_playground(env: BaseEnv, cfg: ExpConfig) -> None:
    env.reset()

    start_x = env.manipulator.get_tcp_pose()[0, 0].item()

    while True:
        action = th.zeros((env.num_envs, 6), device=cfg.get_device())

        x = env.manipulator.get_tcp_pose()[0, 0].item()

        if x < start_x + 0.40:
            action[:, 0] = 1.0
        else:
            action[:, 2] = -0.1

        env.manipulator.ctrl_apply_vel_action(action, open_gripper=True)
        env._scene.step()

        env.manipulator.get_joint_torque_sensor()


def ft_sensor_playground_weld(env: BaseEnv, cfg: ExpConfig) -> None:
    env.reset()

    robot = env.manipulator._robot_entity
    box = env._scene._weld_test_box

    tool_mount_link = robot.get_link("tool_mount_link")
    box_link = box.get_link("box_baselink")

    tcp_pos = env.manipulator.get_tcp_pose()[0, :3]

    box.set_pos(
        tcp_pos
        + th.tensor(
            [0.0, 0.0, -0.10],
            device=cfg.get_device(),
        )
    )

    rigid = env._scene.gs_scene.sim.rigid_solver

    rigid.add_weld_constraint(
        tool_mount_link.idx,
        box_link.idx,
    )

    step = 0

    while True:
        action = th.zeros(
            (env.num_envs, 6),
            device=cfg.get_device(),
        )

        phase = (step // 50) % 9

        if phase == 0:
            pass

        elif phase == 1:
            # +X
            action[:, 0] = 1.0

        elif phase == 2:
            # stop
            pass

        elif phase == 3:
            # -X
            action[:, 0] = -1.0

        elif phase == 4:
            # +Y
            action[:, 1] = 1.0

        elif phase == 5:
            # -Y
            action[:, 1] = -1.0

        elif phase == 6:
            # +Z
            action[:, 2] = 0.5

        elif phase == 7:
            # -Z
            action[:, 2] = -0.5

        elif phase == 8:
            # stop
            pass

        env.manipulator.ctrl_apply_vel_action(
            action,
            open_gripper=True,
        )

        env._scene.step()

        welds = rigid.get_weld_constraints()
        force = welds["force"][0, 0]

        if step % 10 == 0:
            tcp = env.manipulator.get_tcp_pose()[0, :3]

            print(
                f"phase={phase}",
                "TCP=",
                tcp,
                "weld=",
                force,
            )

        step += 1


def ft_sensor_playground_weld_hit_table(env: BaseEnv, cfg: ExpConfig) -> None:
    env.reset()

    robot = env.manipulator._robot_entity
    box = env._scene._weld_test_box

    tool_mount_link = robot.get_link("tool_mount_link")
    box_link = box.get_link("box_baselink")

    device = cfg.get_device()

    tcp_pos = env.manipulator.get_tcp_pose()[0, :3]

    box.set_pos(
        tcp_pos
        + th.tensor(
            [0.0, 0.0, -0.05],
            device=device,
        )
    )

    rigid = env._scene.gs_scene.sim.rigid_solver

    rigid.add_weld_constraint(
        tool_mount_link.idx,
        box_link.idx,
    )

    start_tcp = env.manipulator.get_tcp_pose()[0, :3].clone()

    target_x = start_tcp[0] + 0.20
    target_z = 0.0

    step = 0
    phase_step = 0

    while True:
        tcp = env.manipulator.get_tcp_pose()[0, :3]

        action = th.zeros(
            (env.num_envs, 6),
            device=device,
        )

        if step < 200:
            phase = "GRAVITY"

        elif tcp[0] < target_x:
            phase = "FAST +X"
            action[:, 0] = 1.0

        elif phase_step < 100:
            phase = "PAUSE"
            phase_step += 1

        elif tcp[2] > target_z:
            phase = "SLOW -Z"
            action[:, 2] = -0.05

        else:
            phase = "TABLE"

        env.manipulator.ctrl_apply_vel_action(
            action,
            open_gripper=True,
        )

        env._scene.step()

        welds = rigid.get_weld_constraints()
        wrench = welds["force"][0, 0]

        if step % 10 == 0:
            print(
                f"{phase:10} | "
                f"TCP=({tcp[0]:+.3f}, {tcp[1]:+.3f}, {tcp[2]:+.3f}) | "
                f"W=["
                f"{wrench[0]:+.4f}, "
                f"{wrench[1]:+.4f}, "
                f"{wrench[2]:+.4f}, "
                f"{wrench[3]:+.4f}, "
                f"{wrench[4]:+.4f}, "
                f"{wrench[5]:+.4f}"
                f"]"
            )

        step += 1


def ft_sensor_playground_weld_axes_joint(env: BaseEnv, cfg: ExpConfig) -> None:
    env.reset()

    robot = env.manipulator._robot_entity
    box = env._scene._weld_test_box

    tool_mount_link = robot.get_link("tool_mount_link")
    box_link = box.get_link("box_baselink")

    device = cfg.get_device()

    tcp_pos = env.manipulator.get_tcp_pose()[0, :3]

    box.set_pos(
        tcp_pos
        + th.tensor(
            [0.0, 0.0, -0.10],
            device=device,
        )
    )

    rigid = env._scene.gs_scene.sim.rigid_solver

    rigid.add_weld_constraint(
        tool_mount_link.idx,
        box_link.idx,
    )

    step = 0

    while True:
        joint_diff = th.zeros(
            (env.num_envs, 8),
            device=device,
        )

        if step < 200:
            phase = "GRAVITY"

        elif step < 700:
            phase = "ROTATE 90"
            joint_diff[:, -5] = -1.0 * th.pi / 90.0

        else:
            phase = "STOP"

        env.manipulator.ctrl_apply_joints_diff_action(joint_diff)

        env._scene.step()

        welds = rigid.get_weld_constraints()
        wrench = welds["force"][0, 0]

        if step % 10 == 0:
            print(
                f"{phase:10} | "
                f"W=["
                f"{wrench[0]:+.4f}, "
                f"{wrench[1]:+.4f}, "
                f"{wrench[2]:+.4f}, "
                f"{wrench[3]:+.4f}, "
                f"{wrench[4]:+.4f}, "
                f"{wrench[5]:+.4f}"
                f"]"
            )

        step += 1


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n\n[Train] > Exiting (invoked by user)")
