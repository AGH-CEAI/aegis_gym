import sys

import genesis as gs
import genesis.utils.geom as gu
import torch as th
from clearml import Task

from aegis_gym.aux.geom import transform_by_quat
from aegis_gym.aux.logging import get_logger, setup_logger
from aegis_gym.config import (
    ConfigManager,
    LaunchArgs,
    parse_arguments,
)
from aegis_gym.config.types import Algorithm, Control, ExpConfig
from aegis_gym.envs import BaseEnv, ReacherEnv
from aegis_gym.envs.manipulator import BaseManipulator
from aegis_gym.envs.scene import GenesisScene, RosGrcpScene

# Robotiq Hand-E's own mass isn't modeled as a real load in the sim: `gravity_compensation=1.0`
# on the robot's material (genesis_manipulator.py) cancels gravity for the WHOLE URDF entity,
# gripper included, because Genesis only supports gravity compensation per-entity, not per-link.
# A real F/T sensor has no such compensation - it always feels whatever hangs beyond it. So we
# weld a stand-in box of the gripper's real mass onto fts_link, the same way a task payload would
# be welded, to make the baseline reading behave like a real sensor's.
GRIPPER_MASS_KG = 0.9
GRIPPER_BOX_DIMS = (0.08, 0.08, 0.08)


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

    if args.calibration_move or args.calibration_move_cartesian:
        env = create_env(cfg)
        logger.info("Setup done")
        logger.info("Proceeding to calibration movement")
        calibration_movment(env, cfg)
        return

    env, gripper_mass = create_env_with_gripper_mass(cfg)
    logger.info("Setup done")
    ft_sensor_playgraund(env=env, gripper_mass=gripper_mass, cfg=cfg)

    # logger.info("Proceeding training")
    # train_runner(env=env, cfg=cfg)


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


def create_env_with_gripper_mass(cfg: ExpConfig):
    """
    Like `create_env()`, but also spawns a free box standing in for the gripper's own mass
    (see the module docstring above), welded onto `fts_link` once the scene is built.
    Genesis entities can only be added before `Scene.build()`, so the box must exist
    before `ReacherEnv.__init__()` builds the scene.
    """
    args: LaunchArgs = cfg.args
    if args.control_type != Control.SIM:
        raise ValueError("The F/T sensor test is only supported in simulation.")

    gs.init(logging_level="info", precision="32")
    scene = GenesisScene(cfg=cfg, device=cfg.get_device())

    volume = GRIPPER_BOX_DIMS[0] * GRIPPER_BOX_DIMS[1] * GRIPPER_BOX_DIMS[2]
    gripper_mass = scene.gs_scene.add_entity(
        gs.morphs.Box(
            size=GRIPPER_BOX_DIMS,
            pos=(0.0, 0.0, -1.0),  # placeholder; repositioned onto fts_link once built
            fixed=False,
            collision=False,  # avoid spurious contact with the gripper before it's welded
        ),
        material=gs.materials.Rigid(rho=GRIPPER_MASS_KG / volume),
        surface=gs.surfaces.Default(color=(0.1, 0.3, 0.9), opacity=0.35),
    )

    env = ReacherEnv(scene=scene, cfg=cfg)
    return env, gripper_mass


def weld_box_to_fts(
    scene: GenesisScene,
    box,
    local_offset: tuple[float, float, float],
) -> None:
    """Welds `box` onto `fts_link`, fixed at `local_offset` in the sensor's local frame."""
    manipulator = scene.get_manipulator()
    robot_entity = manipulator._robot_entity
    fts_link = manipulator._fts_link

    pos = fts_link.get_pos()  # [num_envs, 3]
    quat = fts_link.get_quat()  # [num_envs, 4], WXYZ
    offset = th.tensor(local_offset, dtype=pos.dtype, device=pos.device).expand_as(pos)
    world_offset = transform_by_quat(offset, quat)

    box.set_pos(pos + world_offset)
    box.set_quat(quat)

    robot_entity.solver.add_weld_constraint(fts_link.idx, box.base_link.idx)


def read_weld_wrench(manipulator: BaseManipulator) -> th.Tensor:
    """
    Reads the fts_link<->gripper-mass weld's actual constraint (Lagrange-multiplier) force
    straight from Genesis's constraint solver, instead of estimating it from actuator torque.

    This is the force the solver applies to keep that joint rigid under whatever the welded
    box is doing (gravity, inertia, any contact reaching it) - i.e. exactly what a physical
    strain-gauge sensor bolted there would feel, with no dependency on `gravity_compensation`
    or the rest of the arm's dynamics like `get_ft_wrench()`'s Jacobian/actuator-torque
    estimate has. NOTE: the frame/sign of Genesis's `efc_force` for a WELD isn't verified yet
    (assumed world-frame Cartesian, MuJoCo-style) - compare it against `get_ft_wrench()` and
    calibrate before trusting it.
    """
    welds = manipulator._robot_entity.solver.get_weld_constraints()
    return welds["force"][0, 0]  # first (only) weld, env 0: [fx, fy, fz, tx, ty, tz]


def _draw_ft_debug(scene: GenesisScene, manipulator: BaseManipulator) -> None:
    """
    Draws the fts_link frame (red=x, green=y, blue=z), the Jacobian/actuator-torque force
    reading from `get_ft_wrench()` as an orange arrow, and the raw weld constraint force
    from `read_weld_wrench()` as a magenta arrow, so the two can be compared visually.
    """
    gs_scene = scene.gs_scene
    gs_scene.clear_debug_objects()

    fts_link = manipulator._fts_link
    pos = fts_link.get_pos()[0]
    quat = fts_link.get_quat()[0]
    pos_np = pos.cpu().numpy()

    T = gu.trans_quat_to_T(pos, quat).cpu().numpy()
    gs_scene.draw_debug_frame(T, axis_length=0.15, axis_radius=0.004)

    force_arrow_scale = 0.02  # meters per Newton, tune for visibility

    force_local = manipulator.get_ft_wrench()[0, :3]
    force_world = transform_by_quat(force_local.unsqueeze(0), quat.unsqueeze(0))[0]
    gs_scene.draw_debug_arrow(
        pos=pos_np,
        vec=(force_world * force_arrow_scale).cpu().numpy(),
        radius=0.006,
        color=(1.0, 0.6, 0.0, 0.9),
    )

    weld_force_world = read_weld_wrench(manipulator)[:3]
    gs_scene.draw_debug_arrow(
        pos=pos_np,
        vec=(weld_force_world * force_arrow_scale).cpu().numpy(),
        radius=0.006,
        color=(1.0, 0.0, 1.0, 0.9),
    )


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


def ft_sensor_playgraund(env: BaseEnv, gripper_mass, cfg: ExpConfig) -> None:
    """Simplest F/T sanity check: robot stands still at home, with the gripper's own mass
    welded on (see module docstring) so the baseline reads like a real sensor would.
    Draws the fts_link frame live (red=x, green=y, blue=z) plus the measured force
    as an orange arrow, so you can see the coordinate system `get_ft_wrench()` uses."""
    logger = get_logger("ft_sensor")
    logger.info("Starting the F/T sensor test")

    manipulator = env.manipulator
    scene = env._scene

    manipulator.ctrl_go_to_home()
    for _ in range(20):
        scene.step()

    # Along fts_link's local +z, just past the box's half-size (0.04 m) for a small clearance.
    GRIPPER_MASS_OFFSET_M = (0.0, 0.0, 0.05)
    logger.info("Welding the gripper-mass box onto fts_link")
    weld_box_to_fts(scene=scene, box=gripper_mass, local_offset=GRIPPER_MASS_OFFSET_M)

    logger.info(
        "Holding home configuration. Live frame at fts_link (red=x, green=y, blue=z); "
        "orange arrow is get_ft_wrench(), magenta arrow is the raw weld constraint force. "
        "Press Ctrl+C to stop."
    )
    step = 0
    while True:
        scene.step()
        _draw_ft_debug(scene, manipulator)
        step += 1


if __name__ == "__main__":
    setup_logger("INFO")
    logger = get_logger("Train")

    try:
        main()
    except KeyboardInterrupt:
        logger.info("\n\n\nExiting (invoked by user)")
