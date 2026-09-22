import genesis.utils.geom as gu
import torch as th

from aegis_gym.aux.geom import transform_by_quat
from aegis_gym.aux.logging import get_logger, setup_logger
from aegis_gym.config import (
    ConfigManager,
    LaunchArgs,
    parse_arguments,
)
from aegis_gym.config.types import ExpConfig
from aegis_gym.envs import BaseEnv
from aegis_gym.envs.manipulator import BaseManipulator
from aegis_gym.envs.scene import GenesisScene
from train import calibration_movment, create_env, init_clearml_task


def main():
    logger = get_logger("ft_sensor")

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

    ft_sensor_gravity_test(env=env, cfg=cfg)
    # ft_sensor_playgraund_move(env=env, cfg=cfg)


def _draw_ft_debug(scene: GenesisScene, manipulator: BaseManipulator) -> None:
    """Draws the fts_link frame (red=x, green=y, blue=z) and the measured force as an arrow."""
    gs_scene = scene.gs_scene
    gs_scene.clear_debug_objects()

    fts_link = manipulator._fts_link
    pos = fts_link.get_pos()[0]
    quat = fts_link.get_quat()[0]

    T = gu.trans_quat_to_T(pos, quat).cpu().numpy()
    gs_scene.draw_debug_frame(T, axis_length=0.15, axis_radius=0.004)

    force_local = manipulator.get_ft_wrench()[0, :3]
    force_world = transform_by_quat(force_local.unsqueeze(0), quat.unsqueeze(0))[0]
    force_arrow_scale = 0.02  # meters per Newton, tune for visibility
    gs_scene.draw_debug_arrow(
        pos=pos.cpu().numpy(),
        vec=(force_world * force_arrow_scale).cpu().numpy(),
        radius=0.006,
        color=(1.0, 0.6, 0.0, 0.9),
    )


def ft_sensor_playgraund(env: BaseEnv, cfg: ExpConfig) -> None:
    """Simplest possible F/T sanity check: robot stands still at home, no payload.
    Draws the fts_link frame live (red=x, green=y, blue=z) plus the measured force
    as an orange arrow"""
    logger = get_logger("ft_sensor")

    manipulator = env.manipulator
    scene = env._scene

    manipulator.ctrl_go_to_home()

    logger.info("Holding home configuration. Press Ctrl+C to stop.")
    while True:
        scene.step()
        _draw_ft_debug(scene, manipulator)


def _log_gravity_links(manipulator: BaseManipulator) -> None:
    """Logs every link counted as "past the F/T sensor" for gravity compensation,
    together with its attachment point: the link's center of mass, both in the
    link's own frame (local COM) and in world coordinates."""
    logger = get_logger("ft_sensor")

    manipulator._gravity_wrench_world()

    logger.info(
        f"Links counted past '{manipulator._fts_link.name}' for gravity compensation "
        f"({len(manipulator._gravity_links)} total):"
    )
    # Both tensors are [num_envs, n_links(, 3)] since genesis-world 1.x, so they
    # are indexed per link on dim 1; env 0 is the one reported.
    for i, link in enumerate(manipulator._gravity_links):
        mass = manipulator._gravity_link_masses[0, i]
        local_com = manipulator._gravity_link_local_coms[0, i]
        world_com = (
            link.get_pos()[0]
            + transform_by_quat(local_com.unsqueeze(0), link.get_quat()[:1])[0]
        )
        logger.info(
            f"  - {link.name:<22} mass={float(mass):7.4f} kg  "
            f"cord_in_link_frame={local_com.cpu().numpy()}  "
            f"cord_in_world={world_com.cpu().numpy()}"
        )


def _fmt_wrench(wrench: th.Tensor) -> str:
    """A [fx, fy, fz, tx, ty, tz] row, 3 decimals per component and kept in
    fixed-width columns so successive steps line up in the log."""
    return "[" + " ".join(f"{v:8.3f}" for v in wrench.tolist()) + "]"


def _quat_to_matrix(quat: th.Tensor) -> th.Tensor:
    """Rotation matrix of a WXYZ quaternion, columns being the local axes in world."""
    eye = th.eye(3, dtype=quat.dtype, device=quat.device)
    return transform_by_quat(eye, quat.unsqueeze(0).expand(3, 4)).T


def _log_frame(manipulator: BaseManipulator, label: str) -> None:
    """Where the sensor link's own axes point, in world. Without this the wrench
    cannot be read: 'rotate 90 deg about X' means one thing in world axes and
    another in the tool's, and at the home pose the two differ by ~90 deg."""
    logger = get_logger("ft_sensor")
    rot = _quat_to_matrix(manipulator._fts_link.get_quat()[0])
    axes = "  ".join(
        f"{ax}->[{' '.join(f'{v:6.3f}' for v in rot[:, i].tolist())}]"
        for i, ax in enumerate("xyz")
    )
    logger.info(f"  {label} frame ({manipulator._fts_link.name}): {axes}")


def _rotation_since(
    quat_ref: th.Tensor, quat_now: th.Tensor
) -> tuple[float, list[float]]:
    """Angle (deg) and world axis of the rotation taking `quat_ref` to `quat_now`."""
    rel = _quat_to_matrix(quat_now) @ _quat_to_matrix(quat_ref).T
    cos = th.clamp((th.diagonal(rel).sum() - 1.0) / 2.0, -1.0, 1.0)
    angle = th.rad2deg(th.acos(cos))
    axis = th.tensor(
        [rel[2, 1] - rel[1, 2], rel[0, 2] - rel[2, 0], rel[1, 0] - rel[0, 1]],
        device=rel.device,
    )
    norm = th.linalg.norm(axis)
    if float(norm) > 1e-6:
        axis = axis / norm
    return float(angle), [round(v, 3) for v in axis.tolist()]


def _log_wrench(manipulator: BaseManipulator, step: int) -> None:
    """Logs the raw reading next to the tared one. The real robot is always tared
    before a measurement, so the tared column is what the real data compares to;
    the raw column is what the simulated physics actually produces."""
    logger = get_logger("ft_sensor")
    gravity_wrench_world = manipulator._gravity_wrench_world()
    raw = manipulator.get_ft_wrench(biased=False)
    line = (
        f"[step {step}] gravity_wrench(world)={_fmt_wrench(gravity_wrench_world[0])}  "
        f"sensor_raw(local)={_fmt_wrench(raw[0])}"
    )
    if manipulator.is_ft_biased():
        line += f"  sensor_tared(local)={_fmt_wrench(manipulator.get_ft_wrench()[0])}"
    logger.info(line)


def ft_sensor_gravity_test(env: BaseEnv, cfg: ExpConfig) -> None:
    """Sanity check for the gripper's gravity compensation added in `get_ft_wrench()`."""
    logger = get_logger("ft_sensor")
    logger.info("Starting the F/T sensor gravity-compensation test")

    manipulator = env.manipulator
    scene = env._scene
    device = cfg.get_device()
    dt = env.get_policy_dt()

    manipulator.ctrl_go_to_home()
    for _ in range(5):
        scene.step()

    _log_gravity_links(manipulator)

    HOLD_SECONDS = 10.0
    ROTATION_DEG = 90.0
    ROTATION_SPEED_DPS = 30.0
    # The axis to swing the gripper about, and the frame it is given in:
    #   "world" -- a fixed direction in the cell. World X swings the tool to the
    #              RIGHT, which is the motion this test wants; the tool's own axes
    #              turn under it as it goes.
    #   "tool"  -- fixed in the sensor, so it turns with the tool. World X and tool
    #              X are different motions: at home the tool's axes point roughly
    #              along world (+Y, +X, -Z), so tool X is world Y, i.e. FORWARD.
    # This also decides which component the load lands on: swinging right about
    # world X puts gravity on the tool's local X, not its Y. The "home frame" and
    # "rotated frame" lines below are what makes the resulting wrench readable.
    ROTATION_AXIS = (1.0, 0.0, 0.0)
    ROTATION_FRAME = "world"

    if ROTATION_FRAME not in ("world", "tool"):
        raise ValueError(
            f"ROTATION_FRAME must be 'world' or 'tool', got {ROTATION_FRAME!r}"
        )

    hold_steps = max(1, round(HOLD_SECONDS / dt))
    log_every = max(1, round(1.0 / dt))  # once per simulated second

    quat_home = manipulator._fts_link.get_quat()[0].clone()
    _log_frame(manipulator, "home")

    # Tared in the home pose with no payload, exactly as the real sensor is zeroed
    # before a measurement, so the bias absorbs the tool's own weight.
    bias = manipulator.set_ft_bias()
    # None on a hardware-tared backend (the robot keeps the offset to itself).
    logger.info(
        f"  tared at home, bias={_fmt_wrench(bias[0])}"
        if bias is not None
        else "  tared at home (bias held in hardware)"
    )

    logger.info(
        f"Holding home configuration for {HOLD_SECONDS:.0f}s ({hold_steps} steps):"
    )
    for i in range(hold_steps):
        scene.step()
        if i % log_every == 0:
            _log_wrench(manipulator, i)
    home = _read_both(manipulator)

    rotation_steps = max(1, round((ROTATION_DEG / ROTATION_SPEED_DPS) / dt))
    logger.info(
        f"Rotating wrist {ROTATION_DEG:.0f} deg about the {ROTATION_FRAME} "
        f"{ROTATION_AXIS} axis to swing the gripper off-axis:"
    )
    axis = th.tensor(ROTATION_AXIS, dtype=th.float32, device=device)
    for _ in range(rotation_steps):
        if ROTATION_FRAME == "tool":
            # Recomputed every step: the axis is fixed in the tool, so its world
            # direction turns with the tool as the rotation proceeds.
            quat = manipulator._fts_link.get_quat()
            axis_world = transform_by_quat(axis.expand_as(quat[:, :3]), quat)
        else:
            # Fixed in the cell, so the same world direction on every step.
            axis_world = axis.expand(env.num_envs, 3)
        action = th.zeros(env.num_envs, 6, device=device)
        action[:, 3:] = axis_world * (
            (ROTATION_SPEED_DPS * th.pi / 180.0) / manipulator.max_angular_speed
        )
        manipulator.ctrl_apply_vel_action(action, open_gripper=None)
        scene.step()

    manipulator.ctrl_apply_vel_action(
        th.zeros(env.num_envs, 6, device=device), open_gripper=None
    )
    for _ in range(round(0.5 / dt)):  # let the wrist settle before reading
        scene.step()

    # Never assume the commanded rotation happened: the wrist joint limits are
    # narrow and `enable_joint_limit=True`, so a blocked rotation would look
    # exactly like a frame bug in the wrench below.
    angle, axis = _rotation_since(quat_home, manipulator._fts_link.get_quat()[0])
    logger.info(
        f"  achieved {angle:.2f} deg about world axis {axis} "
        f"(commanded {ROTATION_DEG:.0f} deg about {ROTATION_FRAME} {ROTATION_AXIS})"
    )
    if abs(angle - ROTATION_DEG) > 5.0:
        logger.warning(
            f"  the wrist reached {angle:.2f} deg, not {ROTATION_DEG:.0f} deg -- "
            "joint limits or IK, NOT the sensor frame. Read the wrench accordingly."
        )
    _log_frame(manipulator, "rotated")

    logger.info(
        f"Holding rotated configuration for {HOLD_SECONDS:.0f}s ({hold_steps} steps):"
    )
    for i in range(hold_steps):
        scene.step()
        if i % log_every == 0:
            _log_wrench(manipulator, i)
    rotated = _read_both(manipulator)

    _log_bias_report(manipulator, home=home, rotated=rotated, angle=angle)


def _read_both(manipulator: BaseManipulator) -> tuple[th.Tensor, th.Tensor]:
    """The current (raw, tared) wrench for env 0."""
    return (
        manipulator.get_ft_wrench(biased=False)[0].clone(),
        manipulator.get_ft_wrench()[0].clone(),
    )


def _log_bias_report(
    manipulator: BaseManipulator,
    home: tuple[th.Tensor, th.Tensor],
    rotated: tuple[th.Tensor, th.Tensor],
    angle: float,
) -> None:
    """Side-by-side raw and tared readings at both poses.

    The tared rows are the ones to hold against real measurements: the real sensor
    is zeroed at home, so its home reading is 0 by construction and only the change
    after the rotation carries information. The raw rows show what the simulated
    physics produced before taring, which is where a sign or frame error shows up.
    """
    logger = get_logger("ft_sensor")
    bias = manipulator.get_ft_bias()

    logger.info("=" * 78)
    logger.info(
        "F/T bias report            [   fx       fy       fz       tx       ty       tz ]"
    )
    if bias is not None:
        logger.info(f"  bias (raw @ home)        {_fmt_wrench(bias[0])}")
    logger.info(f"  home    raw              {_fmt_wrench(home[0])}")
    logger.info(
        f"  home    tared            {_fmt_wrench(home[1])}   <- ~0 by construction"
    )
    logger.info(f"  rotated raw              {_fmt_wrench(rotated[0])}")
    logger.info(
        f"  rotated tared            {_fmt_wrench(rotated[1])}   <- compare with the real robot"
    )
    logger.info(f"  (rotation actually achieved: {angle:.2f} deg)")
    logger.info("=" * 78)


def ft_sensor_playgraund_move(env: BaseEnv, cfg: ExpConfig) -> None:
    """Drives the arm forward 20 cm, then slowly toward the table to study contact:
    watch the fts_link frame/force arrow and the logged wrench as it makes contact."""
    logger = get_logger("ft_sensor")
    logger.info("Starting the F/T sensor contact test")

    manipulator = env.manipulator
    scene = env._scene
    device = cfg.get_device()
    dt = env.get_policy_dt()

    manipulator.ctrl_go_to_home()

    FORWARD_DISTANCE_M = 0.20
    FORWARD_SPEED_MPS = 0.1
    PUSH_SPEED_MPS = 0.05

    step = 0

    def move(direction: list[float], speed: float, n_steps: int | None) -> None:
        nonlocal step
        vel_dir = th.tensor(direction, dtype=th.float32, device=device)
        i = 0
        while n_steps is None or i < n_steps:
            action = th.zeros(env.num_envs, 6, device=device)
            action[:, :3] = vel_dir * (speed / manipulator.max_linear_speed)
            manipulator.ctrl_apply_vel_action(action, open_gripper=None)
            scene.step()
            _draw_ft_debug(scene, manipulator)
            step += 1
            i += 1

    logger.info(f"Phase 1: moving forward {FORWARD_DISTANCE_M} m")
    move(
        direction=[1.0, 0.0, 0.0],
        speed=FORWARD_SPEED_MPS,
        n_steps=max(1, round((FORWARD_DISTANCE_M / FORWARD_SPEED_MPS) / dt)),
    )

    logger.info("Phase 2: pushing down into the table slowly (Ctrl+C to stop)")
    move(direction=[0.0, 0.0, -1.0], speed=PUSH_SPEED_MPS, n_steps=None)


if __name__ == "__main__":
    setup_logger("INFO")
    logger = get_logger("ft_sensor")

    try:
        main()
    except KeyboardInterrupt:
        logger.info("\n\n\nExiting (invoked by user)")
