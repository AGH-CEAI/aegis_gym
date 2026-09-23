import time

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

    # ft_sensor_gravity_test(env=env, cfg=cfg)
    # ft_sensor_identify_payload(env=env, cfg=cfg)
    ft_sensor_contact_drag_test(env=env, cfg=cfg)
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


def _step(scene) -> None:
    """One environment step.

    `pre_step()` is what paces the loop on the real robot -- it blocks until a full
    policy period has elapsed -- and is a no-op in simulation. Without it the hardware
    loop free-runs at gRPC speed, so "hold for 10 s" and "rotate for 3 s" would mean
    nothing there.
    """
    scene.pre_step()
    scene.step()


def _is_modelled(manipulator: BaseManipulator) -> bool:
    """Whether this backend exposes the simulation-only modelling internals.

    The real robot reports one measurement over the bridge and nothing else: there is
    no gravity term to inspect, no untared reading, and no sensor-link handle. Every
    block guarded by this is a sim-side cross-check, not part of the test itself.
    """
    return hasattr(manipulator, "get_ft_wrench_raw")


def _log_gravity_links(manipulator: BaseManipulator) -> None:
    """Logs every link counted as "past the F/T sensor" for gravity compensation,
    together with its attachment point: the link's center of mass, both in the
    link's own frame (local COM) and in world coordinates."""
    logger = get_logger("ft_sensor")

    if not _is_modelled(manipulator):
        logger.info(
            "Real sensor: no gravity-compensation model to report (the hardware "
            "measures the load directly)."
        )
        return

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


def _sensor_quat(manipulator: BaseManipulator) -> tuple[th.Tensor, str]:
    """Orientation to reason about the wrench with, and the name of its frame.

    Both backends answer in the frame the wrench is actually reported in. In
    simulation that is `get_ft_frame_quat()`. On the real robot the bridge does not
    publish the sensor link, but it does not need to: per the URDF the only rotation
    between `tool_mount_link` and the TCP is the +90 deg about Z of
    `adapter_from_sensor_end_joint`, which is exactly the sensor's measured output
    offset -- so the TCP orientation *is* the sensor frame.
    """
    if _is_modelled(manipulator):
        return manipulator.get_ft_frame_quat()[0], "sensor output frame"
    return manipulator.get_tcp_orientation()[0], "sensor output frame (via TCP)"


def _log_frame(manipulator: BaseManipulator, label: str) -> None:
    """Where the frame's own axes point, in world. Without this the wrench cannot be
    read: 'rotate 90 deg about X' means one thing in world axes and another in the
    tool's, and at the home pose the two differ by ~90 deg."""
    logger = get_logger("ft_sensor")
    quat, name = _sensor_quat(manipulator)
    rot = _quat_to_matrix(quat)
    axes = "  ".join(
        f"{ax}->[{' '.join(f'{v:6.3f}' for v in rot[:, i].tolist())}]"
        for i, ax in enumerate("xyz")
    )
    logger.info(f"  {label} frame ({name}): {axes}")


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
    """Logs the measurement, plus -- in simulation only -- the untared reading and the
    gravity term behind it. The real robot reports a single value, so that is all there
    is to log there, and it is the column the simulated one must match."""
    logger = get_logger("ft_sensor")
    line = f"[step {step}] wrench={_fmt_wrench(manipulator.get_ft_wrench()[0])}"
    if manipulator.is_ft_biased():
        line += " (tared)"
    if _is_modelled(manipulator):
        line += (
            f"  sim_raw={_fmt_wrench(manipulator.get_ft_wrench_raw()[0])}"
            f"  sim_gravity(world)={_fmt_wrench(manipulator._gravity_wrench_world()[0])}"
        )
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
        _step(scene)

    _log_gravity_links(manipulator)

    HOLD_SECONDS = 10.0
    # How long to keep stepping after the goal is issued, so the arm reaches it and
    # the reading settles before it is logged.
    MOVE_SETTLE_SECONDS = 5.0
    # The arm configuration in which the TCP sits 90 deg from home, measured on the
    # real robot from /joint_states and reordered into kinematic order. Commanded as
    # joints rather than as a TCP pose on purpose: the configuration is stated
    # outright, so no IK or planner can put the simulator on a different branch from
    # the robot, and there is nothing left to "not accept".
    ROTATED_ARM_DOF = (
        -0.999880615864889,  # shoulder_pan
        -1.5083312888494511,  # shoulder_lift
        2.2797167936908167,  # elbow
        -0.7881105703166504,  # wrist_1
        -0.9998858610736292,  # wrist_2
        -1.5612619558917444,  # wrist_3
    )
    # What that configuration is expected to be relative to home, for the check below.
    ROTATION_DEG = 90.0

    hold_steps = max(1, round(HOLD_SECONDS / dt))
    log_every = max(1, round(1.0 / dt))  # once per simulated second

    quat_home = _sensor_quat(manipulator)[0].clone()
    _log_frame(manipulator, "home")

    # Tared in the home pose with no payload, exactly as the real sensor is zeroed
    # before a measurement, so the bias absorbs the tool's own weight.
    manipulator.set_ft_bias()
    bias = _sim_bias(manipulator)
    logger.info(
        f"  tared at home, bias={_fmt_wrench(bias[0])}"
        if bias is not None
        else "  tared at home (offset held by the sensor)"
    )

    logger.info(
        f"Holding home configuration for {HOLD_SECONDS:.0f}s ({hold_steps} steps):"
    )
    for i in range(hold_steps):
        _step(scene)
        if i % log_every == 0:
            _log_wrench(manipulator, i)
    home = _read_both(manipulator)

    logger.info(f"Moving to the rotated configuration (joint goal): {ROTATED_ARM_DOF}")
    manipulator.ctrl_go_to_joints(
        th.tensor(ROTATED_ARM_DOF, dtype=th.float32, device=device)
    )
    for _ in range(max(1, round(MOVE_SETTLE_SECONDS / dt))):
        _step(scene)

    # Never assume the arm arrived: a joint goal cannot be refused for want of a
    # solution, but it can still be clipped by limits or simply not reached yet, and
    # either would look exactly like a frame bug in the wrench below.
    angle, axis = _rotation_since(quat_home, _sensor_quat(manipulator)[0])
    logger.info(
        f"  achieved {angle:.2f} deg about world axis {axis} "
        f"(the measured configuration should sit {ROTATION_DEG:.0f} deg from home)"
    )
    if abs(angle - ROTATION_DEG) > 5.0:
        logger.warning(
            f"  the arm sits {angle:.2f} deg from home, not {ROTATION_DEG:.0f} deg -- "
            "it did not reach the commanded joints (limits, or still moving), NOT the "
            "sensor frame. Read the wrench accordingly."
        )
    _log_frame(manipulator, "rotated")

    logger.info(
        f"Holding rotated configuration for {HOLD_SECONDS:.0f}s ({hold_steps} steps):"
    )
    for i in range(hold_steps):
        _step(scene)
        if i % log_every == 0:
            _log_wrench(manipulator, i)
    rotated = _read_both(manipulator)

    _log_bias_report(manipulator, home=home, rotated=rotated, angle=angle)


def _sim_bias(manipulator: BaseManipulator) -> th.Tensor | None:
    """The numeric tare offset when the backend can report one. The real sensor keeps
    its offset inside the hardware, so there it is simply unknown."""
    getter = getattr(manipulator, "get_ft_bias", None)
    return getter() if getter is not None else None


def _read_both(manipulator: BaseManipulator) -> tuple[th.Tensor | None, th.Tensor]:
    """The current (untared, measured) wrench for env 0. The untared half is None on
    the real robot, which reports one value and no more."""
    raw = (
        manipulator.get_ft_wrench_raw()[0].clone()
        if _is_modelled(manipulator)
        else None
    )
    return raw, manipulator.get_ft_wrench()[0].clone()


def _log_bias_report(
    manipulator: BaseManipulator,
    home: tuple[th.Tensor | None, th.Tensor],
    rotated: tuple[th.Tensor | None, th.Tensor],
    angle: float,
) -> None:
    """Measured readings at both poses, with the untared ones alongside in simulation.

    The measured rows are the comparable ones: the sensor is zeroed at home, so its
    home reading is 0 by construction and only the change after the rotation carries
    information. The sim-only rows show what the model produced before taring, which
    is where a sign or frame error shows up.
    """
    logger = get_logger("ft_sensor")
    bias = _sim_bias(manipulator)

    logger.info("=" * 78)
    logger.info(
        "F/T bias report            [   fx       fy       fz       tx       ty       tz ]"
    )
    if bias is not None:
        logger.info(f"  bias (untared @ home)    {_fmt_wrench(bias[0])}")
    if home[0] is not None:
        logger.info(f"  home    untared (sim)    {_fmt_wrench(home[0])}")
    logger.info(
        f"  home    measured         {_fmt_wrench(home[1])}   <- ~0 by construction"
    )
    if rotated[0] is not None:
        logger.info(f"  rotated untared (sim)    {_fmt_wrench(rotated[0])}")
    logger.info(
        f"  rotated measured         {_fmt_wrench(rotated[1])}   <- compare sim vs real"
    )
    logger.info(f"  (rotation actually achieved: {angle:.2f} deg)")
    logger.info("=" * 78)


def _skew(vec: th.Tensor) -> th.Tensor:
    """[3, 3] cross-product matrix, so that `_skew(a) @ b == cross(a, b)`."""
    zero = th.zeros((), dtype=vec.dtype, device=vec.device)
    return th.stack(
        [
            th.stack([zero, -vec[2], vec[1]]),
            th.stack([vec[2], zero, -vec[0]]),
            th.stack([-vec[1], vec[0], zero]),
        ]
    )


def _identify_payload(
    rots: list[th.Tensor],
    forces: list[th.Tensor],
    torques: list[th.Tensor],
    gravity: th.Tensor,
) -> tuple[th.Tensor, th.Tensor, th.Tensor, th.Tensor]:
    """Least-squares payload identification from wrenches taken at several orientations.

    With the load hanging rigidly past the sensor, each pose contributes

        F_i = m * (R_i^T g) + F0
        T_i = (m*c) x (R_i^T g) + T0

    where `R_i` turns the sensor frame into world, `m` is the payload mass, `c` its
    centre of mass in the sensor frame, and `F0`/`T0` a constant sensor offset. Both
    are linear in the unknowns, so this is one least-squares solve each: 4 unknowns for
    the force, 6 for the torque. Solving for `m*c` rather than `c` keeps it linear.

    The readings must be UNTARED -- a tare subtracts a constant, which is precisely the
    signal `F0`/`T0` absorb, and it would bias `m` if left in.

    Returns (mass, com, force_offset, torque_offset).
    """
    device, n = gravity.device, len(rots)
    eye = th.eye(3, dtype=th.float32, device=device)

    a_mat = th.zeros(3 * n, 4, device=device)
    b_vec = th.zeros(3 * n, device=device)
    for i, (rot, force) in enumerate(zip(rots, forces)):
        a_mat[3 * i : 3 * i + 3, 0] = rot.T @ gravity
        a_mat[3 * i : 3 * i + 3, 1:4] = eye
        b_vec[3 * i : 3 * i + 3] = force
    sol = th.linalg.lstsq(a_mat, b_vec.unsqueeze(-1)).solution.squeeze(-1)
    mass, force_offset = sol[0], sol[1:4]

    c_mat = th.zeros(3 * n, 6, device=device)
    d_vec = th.zeros(3 * n, device=device)
    for i, (rot, torque) in enumerate(zip(rots, torques)):
        c_mat[3 * i : 3 * i + 3, 0:3] = -_skew(rot.T @ gravity)
        c_mat[3 * i : 3 * i + 3, 3:6] = eye
        d_vec[3 * i : 3 * i + 3] = torque
    sol = th.linalg.lstsq(c_mat, d_vec.unsqueeze(-1)).solution.squeeze(-1)
    return mass, sol[0:3] / mass, force_offset, sol[3:6]


def ft_sensor_identify_payload(env: BaseEnv, cfg: ExpConfig) -> None:
    """Measures the mass and centre of mass of whatever hangs past the F/T sensor.

    Visits a spread of wrist orientations, records the untared wrench at each, and fits
    the payload from them. Run it on both backends: the difference between the two
    answers IS the sim-to-real gap, expressed in the two numbers that cause it, rather
    than as an unexplained offset in the readings.

    On the real robot the joint targets below must be reachable and collision-free from
    home -- check them in RViz before running.
    """
    logger = get_logger("ft_sensor")
    logger.info("Starting F/T payload identification")

    manipulator = env.manipulator
    scene = env._scene
    device = cfg.get_device()
    dt = env.get_policy_dt()

    SETTLE_SECONDS = 4.0
    # Offsets in degrees from the home configuration, applied to
    # [shoulder_pan, shoulder_lift, elbow, wrist_1, wrist_2, wrist_3]. Only the wrist
    # moves, so the tool stays put while gravity sweeps around the sensor frame -- that
    # spread is what separates mass from centre of mass. Kept inside the URDF wrist
    # limits (wrist_1 [-2.79, -0.70], wrist_2 [-2.27, -0.87], wrist_3 [-2.27, 2.27]).
    POSE_OFFSETS_DEG = (
        (0.0, 0.0, 0.0, 0.0, 0.0, 0.0),
        (0.0, 0.0, 0.0, 40.0, 0.0, 0.0),
        (0.0, 0.0, 0.0, -40.0, 0.0, 0.0),
        (0.0, 0.0, 0.0, 0.0, 35.0, 0.0),
        (0.0, 0.0, 0.0, 0.0, -35.0, 0.0),
        (0.0, 0.0, 0.0, 30.0, 25.0, 90.0),
        (0.0, 0.0, 0.0, -30.0, -25.0, -90.0),
    )

    home = th.tensor(cfg.robot_cfg.default_arm_dof, dtype=th.float32, device=device)
    settle_steps = max(1, round(SETTLE_SECONDS / dt))

    manipulator.ctrl_go_to_home()
    for _ in range(settle_steps):
        _step(scene)

    # Identification needs the untared signal; a tare would hide the very offset the
    # fit estimates and drag the mass with it.
    manipulator.clear_ft_bias()

    rots, forces, torques = [], [], []
    for i, offsets in enumerate(POSE_OFFSETS_DEG):
        target = home + th.deg2rad(th.tensor(offsets, dtype=th.float32, device=device))
        manipulator.ctrl_go_to_joints(target)
        for _ in range(settle_steps):
            _step(scene)

        quat, _ = _sensor_quat(manipulator)
        wrench = manipulator.get_ft_wrench()[0]
        rots.append(_quat_to_matrix(quat))
        forces.append(wrench[:3].clone())
        torques.append(wrench[3:].clone())
        logger.info(f"  pose {i}: offsets={offsets} wrench={_fmt_wrench(wrench)}")

    gravity = th.tensor([0.0, 0.0, -9.81], dtype=th.float32, device=device)
    mass, com, force_offset, torque_offset = _identify_payload(
        rots, forces, torques, gravity
    )

    # Residual: what the fitted model fails to explain, in Newtons. Large values mean
    # the load is not rigid, the arm had not settled, or a pose was not reached.
    residuals = [
        th.linalg.norm(f - (mass * (r.T @ gravity) + force_offset))
        for r, f in zip(rots, forces)
    ]

    logger.info("=" * 78)
    logger.info(f"Payload identification over {len(rots)} poses")
    logger.info(f"  mass            = {float(mass):8.4f} kg")
    logger.info(
        f"  centre of mass  = [{' '.join(f'{v:8.4f}' for v in com.tolist())}] m "
        "(sensor frame)"
    )
    logger.info(
        f"  force offset    = {_fmt_wrench(th.cat([force_offset, torque_offset]))}"
    )
    logger.info(
        f"  fit residual    = {float(th.stack(residuals).max()):.4f} N max, "
        f"{float(th.stack(residuals).mean()):.4f} N mean"
    )
    if _is_modelled(manipulator):
        modelled = float(manipulator._gravity_link_masses[0].sum())
        logger.info(
            f"  URDF model says   {modelled:8.4f} kg "
            f"-> gap {1000 * (float(mass) - modelled):+.1f} g"
        )
    logger.info("=" * 78)


def _tcp_velocity(
    env: BaseEnv,
    manipulator: BaseManipulator,
    direction_tcp: tuple[float, float, float],
    speed: float,
    device: th.device,
) -> th.Tensor:
    """A world-frame twist, [num_envs, 6], moving at `speed` along a TCP-frame direction.

    `ctrl_apply_vel_action` wants world axes on both backends -- the sim IK uses a
    world-frame Jacobian and the bridge forwards the twist to `servo_tcp` unscaled -- so
    the direction is rotated out of the TCP frame here. Recomputed each step, because
    the TCP turns as the arm moves and a direction fixed in it does not stay fixed in
    the cell. The sensor's output frame and the TCP share an orientation, so the axes
    named here are the same ones the wrench is reported in.
    """
    quat, _ = _sensor_quat(manipulator)
    vec = th.tensor(direction_tcp, dtype=th.float32, device=device).unsqueeze(0)
    vec = vec / th.linalg.norm(vec)
    world = transform_by_quat(vec, quat.unsqueeze(0))
    action = th.zeros(env.num_envs, 6, device=device)
    action[:, :3] = world * speed
    return action


def _travelled(manipulator: BaseManipulator, origin: th.Tensor) -> float:
    """Distance the TCP has actually covered since `origin`, in metres.

    Measured, never integrated from the command. A velocity command is a request: the
    sim tracks it only as well as its gains allow, and on the real robot `servo_tcp`
    keeps running at that velocity until it is superseded, so a slow loop iteration
    carries the arm further than the commanded distance implies. Dead reckoning
    under-reports both, which is the wrong direction for a safety budget.
    """
    return float(th.linalg.norm(manipulator.get_tcp_position()[0] - origin))


def _stop(env: BaseEnv, manipulator: BaseManipulator, device: th.device) -> None:
    """Commands zero velocity. Called on every exit path, including failures."""
    manipulator.ctrl_apply_vel_action(
        th.zeros(env.num_envs, 6, device=device), open_gripper=None
    )


def _clamp_speed(env: BaseEnv, speed: float, label: str) -> float:
    """Keeps a commanded speed inside the configured limit for the cell."""
    logger = get_logger("ft_sensor")
    limit = float(env.max_linear_speed)
    if speed > limit:
        logger.warning(
            f"  {label} speed {speed:.4f} m/s exceeds the configured limit "
            f"{limit:.4f} m/s; clamping."
        )
        return limit
    return speed


def ft_sensor_contact_drag_test(env: BaseEnv, cfg: ExpConfig) -> None:
    """Touches the table under velocity control, then drags along the surface.

    Three phases: descend along the TCP's own Z until the sensor feels CONTACT_FORCE_N,
    stop and settle, then translate along the TCP's X while logging. The point is the
    drag: the normal force shows how well the height is held, and the tangential force
    is friction building up and breaking away, which is the first reading in this whole
    exercise that is not just gravity.

    Velocity control on purpose -- a planned move cannot stop on a force threshold.

    SAFETY: this drives the arm into a surface. Every phase is bounded by distance and
    by an abort force, the approach speed is clamped to the configured limit, and the
    arm is stopped and retracted on every exit path including an exception. Run it in
    simulation first and watch the numbers before letting it near the real cell.
    """
    logger = get_logger("ft_sensor")
    logger.info("Starting the F/T contact and drag test")

    manipulator = env.manipulator
    scene = env._scene
    device = cfg.get_device()
    dt = env.get_policy_dt()

    # Axes are given in the TCP frame, which is also the sensor's output frame, so a
    # force reported on Z is the force along APPROACH_AXIS_TCP.
    APPROACH_AXIS_TCP = (0.0, 0.0, 1.0)
    DRAG_AXIS_TCP = (1.0, 0.0, 0.0)
    # Home puts the tool over the cage, not the table, so step out along the TCP's own
    # X before descending. Same axis the drag runs along, so the drag continues the
    # direction this move starts in.
    START_OFFSET_TCP = (1.0, 0.0, 2.0)
    START_OFFSET_M = 0.15
    # The approach speed sets how finely contact can be resolved, because the force
    # jumps by `stiffness * speed * dt` between two samples and that jump is the only
    # warning before the threshold is passed. At 25 Hz, stopping within 1 N needs
    # roughly 28 mm/s on something soft (900 N/m), 2.5 mm/s at 10 kN/m and 0.25 mm/s on
    # a rigid table. Too fast and the reading goes from 0 N to past ABORT_FORCE_N in one
    # step, never showing the contact at all.
    APPROACH_SPEED_MPS = 0.15
    DRAG_SPEED_MPS = 0.010

    CONTACT_FORCE_N = 1.0  # what counts as "touching"
    ABORT_FORCE_N = 100.0  # anything past this and the test gives up
    MAX_APPROACH_M = 2.00  # travel budget if the surface is never felt
    DRAG_DISTANCE_M = 0.05
    RETRACT_M = 0.03
    SETTLE_SECONDS = 1.0
    MOVE_SETTLE_SECONDS = 5.0  # budget for the planned offset move to complete

    approach_speed = _clamp_speed(env, APPROACH_SPEED_MPS, "approach")
    drag_speed = _clamp_speed(env, DRAG_SPEED_MPS, "drag")
    settle_steps = max(1, round(SETTLE_SECONDS / dt))
    log_every = max(1, round(0.25 / dt))

    def _unit(vec: tuple[float, float, float]) -> th.Tensor:
        out = th.tensor(vec, dtype=th.float32, device=device)
        return out / th.linalg.norm(out)

    axis = _unit(APPROACH_AXIS_TCP)
    drag_axis = _unit(DRAG_AXIS_TCP)

    def force_along(unit_axis: th.Tensor) -> float:
        """Signed force on a TCP-frame axis. The sensor reports in that same frame, so
        this is a projection and not a change of frame."""
        return float(manipulator.get_ft_wrench()[0, :3] @ unit_axis)

    manipulator.ctrl_go_to_home()
    for _ in range(settle_steps):
        _step(scene)
    _log_frame(manipulator, "home")

    # A planned TCP goal, not a velocity ramp: this is free-space and point-to-point,
    # with no force threshold to stop on, so the manipulation stack should own the
    # trajectory, the joint limits and -- the reason it matters here -- the collision
    # checking. Clearing the cage is exactly the move that wants a planner rather than
    # a blind velocity command.
    logger.info(
        f"Moving {START_OFFSET_M * 100:.0f} cm along TCP{START_OFFSET_TCP} to clear "
        "the cage and sit over the table"
    )
    pose = manipulator.get_tcp_pose().clone()
    quat, _ = _sensor_quat(manipulator)
    offset_world = transform_by_quat(
        _unit(START_OFFSET_TCP).unsqueeze(0), quat.unsqueeze(0)
    )
    origin = manipulator.get_tcp_position()[0].clone()
    manipulator.ctrl_go_to_goal(
        th.cat([pose[:, :3] + offset_world * START_OFFSET_M, pose[:, 3:]], dim=-1),
        open_gripper=None,
    )
    for _ in range(max(1, round(MOVE_SETTLE_SECONDS / dt))):
        _step(scene)
    moved = _travelled(manipulator, origin)
    logger.info(f"  moved {moved * 1000:.1f} mm of {START_OFFSET_M * 1000:.0f} mm")
    if abs(moved - START_OFFSET_M) > 0.005:
        logger.warning(
            "  the arm did not reach the offset -- it may still be over the cage. "
            "Check that before letting it descend."
        )

    # Tared here rather than at home, so the offset is taken in the pose the approach
    # actually starts from.
    # Tared in free space, so every force below is the contact, not the tool's weight.
    manipulator.set_ft_bias()
    logger.info("  tared in free space before the approach")

    # Closed after the tare, so the fingers meet the surface rather than straddle it.
    # The two fingers travel the same distance in opposite directions, so a symmetric
    # close shifts no mass across the sensor and the tare stays valid -- the reading
    # logged below is the check on that, and should still be ~0.
    manipulator.ctrl_gripper_close()
    for _ in range(settle_steps):
        _step(scene)
    logger.info(
        f"  gripper closed, wrench now {_fmt_wrench(manipulator.get_ft_wrench()[0])}"
    )

    # Said out loud before anything moves: if the sign is wrong this is the line that
    # shows it, and the approach axis points wherever the tool is actually facing.
    quat, _ = _sensor_quat(manipulator)
    world_dir = transform_by_quat(axis.unsqueeze(0), quat.unsqueeze(0))[0]
    logger.info(
        f"  approach axis TCP{APPROACH_AXIS_TCP} points along world "
        f"[{' '.join(f'{v:+.3f}' for v in world_dir.tolist())}] "
        f"({'downward' if world_dir[2] < -0.5 else 'NOT downward -- check this'})"
    )

    try:
        # --- phase 1: approach until the surface is felt ----------------------
        # A backstop for a stalled or frozen pose, not the budget -- generous, so
        # ordinary tracking error never trips it. The budget is measured travel.
        max_steps = max(1, round(3.0 * MAX_APPROACH_M / approach_speed / dt))
        logger.info(
            f"Phase 1: descending at {approach_speed * 1000:.1f} mm/s until "
            f"|Fz| >= {CONTACT_FORCE_N:.2f} N (budget {MAX_APPROACH_M * 100:.0f} cm)"
        )
        origin = manipulator.get_tcp_position()[0].clone()
        touched = False
        moved = 0.0
        for i in range(max_steps):
            manipulator.ctrl_apply_vel_action(
                _tcp_velocity(
                    env, manipulator, APPROACH_AXIS_TCP, approach_speed, device
                ),
                open_gripper=None,
            )
            _step(scene)

            moved = _travelled(manipulator, origin)
            f_n = force_along(axis)
            if i % log_every == 0:
                logger.info(
                    f"  [{moved * 1000:6.1f} mm] "
                    f"Fz={f_n:+7.3f} N  {_fmt_wrench(manipulator.get_ft_wrench()[0])}"
                )
            if abs(f_n) >= ABORT_FORCE_N:
                raise RuntimeError(
                    f"Contact force {f_n:+.3f} N exceeded the abort limit "
                    f"{ABORT_FORCE_N:.1f} N after {moved * 1000:.1f} mm -- stopping."
                )
            if abs(f_n) >= CONTACT_FORCE_N:
                touched = True
                logger.info(f"  contact after {moved * 1000:.1f} mm, Fz={f_n:+.3f} N")
                break
            if moved >= MAX_APPROACH_M:
                break

        _stop(env, manipulator, device)

        if not touched:
            logger.warning(
                f"  no contact after {moved * 1000:.1f} mm of travel (budget "
                f"{MAX_APPROACH_M * 100:.0f} cm) -- the surface is out of reach, or "
                "the sensor is not reporting. Skipping the drag."
            )
            return

        for _ in range(settle_steps):
            _step(scene)
        settled = manipulator.get_ft_wrench()[0].clone()
        logger.info(f"  touch the the surface (now wait 5s): {_fmt_wrench(settled)}")
        time.sleep(5)
        settled = manipulator.get_ft_wrench()[0].clone()
        logger.info(f"  settled on the surface: {_fmt_wrench(settled)}")

        # --- phase 2: drag along the surface ----------------------------------
        drag_steps = max(1, round(3.0 * DRAG_DISTANCE_M / drag_speed / dt))
        logger.info(
            f"Phase 2: dragging {DRAG_DISTANCE_M * 100:.0f} cm along TCP"
            f"{DRAG_AXIS_TCP} at {drag_speed * 1000:.1f} mm/s"
        )
        origin = manipulator.get_tcp_position()[0].clone()
        peak_tangential = 0.0
        for i in range(drag_steps):
            manipulator.ctrl_apply_vel_action(
                _tcp_velocity(env, manipulator, DRAG_AXIS_TCP, drag_speed, device),
                open_gripper=None,
            )
            _step(scene)

            moved = _travelled(manipulator, origin)
            wrench = manipulator.get_ft_wrench()[0]
            f_n = force_along(axis)
            f_t = float(wrench[:3] @ drag_axis)
            peak_tangential = max(peak_tangential, abs(f_t))
            if abs(f_n) >= ABORT_FORCE_N:
                raise RuntimeError(
                    f"Normal force {f_n:+.3f} N exceeded the abort limit during the "
                    f"drag, after {moved * 1000:.1f} mm -- stopping."
                )
            if i % log_every == 0:
                mu = abs(f_t / f_n) if abs(f_n) > 1e-6 else float("nan")
                logger.info(
                    f"  [{moved * 1000:6.1f} mm] "
                    f"Fnormal={f_n:+7.3f} N  Ftangential={f_t:+7.3f} N  "
                    f"ratio={mu:5.2f}  {_fmt_wrench(wrench)}"
                )
            if moved >= DRAG_DISTANCE_M:
                logger.info(f"  drag complete at {moved * 1000:.1f} mm")
                break
        _stop(env, manipulator, device)
        for _ in range(settle_steps):
            _step(scene)

        logger.info("=" * 78)
        logger.info("Contact and drag summary")
        logger.info(f"  on contact, settled   {_fmt_wrench(settled)}")
        logger.info(
            f"  after the drag        {_fmt_wrench(manipulator.get_ft_wrench()[0])}"
        )
        logger.info(f"  peak tangential force {peak_tangential:.3f} N")
        logger.info(
            "  the tangential/normal ratio is an apparent friction coefficient; it "
            "rises while the tool sticks and levels off once it slides"
        )
        logger.info("=" * 78)
    finally:
        # Whatever happened, do not leave the arm loaded against the table. The
        # velocity command is dropped first, so the arm is already still before the
        # planner is asked for anything.
        _stop(env, manipulator, device)
        logger.info(f"Retracting {RETRACT_M * 100:.0f} cm")
        try:
            pose = manipulator.get_tcp_pose().clone()
            quat, _ = _sensor_quat(manipulator)
            back = transform_by_quat(
                _unit(tuple(-v for v in APPROACH_AXIS_TCP)).unsqueeze(0),
                quat.unsqueeze(0),
            )
            origin = manipulator.get_tcp_position()[0].clone()
            manipulator.ctrl_go_to_goal(
                th.cat([pose[:, :3] + back * RETRACT_M, pose[:, 3:]], dim=-1),
                open_gripper=None,
            )
            for _ in range(max(1, round(MOVE_SETTLE_SECONDS / dt))):
                _step(scene)
            logger.info(
                f"  retracted {_travelled(manipulator, origin) * 1000:.1f} mm, "
                f"clear of the surface: "
                f"{_fmt_wrench(manipulator.get_ft_wrench()[0])}"
            )
        except Exception as exc:  # noqa: BLE001 - see below; nothing may escape here
            # This runs while another exception may already be propagating, and that
            # one is the one worth seeing. Report the failed retreat loudly, but do
            # not let it replace the cause.
            logger.error(
                f"  RETRACT FAILED ({exc}) -- the tool may still be loaded against "
                "the surface. Clear it by hand before running again."
            )


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
