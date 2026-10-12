"""
Servo stiffness probes: compare simulated and real contact dynamics.

- Torque probe: presses a tilted open gripper into the table and fits the servo
  stiffness K_eff, command latency and torque moment arm.
  Set `servo_stiffness_scale` in `robot.yaml` to K_eff_real / K_eff_sim.
- Contact and drag test: touches the table and drags along it, logging forces.

Run on both backends and compare:

    python -m scripts.ft_sensor.stiffness_probe --env reacher -B 1 --vis      # simulation
    python -m scripts.ft_sensor.stiffness_probe --env reacher --control ros   # real robot

SAFETY: both tests drive the arm into a surface. Run them in simulation first.
"""

import math

import torch as th

from aegis_gym.aux.geom import transform_by_quat, transform_quat_by_quat
from aegis_gym.aux.logging import get_logger, setup_logger
from aegis_gym.config import ConfigManager, parse_arguments
from aegis_gym.config.types import ExpConfig
from aegis_gym.envs import BaseEnv
from aegis_gym.envs.manipulator import BaseManipulator
from aegis_gym.envs.plotjuggler_udp import PlotJugglerUDP
from scripts.ft_sensor.common import (
    fmt_wrench,
    is_modelled,
    log_frame,
    rotation_since,
    sensor_quat,
    step,
)
from train import create_env, init_clearml_task

# Which test `main()` runs: "torque_probe" or "contact_drag".
TEST = "torque_probe"


def _log_contacts(manipulator: BaseManipulator, label: str) -> None:
    """Simulation only: logs the solver's contacts and compares its contact force
    with the F/T model's (sensor frame)."""
    if not is_modelled(manipulator):
        return
    logger = get_logger("ft_stiffness_probe")
    entity = manipulator._robot_entity
    contacts = entity.get_contacts(exclude_self_contact=True)

    mask = contacts["valid_mask"][0]
    hits = mask.nonzero().reshape(-1).tolist() if mask.numel() else []
    if not hits:
        logger.info(f"  {label}: solver reports NO contact")
        return

    links = entity.solver.links

    def _name(link_idx: int) -> str:
        """Link name with entity index; boxes share the name `box_baselink`."""
        link = links[link_idx]
        return f"{link.name}@e{link.entity.idx}"

    for i in hits[:4]:
        name_a = _name(int(contacts["link_a"][0, i]))
        name_b = _name(int(contacts["link_b"][0, i]))
        force = float(th.linalg.norm(contacts["force_a"][0, i]))
        pen = float(contacts["penetration"][0, i]) * 1000.0
        # A tilted normal means an edge contact, not friction.
        normal = contacts["normal"][0, i]
        tilt = float(th.rad2deg(th.acos(th.clamp(abs(normal[2]), 0.0, 1.0))))
        logger.info(
            f"  {label}: {name_a} <-> {name_b}  |F|={force:6.3f} N  pen={pen:5.2f} mm  "
            f"normal=[{' '.join(f'{v:+.2f}' for v in normal.tolist())}] "
            f"({tilt:4.1f} deg off vertical)"
        )
    if len(hits) > 4:
        logger.info(f"  {label}: ... and {len(hits) - 4} more contact point(s)")

    net = th.zeros(3, device=manipulator.device)
    for i in hits:
        a_is_ours = bool(
            th.isin(contacts["link_a"][0, i], manipulator._gravity_link_idx)
        )
        net = net + (
            contacts["force_a"][0, i] if a_is_ours else contacts["force_b"][0, i]
        )
    normal = abs(float(net[2]))
    tangential = float(th.linalg.norm(net[:2]))
    if normal > 1e-6:
        logger.info(
            f"  {label}: world contact force [{net[0]:+7.3f} {net[1]:+7.3f} "
            f"{net[2]:+7.3f}] N -> mobilised mu = {tangential / normal:.3f} "
            f"(the robot measured 0.30)"
        )

    # Both gravity-free, so any difference is a model bug.
    quat = manipulator.get_ft_frame_quat()
    quat_conj = quat * th.tensor(
        [1.0, -1.0, -1.0, -1.0], device=quat.device, dtype=quat.dtype
    )
    net = entity.get_links_net_contact_force()[0]
    rows = [link.idx - entity.link_start for link in manipulator._gravity_links]
    truth = transform_by_quat(net[rows].sum(dim=0).unsqueeze(0), quat_conj)[0]
    model = transform_by_quat(manipulator._contact_wrench_world()[:1, :3], quat_conj)[0]
    logger.info(
        f"  {label}: contact force, solver "
        f"[{' '.join(f'{v:7.3f}' for v in truth.tolist())}] "
        f"vs model [{' '.join(f'{v:7.3f}' for v in model.tolist())}]"
    )


# Phase as a number, so PlotJuggler can plot it.
_PHASE_ID = {"approach": 0, "hold": 1, "release": 2, "unload": 3}


class _ProbeStream:
    """Sends probe signals (commanded vs actual travel, lag, forces) to PlotJuggler.
    Same keys in sim and real, so runs can be overlaid."""

    def __init__(self, cfg: ExpConfig) -> None:
        self._pj = (
            PlotJugglerUDP(host="127.0.0.1", port=9870)
            if cfg.args.enable_plotjuggler
            else None
        )
        self._n = 0

    def send(
        self,
        manipulator: BaseManipulator,
        dt: float,
        phase: str,
        cmd_travel_m: float,
        travel_m: float,
        cmd_speed_mps: float,
    ) -> None:
        if self._pj is None:
            return
        wrench = manipulator.get_ft_wrench_compensated()[0].tolist()
        data = {
            "probe/t_s": self._n * dt,
            "probe/phase": _PHASE_ID.get(phase, -1),
            "probe/cmd_travel_mm": cmd_travel_m * 1000.0,
            "probe/travel_mm": travel_m * 1000.0,
            "probe/lag_mm": (cmd_travel_m - travel_m) * 1000.0,
            "probe/cmd_speed_mm_s": cmd_speed_mps * 1000.0,
        }
        for i, axis in enumerate("xyz"):
            data[f"probe/force/{axis}"] = wrench[i]
            data[f"probe/torque/{axis}"] = wrench[i + 3]
        self._pj.send(data)
        self._n += 1


# Smallest action the real servo executes (deadband); sim has none.
MIN_ACTION_FRACTION = 0.01
MIN_ACTION_FRACTION_SIM = 2.0e-4
_ACTION_WARNED: set[int] = set()

# Out of reach, so the task object can't be hit instead of the table.
OBJECT_PARK_POS = (1.5, 1.5, 0.05)
START_OFFSET_TCP = (3.0, 0.0, 3.0)
START_OFFSET_M = 0.2
MOVE_SETTLE_SECONDS = 5.0


def _prepare_over_table(
    env: BaseEnv,
    cfg: ExpConfig,
    settle_steps: int,
    tilt_quat: th.Tensor | None = None,
    close_gripper: bool = True,
) -> None:
    """Home, park the task object, move over the table, tare, set the gripper.

    `tilt_quat`: extra rotation in the TCP frame. Open gripper gives a lateral
    contact offset (moment arm); closed gives none.
    """
    logger = get_logger("ft_stiffness_probe")
    manipulator = env.manipulator
    scene = env._scene
    device = cfg.get_device()
    dt = env.get_policy_dt()

    manipulator.ctrl_go_to_home()
    for _ in range(settle_steps):
        step(scene)
    log_frame(manipulator, "home")

    obj = getattr(env, "object", None)
    if obj is not None:
        parked = th.tensor(
            [[*OBJECT_PARK_POS, 1.0, 0.0, 0.0, 0.0]], dtype=th.float32, device=device
        ).repeat(env.num_envs, 1)
        obj.set_pose(pose=parked)
        for _ in range(settle_steps):
            step(scene)
        logger.info(f"  parked the task object at {OBJECT_PARK_POS} to clear the path")

    # Planned move, so the planner checks collisions with the cage.
    logger.info(
        f"Moving {START_OFFSET_M * 100:.0f} cm along TCP{START_OFFSET_TCP} to clear "
        "the cage and sit over the table"
    )
    pose = manipulator.get_tcp_pose().clone()
    quat = sensor_quat(manipulator)
    offset = th.tensor(START_OFFSET_TCP, dtype=th.float32, device=device).unsqueeze(0)
    offset = offset / th.linalg.norm(offset)
    offset_world = transform_by_quat(offset, quat.unsqueeze(0))
    goal_quat = pose[:, 3:]
    if tilt_quat is not None:
        # Rotation about the TCP axes.
        goal_quat = transform_quat_by_quat(goal_quat, tilt_quat.expand_as(goal_quat))
    origin = manipulator.get_tcp_position()[0].clone()
    manipulator.ctrl_go_to_goal(
        th.cat([pose[:, :3] + offset_world * START_OFFSET_M, goal_quat], dim=-1),
        open_gripper=None,
    )
    for _ in range(max(1, round(MOVE_SETTLE_SECONDS / dt))):
        step(scene)
    moved = _travelled(manipulator, origin)
    logger.info(f"  moved {moved * 1000:.1f} mm of {START_OFFSET_M * 1000:.0f} mm")
    if abs(moved - START_OFFSET_M) > 0.005:
        logger.warning(
            "  the arm did not reach the offset -- it may still be over the cage. "
            "Check that before letting it descend."
        )

    manipulator.set_ft_bias()
    logger.info("  tared in free space before the approach")
    if close_gripper:
        manipulator.ctrl_gripper_close()
    else:
        manipulator.ctrl_gripper_open()
    for _ in range(settle_steps):
        step(scene)
    logger.info(
        f"  gripper {'closed' if close_gripper else 'open'}, wrench now "
        f"{fmt_wrench(manipulator.get_ft_wrench()[0])}"
    )


def _tcp_velocity(
    env: BaseEnv,
    manipulator: BaseManipulator,
    direction_tcp: tuple[float, float, float],
    speed: float,
    device: th.device,
) -> th.Tensor:
    """World-frame action [num_envs, 6] moving at `speed` [m/s] along a TCP-frame
    direction (TCP and sensor frames share orientation).

    Normalised by `env.max_linear_speed`; warns when clipped above 1 or below the
    servo deadband (no motion).
    """
    quat = sensor_quat(manipulator)
    vec = th.tensor(direction_tcp, dtype=th.float32, device=device).unsqueeze(0)
    vec = vec / th.linalg.norm(vec)
    world = transform_by_quat(vec, quat.unsqueeze(0))
    scale = speed / float(env.max_linear_speed)
    if abs(scale) > 1.0:
        get_logger("ft_stiffness_probe").warning(
            f"  {speed * 1000:.3f} mm/s is {abs(scale):.2f} of full scale -- the action "
            f"saturates at 1.0, so the arm will move at "
            f"{float(env.max_linear_speed) * 1000:.1f} mm/s instead."
        )
        scale = math.copysign(1.0, scale)
    elif (
        not is_modelled(manipulator)
        and 0.0 < abs(scale) < MIN_ACTION_FRACTION
        and round(speed * 1e6) not in _ACTION_WARNED
    ):
        # Warn once per speed, not every step.
        _ACTION_WARNED.add(round(speed * 1e6))
        get_logger("ft_stiffness_probe").warning(
            f"  {speed * 1e6:.0f} um/s is an action of {abs(scale):.4f}, under the "
            f"{MIN_ACTION_FRACTION:.3f} the servo resolves -- expect no motion."
        )
    action = th.zeros(env.num_envs, 6, device=device)
    action[:, :3] = world * scale
    return action


def _travelled(manipulator: BaseManipulator, origin: th.Tensor) -> float:
    """Measured TCP distance from `origin` [m] (not integrated from commands)."""
    return float(th.linalg.norm(manipulator.get_tcp_position()[0] - origin))


def _stop(env: BaseEnv, manipulator: BaseManipulator, device: th.device) -> None:
    """Commands zero velocity."""
    manipulator.ctrl_apply_vel_action(
        th.zeros(env.num_envs, 6, device=device), open_gripper=None
    )


def _clamp_speed(env: BaseEnv, speed: float, label: str) -> float:
    """Clamps the speed to `env.max_linear_speed`."""
    logger = get_logger("ft_stiffness_probe")
    limit = float(env.max_linear_speed)
    if speed > limit:
        logger.warning(
            f"  {label} speed {speed:.4f} m/s exceeds the configured limit "
            f"{limit:.4f} m/s; clamping."
        )
        return limit
    return speed


def ft_sensor_contact_drag_test(env: BaseEnv, cfg: ExpConfig) -> None:
    """Descends along TCP Z until contact, holds CONTACT_FORCE_N, then drags along
    TCP X and logs normal and tangential (friction) force.

    SAFETY: bounded by distance and abort force; always stops and retracts.
    """
    logger = get_logger("ft_stiffness_probe")
    logger.info("Starting the F/T contact and drag test")

    manipulator = env.manipulator
    scene = env._scene
    device = cfg.get_device()
    dt = env.get_policy_dt()

    # Axes in the TCP frame (= sensor frame).
    APPROACH_AXIS_TCP = (0.0, 0.0, 1.0)
    DRAG_AXIS_TCP = (1.0, 0.0, 0.0)
    # Force jumps by stiffness * speed * dt per step; too fast skips past the threshold.
    APPROACH_SPEED_MPS = 0.15
    DRAG_SPEED_MPS = 0.010

    CONTACT_FORCE_N = 1.0  # contact threshold and held force
    # PI control of the normal force; the integral removes the steady-state error.
    FORCE_GAIN_MPS_PER_N = 0.004
    FORCE_INTEGRAL_MPS_PER_NS = 0.020
    MAX_NORMAL_SPEED_MPS = 0.020
    # Anti-windup.
    FORCE_INTEGRAL_CLAMP_NS = MAX_NORMAL_SPEED_MPS / FORCE_INTEGRAL_MPS_PER_NS
    ABORT_FORCE_N = 100.0
    MAX_APPROACH_M = 2.00  # travel budget
    DRAG_DISTANCE_M = 0.05
    RETRACT_M = 0.03
    SETTLE_SECONDS = 1.0

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
        """Signed force along a TCP-frame axis."""
        return float(manipulator.get_ft_wrench()[0, :3] @ unit_axis)

    force_integral = 0.0

    def normal_speed() -> float:
        """PI speed along the approach axis towards CONTACT_FORCE_N (+ presses in)."""
        nonlocal force_integral
        error = CONTACT_FORCE_N - abs(force_along(axis))
        force_integral = max(
            -FORCE_INTEGRAL_CLAMP_NS,
            min(FORCE_INTEGRAL_CLAMP_NS, force_integral + error * dt),
        )
        speed = (
            FORCE_GAIN_MPS_PER_N * error + FORCE_INTEGRAL_MPS_PER_NS * force_integral
        )
        return max(-MAX_NORMAL_SPEED_MPS, min(MAX_NORMAL_SPEED_MPS, speed))

    _prepare_over_table(env, cfg, settle_steps)

    # Sanity check of the approach direction before moving.
    quat = sensor_quat(manipulator)
    world_dir = transform_by_quat(axis.unsqueeze(0), quat.unsqueeze(0))[0]
    logger.info(
        f"  approach axis TCP{APPROACH_AXIS_TCP} points along world "
        f"[{' '.join(f'{v:+.3f}' for v in world_dir.tolist())}] "
        f"({'downward' if world_dir[2] < -0.5 else 'NOT downward -- check this'})"
    )

    try:
        # --- phase 1: approach until contact ---
        # Step limit is only a backstop; the real budget is measured travel.
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
            step(scene)

            moved = _travelled(manipulator, origin)
            f_n = force_along(axis)
            if i % log_every == 0:
                logger.info(
                    f"  [{moved * 1000:6.1f} mm] "
                    f"Fz={f_n:+7.3f} N  {fmt_wrench(manipulator.get_ft_wrench()[0])}"
                )
            if abs(f_n) >= ABORT_FORCE_N:
                raise RuntimeError(
                    f"Contact force {f_n:+.3f} N exceeded the abort limit "
                    f"{ABORT_FORCE_N:.1f} N after {moved * 1000:.1f} mm -- stopping."
                )
            if abs(f_n) >= CONTACT_FORCE_N:
                touched = True
                logger.info(f"  contact after {moved * 1000:.1f} mm, Fz={f_n:+.3f} N")
                _log_contacts(manipulator, "at contact")
                break
            if moved >= MAX_APPROACH_M:
                break

        if not touched:
            _stop(env, manipulator, device)
            logger.warning(
                f"  no contact after {moved * 1000:.1f} mm of travel (budget "
                f"{MAX_APPROACH_M * 100:.0f} cm) -- the surface is out of reach, or "
                "the sensor is not reporting. Skipping the drag."
            )
            return

        # No `_stop()`: keep regulating force so the tool doesn't bounce off.
        logger.info(
            f"  holding {CONTACT_FORCE_N:.2f} N for {SETTLE_SECONDS:.0f}s to settle"
        )
        for i in range(settle_steps):
            manipulator.ctrl_apply_vel_action(
                _tcp_velocity(
                    env, manipulator, APPROACH_AXIS_TCP, normal_speed(), device
                ),
                open_gripper=None,
            )
            step(scene)
            if i % log_every == 0:
                logger.info(
                    f"    Fz={force_along(axis):+7.3f} N  "
                    f"{fmt_wrench(manipulator.get_ft_wrench()[0])}"
                )
        settled = manipulator.get_ft_wrench()[0].clone()
        logger.info(f"  settled on the surface: {fmt_wrench(settled)}")
        _log_contacts(manipulator, "settled")
        if abs(force_along(axis)) < 0.5 * CONTACT_FORCE_N:
            logger.warning(
                "  the normal force collapsed while settling -- the tool is off the "
                "surface. The drag below will measure nothing."
            )

        # --- phase 2: drag along the surface ---
        drag_steps = max(1, round(3.0 * DRAG_DISTANCE_M / drag_speed / dt))
        logger.info(
            f"Phase 2: dragging {DRAG_DISTANCE_M * 100:.0f} cm along TCP"
            f"{DRAG_AXIS_TCP} at {drag_speed * 1000:.1f} mm/s"
        )
        origin = manipulator.get_tcp_position()[0].clone()
        # Tilt drift leaks payload weight into the tangential force (~0.23 N/deg).
        quat_contact = sensor_quat(manipulator).clone()
        peak_tangential = 0.0
        tilt = 0.0
        for i in range(drag_steps):
            # Drag plus normal force correction.
            manipulator.ctrl_apply_vel_action(
                _tcp_velocity(env, manipulator, DRAG_AXIS_TCP, drag_speed, device)
                + _tcp_velocity(
                    env, manipulator, APPROACH_AXIS_TCP, normal_speed(), device
                ),
                open_gripper=None,
            )
            step(scene)

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
                tilt = rotation_since(quat_contact, sensor_quat(manipulator))[0]
                logger.info(
                    f"  [{moved * 1000:6.1f} mm] "
                    f"Fnormal={f_n:+7.3f} N  Ftangential={f_t:+7.3f} N  "
                    f"ratio={mu:5.2f}  tilt={tilt:4.2f} deg  {fmt_wrench(wrench)}"
                )
            if i % (8 * log_every) == 0:
                _log_contacts(manipulator, "dragging")
            if moved >= DRAG_DISTANCE_M:
                logger.info(f"  drag complete at {moved * 1000:.1f} mm")
                break
        _stop(env, manipulator, device)
        for _ in range(settle_steps):
            step(scene)

        logger.info("=" * 78)
        logger.info("Contact and drag summary")
        logger.info(f"  on contact, settled   {fmt_wrench(settled)}")
        logger.info(
            f"  after the drag        {fmt_wrench(manipulator.get_ft_wrench()[0])}"
        )
        logger.info(f"  peak tangential force {peak_tangential:.3f} N")
        logger.info(
            f"  orientation drift over the drag {tilt:.2f} deg "
            f"(~{13.33 * abs(th.sin(th.deg2rad(th.tensor(tilt)))):.3f} N of the "
            "tangential reading is the payload's weight, not friction)"
        )
        logger.info(
            "  the tangential/normal ratio is an apparent friction coefficient; it "
            "rises while the tool sticks and levels off once it slides"
        )
        logger.info("=" * 78)
    finally:
        # Always stop and retract from the table.
        _stop(env, manipulator, device)
        logger.info(f"Retracting {RETRACT_M * 100:.0f} cm")
        try:
            pose = manipulator.get_tcp_pose().clone()
            quat = sensor_quat(manipulator)
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
                step(scene)
            logger.info(
                f"  retracted {_travelled(manipulator, origin) * 1000:.1f} mm, "
                f"clear of the surface: "
                f"{fmt_wrench(manipulator.get_ft_wrench()[0])}"
            )
        except Exception as exc:  # noqa: BLE001 - must not mask the real failure
            logger.error(
                f"  RETRACT FAILED ({exc}) -- the tool may still be loaded against "
                "the surface. Clear it by hand before running again."
            )


def ft_sensor_torque_probe(env: BaseEnv, cfg: ExpConfig) -> None:
    """Measures servo stiffness, command latency and the torque channel.

    The tool is tilted with the gripper open, so one finger (~25 mm off-axis) touches
    first. Presses at constant speed to MAX_FORCE_N, holds, then unloads slowly and
    fits torque against force (slope = moment arm). No force control: the contact is
    too stiff for a stable loop.
    """
    logger = get_logger("ft_stiffness_probe")
    logger.info("Starting the F/T torque-channel probe")

    manipulator = env.manipulator
    scene = env._scene
    device = cfg.get_device()
    dt = env.get_policy_dt()

    TILT_DEG = 6.0
    TILT_AXIS_TCP = (0.0, 1.0, 0.0)  # fingers separate along TCP X
    APPROACH_AXIS_TCP = (0.0, 0.0, 1.0)
    TORQUE_AXIS_TCP = (0.0, 1.0, 0.0)
    TANGENT_AXIS_TCP = (1.0, 0.0, 0.0)  # friction direction
    # Nominal finger offset from the URDF; sim overrides it from contacts.
    EXPECTED_ARM_M = 0.025
    # Sensor to fingertip along the tool axis (URDF); sim overrides it from contacts.
    EXPECTED_RZ_M = 0.1754

    # Slow enough that the first touch stays below ABORT_FORCE_N (~24 N at 400 kN/m).
    # Constant: with ~1 s command delay, slowing down on touch corrupts the estimate.
    APPROACH_SPEED_MPS = 0.0015
    TOUCH_FORCE_N = 0.3  # start of the stiffness samples
    MAX_FORCE_N = 5.0  # press target
    MIN_FIT_FORCE_N = 0.5  # below this it's offset, not signal
    ABORT_FORCE_N = 30.0
    MAX_APPROACH_M = 0.20
    UNLOAD_SECONDS = 120.0

    # After the press the setpoint is still inside the table, so force keeps rising
    # (5.6 -> 16.2 N on the robot). It must be released before the unload sweep.
    HOLD_SECONDS = 5.0
    RELEASE_SECONDS = 30.0
    UNLOAD_TARGET_N_PER_STEP = 0.10  # sweep resolution
    # Above the servo deadband.
    UNLOAD_MIN_SPEED_MPS = float(env.max_linear_speed) * (
        MIN_ACTION_FRACTION_SIM if is_modelled(manipulator) else MIN_ACTION_FRACTION
    )
    UNLOAD_MAX_SPEED_MPS = 5.0e-3
    UNLOAD_ADAPT_EVERY = 25  # steps between speed corrections
    UNLOAD_STALL_SECONDS = 8.0  # give up if force doesn't fall this long
    DESCEND_MARGIN_N = 0.4  # drop below the peak that starts the sweep

    SETTLE_SECONDS = 1.0
    RETRACT_M = 0.03

    stream = _ProbeStream(cfg)
    approach_speed = _clamp_speed(env, APPROACH_SPEED_MPS, "approach")
    settle_steps = max(1, round(SETTLE_SECONDS / dt))

    def _unit(vec: tuple[float, float, float]) -> th.Tensor:
        out = th.tensor(vec, dtype=th.float32, device=device)
        return out / th.linalg.norm(out)

    axis = _unit(APPROACH_AXIS_TCP)
    torque_axis = _unit(TORQUE_AXIS_TCP)
    tangent_axis = _unit(TANGENT_AXIS_TCP)

    def reading() -> tuple[float, float, float, float]:
        """(|Fz|, Fz, Fx, tau_y), gravity compensated.

        Fx is needed because tau_y = r_z * Fx - r_x * Fz: friction on the long
        axial lever also creates torque.
        """
        wrench = manipulator.get_ft_wrench_compensated()[0]
        f_z = float(wrench[:3] @ axis)
        f_x = float(wrench[:3] @ tangent_axis)
        return abs(f_z), f_z, f_x, float(wrench[3:] @ torque_axis)

    half = math.radians(TILT_DEG) / 2.0
    tilt_vec = _unit(TILT_AXIS_TCP) * math.sin(half)
    tilt_quat = th.cat(
        [th.tensor([math.cos(half)], device=device), tilt_vec]
    ).unsqueeze(0)

    # Open gripper, so the contact is off the tool axis.
    _prepare_over_table(
        env, cfg, settle_steps, tilt_quat=tilt_quat, close_gripper=False
    )
    logger.info(
        f"  tool tilted {TILT_DEG:.1f} deg about TCP{TILT_AXIS_TCP} with the gripper "
        f"open, so one finger lands first with an expected arm of "
        f"{EXPECTED_ARM_M * 1000:.0f} mm"
    )

    samples: list[tuple[float, float, float, float]] = []
    try:
        # Press in, then measure while unloading (loading is too stiff to resolve).
        logger.info(
            f"Pressing in at {approach_speed * 1000:.1f} mm/s until "
            f"|Fz| >= {MAX_FORCE_N:.1f} N"
        )
        origin = manipulator.get_tcp_position()[0].clone()
        # (lag, force), where F = K_eff * lag and lag = commanded - actual travel.
        probe: list[tuple[float, float]] = []
        # (commanded travel, force): estimate that doesn't use the measured pose.
        probe_cmd: list[tuple[float, float]] = []
        cmd_travel = 0.0
        touched_logged = False
        touched = False
        # Periodic progress log.
        press_log_every = max(1, round(2.0 / dt))
        stalled_since: float | None = None
        speed = approach_speed
        for i in range(max(1, round(3.0 * MAX_APPROACH_M / approach_speed / dt))):
            manipulator.ctrl_apply_vel_action(
                _tcp_velocity(env, manipulator, APPROACH_AXIS_TCP, speed, device),
                open_gripper=None,
            )
            step(scene)
            cmd_travel += speed * dt
            f_n, _, _, _ = reading()
            moved = _travelled(manipulator, origin)
            lag = cmd_travel - moved
            stream.send(manipulator, dt, "approach", cmd_travel, moved, speed)
            if i % press_log_every == 0:
                logger.info(
                    f"    t={i * dt:6.1f} s  cmd={cmd_travel * 1000:7.2f} mm  "
                    f"travel={moved * 1000:7.2f} mm  lag={lag * 1000:6.3f} mm  "
                    f"Fn={f_n:6.2f} N"
                )
                # Blocked but no force: the servo isn't accumulating error.
                if stalled_since is not None and moved - stalled_since < 1e-4:
                    logger.warning(
                        f"    stalled at {moved * 1000:.2f} mm with only "
                        f"{f_n:.2f} N -- blocked but not loading"
                    )
                stalled_since = moved
            if f_n >= TOUCH_FORCE_N:
                if not touched_logged:
                    touched_logged = True
                    logger.info(
                        f"  touched at {moved * 1000:.2f} mm ({f_n:.2f} N) -- holding "
                        f"{speed * 1000:.2f} mm/s through the loading curve"
                    )
                probe.append((lag, f_n))
                probe_cmd.append((cmd_travel, f_n))
            if f_n >= ABORT_FORCE_N:
                raise RuntimeError(
                    f"Contact force {f_n:.2f} N exceeded the abort limit"
                )
            if f_n >= MAX_FORCE_N:
                touched = True
                logger.info(
                    f"  reached {f_n:.2f} N after {moved * 1000:.1f} mm of travel "
                    f"({cmd_travel * 1000:.1f} mm commanded, {lag * 1000:.3f} mm lag)"
                )
                break
            if moved >= MAX_APPROACH_M:
                break
        _stop(env, manipulator, device)
        if not touched:
            logger.warning("  never reached the press force -- nothing to measure.")
            return
        _log_contacts(manipulator, "at full press")
        # Reference geometry: from the solver in sim, else from the URDF.
        r_z_m = EXPECTED_RZ_M
        expected_arm_m = EXPECTED_ARM_M
        lever = _contact_lever(manipulator)
        if lever is not None:
            r_z_m = abs(float(lever[2]))
            expected_arm_m = float(lever[0])
            logger.info(
                f"  contact centroid at r=[{lever[0] * 1000:+7.2f}, "
                f"{lever[1] * 1000:+7.2f}, {lever[2] * 1000:+7.2f}] mm in the sensor "
                f"frame -- lateral arm {abs(float(lever[0])) * 1000:.2f} mm, axial "
                f"lever {r_z_m * 1000:.1f} mm (URDF says {EXPECTED_RZ_M * 1000:.1f})"
            )

        log_every = max(1, round(1.0 / dt))
        back = tuple(-v for v in APPROACH_AXIS_TCP)

        # The value sim and real must agree on.
        stiffness = _estimate_stiffness(probe)
        logger.info("-" * 78)
        logger.info(
            f"  K_eff(lag)  = {stiffness / 1000:7.1f} kN/m   over {len(probe)} samples"
        )
        k_cmd = _estimate_stiffness_cmd(probe_cmd)
        if k_cmd is not None:
            logger.info(
                f"  K_eff(cmd)  = {k_cmd / 1000:7.1f} kN/m   force against commanded "
                f"travel, pose not used"
            )
            # K(cmd) is a lower bound; the ratio is the share of command that became
            # deflection.
            absorbed = k_cmd / max(stiffness, 1e-9)
            logger.info(
                f"  -> the tool kept moving through {100 * (1 - absorbed):.0f} % of "
                f"the commanded press; only {100 * absorbed:.0f} % became deflection"
            )
            if absorbed < 0.25:
                logger.warning(
                    "  most of the command went into motion, not load, so K_eff(lag) "
                    "rests almost entirely on the reported pose being right. Check it "
                    "against an independent measurement before trusting the number."
                )
        logger.info(
            "  force = K_eff * following error. This slope IS the contact force "
            "dynamic; to match the robot set"
        )
        logger.info("    servo_stiffness_scale = K_eff_real / K_eff_sim")
        logger.info("-" * 78)

        # Hold at zero velocity: force keeps rising while delayed commands finish.
        logger.info(f"Holding {HOLD_SECONDS:.0f} s at zero commanded velocity")
        f_hold_0 = f_n
        hold_trace: list[float] = []
        for i in range(max(1, round(HOLD_SECONDS / dt))):
            _stop(env, manipulator, device)
            step(scene)
            f_n, _, _, _ = reading()
            hold_trace.append(f_n)
            stream.send(
                manipulator,
                dt,
                "hold",
                cmd_travel,
                _travelled(manipulator, origin),
                0.0,
            )
            if i % log_every == 0:
                logger.info(f"    F={f_n:7.3f} N")
        logger.info(
            f"  hold: {f_hold_0:.2f} N -> {f_n:.2f} N  "
            f"(x{f_n / max(f_hold_0, 1e-9):.2f} after the command stopped)"
        )
        _log_contacts(manipulator, "at hold end")
        logger.info(
            f"  at hold end: wrench {fmt_wrench(manipulator.get_ft_wrench_compensated()[0])}"
        )

        # Time to peak force = command latency (1.04 s measured on the robot).
        if hold_trace:
            i_pk = max(range(len(hold_trace)), key=lambda k: hold_trace[k])
            f_pk = hold_trace[i_pk]
            logger.info(
                f"  command latency ~{i_pk * dt:.2f} s (force peaked at {f_pk:.2f} N, "
                f"{f_pk - f_hold_0:+.2f} N after the command stopped, then relaxed "
                f"{100 * (f_n / max(f_pk, 1e-9) - 1):+.1f} %)"
            )

        phase = "release"

        def retreat(speed: float) -> tuple[float, float, float, float]:
            nonlocal cmd_travel
            manipulator.ctrl_apply_vel_action(
                _tcp_velocity(env, manipulator, back, speed, device),
                open_gripper=None,
            )
            step(scene)
            cmd_travel -= speed * dt
            stream.send(
                manipulator,
                dt,
                phase,
                cmd_travel,
                _travelled(manipulator, origin),
                -speed,
            )
            return reading()

        # Release: back off until force is under MAX_FORCE_N.
        release_speed = _clamp_speed(env, approach_speed, "release")
        logger.info(
            f"Releasing the press overshoot at {release_speed * 1000:.1f} mm/s until "
            f"|Fz| <= {MAX_FORCE_N:.1f} N"
        )
        f_n, _, _, _ = reading()
        f_release_peak = f_n
        for i in range(max(1, round(RELEASE_SECONDS / dt))):
            f_n, _, _, _ = retreat(release_speed)
            f_release_peak = max(f_release_peak, f_n)
            if i % log_every == 0:
                logger.info(f"    F={f_n:7.3f} N")
            if f_n <= MAX_FORCE_N:
                break
        else:
            logger.warning(
                f"  still {f_n:.2f} N after {RELEASE_SECONDS:.0f} s of retreat -- the "
                f"arm is not backing off. Nothing below can be trusted."
            )
        logger.info(
            f"  released: peaked at {f_release_peak:.2f} N after the press stopped "
            f"(pressed to {MAX_FORCE_N:.1f} N), now {f_n:.2f} N"
        )

        # Unload sweep: speed adapts to the observed force drop.
        logger.info(f"Unloading and recording down to {MIN_FIT_FORCE_N:.1f} N")
        phase = "unload"
        unload_speed = min(
            max(
                UNLOAD_TARGET_N_PER_STEP / max(stiffness * dt, 1e-9),
                UNLOAD_MIN_SPEED_MPS,
            ),
            UNLOAD_MAX_SPEED_MPS,
        )
        unload_speed = _clamp_speed(env, unload_speed, "unload")
        f_peak = f_n
        descending = False
        last_kept = float("inf")
        rejected = 0
        window_start_f = f_n
        stalled_steps = 0
        stall_limit = max(1, round(UNLOAD_STALL_SECONDS / dt))
        for i in range(max(1, round(UNLOAD_SECONDS / dt))):
            f_n, f_z, f_x, t_y = retreat(unload_speed)
            f_peak = max(f_peak, f_n)
            # Keep only samples where force is falling.
            if not descending and f_n <= f_peak - DESCEND_MARGIN_N:
                descending = True
                last_kept = f_n + DESCEND_MARGIN_N
            if descending and MIN_FIT_FORCE_N <= f_n <= last_kept + 0.2:
                samples.append((f_n, f_z, f_x, t_y))
                last_kept = min(last_kept, f_n)
            elif descending:
                rejected += 1
            if i % log_every == 0:
                arm = 1000 * abs(t_y) / f_n if f_n > 1e-6 else float("nan")
                logger.info(
                    f"  F={f_n:7.3f} N  Ft={f_x:+7.3f} N  T={t_y:+8.4f} Nm  "
                    f"arm={arm:7.2f} mm  v={unload_speed * 1e6:5.0f} um/s  "
                    f"n={len(samples)}"
                )
            if descending and f_n < MIN_FIT_FORCE_N and samples:
                logger.info(f"  unloaded after {i + 1} steps")
                break
            if (i + 1) % UNLOAD_ADAPT_EVERY == 0:
                drop = (window_start_f - f_n) / UNLOAD_ADAPT_EVERY
                window_start_f = f_n
                if drop <= 0.0:
                    stalled_steps += UNLOAD_ADAPT_EVERY
                    unload_speed = min(unload_speed * 2.0, UNLOAD_MAX_SPEED_MPS)
                    logger.warning(
                        f"    force not falling ({f_n:.2f} N) -- retreating faster, "
                        f"{unload_speed * 1e6:.0f} um/s"
                    )
                    if stalled_steps >= stall_limit:
                        logger.error(
                            f"  force has refused to fall for {UNLOAD_STALL_SECONDS:.0f} s "
                            f"at up to {unload_speed * 1e6:.0f} um/s. The arm is not "
                            f"tracking the retreat; abandoning the sweep."
                        )
                        break
                else:
                    stalled_steps = 0
                    unload_speed = min(
                        max(
                            unload_speed * UNLOAD_TARGET_N_PER_STEP / drop,
                            UNLOAD_MIN_SPEED_MPS,
                        ),
                        UNLOAD_MAX_SPEED_MPS,
                    )
        if rejected:
            logger.info(
                f"  skipped {rejected} samples where the force was rising again"
            )
        _stop(env, manipulator, device)
        _log_torque_fit(samples, expected_arm_m, r_z_m)
    finally:
        _stop(env, manipulator, device)
        logger.info(f"Retracting {RETRACT_M * 100:.0f} cm")
        try:
            pose = manipulator.get_tcp_pose().clone()
            quat = sensor_quat(manipulator)
            back = transform_by_quat(
                (-_unit(APPROACH_AXIS_TCP)).unsqueeze(0), quat.unsqueeze(0)
            )
            manipulator.ctrl_go_to_goal(
                th.cat([pose[:, :3] + back * RETRACT_M, pose[:, 3:]], dim=-1),
                open_gripper=None,
            )
            for _ in range(max(1, round(MOVE_SETTLE_SECONDS / dt))):
                step(scene)
            logger.info(f"  clear: {fmt_wrench(manipulator.get_ft_wrench()[0])}")
        except Exception as exc:  # noqa: BLE001 - must not mask the real failure
            logger.error(f"  RETRACT FAILED ({exc}) -- clear the tool by hand.")


def _estimate_stiffness(probe: list[tuple[float, float]]) -> float:
    """K_eff [N/m]: slope of force against following error (lag)."""
    if len(probe) < 2:
        return 1e5  # no data; assume stiff (slower sweep)
    travel = th.tensor([d for d, _ in probe], dtype=th.float32)
    force = th.tensor([f for _, f in probe], dtype=th.float32)
    design = th.stack([travel, th.ones_like(travel)], dim=-1)
    slope = float(th.linalg.lstsq(design, force.unsqueeze(-1)).solution.reshape(2)[0])
    return max(slope, 1e3)


def _contact_lever(manipulator: BaseManipulator) -> th.Tensor | None:
    """Simulation only: force-weighted contact centroid relative to the sensor,
    in the sensor frame [m]. The true moment arm."""
    if not is_modelled(manipulator):
        return None
    entity = manipulator._robot_entity
    contacts = entity.get_contacts(exclude_self_contact=True, is_padded=True)
    valid = contacts["valid_mask"]
    if valid.numel() == 0:
        return None

    idx = manipulator._gravity_link_idx
    is_a = th.isin(contacts["link_a"], idx)
    is_b = th.isin(contacts["link_b"], idx)
    force = th.where(is_b.unsqueeze(-1), contacts["force_b"], contacts["force_a"])
    keep = (valid & (is_a | is_b)).unsqueeze(-1)
    force = th.where(keep, force, th.zeros_like(force))

    weight = th.linalg.norm(force, dim=-1)[0]  # [n_contacts]
    total = float(weight.sum())
    if total <= 1e-9:
        return None
    lever = contacts["position"][0] - manipulator._fts_link.get_pos()[0]
    centroid_world = (lever * weight.unsqueeze(-1)).sum(dim=0) / total

    quat = sensor_quat(manipulator)
    quat_conj = quat * th.tensor([1.0, -1.0, -1.0, -1.0], device=quat.device)
    return transform_by_quat(centroid_world.unsqueeze(0), quat_conj.unsqueeze(0))[0]


def _estimate_stiffness_cmd(
    probe_cmd: list[tuple[float, float]],
) -> float | None:
    """Lower bound of K_eff [N/m]: slope of force against commanded travel.
    Doesn't use the measured pose."""
    if len(probe_cmd) < 4:
        return None
    x = th.tensor([v for v, _ in probe_cmd], dtype=th.float32)
    f = th.tensor([v for _, v in probe_cmd], dtype=th.float32)
    if float(x.max() - x.min()) <= 0.0:
        return None
    design = th.stack([x, th.ones_like(x)], dim=-1)
    slope = float(th.linalg.lstsq(design, f.unsqueeze(-1)).solution.reshape(2)[0])
    return max(slope, 0.0)


def _log_torque_fit(
    samples: list[tuple[float, float, float, float]],
    expected_arm_m: float,
    r_z_m: float,
) -> None:
    """Fits the moment arm from torque against force (a slope ignores offsets).

    tau_y = r_z * Fx - r_x * Fz. First fits r_x alone, then subtracts the friction
    term with the known `r_z` (r_x and r_z can't both be fitted: Fx ~ mu * Fz).
    """
    logger = get_logger("ft_stiffness_probe")
    if len(samples) < 2:
        logger.warning("  too few samples to fit a moment arm.")
        return

    forces = th.tensor([s[0] for s in samples], dtype=th.float32)
    f_z = th.tensor([s[1] for s in samples], dtype=th.float32)
    f_x = th.tensor([s[2] for s in samples], dtype=th.float32)
    t_y = th.tensor([s[3] for s in samples], dtype=th.float32)
    # Signed values; abs() would hide the contact side.
    torques = t_y

    def _fit(mask: th.Tensor) -> tuple[th.Tensor, th.Tensor, th.Tensor]:
        design = th.stack([f_z[mask], th.ones_like(f_z[mask])], dim=-1)
        slope, intercept = th.linalg.lstsq(
            design, torques[mask].unsqueeze(-1)
        ).solution.reshape(2)
        return slope, intercept, torques - (slope * f_z + intercept)

    # Fit, drop outliers (unload transient), refit.
    keep = th.ones_like(forces, dtype=th.bool)
    slope, intercept, residual = _fit(keep)
    if len(samples) >= 6:
        for _ in range(3):
            scale = float(residual[keep].abs().median())
            if scale <= 0.0:
                break
            candidate = residual.abs() <= 2.5 * scale
            if int(candidate.sum()) < 3 or bool((candidate == keep).all()):
                break
            keep = candidate
            slope, intercept, residual = _fit(keep)
        dropped = len(samples) - int(keep.sum())
        if dropped:
            logger.info(
                f"  dropped {dropped} of {len(samples)} samples as off-line "
                "(the unload transient)"
            )
    residual = residual[keep]

    logger.info("=" * 78)
    logger.info(
        f"Torque channel over {len(samples)} samples: torque = arm * force + offset"
    )
    arm = -float(slope)  # tau_y = -r_x * Fz
    logger.info(f"  measured arm  = {arm * 1000:7.2f} mm")
    logger.info(f"  expected arm  = {expected_arm_m * 1000:7.2f} mm")
    logger.info(f"  ratio         = {arm / expected_arm_m:7.3f}")
    logger.info(
        f"  offset        = {float(intercept):7.4f} Nm (a tare residual, not an arm)"
    )
    worst = float(residual.abs().max())
    span = float(torques[keep].max() - torques[keep].min())
    logger.info(f"  fit residual  = {worst:7.4f} Nm max")

    # Friction correction with the known `r_z`.
    tan_span = float(f_x[keep].max() - f_x[keep].min())
    nrm_span = float(f_z[keep].max() - f_z[keep].min())
    corrected_ok = False
    logger.info("-" * 78)
    if int(keep.sum()) < 4 or tan_span < 0.05 * max(nrm_span, 1e-9):
        logger.info(
            f"  tangential force spans only {tan_span:.3f} N against {nrm_span:.3f} N "
            f"normal -- contact is effectively frictionless, so the arm above is the "
            f"whole story."
        )
    else:
        corrected = t_y[keep] - r_z_m * f_x[keep]
        design2 = th.stack([f_z[keep], th.ones_like(f_z[keep])], dim=-1)
        slope2, off2 = th.linalg.lstsq(
            design2, corrected.unsqueeze(-1)
        ).solution.reshape(2)
        resid2 = corrected - (slope2 * f_z[keep] + off2)
        mu = tan_span / max(nrm_span, 1e-9)
        logger.info(
            f"  tangential force present (mu ~ {mu:.2f}), so part of that torque is "
            f"friction at the {r_z_m * 1000:.1f} mm axial lever, not finger offset:"
        )
        logger.info(
            f"    lateral arm r_x = {-float(slope2) * 1000:8.2f} mm (corrected)"
        )
        logger.info(f"    offset          = {float(off2):8.4f} Nm")
        logger.info(f"    residual        = {float(resid2.abs().max()):8.4f} Nm max")
        corrected_ok = float(resid2.abs().max()) <= 0.1 * max(
            float(corrected.max() - corrected.min()), 1e-9
        )
        share_fric = abs(r_z_m * tan_span)
        share_arm = abs(float(slope2) * nrm_span)
        total = share_fric + share_arm
        if total > 0:
            frac = share_fric / total
            logger.info(
                f"    of the torque swing, {100 * (1 - frac):.0f} % is the "
                f"finger offset and {100 * frac:.0f} % is friction at "
                f"the axial lever"
            )
            if frac > 0.5:
                # Friction dominates, so errors in r_z are amplified in r_x.
                logger.warning(
                    f"    friction carries {100 * frac:.0f} % of the torque, so the "
                    f"corrected arm inherits the error in r_z amplified about "
                    f"{frac / max(1 - frac, 1e-9):.1f}x. To measure the finger offset "
                    f"properly, sweep at two tilt angles or on a lower-friction surface."
                )
    logger.info("=" * 78)
    if span > 0 and worst > 0.1 * span and not corrected_ok:
        logger.warning(
            f"  the residual is {100 * worst / span:.0f} % of the torque span -- these "
            "points are not on a line, so the slope is not a moment arm. Look for a "
            "transient left in the sweep, or contact moving between fingers."
        )
    elif span > 0 and worst > 0.1 * span:
        # Expected: the one-term fit ignores friction.
        logger.info(
            f"  (the one-term residual is {100 * worst / span:.0f} % of the span, as "
            f"expected with friction present -- read the corrected arm above)"
        )


def main() -> None:
    th.set_default_dtype(th.float32)
    args = parse_arguments()
    task = init_clearml_task(
        project_name=args.project_name,
        algorithm=args.algorithm,
        control=args.control_type,
        exp_name=args.experiment_name or "ft_stiffness_probe",
    )
    device = th.device("cuda" if th.cuda.is_available() else "cpu")
    ConfigManager.setup_config(argv=args, device=device, task=task)
    cfg = ConfigManager.get_config()

    env = create_env(cfg)
    env = getattr(env, "unwrapped", env)

    match TEST:
        case "torque_probe":
            ft_sensor_torque_probe(env=env, cfg=cfg)
        case "contact_drag":
            ft_sensor_contact_drag_test(env=env, cfg=cfg)
        case _:
            raise ValueError(f"Unknown test: {TEST}")


if __name__ == "__main__":
    setup_logger("INFO")
    try:
        main()
    except KeyboardInterrupt:
        get_logger("ft_stiffness_probe").info("Stopped by user")
