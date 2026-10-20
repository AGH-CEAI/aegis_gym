"""
F/T sensor payload calibration, for both the simulation and the real robot.

0. Frame test: tares at home, moves to a configuration ~90 deg away (measured on the
   real robot) and reports the readings; they must match between the simulation and
   the real robot, a rotation between them is an error in the sensor frame.
1. Identification: visits a spread of wrist orientations, records the UNTARED wrench
   at each and fits the mass and the centre of mass of everything hanging past the
   sensor (plus a constant sensor offset).
2. Verification: tares at home, revisits the same poses and checks that the reading
   with the configured payload's weight modelled out (`get_ft_wrench_compensated()`)
   stays ~0 everywhere.

Run it on both backends; the difference between the two fits is the sim-to-real gap
in the two numbers that cause it. Paste the printed values into `robot.yaml`
(`ft_payload_mass`, `ft_payload_com`).

    python -m scripts.ft_sensor.calibration --env reacher -B 1 --vis      # simulation
    python -m scripts.ft_sensor.calibration --env reacher --control ros   # real robot

On the real robot the joint targets below must be reachable and collision-free from
home -- check them in RViz before running.
"""

import torch as th

from aegis_gym.aux.geom import transform_by_quat
from aegis_gym.aux.logging import get_logger, setup_logger
from aegis_gym.config import ConfigManager, parse_arguments
from aegis_gym.envs.manipulator import BaseManipulator
from scripts.ft_sensor.common import (
    fmt_wrench,
    is_modelled,
    log_frame,
    quat_to_matrix,
    rotation_since,
    sensor_quat,
    step,
)
from train import create_env, init_clearml_task

SETTLE_SECONDS = 4.0

# --- Frame test --------------------------------------------------------------------
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

# --- Payload identification --------------------------------------------------------
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

logger = get_logger("ft_calibration")


def _log_gravity_links(manipulator: BaseManipulator) -> None:
    """Logs every link counted as "past the F/T sensor" for gravity compensation,
    together with its attachment point: the link's center of mass, both in the
    link's own frame (local COM) and in world coordinates."""
    if not is_modelled(manipulator):
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
    # Both tensors are [num_envs, n_links(, 3)], so they are indexed per link on
    # dim 1; env 0 is the one reported.
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


def _log_wrench(manipulator: BaseManipulator, step_no: int) -> None:
    """Logs the measurement, plus -- in simulation only -- the untared reading and the
    gravity term behind it. The real robot reports a single value, so that is all there
    is to log there, and it is the column the simulated one must match."""
    line = f"[step {step_no}] wrench={fmt_wrench(manipulator.get_ft_wrench()[0])}"
    if manipulator.is_ft_biased():
        line += " (tared)"
    if is_modelled(manipulator):
        line += (
            f"  sim_raw={fmt_wrench(manipulator.get_ft_wrench_raw()[0])}"
            f"  sim_gravity(world)={fmt_wrench(manipulator._gravity_wrench_world()[0])}"
        )
    logger.info(line)


def _sim_bias(manipulator: BaseManipulator) -> th.Tensor | None:
    """The numeric tare offset when the backend can report one. The real sensor keeps
    its offset inside the hardware, so there it is simply unknown."""
    getter = getattr(manipulator, "get_ft_bias", None)
    return getter() if getter is not None else None


def _read_both(manipulator: BaseManipulator) -> tuple[th.Tensor | None, th.Tensor]:
    """The current (untared, measured) wrench for env 0. The untared half is None on
    the real robot, which reports one value and no more."""
    raw = (
        manipulator.get_ft_wrench_raw()[0].clone() if is_modelled(manipulator) else None
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
    bias = _sim_bias(manipulator)

    logger.info("=" * 78)
    logger.info(
        "F/T bias report            [   fx       fy       fz       tx       ty       tz ]"
    )
    if bias is not None:
        logger.info(f"  bias (untared @ home)    {fmt_wrench(bias[0])}")
    if home[0] is not None:
        logger.info(f"  home    untared (sim)    {fmt_wrench(home[0])}")
    logger.info(
        f"  home    measured         {fmt_wrench(home[1])}   <- ~0 by construction"
    )
    if rotated[0] is not None:
        logger.info(f"  rotated untared (sim)    {fmt_wrench(rotated[0])}")
    logger.info(
        f"  rotated measured         {fmt_wrench(rotated[1])}   <- compare sim vs real"
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


def _frame_test(manipulator: BaseManipulator, scene, device: th.device, dt: float):
    """Checks the sensor frame and the gravity term: tares at home, moves to a
    configuration ~90 deg away and reports both readings. Run on both backends -- the
    rotated readings must match between the simulation and the real robot; a rotation
    between them is an error in the sensor frame (`_FTS_OUTPUT_YAW_DEG`)."""
    manipulator.ctrl_go_to_home()
    for _ in range(5):
        step(scene)

    _log_gravity_links(manipulator)

    hold_steps = max(1, round(HOLD_SECONDS / dt))
    log_every = max(1, round(1.0 / dt))  # once per simulated second

    quat_home = sensor_quat(manipulator).clone()
    log_frame(manipulator, "home")

    # Tared in the home pose with no payload, exactly as the real sensor is zeroed
    # before a measurement, so the bias absorbs the tool's own weight.
    manipulator.set_ft_bias()
    bias = _sim_bias(manipulator)
    logger.info(
        f"  tared at home, bias={fmt_wrench(bias[0])}"
        if bias is not None
        else "  tared at home (offset held by the sensor)"
    )

    logger.info(
        f"Holding home configuration for {HOLD_SECONDS:.0f}s ({hold_steps} steps):"
    )
    for i in range(hold_steps):
        step(scene)
        if i % log_every == 0:
            _log_wrench(manipulator, i)
    home = _read_both(manipulator)

    logger.info(f"Moving to the rotated configuration (joint goal): {ROTATED_ARM_DOF}")
    manipulator.ctrl_go_to_joints(
        th.tensor(ROTATED_ARM_DOF, dtype=th.float32, device=device)
    )
    for _ in range(max(1, round(MOVE_SETTLE_SECONDS / dt))):
        step(scene)

    # Never assume the arm arrived: a joint goal cannot be refused for want of a
    # solution, but it can still be clipped by limits or simply not reached yet, and
    # either would look exactly like a frame bug in the wrench below.
    angle, axis = rotation_since(quat_home, sensor_quat(manipulator))
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
    log_frame(manipulator, "rotated")

    logger.info(
        f"Holding rotated configuration for {HOLD_SECONDS:.0f}s ({hold_steps} steps):"
    )
    for i in range(hold_steps):
        step(scene)
        if i % log_every == 0:
            _log_wrench(manipulator, i)
    rotated = _read_both(manipulator)

    _log_bias_report(manipulator, home=home, rotated=rotated, angle=angle)


def main() -> None:
    th.set_default_dtype(th.float32)
    args = parse_arguments()
    task = init_clearml_task(
        project_name=args.project_name,
        algorithm=args.algorithm,
        control=args.control_type,
        exp_name=args.experiment_name or "ft_calibration",
    )
    device = th.device("cuda" if th.cuda.is_available() else "cpu")
    ConfigManager.setup_config(argv=args, device=device, task=task)
    cfg = ConfigManager.get_config()

    env = create_env(cfg)
    env = getattr(env, "unwrapped", env)
    manipulator = env.manipulator
    scene = env._scene
    dt = env.get_policy_dt()
    settle_steps = max(1, round(SETTLE_SECONDS / dt))

    # Only the arm: `GenesisManipulator` appends the gripper DOFs to this config list
    # in place, so after `create_env()` it may hold 8 values.
    home = th.tensor(cfg.robot_cfg.default_arm_dof[:6], dtype=th.float32, device=device)
    targets = [
        home + th.deg2rad(th.tensor(offsets, dtype=th.float32, device=device))
        for offsets in POSE_OFFSETS_DEG
    ]

    def go_to(target: th.Tensor) -> None:
        manipulator.ctrl_go_to_joints(target)
        for _ in range(settle_steps):
            step(scene)

    # --- 0. Frame test (tared at home, ~90 deg rotation) -----------------------------
    logger.info("0. Sensor frame and gravity test")
    _frame_test(manipulator, scene, device, dt)

    # --- 1. Identification (untared) -------------------------------------------------
    logger.info("1. Payload identification")
    go_to(home)
    # Identification needs the untared signal; a tare would hide the very offset the
    # fit estimates and drag the mass with it.
    manipulator.clear_ft_bias()

    rots, forces, torques = [], [], []
    for i, (offsets, target) in enumerate(zip(POSE_OFFSETS_DEG, targets)):
        go_to(target)
        wrench = manipulator.get_ft_wrench()[0]
        rots.append(quat_to_matrix(sensor_quat(manipulator)))
        forces.append(wrench[:3].clone())
        torques.append(wrench[3:].clone())
        logger.info(f"  pose {i}: offsets={offsets} wrench={fmt_wrench(wrench)}")

    gravity = th.tensor(BaseManipulator.GRAVITY_WORLD, dtype=th.float32, device=device)
    mass, com, force_offset, torque_offset = _identify_payload(
        rots, forces, torques, gravity
    )
    # What the fitted model fails to explain, in Newtons. Large values mean the load
    # is not rigid, the arm had not settled, or a pose was not reached.
    residuals = th.stack(
        [
            th.linalg.norm(f - (mass * (r.T @ gravity) + force_offset))
            for r, f in zip(rots, forces)
        ]
    )

    # --- 2. Verification with the configured payload (tared at home) ----------------
    logger.info("2. Verification of the configured payload (tared at home)")
    go_to(home)
    manipulator.set_ft_bias()
    worst_f, worst_t = 0.0, 0.0
    for i, target in enumerate(targets):
        go_to(target)
        measured = manipulator.get_ft_wrench()[0]
        compensated = manipulator.get_ft_wrench_compensated()[0]
        worst_f = max(worst_f, compensated[:3].norm().item())
        worst_t = max(worst_t, compensated[3:].norm().item())
        logger.info(
            f"  pose {i}: measured={fmt_wrench(measured)}  "
            f"compensated={fmt_wrench(compensated)}"
        )
    go_to(home)

    # --- Report ----------------------------------------------------------------------
    cfg_mass, cfg_com = cfg.robot_cfg.ft_payload_mass, cfg.robot_cfg.ft_payload_com
    logger.info("=" * 78)
    logger.info(f"Payload identification over {len(rots)} poses")
    logger.info(f"  mass            = {float(mass):8.4f} kg")
    logger.info(
        f"  centre of mass  = [{' '.join(f'{v:8.4f}' for v in com.tolist())}] m "
        "(sensor frame)"
    )
    logger.info(
        f"  sensor offset   = {fmt_wrench(th.cat([force_offset, torque_offset]))}"
    )
    logger.info(
        f"  fit residual    = {float(residuals.max()):.4f} N max, "
        f"{float(residuals.mean()):.4f} N mean"
    )
    if cfg_mass is not None:
        logger.info(
            f"  configured      = {cfg_mass:8.4f} kg at {cfg_com} "
            f"-> gap {1000 * (float(mass) - cfg_mass):+.1f} g"
        )
    if is_modelled(manipulator):
        modelled = float(manipulator._gravity_link_masses[0].sum())
        logger.info(
            f"  URDF model says   {modelled:8.4f} kg "
            f"-> gap {1000 * (float(mass) - modelled):+.1f} g"
        )
    logger.info(
        f"  verification    = |compensated| max {worst_f:.3f} N, {worst_t:.4f} Nm "
        "(~0 when the configured payload is right)"
    )
    logger.info("  robot.yaml:")
    logger.info(f"    ft_payload_mass: {float(mass):.4f}")
    logger.info(f"    ft_payload_com: [{', '.join(f'{v:.4f}' for v in com.tolist())}]")
    logger.info("=" * 78)


if __name__ == "__main__":
    setup_logger("INFO")
    try:
        main()
    except KeyboardInterrupt:
        logger.info("Stopped by user")
