"""
Simulated F/T sanity check: moves the TCP over the table, descends, touches the table,
presses for a moment and lifts, then sweeps the wrist. Logs the F/T reading on the way.
Simulation only (the wrist sweep uses joint diff actions).

Run from the repo root, e.g.:
    python -m scripts.ft_sensor.touch_test --env reacher -B 1 --vis --plotjuggler -e ft_touch

Expected:
    - at rest: |F| ~ weight of the links past the sensor (~12.5 N from the URDF),
      |T| small (<~1 Nm);
    - on contact: a clear jump of dF, mostly along the sensor axis pointing down;
    - arm tilted, then wrist_3 swept over 360 deg: |F| and Fz constant, (Fx, Fy)
      rotating with the wrist, Tz ~0 and the COM lever d = |T| / |F_xy| constant
      (a few cm; the URDF puts the COM ~6.6 cm past the sensor).
"""

import math

import torch as th

from aegis_gym.aux.logging import get_logger, setup_logger
from aegis_gym.config import ConfigManager, parse_arguments
from train import create_env, init_clearml_task

# Target over the table (robot base frame, metres). The table spans x ~0.32..0.87 m and
# its top is at z ~0.0 m (table_size[2] - workbench_size[2]).
HOVER_POS = (0.55, 0.0, 0.10)
TABLE_TOP_Z = 0.0
APPROACH_SPEED = 0.05  # m/s, free-space moves
PRESS_SPEED = 0.01  # m/s, the final descent
CONTACT_THRESHOLD_N = 2.0  # |dF| that counts as a touch
PRESS_HOLD_S = 2.0
LOG_EVERY = 5  # steps
# Arm more upright: shoulder_lift from -120 deg towards -90 deg tilts the tool by ~30 deg.
TILT_JOINTS_DIFF = {1: math.radians(30.0)}  # {joint index: rad}
WRIST_3 = 5  # joint index of the 6th axis
SWEEP_STEP_DEG = 30.0
JOINT_SPEED = 0.5  # rad/s

logger = get_logger("ft_touch")


def main() -> None:
    th.set_default_dtype(th.float32)
    args = parse_arguments()
    task = init_clearml_task(
        project_name=args.project_name,
        algorithm=args.algorithm,
        control=args.control_type,
        exp_name=args.experiment_name,
    )
    device = th.device("cuda" if th.cuda.is_available() else "cpu")
    ConfigManager.setup_config(argv=args, device=device, task=task)
    cfg = ConfigManager.get_config()

    env = create_env(cfg)
    env = getattr(env, "unwrapped", env)
    env.reset()
    manip = env.manipulator
    scene = env._scene
    dt = env.get_policy_dt()
    step_no = 0

    def step(lin_vel: th.Tensor | None = None) -> None:
        nonlocal step_no
        action = th.zeros(env.num_envs, 6, device=device)
        if lin_vel is not None:
            action[:, :3] = lin_vel / manip.max_linear_speed
        scene.pre_step()
        manip.ctrl_apply_vel_action(action, open_gripper=None)
        scene.step()
        step_no += 1

    def wrench() -> th.Tensor:
        return manip.get_ft_wrench()[0]

    def fmt(w: th.Tensor) -> str:
        f, t = w[:3].tolist(), w[3:].tolist()
        return (
            f"F=[{f[0]:7.2f} {f[1]:7.2f} {f[2]:7.2f}] |F|={w[:3].norm():6.2f} N  "
            f"T=[{t[0]:6.3f} {t[1]:6.3f} {t[2]:6.3f}] |T|={w[3:].norm():5.3f} Nm"
        )

    # --- Phase 0: rest at home -------------------------------------------------------
    manip.ctrl_go_to_home()
    for _ in range(round(1.0 / dt)):
        step()
    rest = wrench().clone()
    logger.info(f"[rest]     {fmt(rest)}")

    # --- Phase 1: move over the table ------------------------------------------------
    target = th.tensor(HOVER_POS, device=device)
    for _ in range(round(15.0 / dt)):
        err = target - manip.get_tcp_position()[0]
        if err.norm() < 0.005:
            break
        vel = err * 2.0
        if vel.norm() > APPROACH_SPEED:
            vel = vel / vel.norm() * APPROACH_SPEED
        step(vel)
    for _ in range(round(1.0 / dt)):
        step()
    hover = wrench().clone()
    logger.info(f"[hover]    tcp={manip.get_tcp_position()[0].tolist()}  {fmt(hover)}")

    # --- Phase 2: descend until contact ----------------------------------------------
    down_fast = th.tensor([0.0, 0.0, -APPROACH_SPEED], device=device)
    down_slow = th.tensor([0.0, 0.0, -PRESS_SPEED], device=device)
    contact_z = None
    for _ in range(round(20.0 / dt)):
        z = manip.get_tcp_position()[0, 2].item()
        step(down_fast if z > TABLE_TOP_Z + 0.03 else down_slow)
        d = wrench() - hover
        if step_no % LOG_EVERY == 0:
            logger.info(f"[descend]  z={z:.4f}  dW: {fmt(d)}")
        if d[:3].norm() > CONTACT_THRESHOLD_N:
            contact_z = z
            logger.info(f"[contact]  z={z:.4f}  dW: {fmt(d)}")
            break
    if contact_z is None:
        logger.warning("No contact detected -- check HOVER_POS / TABLE_TOP_Z.")

    # --- Phase 3: keep pressing ------------------------------------------------------
    peak = th.zeros(6, device=device)
    for _ in range(round(PRESS_HOLD_S / dt)):
        step(down_slow)
        d = wrench() - hover
        if d[:3].norm() > peak[:3].norm():
            peak = d.clone()
        if step_no % LOG_EVERY == 0:
            z = manip.get_tcp_position()[0, 2].item()
            logger.info(f"[press]    z={z:.4f}  dW: {fmt(d)}")

    # --- Phase 4: lift ---------------------------------------------------------------
    up = th.tensor([0.0, 0.0, APPROACH_SPEED], device=device)
    for _ in range(round(1.0 / dt)):
        step(up)
    for _ in range(round(1.0 / dt)):
        step()
    after = wrench().clone()

    logger.info("=== Summary ===")
    logger.info(f"rest           {fmt(rest)}")
    logger.info(f"peak dW press  {fmt(peak)}")
    logger.info(f"after lift dW  {fmt(after - hover)}  (should be ~0)")
    logger.info(f"contact at z = {contact_z}")

    # --- Phase 5: arm more upright, then sweep the 6th axis --------------------------
    def move_joints(target: th.Tensor) -> None:
        """Drives the joints to `target` [n_dofs] at JOINT_SPEED, then settles."""
        max_step = JOINT_SPEED * dt
        for _ in range(round(20.0 / dt)):
            err = target - manip.get_joints_positions()[0]
            if err.abs().max() < 1e-3:
                break
            diff = err.clamp(-max_step, max_step).expand(env.num_envs, -1)
            scene.pre_step()
            manip.ctrl_apply_joints_diff_action(diff)
            scene.step()
        for _ in range(round(0.5 / dt)):
            scene.pre_step()
            manip.ctrl_apply_joints_diff_action(
                (target - manip.get_joints_positions()[0]).expand(env.num_envs, -1)
            )
            scene.step()

    manip.ctrl_go_to_home()
    for _ in range(round(1.0 / dt)):
        step()
    q = manip.get_joints_positions()[0].clone()
    for joint, delta in TILT_JOINTS_DIFF.items():
        q[joint] += delta
    move_joints(q)
    logger.info(f"[tilted]   {fmt(wrench())}")

    logger.info("=== wrist_3 sweep ===")
    logger.info("   deg |      Fx      Fy      Fz    |F| |      Tz   d=|T|/|Fxy|")
    levers = []
    q0 = q.clone()
    n_steps = round(360.0 / SWEEP_STEP_DEG)
    for k in range(n_steps + 1):
        angle = k * SWEEP_STEP_DEG
        q = q0.clone()
        q[WRIST_3] += math.radians(angle)
        move_joints(q)
        w = wrench()
        f_xy = w[:2].norm().item()
        lever = w[3:].norm().item() / f_xy if f_xy > 0.5 else float("nan")
        levers.append(lever)
        logger.info(
            f"  {angle:5.0f} | {w[0]:7.2f} {w[1]:7.2f} {w[2]:7.2f} {w[:3].norm():6.2f} |"
            f" {w[5]:7.3f}   {lever:.4f} m"
        )
    valid = [x for x in levers if not math.isnan(x)]
    if valid:
        logger.info(
            f"COM lever d: min {min(valid):.4f} m, max {max(valid):.4f} m "
            "(should be equal and a few cm)"
        )


if __name__ == "__main__":
    setup_logger("INFO")
    try:
        main()
    except KeyboardInterrupt:
        logger.info("Stopped by user")
