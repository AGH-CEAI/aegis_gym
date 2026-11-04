"""
Check of `ctrl_apply_delta_pose_action` (A1 admittance control mode), on the real robot or in
Genesis (`--control sim -B 1`), so the two logs can be overlaid.

Phases, all at the policy rate (`policy_dt`, must match the server's
`admittance_step_duration_s`, 0.04 s):
    1. home (MoveIt), tare the F/T sensor;
    2. square in the horizontal plane through `ctrl_apply_delta_pose_action`;
    3. yaw about the world Z there and back;
    4. hold (zero deltas) -- push the gripper by hand, it must yield and come back;
    5. home again (MoveIt), which leaves admittance mode.
Logs every step to <local_log_dir>/delta_pose_test.csv and prints a summary: loop period
(a step longer than `policy_dt` stalls the JTC between segments), tracking of the commanded
reference by the measured TCP, and clipped steps.

Run from the repo root, e.g.:
    python -m scripts.admittance.delta_pose_test --control ros --env reacher --disable-cameras -e a1_delta
    python -m scripts.admittance.delta_pose_test --control sim --env reacher -B 1 -e a1_delta_sim
(the real robot needs bringup and the gRPC server up; the hold phase has nobody to push in sim)
"""

import csv
import math
import sys
import time
from pathlib import Path

import numpy as np
import torch as th

from aegis_gym.aux.logging import get_logger, setup_logger
from aegis_gym.config import ConfigManager, parse_arguments
from aegis_gym.config.types import Control
from scripts.ft_sensor.common import rotation_since
from train import create_env, init_clearml_task

logger = get_logger("a1_delta_pose")

SETTLE_S = 1.0
LOG_EVERY_S = 1.0


def extra_args(p) -> None:
    p.add_argument("--side", type=float, default=0.03, help="[m] square side")
    p.add_argument("--speed", type=float, default=0.025, help="[m/s] square speed")
    p.add_argument("--yaw", type=float, default=10.0, help="[deg] yaw amplitude")
    p.add_argument("--yaw-speed", type=float, default=5.0, help="[deg/s]")
    p.add_argument("--hold", type=float, default=15.0, help="[s] hand-push window")


def main() -> None:
    th.set_default_dtype(th.float32)
    args = parse_arguments(extra_argparser=extra_args)
    opts = args.get_raw_args()
    if args.control_type == Control.SIM and args.num_envs != 1:
        logger.error("Run the simulation with a single environment (-B 1).")
        sys.exit(1)
    task = init_clearml_task(
        project_name=args.project_name,
        algorithm=args.algorithm,
        control=args.control_type,
        exp_name=args.experiment_name,
    )
    device = th.device(
        "cuda" if args.control_type == Control.SIM and th.cuda.is_available() else "cpu"
    )
    ConfigManager.setup_config(argv=args, device=device, task=task)
    cfg = ConfigManager.get_config()

    env = create_env(cfg)
    env = getattr(env, "unwrapped", env)
    env.reset()
    manip = env.manipulator
    scene = env._scene
    dt = env.get_policy_dt()

    log_path = Path(cfg.logger_cfg.local_log_dir) / "delta_pose_test.csv"
    rows: list[list] = [
        [
            "t",
            "phase",
            "clipped",
            "period_ms",
            "ref_x",
            "ref_y",
            "ref_z",
            "ref_yaw_deg",
            "tcp_x",
            "tcp_y",
            "tcp_z",
            "yaw_deg",
            "f_sensor_x",
            "f_sensor_y",
            "f_sensor_z",
        ]
    ]

    def write_log() -> None:
        with open(log_path, "w", newline="") as log_file:
            csv.writer(log_file).writerows(rows)

    t0 = time.perf_counter()
    last = [t0]
    periods: list[float] = []
    clipped_steps: dict[str, int] = {}

    def settle() -> None:
        for _ in range(round(SETTLE_S / dt)):
            scene.pre_step()
            scene.step()

    manip.ctrl_go_to_home()
    settle()
    manip.set_ft_bias()
    settle()
    start = manip.get_tcp_pose()[0].clone()
    reference = start[:3].clone()
    ref_yaw = 0.0
    logger.info(f"start TCP {np.round(start[:3].tolist(), 4)}; log: {log_path}")

    def act(phase: str, translation=(0.0, 0.0, 0.0), yaw: float = 0.0) -> None:
        nonlocal reference, ref_yaw
        delta = th.zeros(1, 6, device=device)
        delta[0, :3] = th.tensor(translation, device=device)
        delta[0, 5] = yaw
        scene.pre_step()
        now = time.perf_counter()
        periods.append(now - last[0])
        last[0] = now
        clipped = bool(manip.ctrl_apply_delta_pose_action(delta)[0])
        scene.step()
        reference = manip.get_tcp_reference()[0, :3]
        ref_yaw += math.degrees(yaw)
        clipped_steps[phase] = clipped_steps.get(phase, 0) + int(clipped)

        pose = manip.get_tcp_pose()[0]
        force = manip.get_ft_wrench()[0, :3]
        rows.append(
            [
                f"{now - t0:.4f}",
                phase,
                int(clipped),
                f"{periods[-1] * 1e3:.2f}",
                *(f"{v:.5f}" for v in reference.tolist()),
                f"{ref_yaw:.3f}",
                *(f"{v:.5f}" for v in pose[:3].tolist()),
                f"{measured_yaw_deg():.3f}",
                *(f"{v:.3f}" for v in force.tolist()),
            ]
        )

    def measured_yaw_deg() -> float:
        angle, axis = rotation_since(start[3:], manip.get_tcp_orientation()[0])
        return angle if axis[2] >= 0 else -angle

    def tracking_error_mm() -> float:
        return float((manip.get_tcp_position()[0] - reference).norm()) * 1e3

    try:
        # --- square ----------------------------------------------------------------------
        step_len = opts.speed * dt
        n_side = max(1, round(opts.side / step_len))
        corner_errors = []
        for direction in ((1, 0, 0), (0, 1, 0), (-1, 0, 0), (0, -1, 0)):
            for _ in range(n_side):
                act("square", tuple(step_len * c for c in direction))
            for _ in range(round(0.5 / dt)):
                act("square")
            corner_errors.append(tracking_error_mm())
            logger.info(
                f"[square] corner, |TCP - reference| = {corner_errors[-1]:.2f} mm"
            )

        # --- yaw -------------------------------------------------------------------------
        yaw_step = math.radians(opts.yaw_speed) * dt
        n_yaw = max(1, round(opts.yaw / opts.yaw_speed / dt))
        for sign in (1, -1, -1, 1):
            for _ in range(n_yaw):
                act("yaw", yaw=sign * yaw_step)
            for _ in range(round(0.5 / dt)):
                act("yaw")
            logger.info(
                f"[yaw] reference {ref_yaw:+.1f} deg, measured {measured_yaw_deg():+.1f} deg, "
                f"|TCP - reference| {tracking_error_mm():.2f} mm"
            )

        # --- hold, push by hand ----------------------------------------------------------
        logger.info(f"[hold] {opts.hold:.0f} s: push the gripper by hand now")
        max_dev, max_force = 0.0, 0.0
        for k in range(round(opts.hold / dt)):
            act("hold")
            max_dev = max(max_dev, tracking_error_mm())
            max_force = max(max_force, float(manip.get_ft_wrench()[0, :3].norm()))
            if k % round(LOG_EVERY_S / dt) == 0:
                logger.info(
                    f"[hold] |TCP - reference| {tracking_error_mm():5.1f} mm, "
                    f"|F| {float(manip.get_ft_wrench()[0, :3].norm()):5.1f} N"
                )
        for _ in range(round(SETTLE_S / dt)):
            act("hold")
        hold_residual = tracking_error_mm()

    finally:
        write_log()
    manip.ctrl_go_to_home()

    p = np.array(periods[1:]) * 1e3
    logger.info("=== Summary ===")
    logger.info(
        f"loop period [ms]: median {np.median(p):.1f}, p95 {np.percentile(p, 95):.1f}, "
        f"max {p.max():.1f}; {(p > 1.25e3 * dt).sum()} of {len(p)} steps > 1.25 * policy_dt"
    )
    logger.info(
        f"square corners |TCP - reference| [mm]: {np.round(corner_errors, 2).tolist()}"
    )
    logger.info(
        f"hold: max deviation {max_dev:.1f} mm, max |F| {max_force:.1f} N, "
        f"residual after release {hold_residual:.2f} mm"
    )
    logger.info(f"clipped steps per phase: {clipped_steps}")
    logger.info(f"log: {log_path}")


if __name__ == "__main__":
    setup_logger("INFO")
    main()
