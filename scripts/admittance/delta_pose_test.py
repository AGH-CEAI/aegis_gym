"""
Check of `ctrl_apply_delta_pose_action` (A1 admittance control mode), on the real robot or in
Genesis (`--control sim -B 1`), so the two logs can be overlaid. All steps run at the policy rate
(`policy_dt`, must match the server's `admittance_step_duration_s`, 0.04 s).

`--test motion` (default), in free space:
    1. home (MoveIt), tare the F/T sensor;
    2. square in the horizontal plane;
    3. yaw about the world Z there and back;
    4. hold (zero deltas) -- push the gripper by hand, it must yield and come back.
`--test press`, onto a compliant surface below the closed gripper (the sponge on the robot, a
`SpringPlate` in simulation), like the robot driver's `press`:
    1. home, close the gripper, descend --press-drop straight down (MoveIt), tare;
    2. step down until |F| exceeds --press-max-force (or --press-max-distance);
    3. hold for --press-hold s;
    4. retreat to the start reference at the approach speed.
Both end at home (MoveIt), which leaves admittance mode. Every step is logged to
<local_log_dir>/delta_pose_test.csv (plate deflection in simulation only) and a summary printed.

Run from the repo root, e.g.:
    python -m scripts.admittance.delta_pose_test --control ros --env reacher --disable-cameras -e a1_delta
    python -m scripts.admittance.delta_pose_test --control sim --env reacher -B 1 -e a1_delta_sim
    python -m scripts.admittance.delta_pose_test --control ros --env reacher --disable-cameras -e a1_press --test press
    python -m scripts.admittance.delta_pose_test --control sim --env reacher -B 1 -e a1_press_sim --test press
(the real robot needs bringup and the gRPC server up, for contact with admittance_check_collisions:=false)
"""

import csv
import math
import sys
import time
from pathlib import Path

import genesis as gs
import numpy as np
import torch as th

from aegis_gym.aux.logging import get_logger, setup_logger
from aegis_gym.config import ConfigManager, parse_arguments
from aegis_gym.config.types import Control, ExpConfig
from aegis_gym.envs import BaseEnv, get_env_class
from aegis_gym.envs.objects.sim.spring_plate import SpringPlate, SpringPlateParams
from aegis_gym.envs.scene import GenesisScene
from scripts.ft_sensor.common import rotation_since
from train import create_env, init_clearml_task

logger = get_logger("a1_delta_pose")

SETTLE_S = 1.0
LOG_EVERY_S = 1.0
# TCP at the home configuration in Genesis (the robot's is within 2 mm); the plate has to be
# placed before the scene is built, so before the start pose can be read.
SIM_HOME_TCP = (0.2805, 0.1337, 0.2104)


def extra_args(p) -> None:
    p.add_argument("--test", choices=("motion", "press"), default="motion")
    p.add_argument("--side", type=float, default=0.03, help="[m] square side")
    p.add_argument("--speed", type=float, default=0.025, help="[m/s] square speed")
    p.add_argument(
        "--mirror-y",
        action="store_true",
        help="square towards -y (the robot's right) instead of +y",
    )
    p.add_argument("--yaw", type=float, default=10.0, help="[deg] yaw amplitude")
    p.add_argument("--yaw-speed", type=float, default=5.0, help="[deg/s]")
    p.add_argument("--hold", type=float, default=15.0, help="[s] hand-push window")
    p.add_argument("--press-step", type=float, default=0.0002, help="[m] per step")
    p.add_argument("--press-max-distance", type=float, default=0.04, help="[m]")
    p.add_argument("--press-max-force", type=float, default=4.0, help="[N] stop")
    p.add_argument("--press-hold", type=float, default=2.0, help="[s] in contact")
    p.add_argument(
        "--press-kick",
        type=float,
        default=0.0,
        help="[N] sim: sideways (world x) push on the plate for one step, 0.5 s into the hold",
    )
    p.add_argument(
        "--press-drop",
        type=float,
        default=0.0,
        help="[m] from home down to the press start (home TCP z ~0.208 m above the table)",
    )
    sim = SpringPlateParams()
    p.add_argument(
        "--plate-gap", type=float, default=0.0165, help="[m] sim: TCP to plate"
    )
    p.add_argument("--plate-k", type=float, default=sim.normal_stiffness, help="[N/m]")
    p.add_argument("--plate-d", type=float, default=sim.normal_damping, help="[N*s/m]")
    p.add_argument(
        "--plate-shear-k",
        type=float,
        nargs=2,
        default=sim.shear_stiffness,
        help="[N/m] world x y",
    )
    p.add_argument(
        "--plate-shear-d", type=float, default=sim.shear_damping, help="[N*s/m]"
    )
    p.add_argument(
        "--plate-noise",
        type=float,
        nargs=2,
        default=sim.surface_noise_std,
        help="[N] sim: sideways surface noise std on the plate, world x y",
    )
    p.add_argument(
        "--plate-noise-cutoff",
        type=float,
        default=sim.surface_noise_cutoff_hz,
        help="[Hz] sim: low-pass corner of the surface noise",
    )


def create_sim_env_with_plate(cfg: ExpConfig, opts) -> tuple[BaseEnv, SpringPlate]:
    """`train.create_env` for Genesis, with the press plate added before the env builds the scene."""
    gs.init(logging_level="info", precision="32", seed=cfg.rl_cfg.seed)
    scene = GenesisScene(cfg=cfg, device=cfg.get_device())
    params = SpringPlateParams(
        normal_stiffness=opts.plate_k,
        normal_damping=opts.plate_d,
        shear_stiffness=tuple(opts.plate_shear_k),
        shear_damping=opts.plate_shear_d,
        surface_noise_std=tuple(opts.plate_noise),
        surface_noise_cutoff_hz=opts.plate_noise_cutoff,
    )
    top = (
        SIM_HOME_TCP[0],
        SIM_HOME_TCP[1],
        SIM_HOME_TCP[2] - opts.press_drop - opts.plate_gap,
    )
    plate = SpringPlate(scene.gs_scene, top_center=top, params=params)
    plate.create()
    env = get_env_class(cfg.env_cfg.env_name)(scene=scene, cfg=cfg)
    return env, plate


def main() -> None:
    th.set_default_dtype(th.float32)
    args = parse_arguments(extra_argparser=extra_args)
    opts = args.get_raw_args()
    is_sim = args.control_type == Control.SIM
    if is_sim and args.num_envs != 1:
        logger.error("Run the simulation with a single environment (-B 1).")
        sys.exit(1)
    task = init_clearml_task(
        project_name=args.project_name,
        algorithm=args.algorithm,
        control=args.control_type,
        exp_name=args.experiment_name,
    )
    device = th.device("cuda" if is_sim and th.cuda.is_available() else "cpu")
    ConfigManager.setup_config(argv=args, device=device, task=task)
    cfg = ConfigManager.get_config()

    plate = None
    if is_sim and opts.test == "press":
        env, plate = create_sim_env_with_plate(cfg, opts)
    else:
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
            "plate_dx",
            "plate_dy",
            "plate_dz",
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
    if opts.test == "press":
        manip.ctrl_gripper_close()
        if opts.press_drop > 0.0:
            settle()
            goal = manip.get_tcp_pose().clone()
            goal[:, 2] -= opts.press_drop
            manip.ctrl_go_to_goal(goal)
            settle()
    settle()
    manip.set_ft_bias()
    settle()
    start = manip.get_tcp_pose()[0].clone()
    reference = start[:3].clone()
    ref_yaw = 0.0
    logger.info(f"start TCP {np.round(start[:3].tolist(), 4)}; log: {log_path}")
    if plate is not None:
        offset = (start[:2] - th.tensor(SIM_HOME_TCP[:2], device=start.device)).norm()
        if float(offset) > 0.02:
            logger.warning(
                f"The start TCP is {float(offset) * 1e3:.0f} mm off the plate centre"
            )

    def force_norm() -> float:
        return float(manip.get_ft_wrench()[0, :3].norm())

    def act(
        phase: str,
        translation=(0.0, 0.0, 0.0),
        yaw: float = 0.0,
        push: th.Tensor | None = None,
    ) -> None:
        nonlocal reference, ref_yaw
        delta = th.zeros(1, 6, device=device)
        delta[0, :3] = th.as_tensor(translation, device=device)
        delta[0, 5] = yaw
        scene.pre_step()
        now = time.perf_counter()
        periods.append(now - last[0])
        last[0] = now
        clipped = bool(manip.ctrl_apply_delta_pose_action(delta)[0])
        if plate is not None:
            plate.apply_surface_noise(dt)
        if push is not None:
            # On the plate, so it reaches the gripper through the contact and the simulated F/T
            # sees it (a wrench applied to a robot link is not a contact). Acts over the whole
            # scene.step(), i.e. one policy step (measured).
            scene.gs_scene.sim.rigid_solver.apply_links_external_wrench(
                force=push.reshape(1, 1, 3), links_idx=[plate.link.idx]
            )
        scene.step()
        reference = manip.get_tcp_reference()[0, :3]
        ref_yaw += math.degrees(yaw)
        clipped_steps[phase] = clipped_steps.get(phase, 0) + int(clipped)

        pose = manip.get_tcp_pose()[0]
        force = manip.get_ft_wrench()[0, :3]
        deflection = (
            plate.deflection()[0].tolist() if plate is not None else [math.nan] * 3
        )
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
                *(f"{v * 1e3:.3f}" for v in deflection),
            ]
        )

    def measured_yaw_deg() -> float:
        angle, axis = rotation_since(start[3:], manip.get_tcp_orientation()[0])
        return angle if axis[2] >= 0 else -angle

    def tracking_error_mm() -> float:
        return float((manip.get_tcp_position()[0] - reference).norm()) * 1e3

    summary: list[str] = []

    def run_motion() -> None:
        step_len = opts.speed * dt
        n_side = max(1, round(opts.side / step_len))
        corner_errors = []
        y = -1 if opts.mirror_y else 1
        for direction in ((1, 0, 0), (0, y, 0), (-1, 0, 0), (0, -y, 0)):
            for _ in range(n_side):
                act("square", tuple(step_len * c for c in direction))
            for _ in range(round(0.5 / dt)):
                act("square")
            corner_errors.append(tracking_error_mm())
            logger.info(
                f"[square] corner, |TCP - reference| = {corner_errors[-1]:.2f} mm"
            )

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

        logger.info(f"[hold] {opts.hold:.0f} s: push the gripper by hand now")
        max_dev, max_force = 0.0, 0.0
        for k in range(round(opts.hold / dt)):
            act("hold")
            max_dev = max(max_dev, tracking_error_mm())
            max_force = max(max_force, force_norm())
            if k % round(LOG_EVERY_S / dt) == 0:
                logger.info(
                    f"[hold] |TCP - reference| {tracking_error_mm():5.1f} mm, |F| {force_norm():5.1f} N"
                )
        for _ in range(round(SETTLE_S / dt)):
            act("hold")
        summary.append(
            f"square corners |TCP - reference| [mm]: {np.round(corner_errors, 2).tolist()}"
        )
        summary.append(
            f"hold: max deviation {max_dev:.1f} mm, max |F| {max_force:.1f} N, "
            f"residual after release {tracking_error_mm():.2f} mm"
        )

    def run_press() -> None:
        down = (0.0, 0.0, -opts.press_step)
        reason = "max distance"
        n = 0
        for n in range(1, round(opts.press_max_distance / opts.press_step) + 1):
            act("press", down)
            if force_norm() > opts.press_max_force:
                reason = f"|F| > {opts.press_max_force:g} N"
                break
        travel = float(start[2] - reference[2]) * 1e3
        logger.info(
            f"[press] stopped after {n} steps ({reason}), reference {travel:.1f} mm down, "
            f"|F| {force_norm():.2f} N"
        )
        hold_forces = []
        kick_step = round(0.5 / dt) if plate is not None and opts.press_kick else -1
        push = th.tensor([opts.press_kick, 0.0, 0.0], device=device)
        for k in range(round(opts.press_hold / dt)):
            act("hold", push=push if k == kick_step else None)
            hold_forces.append(force_norm())
        logger.info(
            f"[hold] |F| {hold_forces[0]:.2f} -> {hold_forces[-1]:.2f} N over {opts.press_hold:g} s"
        )
        for _ in range(4 * n + 50):
            to_start = start[:3] - reference
            distance = float(to_start.norm())
            if distance < 5e-4:
                break
            act(
                "retreat",
                (to_start / distance * min(opts.press_step, distance)).tolist(),
            )
        for _ in range(round(SETTLE_S / dt)):
            act("retreat")
        logger.info(f"[retreat] back at the start reference, |F| {force_norm():.2f} N")
        summary.append(
            f"press: {n} steps ({reason}), reference travel {travel:.1f} mm, "
            f"hold |F| {hold_forces[0]:.2f} -> {hold_forces[-1]:.2f} N, after retreat {force_norm():.2f} N"
        )

    try:
        run_press() if opts.test == "press" else run_motion()
    finally:
        write_log()
    manip.ctrl_go_to_home()

    p = np.array(periods[1:]) * 1e3
    logger.info("=== Summary ===")
    logger.info(
        f"loop period [ms]: median {np.median(p):.1f}, p95 {np.percentile(p, 95):.1f}, "
        f"max {p.max():.1f}; {(p > 1.25e3 * dt).sum()} of {len(p)} steps > 1.25 * policy_dt"
    )
    for line in summary:
        logger.info(line)
    logger.info(f"clipped steps per phase: {clipped_steps}")
    logger.info(f"log: {log_path}")


if __name__ == "__main__":
    setup_logger("INFO")
    main()
