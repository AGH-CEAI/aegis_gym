"""
Press sweep: quasi-static press staircase for matching the simulated servo stiffness
to the real robot (`servo_stiffness_scale` = K_real / K_sim).

Per pose the tool locates the table with a slow touch, backs off GAP_M and tares, then
steps the commanded position down in small increments, holding each until the force
settles (longer than the command delay), up to MAX_FORCE_N. Steady force against
commanded penetration is the stiffness the policy sees, free of latency and damping.
The flat pose gives the scale; the tilted ones check that one scale fits other wrist
loadings. Every sample goes to `logs/ft_stiffness/` as CSV.

Run on both backends and compare:

    python -m scripts.ft_sensor.stiffness_probe_MAX --env reacher -B 1 --vis      # simulation
    python -m scripts.ft_sensor.stiffness_probe_MAX --env reacher --control ros   # real robot

SAFETY: the start, the touch and its forces are those of `stiffness_probe.py`: descent
at 1.5 mm/s until 0.3 N, after which the delayed commands (~1.2 s) press on to ~13 N
(16 N seen on the robot), abort at 30 N. The staircase then moves only in small
commanded steps and waits out the delay after each, so at most one step (<= 0.3 mm) is
pending when a limit is seen. On top of that: the staircase never exceeds MAX_FORCE_N, the
commanded penetration is capped at MAX_PRESS_M past the surface (the bound if the
sensor stops reporting), and nothing is tared or zeroed while loaded. Abort and Ctrl+C both lift the tool by velocity first.
On the robot: speed slider at 100 % (it scales the commanded steps), hand on the
e-stop.
"""

import csv
import json
import math
import time
from pathlib import Path

import torch as th

from aegis_gym.aux.geom import transform_by_quat
from aegis_gym.aux.logging import get_logger, setup_logger
from aegis_gym.config import ConfigManager, parse_arguments
from aegis_gym.config.types import ExpConfig
from aegis_gym.envs import BaseEnv
from scripts.ft_sensor.common import is_modelled, sensor_quat, step
from scripts.ft_sensor.stiffness_probe import (
    _PHASE_ID,
    MOVE_SETTLE_SECONDS,
    _log_contacts,
    _prepare_over_table,
    _ProbeStream,
    _stop,
)
from train import create_env, init_clearml_task

# (name, tilt axis in the TCP frame, tilt [deg], gripper open). Flat and closed has no
# moment arm and gives the scale; 6 deg about TCP Y and X loads the wrist two ways.
PRESS_POSES: list[tuple[str, tuple[float, float, float], float, bool]] = [
    ("flat_closed", (0.0, 1.0, 0.0), 0.0, False),
    ("tilt_y+6", (0.0, 1.0, 0.0), 6.0, True),
    ("tilt_x+6", (1.0, 0.0, 0.0), 6.0, True),
]
# World XY shifts of the press point [m]; each is another arm configuration. Check the
# first on the robot before adding more, e.g. (0.08, 0.0) and (0.0, 0.08).
PRESS_POSITIONS: list[tuple[float, float]] = [(0.0, 0.0)]
PRESS_REPEATS = 2
PRESS_OUT_DIR = "logs/ft_stiffness"

_PHASE_ID.update({"locate": 4, "backoff": 5, "pulse": 6, "settle": 7})


def _world_velocity(
    env: BaseEnv,
    direction_world: tuple[float, float, float],
    speed: float,
    device: th.device,
) -> th.Tensor:
    """World-frame action [num_envs, 6] moving at `speed` [m/s] along a world
    direction, normalised by `env.max_linear_speed`."""
    vec = th.tensor(direction_world, dtype=th.float32, device=device)
    scale = max(-1.0, min(1.0, speed / float(env.max_linear_speed)))
    action = th.zeros(env.num_envs, 6, device=device)
    action[:, :3] = vec / th.linalg.norm(vec) * scale
    return action


def _slope(x: list[float], y: list[float]) -> float | None:
    """Least-squares slope, None for fewer than 3 points."""
    if len(x) < 3 or max(x) - min(x) <= 0.0:
        return None
    xs = th.tensor(x, dtype=th.float32)
    design = th.stack([xs, th.ones_like(xs)], dim=-1)
    return float(th.linalg.lstsq(design, th.tensor(y).unsqueeze(-1)).solution[0, 0])


def _write_csv(path: str, rows: list[dict]) -> None:
    if not rows:
        return
    with open(path, "w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def ft_sensor_press_sweep(env: BaseEnv, cfg: ExpConfig) -> None:
    """Runs the staircase for every PRESS_POSITIONS x PRESS_POSES x PRESS_REPEATS and
    writes `<PRESS_OUT_DIR>/<sim|real>_<time>_{raw,steps}.csv` and `_meta.json`.
    The motion is world-vertical, so a tilted tool does not slide."""
    logger = get_logger("ft_stiffness_probe")
    logger.info("Starting the F/T press sweep")

    manipulator = env.manipulator
    scene = env._scene
    device = cfg.get_device()
    dt = env.get_policy_dt()

    LOCATE_SPEED_MPS = 0.0015
    STEP_SPEED_MPS = 0.0015  # above the real servo deadband (~1 mm/s)
    BACKOFF_SPEED_MPS = 0.005
    BACKOFF_M = 0.004  # commanded lift after the touch; > its overshoot (~2 mm)
    TOUCH_FORCE_N = 0.3
    MAX_FORCE_N = 5.0  # press target, as in `stiffness_probe.py`
    ABORT_FORCE_N = 30.0
    MAX_APPROACH_M = 0.20
    GAP_M = 0.0005  # start of each staircase above the located surface
    FREE_STEP_M = 0.00025  # increments before contact
    FORCE_STEP_N = 0.6  # target force increment, ~8 points up to MAX_FORCE_N
    MAX_STEP_M = 0.0003
    # Commanded penetration budget past the surface: the bound on force even if the
    # sensor stops reporting (~10 N at 7 kN/m).
    MAX_PRESS_M = 0.0015
    LOCATE_HOLD_S = 3.0
    ZERO_S = 0.5  # force zero averaged before the locate and each staircase
    ZERO_MAX_N = 1.0  # a larger zero means the tool is already loaded
    HOLD_MIN_S = 2.5  # longer than servo_command_delay_s + the pulse
    HOLD_MAX_S = 6.0
    AVG_S = 0.8  # averaging window at the end of a hold
    SETTLED_RATE_N_S = 0.2
    SETTLE_SECONDS = 1.0
    RETRACT_M = 0.03

    up = (0.0, 0.0, 1.0)
    down = (0.0, 0.0, -1.0)
    up_t = th.tensor(up, dtype=th.float32, device=device)
    settle_steps = max(1, round(SETTLE_SECONDS / dt))
    move_steps = max(1, round(MOVE_SETTLE_SECONDS / dt))
    pulse_m = STEP_SPEED_MPS * dt  # smallest commanded increment
    backend = "sim" if is_modelled(manipulator) else "real"
    stream = _ProbeStream(cfg)

    constants = {
        k: v for k, v in locals().items() if k.isupper() and isinstance(v, (int, float))
    }
    out_dir = Path(PRESS_OUT_DIR)
    out_dir.mkdir(parents=True, exist_ok=True)
    stem = out_dir / f"{backend}_{time.strftime('%Y%m%d_%H%M%S')}"
    robot_cfg = cfg.robot_cfg
    meta = {
        "backend": backend,
        "policy_dt": dt,
        "max_linear_speed": float(env.max_linear_speed),
        "servo_stiffness_scale": robot_cfg.servo_stiffness_scale,
        "servo_command_delay_s": robot_cfg.servo_command_delay_s,
        "servo_follow_error_max_rad": robot_cfg.servo_follow_error_max_rad,
        "poses": PRESS_POSES,
        "positions": PRESS_POSITIONS,
        "repeats": PRESS_REPEATS,
        "constants": constants,
    }
    with open(f"{stem}_meta.json", "w") as fh:
        json.dump(meta, fh, indent=2)

    raw_rows: list[dict] = []
    step_rows: list[dict] = []
    # Mutable context stamped on every raw sample.
    ctx = {
        "pose": "",
        "position": -1,
        "repeat": -1,
        "phase": "",
        "cmd": 0.0,
        "abort": True,
    }
    # Start position and world Fz zero; the zero absorbs a residual bias, which would
    # otherwise trip TOUCH_FORCE_N in free space.
    ref = {"pos": manipulator.get_tcp_position()[0].clone(), "fz0": 0.0}
    n_step = [0]

    def sample() -> dict:
        """One raw sample after `step()`; checks the abort force."""
        wrench = manipulator.get_ft_wrench_compensated()[0]
        quat = sensor_quat(manipulator)
        f_world = transform_by_quat(wrench[:3].unsqueeze(0), quat.unsqueeze(0))[0]
        pos = manipulator.get_tcp_position()[0]
        travel = float((ref["pos"] - pos) @ up_t)  # + is downward
        f_n = abs(float(f_world[2]) - ref["fz0"])
        row = {
            "t_s": n_step[0] * dt,
            "pose": ctx["pose"],
            "position": ctx["position"],
            "repeat": ctx["repeat"],
            "phase": ctx["phase"],
            "cmd_mm": ctx["cmd"] * 1000.0,
            "travel_mm": travel * 1000.0,
            "lag_mm": (ctx["cmd"] - travel) * 1000.0,
            "fn": f_n,
        }
        row.update({f"fw_{a}": float(v) for a, v in zip("xyz", f_world)})
        row.update(
            {n: float(v) for n, v in zip(("fx", "fy", "fz", "tx", "ty", "tz"), wrench)}
        )
        row.update({f"p_{a}": float(v) for a, v in zip("xyz", pos)})
        row.update({f"q_{a}": float(v) for a, v in zip("wxyz", quat)})
        q = manipulator.get_joints_positions()[0]
        eff = manipulator.get_joints_efforts()[0]
        row.update({f"j{i}": float(v) for i, v in enumerate(q[:6])})
        row.update({f"e{i}": float(v) for i, v in enumerate(eff[:6])})
        raw_rows.append(row)
        n_step[0] += 1
        stream.send(manipulator, dt, ctx["phase"], ctx["cmd"], travel, 0.0)
        if ctx["abort"] and f_n >= ABORT_FORCE_N:
            raise RuntimeError(
                f"Contact force {f_n:.2f} N exceeded the abort limit "
                f"{ABORT_FORCE_N:.1f} N ({ctx['pose']}, {ctx['phase']})"
            )
        return row

    def drive(direction: tuple[float, float, float], speed: float, n: int) -> None:
        for _ in range(n):
            manipulator.ctrl_apply_vel_action(
                _world_velocity(env, direction, speed, device), open_gripper=None
            )
            step(scene)
            sample()

    def wait(seconds: float) -> None:
        for _ in range(max(1, round(seconds / dt))):
            _stop(env, manipulator, device)
            step(scene)
            sample()

    def zero_force() -> None:
        """Zero command for ZERO_S; the mean world Fz becomes the zero. Refuses a
        large one: zeroing in contact would hide that load from every limit."""
        ref["fz0"] = 0.0
        start = len(raw_rows)
        wait(ZERO_S)
        tail = raw_rows[start:]
        zero = sum(r["fw_z"] for r in tail) / len(tail)
        if abs(zero) > ZERO_MAX_N:
            raise RuntimeError(
                f"{zero:+.2f} N before zeroing, right after a tare -- the tool is "
                f"loaded (touching?) or the sensor drifts. Not zeroing over it."
            )
        ref["fz0"] = zero

    def hold(direction: str, index: int) -> dict:
        """Zero command until the force settles; returns the averaged step row."""
        ctx["phase"] = "settle"
        window = max(4, round(AVG_S / dt))
        n_min = max(window, round(HOLD_MIN_S / dt))
        buf: list[dict] = []
        settled = False
        for i in range(max(n_min, round(HOLD_MAX_S / dt))):
            _stop(env, manipulator, device)
            step(scene)
            buf.append(sample())
            if i + 1 >= n_min:
                tail = [r["fn"] for r in buf[-window:]]
                half = window // 2
                rate = (
                    sum(tail[half:]) / (window - half) - sum(tail[:half]) / half
                ) / (half * dt)
                if abs(rate) < SETTLED_RATE_N_S:
                    settled = True
                    break
        tail = buf[-window:]
        keys = [k for k, v in tail[0].items() if isinstance(v, float)]
        row = {
            "pose": ctx["pose"],
            "position": ctx["position"],
            "xy_offset": PRESS_POSITIONS[ctx["position"]],
            "repeat": ctx["repeat"],
            "index": index,
            "direction": direction,
            "hold_s": len(buf) * dt,
            "settled": settled,
        }
        row.update({k: sum(r[k] for r in tail) / len(tail) for k in keys})
        mean = row["fn"]
        row["fn_std"] = math.sqrt(sum((r["fn"] - mean) ** 2 for r in tail) / len(tail))
        step_rows.append(row)
        return row

    def pulse(distance_m: float) -> None:
        """Commands a displacement (+ down) as whole policy steps at STEP_SPEED_MPS."""
        ctx["phase"] = "pulse"
        for _ in range(max(1, round(abs(distance_m) / pulse_m))):
            ctx["cmd"] += math.copysign(pulse_m, distance_m)
            drive(down if distance_m > 0 else up, STEP_SPEED_MPS, 1)

    def go_to(pos: th.Tensor, quat: th.Tensor) -> None:
        manipulator.ctrl_go_to_goal(
            th.cat([pos.unsqueeze(0), quat.unsqueeze(0)], dim=-1).repeat(
                env.num_envs, 1
            ),
            open_gripper=None,
        )
        for _ in range(move_steps):
            step(scene)
            sample()

    def locate() -> th.Tensor | None:
        """Touch as in `stiffness_probe.py`: continuous descent at LOCATE_SPEED_MPS
        until TOUCH_FORCE_N or MAX_APPROACH_M; returns the TCP position at contact."""
        ctx["phase"] = "locate"
        ctx["cmd"] = 0.0
        ref["pos"] = manipulator.get_tcp_position()[0].clone()
        zero_force()
        contact = None
        for _ in range(max(1, round(3.0 * MAX_APPROACH_M / LOCATE_SPEED_MPS / dt))):
            ctx["cmd"] += LOCATE_SPEED_MPS * dt
            drive(down, LOCATE_SPEED_MPS, 1)
            if raw_rows[-1]["fn"] >= TOUCH_FORCE_N:
                contact = manipulator.get_tcp_position()[0].clone()
                break
            if raw_rows[-1]["travel_mm"] >= MAX_APPROACH_M * 1000.0:
                break
        if contact is None:
            _stop(env, manipulator, device)
            return None
        # Delayed commands keep pressing: the peak is ~K * speed * delay.
        start = len(raw_rows)
        wait(LOCATE_HOLD_S)
        peak = max(r["fn"] for r in raw_rows[start:])
        logger.info(
            f"  touched after {float((ref['pos'] - contact) @ up_t) * 1000:.1f} mm at "
            f"z={float(contact[2]) * 1000:.2f} mm, peak {peak:.2f} N"
        )
        # Lift by velocity first: a planner may refuse to start in contact.
        ctx["phase"] = "backoff"
        drive(up, BACKOFF_SPEED_MPS, max(1, round(BACKOFF_M / BACKOFF_SPEED_MPS / dt)))
        wait(HOLD_MIN_S)
        return contact

    def staircase(gap_pos: th.Tensor, quat: th.Tensor) -> None:
        ctx["phase"] = "backoff"
        go_to(gap_pos, quat)
        wait(SETTLE_SECONDS)
        # Taring in contact would hide that load; the zero from the locate still holds.
        if raw_rows[-1]["fn"] >= TOUCH_FORCE_N:
            raise RuntimeError(
                f"{raw_rows[-1]['fn']:.2f} N at the gap -- the move ended in contact. "
                f"Not taring over it."
            )
        manipulator.set_ft_bias()
        ctx["cmd"] = 0.0
        zero_force()
        ref["pos"] = manipulator.get_tcp_position()[0].clone()
        rows = [hold("start", 0)]
        while rows[-1]["fn"] < MAX_FORCE_N:
            if ctx["cmd"] + pulse_m > GAP_M + MAX_PRESS_M:
                logger.warning(
                    f"  only {rows[-1]['fn']:.2f} N after {ctx['cmd'] * 1000:.2f} mm "
                    f"commanded -- penetration budget used up"
                )
                break
            loaded = [r for r in rows if r["fn"] >= TOUCH_FORCE_N]
            if len(loaded) >= 2:
                d_x = (loaded[-1]["cmd_mm"] - loaded[-2]["cmd_mm"]) / 1000.0
                k = (loaded[-1]["fn"] - loaded[-2]["fn"]) / d_x if d_x > 0 else 0.0
                inc = MAX_STEP_M if k <= 0 else FORCE_STEP_N / k
                inc = max(pulse_m, min(MAX_STEP_M, inc))
                # Stop before a step that would cross MAX_FORCE_N.
                if (
                    k > 0
                    and rows[-1]["fn"] + k * pulse_m * round(inc / pulse_m)
                    > MAX_FORCE_N
                ):
                    break
            elif loaded:
                inc = pulse_m  # stiffness unknown yet
            else:
                inc = FREE_STEP_M
            pulse(inc)
            rows.append(hold("load", len(rows)))
        if is_modelled(manipulator):
            _log_contacts(manipulator, f"{ctx['pose']} top step")
        top = rows[-1]
        # Back to the start in one move; the end row shows tare drift.
        pulse(-ctx["cmd"])
        end = hold("end", len(rows))

        load = [
            r for r in rows if r["direction"] == "load" and r["fn"] >= TOUCH_FORCE_N
        ]
        k_cmd = _slope([r["cmd_mm"] / 1000.0 for r in load], [r["fn"] for r in load])
        k_lag = _slope([r["lag_mm"] / 1000.0 for r in load], [r["fn"] for r in load])
        free = [r for r in rows if r["direction"] == "load" and r["fn"] < TOUCH_FORCE_N]
        gain = (
            free[-1]["travel_mm"] / free[-1]["cmd_mm"]
            if free and free[-1]["cmd_mm"] > 0
            else float("nan")
        )

        def kn(k: float | None) -> str:
            return "n/a" if k is None else f"{k / 1000:.2f}"

        logger.info(
            f"  [{ctx['pose']} p{ctx['position']} r{ctx['repeat']}] K_cmd = "
            f"{kn(k_cmd)} kN/m  K_lag = {kn(k_lag)} kN/m over {len(load)} steps, "
            f"top {top['fn']:.2f} N, free travel/cmd = {gain:.2f}, back at start "
            f"{end['fn']:.2f} N, unsettled {sum(not r['settled'] for r in rows)}"
        )

    try:
        for p_idx, xy in enumerate(PRESS_POSITIONS):
            shift = th.tensor([*xy, 0.0], dtype=th.float32, device=device)
            for name, axis, tilt_deg, gripper_open in PRESS_POSES:
                ctx.update(pose=name, position=p_idx, repeat=-1, cmd=0.0)
                logger.info("=" * 78)
                logger.info(
                    f"Pose {name} at position {p_idx} {xy}: tilt {tilt_deg:+.1f} deg "
                    f"about TCP{axis}, gripper {'open' if gripper_open else 'closed'}"
                )
                half = math.radians(tilt_deg) / 2.0
                vec = th.tensor(axis, dtype=th.float32, device=device)
                tilt_quat = th.cat(
                    [
                        th.tensor([math.cos(half)], device=device),
                        vec / th.linalg.norm(vec) * math.sin(half),
                    ]
                ).unsqueeze(0)
                _prepare_over_table(
                    env,
                    cfg,
                    settle_steps,
                    tilt_quat=tilt_quat,
                    close_gripper=not gripper_open,
                )
                ctx["phase"] = "backoff"
                if any(xy):
                    start = manipulator.get_tcp_position()[0].clone() + shift
                    go_to(start, manipulator.get_tcp_orientation()[0].clone())
                contact = locate()
                if contact is None:
                    logger.warning("  no contact within reach -- skipping this pose")
                    continue
                quat_tcp = manipulator.get_tcp_orientation()[0].clone()
                gap_pos = contact + up_t * GAP_M
                for rep in range(PRESS_REPEATS):
                    ctx["repeat"] = rep
                    staircase(gap_pos, quat_tcp)
                ctx["phase"] = "backoff"
                go_to(gap_pos + up_t * RETRACT_M, quat_tcp)
    finally:
        _stop(env, manipulator, device)
        ctx["abort"] = False  # the lift must run even when overloaded
        try:
            # Lift by velocity, then plan clear of the table.
            ctx["phase"] = "backoff"
            drive(
                up, BACKOFF_SPEED_MPS, max(1, round(BACKOFF_M / BACKOFF_SPEED_MPS / dt))
            )
            wait(HOLD_MIN_S)
            pose = manipulator.get_tcp_pose()[0].clone()
            go_to(pose[:3] + up_t * RETRACT_M, pose[3:])
        except Exception as exc:  # noqa: BLE001 - must not mask the real failure
            logger.error(f"  RETRACT FAILED ({exc}) -- clear the tool by hand.")
        _write_csv(f"{stem}_raw.csv", raw_rows)
        _write_csv(f"{stem}_steps.csv", step_rows)
        logger.info(
            f"Wrote {len(raw_rows)} samples and {len(step_rows)} steps to {stem}_*.csv"
        )


def main() -> None:
    th.set_default_dtype(th.float32)
    args = parse_arguments()
    task = init_clearml_task(
        project_name=args.project_name,
        algorithm=args.algorithm,
        control=args.control_type,
        exp_name=args.experiment_name or "ft_press_sweep",
    )
    device = th.device("cuda" if th.cuda.is_available() else "cpu")
    ConfigManager.setup_config(argv=args, device=device, task=task)
    cfg = ConfigManager.get_config()

    env = create_env(cfg)
    env = getattr(env, "unwrapped", env)
    ft_sensor_press_sweep(env=env, cfg=cfg)


if __name__ == "__main__":
    setup_logger("INFO")
    try:
        main()
    except KeyboardInterrupt:
        get_logger("ft_stiffness_probe").info("Stopped by user")
