"""
Open-loop push_t replay: the actions of a policy's simulated episode, played on the real robot,
then the tee's final pose compared with the simulation's. The tee is measured only at the start
and after the arm has lifted off (`tee_tracker` of aegis_utils on the scene camera), so the arm
hiding it during the pushes doesn't matter. This isolates the contact and robot parameters
(friction, tee mass, the admittance chain) from the closed loop.

`--phase record` (sim): runs the policy (deterministic) on the `--seeds` initial states, nominal
domain, until the first success or the time-out; saves <replay_dir>/sim_<seed>.npz.
`--phase replay` (robot, one `--replay-seed`): the seeded reset (the operator places the tee at the
yellow outline of /tee_tracker/image), the recorded actions at the policy rate, a 10 cm lift,
the tee measurement, home; saves <replay_dir>/real_<seed>.npz.
`--phase compare` (sim, every real_<seed>.npz): re-simulates each run from the tee start measured
on the robot (removes the placement error), as an ensemble over the tracker error (-B members),
and prints where the real final pose falls in the ensemble.
`--env-dict` overrides apply, e.g. "{'friction': 0.3}" to compare another tee friction.

`--no-tee` (replay and compare): the same actions in free space, nothing on the table and no
tracker; checks the robot side alone (action scaling, the admittance reference and its limiter,
tracking lag, inertial F/T forces). Saves / reads <replay_dir>/real_free_<seed>.npz; compare
re-simulates with the tee out of reach and saves <replay_dir>/free_compare_<seed>.npz.

Run from the repo root (the config comes from the policy's training task):
    python -m scripts.sim2real.open_loop_replay --phase record --replay-dir R --seeds 0 1 2 3 4 \
        --control sim -B 5 --env push_t -a rl --load-rl-task-id <id> -e replay_record
    python -m scripts.sim2real.open_loop_replay --phase replay --replay-dir R --replay-seed 0 \
        --control ros --env push_t -a rl --load-rl-task-id <id> -e replay_real --disable-cameras
    python -m scripts.sim2real.open_loop_replay --phase compare --replay-dir R \
        --control sim -B 64 --env push_t -a rl --load-rl-task-id <id> -e replay_compare
    python -m scripts.sim2real.open_loop_replay --phase replay --no-tee --replay-dir R \
        --replay-seed 0 1 2 --control ros --env push_t -a rl --load-rl-task-id <id> \
        -e replay_free --disable-cameras
    python -m scripts.sim2real.open_loop_replay --phase compare --no-tee --replay-dir R \
        --control sim -B 16 --env push_t -a rl --load-rl-task-id <id> -e replay_free_compare
"""

import math
import sys
import time
from pathlib import Path

import numpy as np
import torch as th

from aegis_gym.aux.geom import quat_to_z_euler
from aegis_gym.aux.logging import get_logger, setup_logger
from aegis_gym.aux.utils import load_policy
from aegis_gym.config import ConfigManager, parse_arguments
from aegis_gym.config.types import Control, ExpConfig
from aegis_gym.envs.objects.real.ros_grpc_objects import RosGrpcURDF
from train import create_env

logger = get_logger("open_loop_replay")

LIFT_M = 0.10
# placement accepted within these of the recorded start (the compare phase removes the rest)
PLACE_TOL_M = 0.01
PLACE_TOL_DEG = 5.0
# a fresh tracker pose after the lift: newer than this, with at least one marker
FRESH_S = 0.2
# the tee parked here (on the floor, beside the workbench) for the free-space compare
PARKED_TEE_XY = (0.6, 1.5)
# speed above which the tracking lag is estimated [m/s]
LAG_MIN_SPEED = 0.05


def extra_args(p) -> None:
    p.add_argument("--phase", choices=("record", "replay", "compare"), required=True)
    p.add_argument("--replay-dir", type=Path, required=True)
    p.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    p.add_argument("--replay-seed", type=int, nargs="+", default=[0])
    p.add_argument(
        "--no-tee", action="store_true", help="replay/compare in free space, no tracker"
    )
    p.add_argument(
        "--start-noise-mm", type=float, default=2.0, help="compare: tracker error"
    )
    p.add_argument(
        "--start-noise-deg", type=float, default=1.0, help="compare: tracker error"
    )


def planar(pose: th.Tensor) -> np.ndarray:
    """[N, 7] poses -> [N, 3] (x, y, yaw)."""
    yaw = quat_to_z_euler(pose[:, 3:])
    return th.stack([pose[:, 0], pose[:, 1], yaw], dim=-1).cpu().numpy()


def planar_to_pose(xyyaw: np.ndarray, z: float, device: th.device) -> th.Tensor:
    x, y, yaw = xyyaw
    return th.tensor(
        [[x, y, z, math.cos(yaw / 2), 0.0, 0.0, math.sin(yaw / 2)]],
        dtype=th.float32,
        device=device,
    )


def pose_error(a: np.ndarray, b: np.ndarray) -> tuple[float, float]:
    """Position [m] and yaw [deg] differences of two planar poses."""
    dyaw = math.atan2(math.sin(a[2] - b[2]), math.cos(a[2] - b[2]))
    return float(np.hypot(a[0] - b[0], a[1] - b[1])), math.degrees(dyaw)


def setup(args) -> tuple:
    source_task = ConfigManager.resolve_source_task_id(
        task_id=args.load_rl_task_id, model_id=args.load_rl_model_id
    )
    device = th.device(
        "cuda" if args.control_type == Control.SIM and th.cuda.is_available() else "cpu"
    )
    ConfigManager.setup_config(
        argv=args, device=device, task=None, source_task_id=source_task
    )
    cfg: ExpConfig = ConfigManager.get_config()
    # the success must not end (and reset) the episode in the middle of a replay
    cfg.env_cfg.env_specific_dict["terminate_on_success"] = False
    RosGrpcURDF.tracked = not args.get_raw_args().no_tee
    # the env constructor's random reset isn't replayed: no placement for it
    RosGrpcURDF.prompt_placement = False
    env = create_env(cfg)
    RosGrpcURDF.prompt_placement = True
    return getattr(env, "unwrapped", env), cfg, device


def record(env, cfg: ExpConfig, device: th.device, opts) -> None:
    seeds = opts.seeds
    if env.num_envs != len(seeds):
        raise ValueError(f"Run with -B {len(seeds)}, one env per seed.")
    policy = load_policy(env=env, cfg=cfg)
    envs_idx = th.arange(env.num_envs, device=device)
    actions, tcp, ref, tee = [], [], [], []
    with th.no_grad(), env.nominal_domain():
        obs = env.reset_seeded(envs_idx=envs_idx, seeds=seeds)
        tee.append(planar(env.object.get_pose()))
        goal = planar(env.goal_pose)
        end = np.full(env.num_envs, -1)
        success = np.zeros(env.num_envs, bool)
        # the loop stays within the time-out, so a `done` is the tee off the table: the env has
        # already reset it, the episode is dropped
        off_table = np.zeros(env.num_envs, bool)
        for t in range(env.max_episode_length):
            a = policy(obs)
            obs, _rew, dones, extras = env.step(a)
            actions.append(a.cpu().numpy())
            tcp.append(env.manipulator.get_tcp_pose()[:, :3].cpu().numpy())
            ref.append(env.manipulator.get_tcp_reference()[:, :3].cpu().numpy())
            tee.append(planar(env.object.get_pose()))
            ended = (extras["success"] | dones).cpu().numpy() & (end < 0)
            success |= ended & extras["success"].cpu().numpy()
            off_table |= ended & ~success
            end[ended] = t + 1
            if (end >= 0).all():
                break
    end[end < 0] = len(actions)
    opts.replay_dir.mkdir(parents=True, exist_ok=True)
    A, TCP, REF, TEE = (np.stack(x, axis=1) for x in (actions, tcp, ref, tee))
    for i, seed in enumerate(seeds):
        if off_table[i]:
            logger.warning(
                f"seed {seed}: the tee left the table at step {end[i]}; not saved"
            )
            continue
        n = end[i]
        path = opts.replay_dir / f"sim_{seed}.npz"
        np.savez(
            path,
            seed=seed,
            actions=A[i, :n],
            tcp=TCP[i, :n],
            reference=REF[i, :n],
            tee=TEE[i, : n + 1],
            goal=goal[i],
            success=success[i],
            policy_dt=env.get_policy_dt(),
            source_task=str(cfg.args.load_rl_task_id),
        )
        d, dyaw = pose_error(TEE[i, n], TEE[i, 0])
        logger.info(
            f"seed {seed}: {n} steps ({n * env.get_policy_dt():.1f} s), success {success[i]}, "
            f"tee moved {d * 100:.1f} cm / {dyaw:.0f} deg -> {path}"
        )


def wait_fresh_tee(link, timeout_s: float = 5.0):
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        p = link.latest()
        if (
            p is not None
            and time.monotonic() - p.received < FRESH_S
            and p.n_markers > 0
        ):
            return p
        time.sleep(0.05)
    return None


def replay(env, cfg: ExpConfig, device: th.device, opts) -> None:
    if cfg.args.control_type != Control.ROS:
        raise ValueError("The replay phase runs on the robot (--control ros).")
    for seed in opts.replay_seed:
        if opts.no_tee:
            replay_free(env, device, opts, seed)
        else:
            replay_tee(env, device, opts, seed)


def lift_off(manip) -> None:
    lift = manip.get_tcp_pose().clone()
    lift[:, 2] += LIFT_M
    manip.ctrl_go_to_goal(goal_pose=lift)
    time.sleep(0.5)


def play(env, actions: np.ndarray, device: th.device) -> dict:
    """Steps the actions at the policy rate; per step the TCP, reference, wrench, period."""
    manip = env.manipulator
    log = {"tcp": [], "reference": [], "wrench": [], "period": []}
    last = time.perf_counter()
    for a in actions:
        env.step(th.as_tensor(a, dtype=th.float32, device=device).unsqueeze(0))
        now = time.perf_counter()
        log["period"].append(now - last)
        last = now
        log["tcp"].append(manip.get_tcp_pose()[0, :3].cpu().numpy())
        log["reference"].append(manip.get_tcp_reference()[0, :3].cpu().numpy())
        log["wrench"].append(manip.get_ft_wrench()[0].cpu().numpy())
    return {k: np.array(v) for k, v in log.items()}


def replay_free(env, device: th.device, opts, seed: int) -> None:
    rec = load_recording(env, opts, seed)
    manip = env.manipulator
    envs_idx = th.zeros(1, dtype=th.long, device=device)
    env.reset_seeded(envs_idx=envs_idx, seeds=[seed])
    # the reset's motion ends after the last state read
    manip.read_state()
    tcp_start = manip.get_tcp_pose()[0, :3].cpu().numpy()
    input(
        f"seed {seed}: {len(rec['actions'])} actions in free space; table clear, e-stop at "
        "hand, press Enter... "
    )
    log = play(env, rec["actions"], device)
    lift_off(manip)
    manip.ctrl_go_to_home()
    path = opts.replay_dir / f"real_free_{seed}.npz"
    np.savez(path, seed=seed, tcp_start=tcp_start, **log)
    period = log["period"][1:]
    logger.info(
        f"seed {seed}: period {np.median(period) * 1e3:.1f} ms (max {np.max(period) * 1e3:.0f}), "
        f"peak |F_xy| {np.max(np.hypot(*log['wrench'][:, :2].T)):.1f} N -> {path}"
    )


def load_recording(env, opts, seed: int):
    rec = np.load(opts.replay_dir / f"sim_{seed}.npz")
    if not math.isclose(float(rec["policy_dt"]), env.get_policy_dt()):
        raise ValueError(
            f"policy_dt {env.get_policy_dt()} differs from the recording's {rec['policy_dt']}"
        )
    return rec


def replay_tee(env, device: th.device, opts, seed: int) -> None:
    rec = load_recording(env, opts, seed)
    link = env.object._link
    manip = env.manipulator
    envs_idx = th.zeros(1, dtype=th.long, device=device)

    # the seeded reset: home, start pose, the operator places the tee (prompt in set_pose)
    env.reset_seeded(envs_idx=envs_idx, seeds=[seed])
    while True:
        p = wait_fresh_tee(link)
        if p is None:
            input("The tracker doesn't see the tee; fix the view and press Enter... ")
            continue
        start = np.array([p.x, p.y, p.yaw])
        d, dyaw = pose_error(start, rec["tee"][0])
        logger.info(
            f"Tee placed {d * 1e3:.1f} mm / {dyaw:.1f} deg off the recorded start"
        )
        if d <= PLACE_TOL_M and abs(dyaw) <= PLACE_TOL_DEG:
            break
        if input("Outside the tolerance: Enter to re-measure, `c` to go on: ") == "c":
            break
    goal = planar(env.goal_pose)[0]
    if pose_error(goal, rec["goal"])[0] > 1e-4:
        raise RuntimeError(f"Goal {goal} differs from the recording's {rec['goal']}")

    input(f"Replaying {len(rec['actions'])} actions; e-stop at hand, press Enter... ")
    tcp, ref, wrench, tee_seen, period = [], [], [], [], []
    last = time.perf_counter()
    for a in rec["actions"]:
        env.step(th.as_tensor(a, dtype=th.float32, device=device).unsqueeze(0))
        now = time.perf_counter()
        period.append(now - last)
        last = now
        tcp.append(manip.get_tcp_pose()[0, :3].cpu().numpy())
        ref.append(manip.get_tcp_reference()[0, :3].cpu().numpy())
        wrench.append(manip.get_ft_wrench()[0].cpu().numpy())
        p = link.latest()
        fresh = time.monotonic() - p.received < FRESH_S
        tee_seen.append([p.x, p.y, p.yaw] if fresh else [np.nan] * 3)

    lift_off(manip)
    p = wait_fresh_tee(link)
    final = np.array([p.x, p.y, p.yaw]) if p is not None else np.full(3, np.nan)
    if p is None:
        logger.error("No tee pose after the lift; the final pose is missing")
    manip.ctrl_go_to_home()

    path = opts.replay_dir / f"real_{seed}.npz"
    np.savez(
        path,
        seed=seed,
        tee_start=start,
        tee_final=final,
        tcp=np.array(tcp),
        reference=np.array(ref),
        wrench=np.array(wrench),
        tee_seen=np.array(tee_seen),
        period=np.array(period),
    )
    d_sim, dyaw_sim = pose_error(final, rec["tee"][-1])
    seen = np.mean(~np.isnan(np.array(tee_seen)[:, 0]))
    logger.info(
        f"seed {seed}: real final {np.round(final[:2], 3)} m / {math.degrees(final[2]):.0f} deg; "
        f"vs the recorded sim final {d_sim * 100:.1f} cm / {dyaw_sim:.0f} deg (placement error "
        f"included, see the compare phase); tee seen in {seen * 100:.0f} % of the steps; "
        f"period {np.median(period) * 1e3:.1f} ms (max {np.max(period) * 1e3:.0f}) -> {path}"
    )


def compare(env, cfg: ExpConfig, device: th.device, opts) -> None:
    """
    Env 0 starts at the measured tee start, the others at it plus Gaussian noise of the tracker
    error (`--start-noise-mm`, `--start-noise-deg`). Open-loop pushing is chaotic (a 3 mm / 2 deg
    start offset moved the sim final pose by up to 13 cm / 34 deg), so one real run is judged
    against the spread of the ensemble, not against a single simulation.
    """
    if opts.no_tee:
        compare_free(env, device, opts)
        return
    runs = sorted(opts.replay_dir.glob("real_[0-9]*.npz"))
    if not runs:
        raise FileNotFoundError(f"No real_<seed>.npz in {opts.replay_dir}")
    n_envs = env.num_envs
    envs_idx = th.arange(n_envs, device=device)
    gen = np.random.default_rng(0)
    noise = np.c_[
        gen.normal(0.0, opts.start_noise_mm * 1e-3, (n_envs, 2)),
        np.radians(gen.normal(0.0, opts.start_noise_deg, n_envs)),
    ]
    noise[0] = 0.0
    rows = []
    for run in runs:
        real = np.load(run)
        seed = int(real["seed"])
        rec = np.load(opts.replay_dir / f"sim_{seed}.npz")
        with th.no_grad(), env.nominal_domain():
            env.reset_seeded(envs_idx=envs_idx, seeds=[seed] * n_envs)
            z = float(env.object.get_pose()[0, 2])
            starts = th.cat(
                [planar_to_pose(real["tee_start"] + n, z, device) for n in noise]
            )
            env.object.set_pose(pose=starts, envs_idx=envs_idx)
            tcp = []
            for a in rec["actions"]:
                action = th.as_tensor(a, dtype=th.float32, device=device)
                env.step(action.expand(n_envs, -1))
                tcp.append(env.manipulator.get_tcp_pose()[0, :3].cpu().numpy())
            sim_finals = planar(env.object.get_pose())
        tcp = np.array(tcp)
        n = min(len(tcp), len(real["tcp"]))
        tcp_rms = np.sqrt(
            np.mean(np.sum((tcp[:n, :2] - real["tcp"][:n, :2]) ** 2, axis=1))
        )
        nominal = sim_finals[0]
        d, dyaw = pose_error(real["tee_final"], nominal)
        spread = np.array([pose_error(f, nominal)[0] for f in sim_finals[1:]])
        to_real = np.array([pose_error(f, real["tee_final"])[0] for f in sim_finals])
        # share of the ensemble members farther from the nominal than the real run
        rank = np.mean(spread >= d) if len(spread) else np.nan
        rows.append(
            {
                "seed": seed,
                "d": d,
                "dyaw": dyaw,
                "spread": np.median(spread) if len(spread) else np.nan,
                "nearest": to_real.min(),
                "rank": rank,
                "real_move": pose_error(real["tee_final"], real["tee_start"])[0],
                "sim_move": pose_error(nominal, real["tee_start"])[0],
                "goal_real": pose_error(real["tee_final"], rec["goal"])[0],
                "goal_sim": pose_error(nominal, rec["goal"])[0],
                "tcp_rms": tcp_rms,
            }
        )
    print(
        f"\nensemble of {n_envs} (start noise {opts.start_noise_mm} mm, "
        f"{opts.start_noise_deg} deg); distances in cm\n"
        "seed | real-sim final pos yaw[deg] | ensemble spread | nearest member | "
        "P(member >= real) | travel real/sim | to goal real/sim | TCP RMS [mm]"
    )
    for r in rows:
        print(
            f"{r['seed']:4d} | {r['d'] * 100:12.1f} {r['dyaw']:8.1f} | {r['spread'] * 100:15.1f} | "
            f"{r['nearest'] * 100:14.1f} | {r['rank']:17.2f} | "
            f"{r['real_move'] * 100:6.1f} / {r['sim_move'] * 100:4.1f} | "
            f"{r['goal_real'] * 100:7.1f} / {r['goal_sim'] * 100:4.1f} | {r['tcp_rms'] * 1e3:7.1f}"
        )
    print(
        "P(member >= real) near 0 in most runs: the real outcome is outside what the start "
        "uncertainty explains, i.e. a parameter gap; spread around 0.5: consistent."
    )


def tracking_lag_s(tcp: np.ndarray, ref: np.ndarray, dt: float) -> float:
    """Median along-path distance of the TCP behind the reference over its speed, while moving."""
    vel = np.gradient(ref[:, :2], dt, axis=0)
    speed = np.linalg.norm(vel, axis=1)
    moving = speed > LAG_MIN_SPEED
    if not moving.any():
        return np.nan
    behind = np.sum((ref[:, :2] - tcp[:, :2]) * vel, axis=1) / speed
    return float(np.median(behind[moving] / speed[moving]))


def compare_free(env, device: th.device, opts) -> None:
    """
    Every real_free_<seed>.npz against -B simulated members of the same free-space run (they
    differ only by the simulated F/T noise driving the admittance). The tee is parked on the
    floor and the off-table termination switched off, so nothing resets the episode.
    """
    runs = sorted(opts.replay_dir.glob("real_free_*.npz"))
    if not runs:
        raise FileNotFoundError(f"No real_free_<seed>.npz in {opts.replay_dir}")
    n_envs = env.num_envs
    envs_idx = th.arange(n_envs, device=device)
    env._is_tee_off_table = lambda: th.zeros(n_envs, dtype=th.bool, device=device)
    dt = env.get_policy_dt()
    rows = []
    for run in runs:
        real = np.load(run)
        seed = int(real["seed"])
        rec = np.load(opts.replay_dir / f"sim_{seed}.npz")
        tcp, ref, wrench = [], [], []
        with th.no_grad(), env.nominal_domain():
            env.reset_seeded(envs_idx=envs_idx, seeds=[seed] * n_envs)
            z = float(env.object.get_pose()[0, 2])
            parked = planar_to_pose(np.array([*PARKED_TEE_XY, 0.0]), z, device)
            env.object.set_pose(pose=parked.expand(n_envs, -1), envs_idx=envs_idx)
            tcp_start = env.manipulator.get_tcp_pose()[0, :3].cpu().numpy()
            for a in rec["actions"]:
                action = th.as_tensor(a, dtype=th.float32, device=device)
                env.step(action.expand(n_envs, -1))
                tcp.append(env.manipulator.get_tcp_pose()[:, :3].cpu().numpy())
                ref.append(env.manipulator.get_tcp_reference()[:, :3].cpu().numpy())
                wrench.append(env.manipulator.get_ft_wrench().cpu().numpy())
        # [members, steps, dims]
        tcp, ref, wrench = (np.stack(x, axis=1) for x in (tcp, ref, wrench))
        n = min(tcp.shape[1], len(real["tcp"]))
        r_tcp, r_ref, r_f = real["tcp"][:n], real["reference"][:n], real["wrench"][:n]
        tcp, ref, wrench = tcp[:, :n], ref[:, :n], wrench[:, :n]
        np.savez(
            opts.replay_dir / f"free_compare_{seed}.npz",
            seed=seed,
            sim_tcp=tcp,
            sim_reference=ref,
            sim_wrench=wrench,
            sim_tcp_start=tcp_start,
            **{f"real_{k}": real[k] for k in real.files if k != "seed"},
        )
        mean_tcp = tcp.mean(axis=0)
        f_real = np.hypot(r_f[:, 0], r_f[:, 1])
        f_sim = np.hypot(wrench[..., 0], wrench[..., 1])
        sim_lags = [tracking_lag_s(tcp[m], ref[m], dt) for m in range(n_envs)]
        rows.append(
            {
                "seed": seed,
                "start": np.linalg.norm(real["tcp_start"] - tcp_start),
                "ref": rms_xy(r_ref - ref.mean(axis=0)),
                "tcp": rms_xy(r_tcp - mean_tcp),
                "tcp_max": np.max(
                    np.linalg.norm(r_tcp[:, :2] - mean_tcp[:, :2], axis=1)
                ),
                "spread": rms_xy(tcp - mean_tcp),
                "track_real": rms_xy(r_tcp - r_ref),
                "track_sim": np.mean([rms_xy(tcp[m] - ref[m]) for m in range(n_envs)]),
                "lag_real": tracking_lag_s(r_tcp, r_ref, dt),
                "lag_sim": np.nanmedian(sim_lags),
                "f_peak_real": f_real.max(),
                "f_peak_sim": np.median(f_sim.max(axis=1)),
                "f_rms_real": np.sqrt(np.mean(f_real**2)),
                "f_rms_sim": np.median(np.sqrt(np.mean(f_sim**2, axis=1))),
                "z_real": np.ptp(r_tcp[:, 2]),
                "z_sim": np.median(np.ptp(tcp[..., 2], axis=1)),
                "period": np.median(real["period"][1:]),
                "period_max": np.max(real["period"][1:]),
            }
        )
    print(
        f"\nfree space, {n_envs} sim members; positions in mm, lag in ms, forces in N\n"
        "seed | start | ref real-sim | TCP real-sim rms/max | sim spread | track rms real/sim "
        "| lag real/sim | |F_xy| peak real/sim | rms real/sim | z range real/sim | period/max"
    )
    for r in rows:
        print(
            f"{r['seed']:4d} | {r['start'] * 1e3:5.1f} | {r['ref'] * 1e3:12.1f} | "
            f"{r['tcp'] * 1e3:9.1f} / {r['tcp_max'] * 1e3:5.1f} | {r['spread'] * 1e3:10.1f} | "
            f"{r['track_real'] * 1e3:8.1f} / {r['track_sim'] * 1e3:4.1f} | "
            f"{r['lag_real'] * 1e3:5.0f} / {r['lag_sim'] * 1e3:3.0f} | "
            f"{r['f_peak_real']:10.1f} / {r['f_peak_sim']:4.1f} | "
            f"{r['f_rms_real']:4.1f} / {r['f_rms_sim']:3.1f} | "
            f"{r['z_real'] * 1e3:7.1f} / {r['z_sim'] * 1e3:3.1f} | "
            f"{r['period'] * 1e3:4.1f}/{r['period_max'] * 1e3:3.0f}"
        )


def rms_xy(d: np.ndarray) -> float:
    return float(np.sqrt(np.mean(np.sum(d[..., :2] ** 2, axis=-1))))


def main() -> None:
    setup_logger("INFO")
    th.set_default_dtype(th.float32)
    args = parse_arguments(extra_argparser=extra_args)
    opts = args.get_raw_args()
    env, cfg, device = setup(args)
    {"record": record, "replay": replay, "compare": compare}[opts.phase](
        env, cfg, device, opts
    )


if __name__ == "__main__":
    sys.exit(main())
