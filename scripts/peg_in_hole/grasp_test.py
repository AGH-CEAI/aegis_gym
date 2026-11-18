"""
Peg-in-hole robot scene: the arm holds the peg between the Hand-E fingers over the table, with the board standing on
it, and moves the peg through the air to check that the grasp holds.

The scene is the env's (`GenesisScene` from the default config, robot from the cached ClearML URDF dataset, no
server) plus `aux.pih_assets.scene`: the board and the peg placed from `PihSceneCfg` (`--scene`). After the reset
(`grasp_peg`) the TCP holds still, then translates along and rotates about every axis at the env's maximum speed and
at twice it. Reports per phase the slip of the peg in the hand and the gripper width. Fails (exit code 1) when the
peg slips more than 1 mm (plus 0.15 mm/s of creep with the physical grasp) or 2 deg. Contact: `impratio` 10 unless
`--rigid-contact` says otherwise ('{}': the Genesis defaults). With `--vis` the sequence repeats (`--slow` slows it
down).

Run from the repo root in the toolbox, e.g.:
    python3 -m scripts.peg_in_hole.grasp_test --vis --slow 2
    python3 -m scripts.peg_in_hole.grasp_test -B 16
    python3 -m scripts.peg_in_hole.grasp_test -B 16 --scene '{"grasp": "weld"}'
    python3 -m scripts.peg_in_hole.grasp_test --vis --scene '{"board_xy": [0.5, 0.1], "board_yaw_deg": 30}'
"""

import argparse
import json
import math
import os
import sys
import time
from dataclasses import dataclass
from pathlib import Path

# importing `aegis_gym.envs` pulls in ClearML, which would try to log in to the server; nothing here needs it
os.environ.setdefault("CLEARML_OFFLINE_MODE", "1")

import genesis as gs
import torch as th

from aegis_gym.aux.pih_assets.scene import (
    PihScene,
    PihSceneCfg,
    add_pih_entities,
    grasp_peg,
    grasp_slip,
)
from aegis_gym.config import ConfigManager
from aegis_gym.config.types.env import validate_rigid_contact
from aegis_gym.envs.manipulator.sim.genesis_manipulator import GenesisManipulator
from aegis_gym.envs.scene import GenesisScene

# a physical grasp creeps in Genesis (~0.08 mm/s with impratio 10, see docs/PEG_IN_HOLE.md); a welded one can't
SLIP_LIMIT_M = 1e-3
DRIFT_LIMIT_M_S = {"physical": 0.15e-3, "weld": 0.0}
SLIP_LIMIT_RAD = math.radians(2.0)
# the recommended contact for the robot scene: less grasp creep, the drop test unchanged
DEFAULT_RIGID_CONTACT = '{"impratio": 10.0}'
DROPPED_M = 0.02  # a slip this large means the peg left the hand
CLEARML_DATASETS = Path("~/.clearml/cache/storage_manager/datasets").expanduser()
HOLE_COLOR = (0.0, 0.8, 0.9, 1.0)


@dataclass(frozen=True)
class Phase:
    name: str
    seconds: float
    twist: tuple[float, ...] = (
        0.0,
    ) * 6  # TCP twist in units of the max speeds (vx, vy, vz, wx, wy, wz)


def phases(speed: float) -> list[Phase]:
    """Moves out and back, so the TCP ends where it started; tilts of ~10 deg at the max angular speed (0.1 rad/s)."""
    out = []
    for axis, name in enumerate(("x", "y", "z")):
        twist = [0.0] * 6
        twist[axis] = speed
        out += [Phase(f"+{name} x{speed:g}", 1.0, tuple(twist))]
        out += [Phase(f"-{name} x{speed:g}", 1.0, tuple(-v for v in twist))]
    for axis, name in ((3, "roll"), (4, "pitch"), (5, "yaw")):
        twist = [0.0] * 6
        twist[axis] = speed
        seconds = math.radians(10.0) / (0.1 * speed)
        out += [Phase(f"+{name} x{speed:g}", seconds, tuple(twist))]
        out += [Phase(f"-{name} x{speed:g}", seconds, tuple(-v for v in twist))]
    return out


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("-B", "--num-envs", type=int, default=1)
    p.add_argument(
        "--scene", type=json.loads, default=None, help="JSON, PihSceneCfg keys"
    )
    p.add_argument(
        "--overrides",
        type=json.loads,
        default=None,
        help="JSON merged into nominal.yaml (same as the scene's 'assets')",
    )
    p.add_argument(
        "--rigid-contact",
        type=json.loads,
        default=DEFAULT_RIGID_CONTACT,
        help="JSON, like env.rigid_contact; {} for the Genesis defaults",
    )
    p.add_argument("--hold", type=float, default=1.0, help="[s] still, after the reset")
    p.add_argument(
        "--speeds",
        type=float,
        nargs="*",
        default=[1.0, 2.0],
        help="move speeds, in units of the env's max TCP speeds; none: only hold",
    )
    p.add_argument(
        "--urdf", type=Path, default=None, help="robot URDF (default: ClearML cache)"
    )
    p.add_argument("--vis", action="store_true", help="viewer, the sequence repeats")
    p.add_argument("--slow", type=float, default=1.0, help="viewer slow-motion factor")
    args = p.parse_args(argv)
    validate_rigid_contact(args.rigid_contact)
    return args


def use_cached_urdf(path: Path | None) -> None:
    """Resolves the robot URDF from `path` or the local ClearML dataset cache, without the server."""
    resolve = GenesisManipulator._resolve_aegis_urdf

    def cached(self: GenesisManipulator) -> Path:
        if path is not None:
            return path
        found = sorted((CLEARML_DATASETS / f"ds_{self._urdf_model_id}").rglob("*.urdf"))
        return found[0] if len(found) == 1 else resolve(self)

    GenesisManipulator._resolve_aegis_urdf = cached


def build(args: argparse.Namespace) -> tuple[GenesisScene, PihScene]:
    argv = [
        "grasp_test",
        "--env",
        "reacher",
        "-B",
        str(args.num_envs),
        "--disable-cameras",
    ]
    ConfigManager.setup_config(
        argv=argv + (["--vis"] if args.vis else []),
        device=gs.device,
        task=None,
    )
    cfg = ConfigManager.get_config()
    cfg.env_cfg.rigid_contact = args.rigid_contact
    table_top_z = cfg.env_cfg.table_size[2] - cfg.env_cfg.workbench_size[2]

    scene_dict = dict(args.scene or {})
    if args.overrides:
        scene_dict["assets"] = args.overrides
    scene = GenesisScene(cfg=cfg, device=gs.device)
    scene.add_manipulator(cfg=cfg.robot_cfg)
    pih = add_pih_entities(scene, PihSceneCfg.from_dict(scene_dict), table_top_z)
    scene.build()
    return scene, pih


def draw_holes(scene: GenesisScene, pih: PihScene) -> None:
    side = pih.assets.cfg.peg.side
    for pose, frame in zip(
        pih.hole_poses.tolist(), pih.assets.hole_frames, strict=True
    ):
        half = (side + frame.clearance) / 2
        corners = [(-half, -half), (half, -half), (half, half), (-half, half)]
        yaw = math.radians(pih.cfg.board_yaw_deg)
        c, s = math.cos(yaw), math.sin(yaw)
        pts = [
            (pose[0] + c * x - s * y, pose[1] + s * x + c * y, pose[2] + 5e-4)
            for x, y in corners
        ]
        for a, b in zip(pts, pts[1:] + pts[:1], strict=True):
            scene.gs_scene.draw_debug_line(a, b, radius=4e-4, color=HOLE_COLOR)


def run(
    scene: GenesisScene, pih: PihScene, args: argparse.Namespace, realtime: bool
) -> list[dict]:
    manipulator = scene.get_manipulator()
    n = scene.num_envs
    envs_idx = th.arange(n, device=gs.device)
    grasp_peg(manipulator, pih, envs_idx)
    dt = scene.get_policy_dt()
    t = 0.0  # [s] since the reset

    rows = []
    sequence = [Phase("hold", args.hold)] + [p for s in args.speeds for p in phases(s)]
    sequence.append(Phase("hold", args.hold))
    for phase in sequence:
        cpu = th.device("cpu")  # `gs.init()` made CUDA the default device
        slip_m, slip_rad = th.zeros(n, device=cpu), th.zeros(n, device=cpu)
        width_min = th.full((n,), math.inf, device=cpu)
        width_max = th.zeros(n, device=cpu)
        for _ in range(max(1, round(phase.seconds / dt))):
            if realtime and not scene.gs_scene.viewer.is_alive():
                return rows
            t0 = time.perf_counter()
            action = th.tensor(phase.twist, device=gs.device).repeat(n, 1)
            scene.pre_step()
            manipulator.ctrl_apply_vel_action(action, open_gripper=False)
            scene.step()
            t += dt
            slip = grasp_slip(manipulator, pih).cpu()
            slip_m = th.maximum(slip_m, slip[:, 0])
            slip_rad = th.maximum(slip_rad, slip[:, 1])
            width = manipulator.get_gripper_width().cpu()
            width_min, width_max = (
                th.minimum(width_min, width),
                th.maximum(width_max, width),
            )
            if realtime:
                time.sleep(max(0.0, dt * args.slow - (time.perf_counter() - t0)))
        rows.append(
            {
                "phase": phase.name,
                "slip_max_mm": slip_m.max().item() * 1e3,
                "slip_max_deg": math.degrees(slip_rad.max().item()),
                "t_end": t,
                "dropped": int((slip_m > DROPPED_M).sum()),
                "width_min_mm": width_min.min().item() * 1e3,
                "width_max_mm": width_max.max().item() * 1e3,
            }
        )
    return rows


def print_report(rows: list[dict], pih: PihScene, n_envs: int) -> list[str]:
    print(
        f"peg {pih.assets.cfg.peg.side * 1e3:g} mm, grasp width {pih.grasp_width * 1e3:g} mm, {pih.cfg.grasp} grasp, "
        f"{n_envs} envs; "
        f"slip of the peg w.r.t. its pose in the hand, max over the envs"
    )
    print(f"{'phase':<14}{'slip mm':>9}{'slip deg':>10}{'dropped':>9}{'width mm':>16}")
    for r in rows:
        print(
            f"{r['phase']:<14}{r['slip_max_mm']:9.3f}{r['slip_max_deg']:10.3f}{r['dropped']:9d}"
            f"{r['width_min_mm']:9.2f}-{r['width_max_mm']:.2f}"
        )
    errors = []
    drift = DRIFT_LIMIT_M_S[pih.cfg.grasp]
    for r in rows:
        limit_mm = (SLIP_LIMIT_M + drift * r["t_end"]) * 1e3
        if r["dropped"]:
            errors.append(f"{r['phase']}: {r['dropped']} peg(s) left the hand")
        elif r["slip_max_mm"] > limit_mm:
            errors.append(
                f"{r['phase']}: slip {r['slip_max_mm']:.2f} mm > {limit_mm:.2f} mm at {r['t_end']:.0f} s"
            )
        elif r["slip_max_deg"] > math.degrees(SLIP_LIMIT_RAD):
            errors.append(f"{r['phase']}: slip {r['slip_max_deg']:.2f} deg")
    return errors


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    use_cached_urdf(args.urdf)
    gs.init(backend=gs.gpu, precision="32", logging_level="warning")
    scene, pih = build(args)
    print(f"assets: {pih.assets.directory}")
    print(f"board top at {pih.board_pose[:3]}, TCP start at {pih.tcp_start_pose[:3]}")
    if args.rigid_contact:
        print(f"RigidOptions overrides: {args.rigid_contact}")

    errors = print_report(run(scene, pih, args, realtime=False), pih, scene.num_envs)
    print("FAILED:\n  " + "\n  ".join(errors) if errors else "OK")

    if args.vis:
        draw_holes(scene, pih)
        print("viewer open, the sequence repeats; close the window or Ctrl+C to exit")
        try:
            while scene.gs_scene.viewer.is_alive():
                rows = run(scene, pih, args, realtime=True)
                if rows and len(rows) > 1:
                    print_report(rows, pih, scene.num_envs)
        except KeyboardInterrupt:
            pass
    return 1 if errors else 0


if __name__ == "__main__":
    sys.exit(main())
