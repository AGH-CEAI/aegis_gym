"""
Scripted peg insertion in the peg-in-hole robot scene: the arm holds the peg welded to the hand, brings its tip over
a hole (centred or offset), and pushes it straight down until the peg is seated or blocked.

Every env aims at one slot with one xy offset of the target from the hole centre (`--offsets-mm`, in a seeded
random direction), `-B` envs per (slot, offset). The TCP is driven by P controllers on its position and orientation
(the env's velocity action, capped at the max speeds):
    approach: the peg tip 10 mm above the target; settle: still for 0.5 s; descend: down at `--insert-speed` with
    the xy held on the target (no search, no compliance).
An env ends `inserted` when the tip is within 1 mm of the hole floor, `blocked` when the peg-board contact force
exceeds `--force-max`, `timeout` otherwise. The contact force is read from the simulator: the wrist F/T model sums
the contacts of the robot links, so it does not see the forces passed through the weld.

Without a chamfer a peg goes in when its offset along each board axis is below half the clearance (2.5 / 0.5 /
0.25 mm; the holes are square, so a diagonal offset fits further) and rests on the rim otherwise. Fails (exit code 1)
on physics errors: a centred peg not inserted, a peg inserted with an axis offset over half the clearance (+0.1 mm)
further than the rim pushed it aside in the hand (the weld is a soft constraint, ~0.1-0.2 mm/N), a peg-board
penetration over 0.5 mm, or a timeout.

Run from the repo root in the toolbox, e.g.:
    python3 -m scripts.peg_in_hole.insert_test --vis --slow 2 --offsets-mm 0 1 3
    python3 -m scripts.peg_in_hole.insert_test -B 4
"""

import argparse
import csv
import json
import math
import os
import sys
import time
from datetime import datetime
from pathlib import Path

# importing `aegis_gym.envs` pulls in ClearML, which would try to log in to the server; nothing here needs it
os.environ.setdefault("CLEARML_OFFLINE_MODE", "1")

import genesis as gs
import numpy as np
import torch as th

from aegis_gym.aux.geom.geom import quat_to_rotvec_error
from aegis_gym.aux.pih_assets import PihAssetsCfg
from aegis_gym.aux.pih_assets.metrics import max_penetration_batch
from aegis_gym.aux.pih_assets.scene import PihScene, grasp_peg, grasp_slip
from aegis_gym.config.types.env import validate_rigid_contact
from aegis_gym.envs.scene import GenesisScene

from .grasp_test import DEFAULT_RIGID_CONTACT, build, draw_holes, use_cached_urdf

OUTCOMES = ("inserted", "blocked", "timeout")
PENETRATION_LIMIT_M = 5e-4
SEAT_TOL_M = 1e-3  # the tip this close to the hole floor counts as seated
HOVER_M = 0.01  # the tip above the board top before the descent
APPROACH_TOL_M = 2e-4  # the TCP this close to the approach target ends the approach
SETTLE_S = 0.5
POS_GAIN = 4.0  # [1/s] P gain of the TCP position
ROT_GAIN = 5.0  # [1/s] P gain of the TCP orientation, as in Push-T
OFFSET_MARGIN_M = (
    1e-4  # an insertion with the offset over clearance/2 + this is a physics error
)
DEFAULT_OUT_DIR = Path("logs/pih_spike")
APPROACH, SETTLE, DESCEND, DONE = range(4)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("-B", "--num-envs", type=int, default=1, help="per (slot, offset)")
    p.add_argument(
        "--offsets-mm",
        type=float,
        nargs="+",
        default=[0.0, 0.1, 0.2, 0.5, 1.0, 2.0, 3.0],
        help="xy offsets of the target from the hole centre",
    )
    p.add_argument("--seed", type=int, default=0, help="offset directions")
    p.add_argument(
        "--insert-speed", type=float, default=0.01, help="[m/s] descent speed"
    )
    p.add_argument(
        "--force-max",
        type=float,
        default=20.0,
        help="[N] peg-board contact force that ends the descent as blocked",
    )
    p.add_argument("--timeout", type=float, default=15.0, help="[s] per run")
    p.add_argument(
        "--scene", type=json.loads, default=None, help="JSON, PihSceneCfg keys"
    )
    p.add_argument(
        "--overrides",
        type=json.loads,
        default=None,
        help="JSON merged into nominal.yaml",
    )
    p.add_argument(
        "--rigid-contact",
        type=json.loads,
        default=DEFAULT_RIGID_CONTACT,
        help="JSON, like env.rigid_contact; {} for the Genesis defaults",
    )
    p.add_argument(
        "--urdf", type=Path, default=None, help="robot URDF (default: ClearML cache)"
    )
    p.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    p.add_argument("--no-csv", action="store_true")
    p.add_argument("--vis", action="store_true", help="viewer, the insertion repeats")
    p.add_argument("--slow", type=float, default=1.0, help="viewer slow-motion factor")
    args = p.parse_args(argv)
    validate_rigid_contact(args.rigid_contact)
    return args


class Insertion:
    """The cases of one run (env -> slot, offset) and the scripted insertion over all envs."""

    def __init__(self, scene: GenesisScene, pih: PihScene, args: argparse.Namespace):
        self.scene, self.pih, self.args = scene, pih, args
        self.manipulator = scene.get_manipulator()
        self.n = scene.num_envs
        self.dev = gs.device
        # the cases on the CPU: `gs.init()` made CUDA the default device
        with th.device("cpu"):
            n_slots, offsets = len(pih.hole_poses), args.offsets_mm
            cases = [
                (slot, off)
                for slot in range(n_slots)
                for off in offsets
                for _ in range(args.num_envs)
            ]
            assert len(cases) == self.n
            self.slot = th.tensor([c[0] for c in cases])
            self.offset_m = th.tensor([c[1] * 1e-3 for c in cases], dtype=th.float32)
            angle = th.rand(self.n, generator=th.Generator().manual_seed(args.seed))
            angle = angle * 2 * math.pi
            offset_xy = self.offset_m[:, None] * th.stack(
                [th.cos(angle), th.sin(angle)], 1
            )
            # the holes are square: what counts is the offset along each board axis
            yaw = math.radians(pih.cfg.board_yaw_deg)
            along = offset_xy[:, 0] * math.cos(yaw) + offset_xy[:, 1] * math.sin(yaw)
            across = -offset_xy[:, 0] * math.sin(yaw) + offset_xy[:, 1] * math.cos(yaw)
            self.offset_axis_m = th.maximum(along.abs(), across.abs())

            holes = pih.hole_poses.cpu()
            self.clearance = th.tensor(
                [pih.assets.hole_frames[s].clearance for s in self.slot.tolist()]
            )
            board = pih.assets.cfg.board
            self.floor_z = holes[self.slot, 2] - (board.thickness - board.bottom)
            target_tip = th.zeros(self.n, 3)
            target_tip[:, :2] = holes[self.slot, :2] + offset_xy
            target_tip[:, 2] = holes[self.slot, 2] + HOVER_M
        self.target_tip = target_tip.to(self.dev)
        self.floor_z = self.floor_z.to(self.dev)

    def _tip(self) -> th.Tensor:
        return self.pih.peg.get_pose()[:, :3]

    def _contact_force(self) -> th.Tensor:
        """(n,) [N] norm of the total board force on the peg."""
        contacts = self.pih.peg._obj.get_contacts(
            with_entity=self.pih.board._obj, is_padded=True
        )
        valid = contacts["valid_mask"]
        if valid.numel() == 0:
            return th.zeros(self.n, device=self.dev)
        peg_links = th.tensor([self.pih.peg._obj.base_link.idx], device=valid.device)
        on_a = th.isin(contacts["link_a"], peg_links)
        force = th.where(on_a.unsqueeze(-1), contacts["force_a"], contacts["force_b"])
        force = th.where(valid.unsqueeze(-1), force, th.zeros_like(force))
        return th.linalg.vector_norm(force.sum(dim=1), dim=1)

    def run(self, realtime: bool = False) -> list[dict]:
        m, pih, args = self.manipulator, self.pih, self.args
        all_envs = th.arange(self.n, device=self.dev)
        grasp_peg(m, pih, all_envs)
        dt = self.scene.get_policy_dt()
        hold_quat = th.tensor(pih.tcp_start_pose[3:], device=self.dev).repeat(self.n, 1)
        # the weld keeps the tip fixed in the hand: aim the TCP by the tip
        tcp_to_tip = self._tip() - m.get_tcp_pose()[:, :3]

        phase = th.full((self.n,), APPROACH, device=self.dev)
        outcome = ["timeout"] * self.n
        settle_left = th.full((self.n,), SETTLE_S, device=self.dev)
        t, t_descend = 0.0, th.zeros(self.n, device=self.dev)
        t_end = th.zeros(self.n, device=self.dev)
        force_max = th.zeros(self.n, device=self.dev)
        penetration = np.zeros(self.n)
        slip = th.zeros(self.n, device=self.dev)
        xy_err = th.zeros(self.n, device=self.dev)
        geom = pih.assets.geometry

        while t < args.timeout and bool((phase != DONE).any()):
            if realtime and not self.scene.gs_scene.viewer.is_alive():
                return []
            t0 = time.perf_counter()
            tcp = m.get_tcp_pose()
            target_tcp = self.target_tip - tcp_to_tip
            descending = phase == DESCEND
            # the descent drives the target down at the insertion speed, the xy stays put
            target_tcp[:, 2] = th.where(
                descending,
                target_tcp[:, 2] - args.insert_speed * (t - t_descend),
                target_tcp[:, 2],
            )
            err = target_tcp - tcp[:, :3]
            v = POS_GAIN * err
            # feed-forward of the descent speed, the P term tracks the moving target
            v[:, 2] = th.where(descending, v[:, 2] - args.insert_speed, v[:, 2])
            v = th.where((phase == DONE).unsqueeze(-1), th.zeros_like(v), v)
            twist = th.zeros(self.n, 6, device=self.dev)
            twist[:, :3] = v / m.max_linear_speed
            twist[:, 3:] = (
                ROT_GAIN
                * quat_to_rotvec_error(hold_quat, tcp[:, 3:])
                / m.max_angular_speed
            )
            twist = th.clamp(twist, -1.0, 1.0)

            self.scene.pre_step()
            m.ctrl_apply_vel_action(twist, open_gripper=False)
            self.scene.step()
            t += dt

            poses = pih.peg.get_pose()
            tip = poses[:, :3]
            force = self._contact_force()
            force_max = th.maximum(force_max, force)
            penetration = np.maximum(
                penetration,
                max_penetration_batch(
                    geom.peg_boxes,
                    poses.cpu().numpy(),
                    geom.board_boxes,
                    pih.board_pose,
                ),
            )
            slip = th.maximum(slip, grasp_slip(m, pih)[:, 0])

            # phase transitions
            reached = (phase == APPROACH) & (
                th.linalg.vector_norm(err, dim=1) < APPROACH_TOL_M
            )
            phase = th.where(reached, SETTLE, phase)
            settling = phase == SETTLE
            settle_left = th.where(settling & ~reached, settle_left - dt, settle_left)
            start = settling & (settle_left <= 0)
            xy_err = th.where(
                start,
                th.linalg.vector_norm(tip[:, :2] - self.target_tip[:, :2], dim=1),
                xy_err,
            )
            t_descend = th.where(start, th.full_like(t_descend, t), t_descend)
            phase = th.where(start, DESCEND, phase)

            descending = phase == DESCEND
            seated = descending & (tip[:, 2] <= self.floor_z + SEAT_TOL_M)
            blocked = descending & ~seated & (force > args.force_max)
            for i in th.nonzero(seated | blocked).flatten().tolist():
                outcome[i] = "inserted" if bool(seated[i]) else "blocked"
            t_end = th.where(seated | blocked, th.full_like(t_end, t), t_end)
            phase = th.where(seated | blocked, DONE, phase)

            if realtime:
                time.sleep(max(0.0, dt * args.slow - (time.perf_counter() - t0)))

        tip = self._tip()
        rows = []
        for i in range(self.n):
            rows.append(
                {
                    "env": i,
                    "slot": int(self.slot[i]),
                    "clearance_mm": round(float(self.clearance[i]) * 1e3, 4),
                    "offset_mm": round(float(self.offset_m[i]) * 1e3, 4),
                    "offset_axis_mm": round(float(self.offset_axis_m[i]) * 1e3, 4),
                    "outcome": outcome[i],
                    "insert_time_s": round(float(t_end[i] - t_descend[i]), 3)
                    if outcome[i] != "timeout"
                    else None,
                    "tip_above_floor_mm": round(
                        float(tip[i, 2] - self.floor_z[i]) * 1e3, 3
                    ),
                    "xy_err_at_descent_mm": round(float(xy_err[i]) * 1e3, 3),
                    "force_max_n": round(float(force_max[i]), 2),
                    "penetration_max_mm": round(float(penetration[i]) * 1e3, 3),
                    "grasp_slip_max_mm": round(float(slip[i]) * 1e3, 3),
                }
            )
        return rows


def failures(rows: list[dict]) -> list[str]:
    errors = []
    for r in rows:
        where = f"env {r['env']} (clearance {r['clearance_mm']:g} mm, offset {r['offset_mm']:g} mm)"
        if r["outcome"] == "timeout":
            errors.append(f"{where}: timeout")
        if r["offset_mm"] == 0 and r["outcome"] != "inserted":
            errors.append(f"{where}: a centred peg ended '{r['outcome']}'")
        over = r["offset_axis_mm"] - r["clearance_mm"] / 2
        # beyond it the rim must have pushed the peg aside in the (compliant, welded) hand, or it went through
        pushed = r["grasp_slip_max_mm"]
        if r["outcome"] == "inserted" and over > OFFSET_MARGIN_M * 1e3 + pushed:
            errors.append(
                f"{where}: inserted {over:.2f} mm beyond half the clearance along a board axis"
            )
        if r["penetration_max_mm"] > PENETRATION_LIMIT_M * 1e3:
            errors.append(f"{where}: penetration {r['penetration_max_mm']:.3f} mm")
    return errors


def print_report(rows: list[dict]) -> None:
    print(
        f"{'clearance':>10}{'offset':>8}{'n':>4}"
        + "".join(f"{o:>10}" for o in OUTCOMES)
        + f"{'time s':>8}{'force N':>9}{'pen. mm':>9}{'xy err mm':>11}{'slip mm':>9}"
    )
    keys = sorted(
        {(r["clearance_mm"], r["offset_mm"]) for r in rows}, key=lambda k: (-k[0], k[1])
    )
    for clearance, offset in keys:
        rs = [
            r
            for r in rows
            if r["clearance_mm"] == clearance and r["offset_mm"] == offset
        ]
        times = [r["insert_time_s"] for r in rs if r["outcome"] == "inserted"]
        print(
            f"{clearance:>7.2f} mm{offset:>5.2f} mm{len(rs):>4}"
            + "".join(f"{sum(r['outcome'] == o for r in rs):>10}" for o in OUTCOMES)
            + f"{(sum(times) / len(times) if times else float('nan')):>8.2f}"
            + f"{max(r['force_max_n'] for r in rs):>9.1f}"
            + f"{max(r['penetration_max_mm'] for r in rs):>9.3f}"
            + f"{max(r['xy_err_at_descent_mm'] for r in rs):>11.3f}"
            + f"{max(r['grasp_slip_max_mm'] for r in rs):>9.3f}"
        )


def write_csv(rows: list[dict], args: argparse.Namespace) -> Path:
    args.out_dir.mkdir(parents=True, exist_ok=True)
    path = args.out_dir / f"insert_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
    run = {
        "rigid_contact": json.dumps(args.rigid_contact or {}, sort_keys=True),
        "insert_speed": args.insert_speed,
        "force_max": args.force_max,
        "seed": args.seed,
    }
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=[*run, *rows[0]])
        writer.writeheader()
        writer.writerows({**run, **r} for r in rows)
    return path


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    use_cached_urdf(args.urdf)
    gs.init(backend=gs.gpu, precision="32", logging_level="warning")
    n_slots = len(PihAssetsCfg.from_yaml(overrides=args.overrides).board.clearances)
    scene_args = argparse.Namespace(
        **{
            **vars(args),
            "num_envs": n_slots * len(args.offsets_mm) * args.num_envs,
            "scene": {**(args.scene or {}), "grasp": "weld"},
        }
    )
    scene, pih = build(scene_args)
    print(f"assets: {pih.assets.directory}")
    print(
        f"{scene.num_envs} envs: {n_slots} slots x {len(args.offsets_mm)} offsets x {args.num_envs}, "
        f"weld grasp, descent {args.insert_speed * 1e3:g} mm/s, force limit {args.force_max:g} N"
    )
    if args.rigid_contact:
        print(f"RigidOptions overrides: {args.rigid_contact}")

    insertion = Insertion(scene, pih, args)
    rows = insertion.run()
    print_report(rows)
    errors = failures(rows)
    print("FAILED:\n  " + "\n  ".join(errors) if errors else "OK")
    if not args.no_csv:
        print(f"csv: {write_csv(rows, args)}")

    if args.vis:
        draw_holes(scene, pih)
        print("viewer open, the insertion repeats; close the window or Ctrl+C to exit")
        try:
            while scene.gs_scene.viewer.is_alive():
                again = insertion.run(realtime=True)
                if again:
                    print_report(again)
        except KeyboardInterrupt:
            pass
    return 1 if errors else 0


if __name__ == "__main__":
    sys.exit(main())
