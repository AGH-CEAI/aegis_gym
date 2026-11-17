"""
Peg drops onto the board with the given scenario, contact settings and clearances, without a robot.

Every environment drops one peg onto one slot; `-B` envs per slot. Reports per clearance how the pegs ended (in
the hole, on the board, fallen over, jammed, off the board, still moving), the deepest peg-board
penetration during the drop, the jitter at rest and the simulation speed. Fails (exit code 1) on physics errors:
penetration over 0.5 mm, pegs still moving, a centred drop not ending in the hole. Every (env, slot) is written to a
CSV in `logs/pih_spike/` (ignored by git). With `--vis` the drop repeats in real time (`--slow` slows it down).

Scenarios: see `aegis_gym.aux.pih_assets.drop_sim` (centred, rim, edge-tilt, random).

Run from the repo root, e.g.:
    python3 -m scripts.peg_in_hole.drop_test --scenario rim --vis --slow 4
    python3 -m scripts.peg_in_hole.drop_test --scenario edge-tilt --vis --slow 4
    python3 -m scripts.peg_in_hole.drop_test --scenario random -B 64 --seed 1
    python3 -m scripts.peg_in_hole.drop_test --scenario random -B 64 --rigid-contact '{"box_box_detection": true}'
"""

import argparse
import csv
import json
import os
import sys
import time
from datetime import datetime
from pathlib import Path

# importing `aegis_gym.envs` pulls in ClearML, which would try to log in to the server; nothing here needs it
os.environ.setdefault("CLEARML_OFFLINE_MODE", "1")

import genesis as gs
import numpy as np

from aegis_gym.aux.pih_assets import PihAssets, PihAssetsCfg, generate_pih_assets
from aegis_gym.aux.pih_assets.drop_sim import (
    OUTCOMES,
    SCENARIOS,
    DropCfg,
    DropResult,
    DropSim,
    scenario_starts,
    summarize,
)
from aegis_gym.config.types.env import validate_rigid_contact

HOLE_COLOR = (0.0, 0.8, 0.9, 1.0)
DEFAULT_OUT_DIR = Path("logs/pih_spike")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--scenario", choices=SCENARIOS, default="random")
    p.add_argument(
        "-B", "--num-envs", type=int, default=1, help="parallel environments per slot"
    )
    p.add_argument("--settle", type=float, default=3.0, help="[s] drop duration")
    p.add_argument(
        "--ctrl-dt",
        type=float,
        default=0.004,
        help="[s] rigid solver step (env.ctrl_dt)",
    )
    p.add_argument(
        "--rigid-contact",
        type=json.loads,
        default=None,
        help="JSON, like env.rigid_contact",
    )
    p.add_argument(
        "--overrides",
        type=json.loads,
        default=None,
        help="JSON merged into nominal.yaml",
    )
    p.add_argument("--seed", type=int, default=0, help="random scenario")
    p.add_argument(
        "--offset-max",
        type=float,
        default=None,
        help="[mm] random scenario (default: clearance/2 + 2)",
    )
    p.add_argument("--tilt-max", type=float, default=5.0, help="[deg] random scenario")
    p.add_argument(
        "--tag", default=None, help="name of the CSV (default: the scenario)"
    )
    p.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    p.add_argument("--no-csv", action="store_true")
    p.add_argument("--vis", action="store_true", help="viewer, the drop repeats")
    p.add_argument(
        "--slow",
        type=float,
        default=1.0,
        help="viewer slow-motion factor (1 = real time)",
    )
    p.add_argument("--cpu", action="store_true", help="Genesis CPU backend")
    args = p.parse_args(argv)
    validate_rigid_contact(args.rigid_contact)
    return args


def print_report(result: DropResult, verbose: bool) -> None:
    if verbose:
        print(
            "  env slot clear  scenario    outcome    penetration      low-floor  tilt   xy-off  settle  jitter"
        )
        print(
            "            [mm]                         impact  rest     [mm]       [deg]  [mm]    [s]     [mm/s]"
        )
        for r in result.rows:
            print(
                f"  {r['env']:>3} {r['slot']:>4} {r['clearance_mm']:5.2f}  {r['scenario']:<10}  {r['outcome']:<9}"
                f" {r['transient_penetration_mm']:7.3f} {r['rest_penetration_mm']:6.3f}"
                f"  {r['lowest_above_floor_mm']:+8.2f}  {r['final_tilt_deg']:5.1f}"
                f"  {r['final_xy_offset_mm']:6.2f}  {r['settle_time_s']:5.2f}  {r['jitter_mm_s']:6.2f}"
            )
    print(
        "  clearance   n  "
        + "  ".join(f"{o:>8}" for o in OUTCOMES)
        + "  penetr. rest max  impact max  p95 [mm]  jitter max [mm/s]"
    )
    for s in summarize(result):
        print(
            f"  {s['clearance_mm']:7.2f} mm {s['n']:>3}  "
            + "  ".join(f"{s[o]:>8}" for o in OUTCOMES)
            + f"  {s['rest_penetration_max_mm']:16.3f}  {s['transient_penetration_max_mm']:10.3f}"
            + f"  {s['transient_penetration_p95_mm']:8.3f}  {s['jitter_max_mm_s']:17.2f}"
        )
    print(
        f"  {result.n_envs} envs x {result.steps} steps: {result.fps:,.0f} env-steps/s in scene.step(), "
        f"{result.total_seconds:.1f} s with the metrics"
    )


def write_csv(result: DropResult, args: argparse.Namespace, assets: PihAssets) -> Path:
    args.out_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    path = args.out_dir / f"drop_{args.tag or args.scenario}_{stamp}.csv"
    run = {
        "tag": args.tag or args.scenario,
        "ctrl_dt": args.ctrl_dt,
        "rigid_contact": json.dumps(args.rigid_contact or {}, sort_keys=True),
        "assets": assets.directory.name,
        "seed": args.seed,
        "fps": round(result.fps),
    }
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=[*run, *result.rows[0]])
        writer.writeheader()
        for r in result.rows:
            writer.writerow(
                {
                    **run,
                    **{
                        k: round(v, 6) if isinstance(v, float) else v
                        for k, v in r.items()
                    },
                }
            )
    return path


def draw_holes(sim: DropSim) -> None:
    top, depth = sim.top, sim.assets.cfg.board.depth
    for slot in sim.assets.geometry.slots:
        for h in slot.holes:
            sim.scene.draw_debug_box(
                [[h.x0, h.y0, top - depth], [h.x1, h.y1, top]],
                color=HOLE_COLOR,
                wireframe=True,
                wireframe_radius=0.0003,
            )


def main() -> int:
    args = parse_args()
    assets = generate_pih_assets(PihAssetsCfg.from_yaml(overrides=args.overrides))
    offset_max = None if args.offset_max is None else args.offset_max / 1e3
    starts = scenario_starts(
        assets, args.scenario, args.num_envs, args.seed, offset_max, args.tilt_max
    )
    cfg = DropCfg(
        ctrl_dt=args.ctrl_dt, settle=args.settle, rigid_contact=args.rigid_contact
    )
    print(f"assets: {assets.directory}")
    print(
        f"scenario {args.scenario}, {args.num_envs} envs per slot, ctrl_dt {args.ctrl_dt * 1e3:g} ms, settle {args.settle:g} s"
    )
    if args.rigid_contact:
        print(f"RigidOptions overrides: {args.rigid_contact}")

    gs.init(
        backend=gs.cpu if args.cpu else gs.gpu, precision="32", logging_level="warning"
    )
    sim = DropSim(assets, cfg, starts, show_viewer=args.vis)

    result = sim.run()
    print_report(result, verbose=args.num_envs <= 2)
    errors = result.failures()
    print("FAILED:\n  " + "\n  ".join(errors) if errors else "OK")
    if not args.no_csv:
        print(f"csv: {write_csv(result, args, assets)}")

    if args.vis:
        draw_holes(sim)
        print("viewer open, the drop repeats; close the window or Ctrl+C to exit")
        try:
            while sim.scene.viewer.is_alive():
                sim.reset()
                for _ in range(round(args.settle / args.ctrl_dt)):
                    if not sim.scene.viewer.is_alive():
                        break
                    t0 = time.perf_counter()
                    sim.step()
                    time.sleep(
                        max(0.0, args.ctrl_dt * args.slow - (time.perf_counter() - t0))
                    )
                else:
                    print_report(sim.result(), verbose=args.num_envs <= 2)
        except KeyboardInterrupt:
            pass
    return 1 if errors else 0


if __name__ == "__main__":
    np.set_printoptions(suppress=True)
    sys.exit(main())
