"""
Loads the generated peg-in-hole assets into an empty Genesis scene (no robot), through the same `GenesisURDF`
path the environment uses, and checks them as Genesis sees them:
- every collision geom is a box, as many as the generator made;
- every leg sits in the middle of its hole: the gap on each of its four sides is clearance / 2.

The board lies on the floor with one peg per slot, hovering above its holes. The pegs are fixed, unless:
- `--insert`: the legs start half way in;
- `--drop`: the pegs are free and fall into the holes; after `--settle` seconds the depth they reached is reported
  and checked (the tip must rest on the hole floor). With `--vis` the drop repeats every `--settle` seconds.

`--rigid-contact` takes the same Genesis `RigidOptions` overrides as `env.rigid_contact` in the config (validated
the same way), so their effect on the contact can be seen before running an environment.

With `--vis` the hole outlines (cyan) and the frames are drawn: board (at the top face), every slot inlet, the peg
tip, and the grasp point (green). Shadows are off: they made identical pegs look different.

Run from the repo root, e.g.:
    python3 -m scripts.peg_in_hole.view_assets --vis
    python3 -m scripts.peg_in_hole.view_assets --vis --drop
    python3 -m scripts.peg_in_hole.view_assets --drop --rigid-contact '{"constraint_timeconst": 0.2}'
    python3 -m scripts.peg_in_hole.view_assets --vis --overrides \
        '{"peg": {"legs": {"grid": [2, 2], "length": 0.03}}, "board": {"slot_pitch": 0.1}}'
Without `--vis` it only builds the scene, runs the checks and exits (non-zero on failure).
"""

import argparse
import json
import math
import os
import sys

# importing `aegis_gym.envs` pulls in ClearML, which would try to log in to the server; nothing here needs it
os.environ.setdefault("CLEARML_OFFLINE_MODE", "1")

import genesis as gs
import numpy as np
import torch as th

from aegis_gym.aux.pih_assets import PihAssets, PihAssetsCfg, generate_pih_assets
from aegis_gym.config.types.env import validate_rigid_contact
from aegis_gym.envs.objects import ObjectProperties
from aegis_gym.envs.objects.sim.genesis_objects import GenesisURDF

PEG_COLOR = (0.85, 0.45, 0.1)
BOARD_COLOR = (0.3, 0.3, 0.38)
HOLE_COLOR = (0.0, 0.8, 0.9, 1.0)
HOVER = 0.02  # [m] peg tip above the inlet
GAP_TOL = 1e-6  # [m] Genesis runs in float32
SEAT_TOL = 1e-3  # [m] tip to hole floor after a drop
CTRL_DT = 0.004  # [s] the environments' rigid solver step (env.ctrl_dt)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument(
        "--vis", action="store_true", help="open the viewer and keep it running"
    )
    mode = p.add_mutually_exclusive_group()
    mode.add_argument(
        "--insert", action="store_true", help="fixed pegs, legs half way into the holes"
    )
    mode.add_argument(
        "--drop", action="store_true", help="free pegs falling into the holes"
    )
    p.add_argument(
        "--settle",
        type=float,
        default=2.0,
        help="[s] drop duration before the report / replay",
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
    p.add_argument(
        "--cache-dir",
        default=None,
        help="asset cache (default: ~/.cache/aegis_gym/pih_assets)",
    )
    p.add_argument("--cpu", action="store_true", help="Genesis CPU backend")
    args = p.parse_args()
    validate_rigid_contact(args.rigid_contact)
    return args


def peg_z(assets: PihAssets, insert: bool) -> float:
    """Height of the peg frame (tip) in the world; the board's top face is at `thickness`."""
    top = assets.cfg.board.thickness
    return top - assets.cfg.board.depth / 2 if insert else top + HOVER


def peg_start_poses(assets: PihAssets, insert: bool) -> list[tuple[float, ...]]:
    z = peg_z(assets, insert)
    return [(f.pose[0], f.pose[1], z, 1.0, 0.0, 0.0, 0.0) for f in assets.hole_frames]


def build_scene(
    assets: PihAssets, args: argparse.Namespace
) -> tuple[gs.Scene, GenesisURDF, list[GenesisURDF]]:
    scene = gs.Scene(
        sim_options=gs.options.SimOptions(dt=CTRL_DT),
        # as in GenesisScene: Newton solver, the env's control step, then the config overrides
        rigid_options=gs.options.RigidOptions(
            dt=CTRL_DT,
            constraint_solver=gs.constraint_solver.Newton,
            **(args.rigid_contact or {}),
        ),
        vis_options=gs.options.VisOptions(shadow=False),
        viewer_options=gs.options.ViewerOptions(
            camera_pos=(0.25, -0.35, 0.3), camera_lookat=(0.0, 0.0, 0.02), camera_fov=40
        ),
        show_viewer=args.vis,
    )
    scene.add_entity(gs.morphs.Plane())
    device = th.device("cpu")

    def add(urdf, pose, color, fixed) -> GenesisURDF:
        obj = GenesisURDF(
            gs_scene=scene,
            properties=ObjectProperties(
                dims=(), pose=pose, fixed=fixed, color=color, urdf_path=str(urdf)
            ),
            device=device,
        )
        obj.create()
        return obj

    board_pose = (0.0, 0.0, assets.cfg.board.thickness, 1.0, 0.0, 0.0, 0.0)
    board = add(assets.board_urdf, board_pose, BOARD_COLOR, fixed=True)
    pegs = [
        add(assets.peg_urdf, pose, PEG_COLOR, fixed=not args.drop)
        for pose in peg_start_poses(assets, args.insert)
    ]
    scene.build(n_envs=1)  # batched, like the environments: poses are (n_envs, 7)
    return scene, board, pegs


def _aabb(geom) -> np.ndarray:
    return geom.get_AABB().cpu().numpy().reshape(-1, 2, 3)[0]


def check_collision_geoms(
    name: str, entity: GenesisURDF, expected_boxes: int
) -> list[str]:
    geoms = entity._obj.geoms
    types = sorted({str(g.type) for g in geoms})
    print(f"  {name}: {len(geoms)} collision geoms, types {types}")
    errors = []
    if len(geoms) != expected_boxes:
        errors.append(
            f"{name}: {len(geoms)} collision geoms, expected {expected_boxes}"
        )
    if any(g.type != gs.GEOM_TYPE.BOX for g in geoms):
        errors.append(f"{name}: collision geom types {types}, expected only BOX")
    return errors


def _leg_gaps(lo: np.ndarray, hi: np.ndarray, walls: list[np.ndarray]) -> np.ndarray:
    """Gaps -x, +x, -y, +y between a leg's box and the nearest walls around it."""
    c = (lo + hi) / 2
    row = [
        w for w in walls if w[0, 1] < c[1] < w[1, 1]
    ]  # walls left and right of the leg
    col = [w for w in walls if w[0, 0] < c[0] < w[1, 0]]  # walls in front and behind
    return np.array(
        [
            lo[0]
            - max(
                (w[1, 0] for w in row if w[1, 0] <= lo[0] + GAP_TOL), default=-np.inf
            ),
            min((w[0, 0] for w in row if w[0, 0] >= hi[0] - GAP_TOL), default=np.inf)
            - hi[0],
            lo[1]
            - max(
                (w[1, 1] for w in col if w[1, 1] <= lo[1] + GAP_TOL), default=-np.inf
            ),
            min((w[0, 1] for w in col if w[0, 1] >= hi[1] - GAP_TOL), default=np.inf)
            - hi[1],
        ]
    )


def check_leg_gaps(
    assets: PihAssets, board: GenesisURDF, pegs: list[GenesisURDF]
) -> list[str]:
    """Gap between every leg and the board walls around it, measured on the boxes Genesis loaded."""
    probe_z = (
        assets.cfg.board.thickness - 1e-3
    )  # just below the top face, inside the holes
    walls = [b for b in map(_aabb, board._obj.geoms) if b[0, 2] < probe_z < b[1, 2]]
    errors = []
    print("  slot clearance  leg centre [mm]   gap -x    +x    -y    +y [mm]")
    for k, (frame, peg) in enumerate(zip(assets.hole_frames, pegs, strict=True)):
        boxes = [_aabb(g) for g in peg._obj.geoms]
        tip = min(b[0, 2] for b in boxes)
        legs = [
            b for b in boxes if abs(b[0, 2] - tip) < 1e-6
        ]  # the base starts higher up
        for lo, hi in legs:
            gaps = _leg_gaps(lo, hi, walls)
            c = (lo + hi) / 2
            print(
                f"  {k:>4} {frame.clearance * 1e3:7.2f}     "
                f"({(c[0] - frame.pose[0]) * 1e3:+6.1f},{(c[1] - frame.pose[1]) * 1e3:+6.1f})  "
                + " ".join(f"{g * 1e3:5.3f}" for g in gaps)
            )
            if np.any(np.abs(gaps - frame.clearance / 2) > GAP_TOL):
                errors.append(
                    f"slot {k}: leg at {np.round(c[:2] * 1e3, 1)} mm has gaps {gaps * 1e3} mm"
                )
    return errors


def reset_pegs(assets: PihAssets, pegs: list[GenesisURDF], insert: bool) -> None:
    for peg, pose in zip(pegs, peg_start_poses(assets, insert), strict=True):
        peg.set_pose(th.tensor([pose], device=gs.device))
        peg._obj.zero_all_dofs_velocity()


def check_drop(
    assets: PihAssets, scene: gs.Scene, pegs: list[GenesisURDF], settle: float
) -> list[str]:
    """Lets the pegs fall for `settle` seconds; every tip must end on its hole floor."""
    for _ in range(round(settle / CTRL_DT)):
        scene.step()
    floor = assets.cfg.board.thickness - assets.cfg.board.depth
    errors = []
    print(
        f"  drop, {settle:.1f} s: tip height above the hole floor (negative = penetration)"
    )
    print("  slot clearance  tip-floor [mm]  xy offset [mm]  tilt [deg]  speed [mm/s]")
    for k, (frame, peg) in enumerate(zip(assets.hole_frames, pegs, strict=True)):
        pose = peg.get_pose()[0].cpu().numpy()
        seat = pose[2] - floor
        offset = np.linalg.norm(pose[:2] - np.array(frame.pose[:2]))
        tilt = math.degrees(2 * math.acos(min(1.0, abs(float(pose[3])))))
        speed = float(th.linalg.vector_norm(peg._obj.get_vel()[0]))
        print(
            f"  {k:>4} {frame.clearance * 1e3:7.2f}     {seat * 1e3:+9.3f}     "
            f"{offset * 1e3:9.3f}     {tilt:7.3f}    {speed * 1e3:9.3f}"
        )
        if abs(seat) > SEAT_TOL:
            errors.append(
                f"slot {k}: tip {seat * 1e3:+.2f} mm from the hole floor after the drop"
            )
    return errors


def draw_overlays(scene: gs.Scene, assets: PihAssets, z: float) -> None:
    def tf(pos) -> np.ndarray:
        t = np.eye(4)
        t[:3, 3] = pos
        return t

    top, depth = assets.cfg.board.thickness, assets.cfg.board.depth
    scene.draw_debug_frame(
        tf((0.0, 0.0, top)), axis_length=0.05, origin_size=0.002, axis_radius=0.001
    )
    for slot in assets.geometry.slots:
        for h in slot.holes:
            scene.draw_debug_box(
                [[h.x0, h.y0, top - depth], [h.x1, h.y1, top]],
                color=HOLE_COLOR,
                wireframe=True,
                wireframe_radius=0.0003,
            )
        x, y = slot.pose[:2]
        scene.draw_debug_frame(
            tf((x, y, top)), axis_length=0.02, origin_size=0.0015, axis_radius=0.0006
        )
        scene.draw_debug_frame(
            tf((x, y, z)), axis_length=0.015, origin_size=0.0015, axis_radius=0.0005
        )
        grasp = np.array([x, y, z]) + np.array(assets.peg_grasp_frame[:3])
        scene.draw_debug_sphere(grasp, radius=0.003, color=(0.1, 0.8, 0.1, 1.0))


def main() -> int:
    args = parse_args()
    assets = generate_pih_assets(
        PihAssetsCfg.from_yaml(overrides=args.overrides), cache_dir=args.cache_dir
    )
    print(f"assets: {assets.directory}")
    if args.rigid_contact:
        print(f"RigidOptions overrides: {args.rigid_contact}")

    gs.init(
        backend=gs.cpu if args.cpu else gs.gpu, precision="32", logging_level="warning"
    )
    scene, board, pegs = build_scene(assets, args)

    print("checks:")
    errors = check_collision_geoms("board", board, len(assets.geometry.board_boxes))
    for i, peg in enumerate(pegs):
        errors += check_collision_geoms(
            f"peg[{i}]", peg, len(assets.geometry.peg_boxes)
        )
    errors += check_leg_gaps(assets, board, pegs)
    if args.drop:
        errors += check_drop(assets, scene, pegs, args.settle)
    print("FAILED:\n  " + "\n  ".join(errors) if errors else "OK")

    if args.vis:
        # the overlays mark the start poses; dropped pegs fall away from them
        draw_overlays(scene, assets, peg_z(assets, args.insert))
        print("viewer open, close the window or Ctrl+C to exit")
        replay_steps = round(args.settle / CTRL_DT)
        try:
            step = 0
            if args.drop:
                reset_pegs(assets, pegs, args.insert)
            while scene.viewer.is_alive():
                scene.step()
                step += 1
                if args.drop and step % replay_steps == 0:
                    reset_pegs(assets, pegs, args.insert)
        except KeyboardInterrupt:
            pass
    return 1 if errors else 0


if __name__ == "__main__":
    sys.exit(main())
