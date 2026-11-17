"""
Loads the generated peg-in-hole assets into an empty Genesis scene (no robot), through the same `GenesisURDF`
path the environment uses, and checks them as Genesis sees them:
- every collision geom is a box, as many as the generator made;
- every leg sits in the middle of its hole: the gap on each of its four sides is clearance / 2.

The board lies on the floor with one fixed peg per slot, hovering above its holes (`--insert` puts the legs half
way in instead). With `--vis` the hole outlines (cyan) and the frames are drawn: board (at the top face), every slot
inlet, the peg tip, and the grasp point (green). Shadows are off: they made identical pegs look different.

Run from the repo root, e.g.:
    python3 -m scripts.peg_in_hole.view_assets --vis
    python3 -m scripts.peg_in_hole.view_assets --vis --overrides \
        '{"peg": {"legs": {"grid": [2, 2], "length": 0.03}}, "board": {"slot_pitch": 0.1}}'
Without `--vis` it only builds the scene, runs the checks and exits (non-zero on failure).
"""

import argparse
import json
import os
import sys

# importing `aegis_gym.envs` pulls in ClearML, which would try to log in to the server; nothing here needs it
os.environ.setdefault("CLEARML_OFFLINE_MODE", "1")

import genesis as gs
import numpy as np
import torch as th

from aegis_gym.aux.pih_assets import PihAssets, PihAssetsCfg, generate_pih_assets
from aegis_gym.envs.objects import ObjectProperties
from aegis_gym.envs.objects.sim.genesis_objects import GenesisURDF

PEG_COLOR = (0.85, 0.45, 0.1)
BOARD_COLOR = (0.3, 0.3, 0.38)
HOLE_COLOR = (0.0, 0.8, 0.9, 1.0)
HOVER = 0.02  # [m] peg tip above the inlet
GAP_TOL = 1e-6  # [m] Genesis runs in float32


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument(
        "--vis", action="store_true", help="open the viewer and keep it running"
    )
    p.add_argument("--insert", action="store_true", help="legs half way into the holes")
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
    return p.parse_args()


def peg_z(assets: PihAssets, insert: bool) -> float:
    """Height of the peg frame (tip) in the world; the board's top face is at `thickness`."""
    top = assets.cfg.board.thickness
    return top - assets.cfg.board.depth / 2 if insert else top + HOVER


def build_scene(
    assets: PihAssets, vis: bool, insert: bool
) -> tuple[gs.Scene, GenesisURDF, list[GenesisURDF]]:
    scene = gs.Scene(
        sim_options=gs.options.SimOptions(dt=0.004),
        vis_options=gs.options.VisOptions(shadow=False),
        viewer_options=gs.options.ViewerOptions(
            camera_pos=(0.25, -0.35, 0.3), camera_lookat=(0.0, 0.0, 0.02), camera_fov=40
        ),
        show_viewer=vis,
    )
    scene.add_entity(gs.morphs.Plane())
    device = th.device("cpu")

    def add(urdf, pose, color) -> GenesisURDF:
        obj = GenesisURDF(
            gs_scene=scene,
            properties=ObjectProperties(
                dims=(), pose=pose, fixed=True, color=color, urdf_path=str(urdf)
            ),
            device=device,
        )
        obj.create()
        return obj

    identity = (1.0, 0.0, 0.0, 0.0)
    board = add(
        assets.board_urdf,
        (0.0, 0.0, assets.cfg.board.thickness, *identity),
        BOARD_COLOR,
    )
    z = peg_z(assets, insert)
    pegs = [
        add(assets.peg_urdf, (f.pose[0], f.pose[1], z, *identity), PEG_COLOR)
        for f in assets.hole_frames
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
            c = (lo + hi) / 2
            row = [
                w for w in walls if w[0, 1] < c[1] < w[1, 1]
            ]  # walls left and right of the leg
            col = [
                w for w in walls if w[0, 0] < c[0] < w[1, 0]
            ]  # walls in front and behind
            gaps = np.array(
                [
                    lo[0]
                    - max(
                        (w[1, 0] for w in row if w[1, 0] <= lo[0] + GAP_TOL),
                        default=-np.inf,
                    ),
                    min(
                        (w[0, 0] for w in row if w[0, 0] >= hi[0] - GAP_TOL),
                        default=np.inf,
                    )
                    - hi[0],
                    lo[1]
                    - max(
                        (w[1, 1] for w in col if w[1, 1] <= lo[1] + GAP_TOL),
                        default=-np.inf,
                    ),
                    min(
                        (w[0, 1] for w in col if w[0, 1] >= hi[1] - GAP_TOL),
                        default=np.inf,
                    )
                    - hi[1],
                ]
            )
            print(
                f"  {k:>4} {frame.clearance * 1e3:7.2f}     ({(c[0] - frame.pose[0]) * 1e3:+6.1f},"
                f"{(c[1] - frame.pose[1]) * 1e3:+6.1f})  "
                + " ".join(f"{g * 1e3:5.3f}" for g in gaps)
            )
            if np.any(np.abs(gaps - frame.clearance / 2) > GAP_TOL):
                errors.append(
                    f"slot {k}: leg at {np.round(c[:2] * 1e3, 1)} mm has gaps {gaps * 1e3} mm"
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

    gs.init(
        backend=gs.cpu if args.cpu else gs.gpu, precision="32", logging_level="warning"
    )
    scene, board, pegs = build_scene(assets, args.vis, args.insert)

    print("checks:")
    errors = check_collision_geoms("board", board, len(assets.geometry.board_boxes))
    for i, peg in enumerate(pegs):
        errors += check_collision_geoms(
            f"peg[{i}]", peg, len(assets.geometry.peg_boxes)
        )
    errors += check_leg_gaps(assets, board, pegs)
    print("FAILED:\n  " + "\n  ".join(errors) if errors else "OK")

    if args.vis:
        draw_overlays(scene, assets, peg_z(assets, args.insert))
        print("viewer open, close the window or Ctrl+C to exit")
        try:
            while scene.viewer.is_alive():
                scene.step()
        except KeyboardInterrupt:
            pass
    return 1 if errors else 0


if __name__ == "__main__":
    sys.exit(main())
