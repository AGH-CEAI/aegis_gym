"""
Example peg-in-hole robot scene, built like the Push-T env: the robot, the board with the hole fixed on the table, and
the peg standing on the table next to it, both loaded from the URDF/STL files in `aegis_gym/assets/peg_in_hole/`
(`python3 -m scripts.peg_in_hole.generate_assets` writes them from `nominal.yaml`).

The placement is a dict like an `env_specific_dict` (`POSITIONS`, `--positions` overrides it). The script builds
the scene offline (the default config, cameras off, the robot URDF from the local ClearML dataset cache), moves the
robot home, simulates 2 s and checks that the board stands on the table and the peg stands upright next to it.
With `--vis` the viewer stays open.

Run from the repo root in the toolbox, e.g.:
    python3 -m scripts.peg_in_hole.robot_scene_example --vis
    python3 -m scripts.peg_in_hole.robot_scene_example --positions '{"board_xy": [0.5, 0.1], "board_yaw_deg": 30}'
"""

import argparse
import json
import math
import os
import sys
from pathlib import Path

# importing `aegis_gym.envs` pulls in ClearML, which would try to log in to the server; nothing here needs it
os.environ.setdefault("CLEARML_OFFLINE_MODE", "1")

import genesis as gs
import torch as th

from aegis_gym.aux.pih_assets import PihAssetsCfg, build_geometry
from aegis_gym.config import ConfigManager
from aegis_gym.envs.manipulator.sim.genesis_manipulator import GenesisManipulator
from aegis_gym.envs.objects import BaseURDF, ObjectProperties, ObjectType
from aegis_gym.envs.scene import GenesisScene

# as the tee files of Push-T (`PushTEnv._TEE_URDF_PATH`)
_ASSETS_DIR = (
    Path(__file__).resolve().parents[2] / "aegis_gym" / "assets" / "peg_in_hole"
)
_BOARD_URDF = _ASSETS_DIR / "board.urdf"
_PEG_URDF = _ASSETS_DIR / "peg.urdf"

# [m], [deg] in the robot base frame; the board pose is its top face centre, the peg pose its bottom face centre
POSITIONS = {
    "board_xy": [0.47, 0.0],
    "board_yaw_deg": 0.0,
    "peg_xy": [0.40, 0.10],
    "peg_yaw_deg": 0.0,
}
SIM_SECONDS = 2.0
PEG_HEIGHT_TOL = (
    1e-3  # [m] the peg base this close to the table top counts as standing on it
)
PEG_TILT_TOL = math.radians(1.0)
CLEARML_DATASETS = Path("~/.clearml/cache/storage_manager/datasets").expanduser()


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument(
        "--positions", type=json.loads, default=None, help="JSON, POSITIONS keys"
    )
    p.add_argument(
        "--urdf", type=Path, default=None, help="robot URDF (default: ClearML cache)"
    )
    p.add_argument("--vis", action="store_true", help="keep the viewer open")
    args = p.parse_args(argv)
    unknown = set(args.positions or {}) - set(POSITIONS)
    if unknown:
        p.error(
            f"unknown --positions keys {sorted(unknown)}, allowed: {sorted(POSITIONS)}"
        )
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


def yaw_quat(yaw_deg: float) -> tuple[float, float, float, float]:
    half = math.radians(yaw_deg) / 2
    return (math.cos(half), 0.0, 0.0, math.sin(half))


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    positions = {**POSITIONS, **(args.positions or {})}
    use_cached_urdf(args.urdf)
    gs.init(backend=gs.gpu, precision="32", logging_level="warning")
    argv = ["robot_scene_example", "-B", "1", "--disable-cameras"]
    ConfigManager.setup_config(
        argv=argv + (["--vis"] if args.vis else []), device=gs.device, task=None
    )
    cfg = ConfigManager.get_config()
    env = cfg.env_cfg
    table_top_z = env.table_size[2] - env.workbench_size[2]
    # the dimensions the files were generated with (nominal.yaml)
    pih = PihAssetsCfg.from_yaml()
    hole_in_board = build_geometry(pih).slots[0].pose

    # as `PushTEnv._setup_scene()`: the robot, then the objects, then `build()`
    scene = GenesisScene(cfg=cfg, device=gs.device)
    scene.add_manipulator(cfg=cfg.robot_cfg)
    board_pose = (
        *positions["board_xy"],
        table_top_z + pih.board.thickness,
        *yaw_quat(positions["board_yaw_deg"]),
    )
    peg_pose = (*positions["peg_xy"], table_top_z, *yaw_quat(positions["peg_yaw_deg"]))
    board: BaseURDF = scene.add_entity(
        entity=ObjectType.URDF,
        properties=ObjectProperties(
            dims=(),
            pose=board_pose,
            fixed=True,
            color=(0.25, 0.25, 0.3),
            urdf_path=str(_BOARD_URDF),
        ),
    )
    peg: BaseURDF = scene.add_entity(
        entity=ObjectType.URDF,
        properties=ObjectProperties(
            dims=(),
            pose=peg_pose,
            fixed=False,
            color=(0.85, 0.45, 0.1),
            urdf_path=str(_PEG_URDF),
        ),
    )
    scene.build()
    manipulator = scene.get_manipulator()
    manipulator.ctrl_go_to_home()

    for _ in range(round(SIM_SECONDS / scene.get_policy_dt())):
        scene.step()

    yaw = math.radians(positions["board_yaw_deg"])
    hole = (
        board_pose[0]
        + math.cos(yaw) * hole_in_board[0]
        - math.sin(yaw) * hole_in_board[1],
        board_pose[1]
        + math.sin(yaw) * hole_in_board[0]
        + math.cos(yaw) * hole_in_board[1],
        board_pose[2] + hole_in_board[2],
    )
    board_now = board.get_pose()[0].tolist()
    peg_now = peg.get_pose()[0].tolist()
    # tilt of the peg axis from the vertical
    _, x, y, _ = peg_now[3:]
    tilt = math.acos(max(-1.0, min(1.0, 1 - 2 * (x * x + y * y))))
    print(
        f"peg {pih.peg.side * 1e3:g} mm, hole {(pih.peg.side + pih.board.clearances[0]) * 1e3:g} mm "
        f"(clearance {pih.board.clearances[0] * 1e3:g} mm), table top at z = {table_top_z:.4f} m"
    )
    print(f"board top centre: {[round(v, 4) for v in board_now[:3]]}")
    print(f"hole inlet centre: {[round(v, 4) for v in hole]}")
    print(
        f"peg base centre:  {[round(v, 4) for v in peg_now[:3]]}, tilt {math.degrees(tilt):.2f} deg"
    )
    print(f"TCP: {[round(v, 4) for v in manipulator.get_tcp_pose()[0].tolist()[:3]]}")

    errors = []
    if abs(board_now[2] - board_pose[2]) > 1e-6:
        errors.append(
            f"the board top is at z = {board_now[2]:.4f}, not {board_pose[2]:.4f}"
        )
    if abs(peg_now[2] - table_top_z) > PEG_HEIGHT_TOL or tilt > PEG_TILT_TOL:
        errors.append("the peg does not stand upright on the table")
    print("FAILED:\n  " + "\n  ".join(errors) if errors else "OK")

    if args.vis:
        print("viewer open; close the window or Ctrl+C to exit")
        try:
            while scene.gs_scene.viewer.is_alive():
                scene.step()
        except KeyboardInterrupt:
            pass
    return 1 if errors else 0


if __name__ == "__main__":
    th.set_default_dtype(th.float32)
    sys.exit(main())
