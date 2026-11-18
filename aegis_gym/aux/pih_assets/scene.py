"""
Peg-in-hole robot scene: the board standing on the table and the peg grasped by the Hand-E fingers, placed from config.

Works through `BaseScene` / `BaseManipulator` (no Genesis import), so an env adds it like the tee in Push-T:

    pih = add_pih_entities(scene, PihSceneCfg.from_dict(env_dict["pih_scene"]), table_top_z)  # before build()
    scene.build()
    grasp_peg(manipulator, pih, envs_idx)  # in reset_idx(); the peg is held from the first step on
    slip = grasp_slip(manipulator, pih)  # (n, 2): [m] and [rad] of the peg w.r.t. the hand

Frames: the board pose is its top face centre; the peg pose is its tip (bottom face centre), z along its axis; the
hole poses are the slot inlet centres, z up, in the world (the robot base frame).
"""

import math
from collections.abc import Sequence
from dataclasses import asdict, dataclass, field, fields
from typing import TYPE_CHECKING, Any

import torch as th

from aegis_gym.aux.geom.geom import transform_by_quat, transform_quat_by_quat
from aegis_gym.envs.objects.base_objects import ObjectProperties, ObjectType

from .assets import PihAssets, generate_pih_assets
from .cfg import PihAssetsCfg

if TYPE_CHECKING:
    from aegis_gym.envs.manipulator import BaseManipulator
    from aegis_gym.envs.objects import BaseURDF
    from aegis_gym.envs.scene import BaseScene

_BOARD_COLOR = (0.75, 0.75, 0.7)
_PEG_COLOR = (0.85, 0.35, 0.1)
# the peg axis points out of the hand: half a turn about the tool x (the finger axis)
_FLIP_X = (0.0, 1.0, 0.0, 0.0)
# [m] the widest Hand-E opening (2 x 25 mm finger stroke)
HANDE_MAX_WIDTH = 0.05
GRASP_MODES = ("physical", "weld")


@dataclass(frozen=True, slots=True)
class PihSceneCfg:
    """Where everything goes; keys fit an `env_specific_dict`."""

    # [m] board top centre on the table
    board_xy: tuple[float, float] = (0.47, 0.0)
    board_yaw_deg: float = 0.0
    # [m] x, y and height above the table top
    tcp_start: tuple[float, float, float] = (0.40, 0.0, 0.15)
    # (w, x, y, z), downward
    tcp_quat: tuple[float, float, float, float] = (0.0, 1.0, 0.0, 0.0)
    # about the tool z; 0: the peg faces parallel to the finger pads
    peg_yaw_deg: float = 0.0
    # [m] TCP to the finger pad centres, towards the wrist (Hand-E URDF)
    pad_offset: float = 0.00945
    # "physical": held by the finger friction; "weld": also welded to the hand, so it can't slip
    grasp: str = "physical"
    # None: Genesis default
    peg_friction: float | None = None
    board_friction: float | None = None
    # overrides of nominal.yaml
    assets: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, d: dict[str, Any] | None) -> "PihSceneCfg":
        d = dict(d or {})
        known = {f.name for f in fields(cls)}
        unknown = sorted(set(d) - known)
        if unknown:
            raise ValueError(
                f"Invalid peg-in-hole scene config: unknown key(s) {unknown}; allowed: {sorted(known)}"
            )
        for key in ("board_xy", "tcp_start", "tcp_quat"):
            if key in d:
                d[key] = tuple(float(v) for v in d[key])
        cfg = cls(**d)
        if len(cfg.board_xy) != 2 or len(cfg.tcp_start) != 3 or len(cfg.tcp_quat) != 4:
            raise ValueError(
                "board_xy needs 2 values, tcp_start 3 and tcp_quat 4 (w, x, y, z)"
            )
        if cfg.grasp not in GRASP_MODES:
            raise ValueError(f"grasp must be one of {GRASP_MODES}, got '{cfg.grasp}'")
        return cfg

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _yaw_quat(yaw_deg: float) -> tuple[float, float, float, float]:
    half = math.radians(yaw_deg) / 2
    return (math.cos(half), 0.0, 0.0, math.sin(half))


def grasp_width(assets: PihAssets) -> float:
    """[m] Peg width along its x (between the pads at `peg_yaw_deg` 0) at the grasp height."""
    z = assets.peg_grasp_frame[2]
    spans = [
        b.bounds
        for b in assets.geometry.peg_boxes
        if b.bounds[0][2] <= z <= b.bounds[1][2]
    ]
    if not spans:
        raise ValueError("The grasp frame lies outside the peg")
    return max(hi[0] for _, hi in spans) - min(lo[0] for lo, _ in spans)


def board_pose(
    cfg: PihSceneCfg, assets: PihAssets, table_top_z: float
) -> tuple[float, ...]:
    return (
        *cfg.board_xy,
        table_top_z + assets.cfg.board.thickness,
        *_yaw_quat(cfg.board_yaw_deg),
    )


def tcp_start_pose(cfg: PihSceneCfg, table_top_z: float) -> tuple[float, ...]:
    x, y, height = cfg.tcp_start
    return (x, y, table_top_z + height, *cfg.tcp_quat)


def hole_poses(cfg: PihSceneCfg, assets: PihAssets, table_top_z: float) -> th.Tensor:
    """(n_slots, 7) slot inlet centres in the world, in the order of `assets.hole_frames`."""
    board = th.tensor(board_pose(cfg, assets, table_top_z), dtype=th.float32)
    local = th.tensor([f.pose for f in assets.hole_frames], dtype=th.float32)
    quat = board[3:].expand(len(local), 4)
    pos = board[:3] + transform_by_quat(local[:, :3], quat)
    return th.cat([pos, transform_quat_by_quat(quat, local[:, 3:])], dim=1)


def peg_pose_in_hand(
    tcp_pose: th.Tensor, assets: PihAssets, cfg: PihSceneCfg
) -> th.Tensor:
    """(n, 7) peg pose with its grasp frame at the finger pad centres and its axis along the tool z."""
    tcp_pose = tcp_pose.reshape(-1, 7)
    n, dev, dtype = len(tcp_pose), tcp_pose.device, tcp_pose.dtype

    def const(values: Sequence[float]) -> th.Tensor:
        return th.tensor(values, dtype=dtype, device=dev).expand(n, len(values))

    tcp_pos, tcp_quat = tcp_pose[:, :3], tcp_pose[:, 3:]
    # the peg in the tool frame: turned about the tool z, then flipped to point out of the hand
    peg_in_tool = transform_quat_by_quat(
        const(_yaw_quat(cfg.peg_yaw_deg)), const(_FLIP_X)
    )
    peg_quat = transform_quat_by_quat(tcp_quat, peg_in_tool)
    pads = tcp_pos + transform_by_quat(const((0.0, 0.0, -cfg.pad_offset)), tcp_quat)
    grasp = const(assets.peg_grasp_frame[:3])
    return th.cat([pads - transform_by_quat(grasp, peg_quat), peg_quat], dim=1)


@dataclass(slots=True)
class PihScene:
    cfg: PihSceneCfg
    assets: PihAssets
    board: "BaseURDF"
    peg: "BaseURDF"
    table_top_z: float
    board_pose: tuple[float, ...]
    tcp_start_pose: tuple[float, ...]
    hole_poses: th.Tensor  # (n_slots, 7), see `hole_poses()`
    grasp_width: float  # [m]


def add_pih_entities(
    scene: "BaseScene",
    cfg: PihSceneCfg,
    table_top_z: float,
    cache_dir: str | None = None,
) -> PihScene:
    """Adds the board and the peg; call after `scene.add_manipulator()` and before `scene.build()`."""
    assets = generate_pih_assets(
        PihAssetsCfg.from_yaml(overrides=cfg.assets), cache_dir=cache_dir
    )
    b_pose = board_pose(cfg, assets, table_top_z)
    t_pose = tcp_start_pose(cfg, table_top_z)
    width = grasp_width(assets)
    if width > HANDE_MAX_WIDTH:
        raise ValueError(
            f"The peg is {width * 1e3:.1f} mm wide at the grasp, the Hand-E opens to {HANDE_MAX_WIDTH * 1e3:.0f} mm"
        )
    board = scene.add_entity(
        entity=ObjectType.URDF,
        properties=ObjectProperties(
            dims=(),
            pose=b_pose,
            fixed=True,
            color=_BOARD_COLOR,
            urdf_path=str(assets.board_urdf),
            friction=cfg.board_friction,
        ),
    )
    peg_start = peg_pose_in_hand(th.tensor(t_pose, dtype=th.float32), assets, cfg)[0]
    peg = scene.add_entity(
        entity=ObjectType.URDF,
        properties=ObjectProperties(
            dims=(),
            pose=tuple(peg_start.tolist()),
            fixed=False,
            color=_PEG_COLOR,
            urdf_path=str(assets.peg_urdf),
            friction=cfg.peg_friction,
        ),
    )
    return PihScene(
        cfg=cfg,
        assets=assets,
        board=board,
        peg=peg,
        table_top_z=table_top_z,
        board_pose=b_pose,
        tcp_start_pose=t_pose,
        hole_poses=hole_poses(cfg, assets, table_top_z),
        grasp_width=width,
    )


def grasp_peg(
    manipulator: "BaseManipulator", pih: PihScene, envs_idx: th.Tensor
) -> None:
    """
    Resets the `envs_idx` to the TCP start pose with the peg between the fingers: the fingers are set touching the
    peg and driven closed, so the grip holds from the first step on, without stepping the simulation here. With
    `grasp: weld` the peg is also welded to the hand (`weld_object()`, simulation only).
    """
    if len(envs_idx) == 0:
        return
    device = manipulator.device
    start = th.tensor(pih.tcp_start_pose, dtype=th.float32, device=device).expand(
        len(envs_idx), 7
    )
    manipulator.ctrl_go_to_home(envs_idx)
    manipulator.ctrl_reset_to_pose(pose=start, open_gripper=False, envs_idx=envs_idx)
    manipulator.set_gripper_width(pih.grasp_width, envs_idx=envs_idx)
    tcp = manipulator.get_tcp_pose()[envs_idx]
    pih.peg.set_pose(pose=peg_pose_in_hand(tcp, pih.assets, pih.cfg), envs_idx=envs_idx)
    if pih.cfg.grasp == "weld":
        # Genesis welds a pair of links in all envs at once: re-welding the others keeps their peg where it is
        manipulator.weld_object(pih.peg)


def grasp_slip(manipulator: "BaseManipulator", pih: PihScene) -> th.Tensor:
    """(n_envs, 2): [m] peg offset and [rad] peg rotation w.r.t. its pose in the hand."""
    expected = peg_pose_in_hand(manipulator.get_tcp_pose(), pih.assets, pih.cfg)
    actual = pih.peg.get_pose().to(expected)
    offset = th.linalg.vector_norm(actual[:, :3] - expected[:, :3], dim=1)
    dot = th.abs(th.sum(actual[:, 3:] * expected[:, 3:], dim=1)).clamp(max=1.0)
    return th.stack([offset, 2 * th.acos(dot)], dim=1)
