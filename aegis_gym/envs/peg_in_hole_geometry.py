"""
Geometry of the peg-in-hole task: a peg with N legs and a board with K slots of N holes each.
The 1-1 variant is the case of a single leg at the peg's origin.

The conventions below are the asset interface (to be agreed with the assets' author), the
procedural URDFs of this module follow them as placeholders:
    - the peg frame: the origin is the grasp point (the handle center), the Z axis points up,
      the legs hang below the handle along -Z, at `leg_offsets` in XY;
    - the board frame: the origin lies on the table top, the Z axis points up, the slot `k` is
      centered at `slot_centers()[k]` and its holes are at `leg_offsets` from the slot center,
      with the openings at `z = hole_depth`;
    - the slot `k` has the clearance `clearances[k]` per side, so the parallel envs share one
      geometry and the episode's tolerance is the slot it draws.
"""

import hashlib
import math
import tempfile
from dataclasses import dataclass
from pathlib import Path

import torch as th


@dataclass(frozen=True, slots=True)
class PegInHoleGeometry:
    # [m] XY cross-section of a leg
    leg_size: tuple[float, float]
    leg_length: float
    # [m] XY centers of the legs in the peg frame
    leg_offsets: tuple[tuple[float, float], ...]
    # [m] the gripped part, visual only (the fingers hold it through the weld, see the env)
    handle_size: tuple[float, float, float]
    hole_depth: float
    # [m] per side, one slot per value
    clearances: tuple[float, ...]
    wall_thickness: float
    slot_spacing: float
    peg_mass: float

    @classmethod
    def from_dict(cls, d: dict) -> "PegInHoleGeometry":
        return cls(
            leg_size=tuple(d["leg_size"]),
            leg_length=d["leg_length"],
            leg_offsets=tuple(tuple(o) for o in d["leg_offsets"]),
            handle_size=tuple(d["handle_size"]),
            hole_depth=d["hole_depth"],
            clearances=tuple(d["clearances"]),
            wall_thickness=d["wall_thickness"],
            slot_spacing=d["slot_spacing"],
            peg_mass=d["peg_mass"],
        )

    def __post_init__(self) -> None:
        if not self.leg_offsets:
            raise ValueError("The peg needs at least one leg.")
        if not self.clearances:
            raise ValueError("The board needs at least one slot.")
        if min(self.clearances) <= 0.0:
            raise ValueError(f"The clearances must be positive, got {self.clearances}.")

    @property
    def num_legs(self) -> int:
        return len(self.leg_offsets)

    @property
    def num_slots(self) -> int:
        return len(self.clearances)

    @property
    def tip_z(self) -> float:
        """Z of the leg tips in the peg frame."""
        return -self.handle_size[2] / 2 - self.leg_length

    def leg_tips(self) -> th.Tensor:
        """[num_legs, 3] leg tips in the peg frame."""
        return th.tensor(
            [[x, y, self.tip_z] for x, y in self.leg_offsets], dtype=th.float32
        )

    def tips_centroid(self) -> th.Tensor:
        """[3] centroid of the leg tips in the peg frame, the insertion point of the peg."""
        return self.leg_tips().mean(dim=0)

    def slot_centers(self) -> th.Tensor:
        """[num_slots, 2] slot centers in the board frame, in a row along Y."""
        k = th.arange(self.num_slots, dtype=th.float32)
        y = (k - (self.num_slots - 1) / 2) * self.slot_spacing
        return th.stack([th.zeros_like(y), y], dim=-1)

    def hole_half_size(self, slot: int) -> tuple[float, float]:
        c = self.clearances[slot]
        return self.leg_size[0] / 2 + c, self.leg_size[1] / 2 + c

    def board_boxes(self) -> list[tuple[tuple[float, ...], tuple[float, ...]]]:
        """
        The board as (center, size) boxes in the board frame: four walls around each hole, as
        ManiSkill's `_build_box_with_hole()`, so the collisions stay box-box (no convex
        decomposition). Without a chamfer, it is the assets' part.
        """
        boxes = []
        w, h = self.wall_thickness, self.hole_depth
        for slot, (sx, sy) in enumerate(self.slot_centers().tolist()):
            hx, hy = self.hole_half_size(slot)
            for lx, ly in self.leg_offsets:
                cx, cy = sx + lx, sy + ly
                boxes += [
                    ((cx + hx + w / 2, cy, h / 2), (w, 2 * (hy + w), h)),
                    ((cx - hx - w / 2, cy, h / 2), (w, 2 * (hy + w), h)),
                    ((cx, cy + hy + w / 2, h / 2), (2 * hx, w, h)),
                    ((cx, cy - hy - w / 2, h / 2), (2 * hx, w, h)),
                ]
        return boxes

    def board_urdf(self) -> str:
        geoms = "".join(
            _box_geom(center, size, collision=True)
            for center, size in self.board_boxes()
        )
        return _urdf(name="peg_in_hole_board", link="board", mass=1.0, geoms=geoms)

    def peg_urdf(self) -> str:
        hz = self.handle_size[2]
        geoms = _box_geom((0.0, 0.0, 0.0), self.handle_size, collision=False)
        for lx, ly in self.leg_offsets:
            center = (lx, ly, -hz / 2 - self.leg_length / 2)
            size = (*self.leg_size, self.leg_length)
            geoms += _box_geom(center, size, collision=True)
        return _urdf(name="peg", link="peg", mass=self.peg_mass, geoms=geoms)

    def write_urdfs(self, out_dir: Path | None = None) -> tuple[Path, Path]:
        """Writes the (peg, board) URDFs, named by their content, and returns their paths."""
        out_dir = out_dir or Path(tempfile.gettempdir()) / "aegis_gym_peg_in_hole"
        out_dir.mkdir(parents=True, exist_ok=True)
        return (
            _write_once(out_dir, "peg", self.peg_urdf()),
            _write_once(out_dir, "board", self.board_urdf()),
        )


def _fmt(values: tuple[float, ...]) -> str:
    return " ".join(f"{v:.6g}" for v in values)


def _box_geom(
    center: tuple[float, ...], size: tuple[float, ...], collision: bool
) -> str:
    body = (
        f'<origin xyz="{_fmt(center)}" rpy="0 0 0"/>'
        f'<geometry><box size="{_fmt(size)}"/></geometry>'
    )
    geom = f"<visual>{body}</visual>"
    if collision:
        geom += f"<collision>{body}</collision>"
    return geom


def _urdf(name: str, link: str, mass: float, geoms: str) -> str:
    # the inertia is recomputed from the geometry by the loader (`GenesisURDF`)
    inertia = 'ixx="1e-4" ixy="0" ixz="0" iyy="1e-4" iyz="0" izz="1e-4"'
    return (
        f'<?xml version="1.0"?>\n<robot name="{name}"><link name="{link}">'
        f'<inertial><origin xyz="0 0 0" rpy="0 0 0"/><mass value="{mass:.6g}"/>'
        f"<inertia {inertia}/></inertial>{geoms}</link></robot>\n"
    )


def _write_once(out_dir: Path, kind: str, content: str) -> Path:
    digest = hashlib.sha1(content.encode()).hexdigest()[:12]
    path = out_dir / f"{kind}_{digest}.urdf"
    if not path.exists():
        path.write_text(content, encoding="utf-8")
    return path


def yaw_to_quat(yaw: th.Tensor) -> th.Tensor:
    """[N] yaw angles -> [N, 4] (w, x, y, z) quaternions of the rotations about Z."""
    zeros = th.zeros_like(yaw)
    return th.stack([th.cos(yaw / 2), zeros, zeros, th.sin(yaw / 2)], dim=-1)


def quat_to_yaw(quat: th.Tensor) -> th.Tensor:
    """[N, 4] (w, x, y, z) quaternions -> [N] yaw angles (Z of the ZYX Euler angles)."""
    w, x, y, z = quat.unbind(dim=-1)
    return th.atan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))


def wrap_symmetric(angle: th.Tensor, symmetry: int) -> th.Tensor:
    """The `angle` wrapped into [-pi/symmetry, pi/symmetry), for a part of `symmetry`-fold symmetry."""
    period = 2 * math.pi / symmetry
    return th.remainder(angle + period / 2, period) - period / 2
