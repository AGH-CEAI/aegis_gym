"""
Peg-in-hole geometry built only from axis-aligned boxes, so every contact in the simulation is box-box.

Frames (all z up, quaternions wxyz):
- peg: centre of the tip plane (bottom face of the legs), z along the peg axis;
- board: centre of the top face;
- slot: centre of the hole group's inlet, in the board frame; the peg frame matches it when aligned above the slot.

The board is a plate with rectangular holes, decomposed into boxes by sweeping the hole edges along x: every x strip
gets one full-height box per free y interval, and adjacent strips with the same interval are merged. A blind hole
gets a floor box of height `board.bottom`.
"""

import itertools
from dataclasses import dataclass

from .cfg import PihAssetsCfg

_EPS = 1e-12
_IDENTITY_QUAT = (1.0, 0.0, 0.0, 0.0)

Pose = tuple[float, float, float, float, float, float, float]


@dataclass(frozen=True, slots=True)
class Box:
    """Axis-aligned box in its body frame."""

    center: tuple[float, float, float]
    size: tuple[float, float, float]

    @property
    def volume(self) -> float:
        return self.size[0] * self.size[1] * self.size[2]

    @property
    def bounds(self) -> tuple[tuple[float, float, float], tuple[float, float, float]]:
        lo = tuple(c - s / 2 for c, s in zip(self.center, self.size, strict=True))
        hi = tuple(c + s / 2 for c, s in zip(self.center, self.size, strict=True))
        return lo, hi


@dataclass(frozen=True, slots=True)
class Rect:
    """Axis-aligned rectangle in the xy plane."""

    x0: float
    x1: float
    y0: float
    y1: float


@dataclass(frozen=True, slots=True)
class Slot:
    """One target: a group of holes (one per leg) sharing a clearance."""

    clearance: float
    pose: Pose  # inlet centre in the board frame
    holes: tuple[Rect, ...]


@dataclass(frozen=True, slots=True)
class PihGeometry:
    peg_boxes: tuple[Box, ...]
    board_boxes: tuple[Box, ...]
    board_footprint: Rect
    slots: tuple[Slot, ...]
    peg_tip_frame: Pose
    peg_grasp_frame: Pose


def leg_offsets(cfg: PihAssetsCfg) -> list[tuple[float, float]]:
    """Leg centres in the peg (and slot) xy plane, row-major along x then y."""
    nx, ny = cfg.peg.legs.grid
    p = cfg.peg.legs.pitch
    return [
        ((i - (nx - 1) / 2) * p, (j - (ny - 1) / 2) * p)
        for j in range(ny)
        for i in range(nx)
    ]


def build_peg(cfg: PihAssetsCfg) -> tuple[Box, ...]:
    peg, legs = cfg.peg, cfg.peg.legs
    if legs.count == 1 and peg.leg_length == peg.length:
        return (
            Box(
                center=(0.0, 0.0, peg.length / 2), size=(peg.side, peg.side, peg.length)
            ),
        )

    leg_len = peg.leg_length
    boxes = [
        Box(center=(x, y, leg_len / 2), size=(peg.side, peg.side, leg_len))
        for x, y in leg_offsets(cfg)
    ]
    base_height = peg.length - leg_len
    if base_height > _EPS:
        nx, ny = legs.grid
        base_xy = ((nx - 1) * legs.pitch + peg.side, (ny - 1) * legs.pitch + peg.side)
        boxes.append(
            Box(
                center=(0.0, 0.0, leg_len + base_height / 2),
                size=(*base_xy, base_height),
            )
        )
    return tuple(boxes)


def build_slots(cfg: PihAssetsCfg) -> tuple[Slot, ...]:
    n = len(cfg.board.clearances)
    slots = []
    for k, clearance in enumerate(cfg.board.clearances):
        sx = (k - (n - 1) / 2) * cfg.board.slot_pitch
        half = (cfg.peg.side + clearance) / 2
        holes = tuple(
            Rect(sx + x - half, sx + x + half, y - half, y + half)
            for x, y in leg_offsets(cfg)
        )
        slots.append(
            Slot(clearance=clearance, pose=(sx, 0.0, 0.0, *_IDENTITY_QUAT), holes=holes)
        )
    return tuple(slots)


def board_footprint(cfg: PihAssetsCfg, slots: tuple[Slot, ...]) -> Rect:
    holes = [h for s in slots for h in s.holes]
    m = cfg.board.margin
    # symmetric about the board frame, so its origin is the centre of the top face
    hx = max(max(abs(h.x0), abs(h.x1)) for h in holes) + m
    hy = max(max(abs(h.y0), abs(h.y1)) for h in holes) + m
    return Rect(-hx, hx, -hy, hy)


def decompose_plate(outer: Rect, holes: list[Rect]) -> list[Rect]:
    """Covers `outer` minus `holes` (disjoint, inside `outer`) with disjoint rectangles."""
    xs = _unique_sorted([outer.x0, outer.x1, *(v for h in holes for v in (h.x0, h.x1))])
    open_strips: dict[
        tuple[float, float], float
    ] = {}  # free y interval -> x where it started
    rects: list[Rect] = []
    for xa, xb in itertools.pairwise(xs):
        blocking = sorted(
            (h.y0, h.y1) for h in holes if h.x0 <= xa + _EPS and h.x1 >= xb - _EPS
        )
        free = _free_intervals(outer.y0, outer.y1, blocking)
        for interval in list(open_strips):
            if interval not in free:
                rects.append(Rect(open_strips.pop(interval), xa, *interval))
        for interval in free:
            open_strips.setdefault(interval, xa)
    rects.extend(Rect(x0, xs[-1], *interval) for interval, x0 in open_strips.items())
    return rects


def build_board(
    cfg: PihAssetsCfg, slots: tuple[Slot, ...], outer: Rect
) -> tuple[Box, ...]:
    t, bottom = cfg.board.thickness, cfg.board.bottom
    holes = [h for s in slots for h in s.holes]
    boxes = [_rect_box(r, z0=-t, z1=0.0) for r in decompose_plate(outer, holes)]
    if bottom > _EPS:
        boxes.extend(_rect_box(h, z0=-t, z1=-t + bottom) for h in holes)
    return tuple(boxes)


def build_geometry(cfg: PihAssetsCfg) -> PihGeometry:
    slots = build_slots(cfg)
    outer = board_footprint(cfg, slots)
    return PihGeometry(
        peg_boxes=build_peg(cfg),
        board_boxes=build_board(cfg, slots, outer),
        board_footprint=outer,
        slots=slots,
        peg_tip_frame=(0.0, 0.0, 0.0, *_IDENTITY_QUAT),
        peg_grasp_frame=(
            0.0,
            0.0,
            cfg.peg.length - cfg.peg.grasp_from_top,
            *_IDENTITY_QUAT,
        ),
    )


def _rect_box(r: Rect, z0: float, z1: float) -> Box:
    return Box(
        center=((r.x0 + r.x1) / 2, (r.y0 + r.y1) / 2, (z0 + z1) / 2),
        size=(r.x1 - r.x0, r.y1 - r.y0, z1 - z0),
    )


def _unique_sorted(values: list[float]) -> list[float]:
    out: list[float] = []
    for v in sorted(values):
        if not out or v - out[-1] > _EPS:
            out.append(v)
    return out


def _free_intervals(
    y0: float, y1: float, blocking: list[tuple[float, float]]
) -> list[tuple[float, float]]:
    free, cursor = [], y0
    for b0, b1 in blocking:
        if b0 - cursor > _EPS:
            free.append((cursor, b0))
        cursor = max(cursor, b1)
    if y1 - cursor > _EPS:
        free.append((cursor, y1))
    return free
