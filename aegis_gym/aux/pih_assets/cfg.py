"""
Parameters of the peg-in-hole assets, loaded from `assets/peg_in_hole/nominal.yaml` with optional overrides.

All lengths are in metres. A clearance is total, as in the SCAD sources: hole side minus leg side.
"""

import copy
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import yaml

NOMINAL_YAML = (
    Path(__file__).resolve().parents[2] / "assets" / "peg_in_hole" / "nominal.yaml"
)


@dataclass(frozen=True, slots=True)
class LegsCfg:
    grid: tuple[int, int] = (1, 1)
    pitch: float = 0.04
    length: float | None = None

    @property
    def count(self) -> int:
        return self.grid[0] * self.grid[1]


@dataclass(frozen=True, slots=True)
class PegCfg:
    side: float
    length: float
    mass: float
    grasp_from_top: float
    legs: LegsCfg = field(default_factory=LegsCfg)

    @property
    def leg_length(self) -> float:
        return self.length if self.legs.length is None else self.legs.length


@dataclass(frozen=True, slots=True)
class BoardCfg:
    thickness: float
    bottom: float
    margin: float
    slot_pitch: float
    clearances: tuple[float, ...]

    @property
    def depth(self) -> float:
        return self.thickness - self.bottom


@dataclass(frozen=True, slots=True)
class PihAssetsCfg:
    peg: PegCfg
    board: BoardCfg

    def __post_init__(self) -> None:
        _validate(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "PihAssetsCfg":
        try:
            peg = dict(data["peg"])
            legs = dict(peg.pop("legs", {}))
            if "grid" in legs:
                legs["grid"] = tuple(int(n) for n in legs["grid"])
            board = dict(data["board"])
            board["clearances"] = tuple(float(c) for c in board["clearances"])
            return cls(peg=PegCfg(**peg, legs=LegsCfg(**legs)), board=BoardCfg(**board))
        except (KeyError, TypeError) as e:
            raise ValueError(f"Invalid peg-in-hole assets config: {e}") from e

    @classmethod
    def from_yaml(
        cls, path: str | Path = NOMINAL_YAML, overrides: dict[str, Any] | None = None
    ) -> "PihAssetsCfg":
        """Loads the nominal dimensions and deep-merges `overrides` (e.g. `{"board": {"clearances": [0.001]}}`)."""
        with open(path) as f:
            data = yaml.safe_load(f)
        return cls.from_dict(_deep_merge(data, overrides or {}))

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _deep_merge(base: dict, overrides: dict) -> dict:
    merged = copy.deepcopy(base)
    for key, value in overrides.items():
        if key not in merged:
            raise ValueError(f"Unknown peg-in-hole config key: {key!r}")
        if isinstance(value, dict) and isinstance(merged[key], dict):
            merged[key] = _deep_merge(merged[key], value)
        else:
            merged[key] = copy.deepcopy(value)
    return merged


def _validate(cfg: PihAssetsCfg) -> None:
    peg, legs, board = cfg.peg, cfg.peg.legs, cfg.board
    errors = []

    if min(peg.side, peg.length, peg.mass) <= 0:
        errors.append("peg side, length and mass must be positive")
    if not 0 < peg.grasp_from_top < peg.length:
        errors.append("peg.grasp_from_top must lie within the peg length")
    if len(legs.grid) != 2 or min(legs.grid) < 1:
        errors.append("peg.legs.grid must be two integers >= 1")
    elif legs.count > 1:
        if legs.length is None:
            errors.append("peg.legs.length is required with more than one leg")
        if legs.pitch <= peg.side + max(board.clearances, default=0.0):
            errors.append(
                "peg.legs.pitch must exceed side + clearance, so walls remain between the holes"
            )
    if legs.length is not None and not 0 < legs.length <= peg.length:
        errors.append("peg.legs.length must lie within (0, peg.length]")

    if min(board.thickness, board.margin) <= 0:
        errors.append("board thickness and margin must be positive")
    if not 0 <= board.bottom < board.thickness:
        errors.append("board.bottom must lie within [0, thickness)")
    if not board.clearances:
        errors.append("board.clearances must not be empty")
    elif min(board.clearances) < 0:
        errors.append("board.clearances must be non-negative")
    elif len(board.clearances) > 1:
        slot_span = (legs.grid[0] - 1) * legs.pitch + peg.side + max(board.clearances)
        if board.slot_pitch <= slot_span:
            errors.append(
                f"board.slot_pitch must exceed the slot width ({slot_span:.4f} m)"
            )

    if errors:
        raise ValueError("Invalid peg-in-hole assets config: " + "; ".join(errors))
