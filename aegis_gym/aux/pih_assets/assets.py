"""
Generates the peg-in-hole asset files: on demand, cached by a hash of the parameters (`generate_pih_assets`), or into
a given directory (`export_pih_assets`, used for the files in `aegis_gym/assets/peg_in_hole/`).

The same parameters always give the same cache directory, so parallel processes and repeated runs reuse the files;
a new tolerance or size just produces a new directory.
"""

import hashlib
import json
import os
import shutil
import tempfile
from dataclasses import dataclass
from pathlib import Path

import yaml

from .cfg import PihAssetsCfg
from .export import export_stl
from .geometry import PihGeometry, Pose, build_geometry
from .urdf import write_urdf

# Bump when the generated files change for the same parameters, so stale caches are not reused.
GENERATOR_VERSION = 2
CACHE_ENV_VAR = "AEGIS_GYM_PIH_CACHE"
DEFAULT_CACHE_DIR = Path.home() / ".cache" / "aegis_gym" / "pih_assets"

PLA_DENSITY = 1240.0  # [kg/m^3], gives the (fixed) board a plausible mass
_PEG_RGBA = (0.85, 0.45, 0.1, 1.0)
_BOARD_RGBA = (0.25, 0.25, 0.3, 1.0)


@dataclass(frozen=True, slots=True)
class HoleFrame:
    clearance: float
    pose: Pose  # slot inlet centre in the board frame


@dataclass(frozen=True, slots=True)
class PihAssets:
    cfg: PihAssetsCfg
    geometry: PihGeometry
    directory: Path
    peg_urdf: Path
    board_urdf: Path
    peg_stl: Path
    board_stl: Path

    @property
    def hole_frames(self) -> tuple[HoleFrame, ...]:
        return tuple(HoleFrame(s.clearance, s.pose) for s in self.geometry.slots)

    @property
    def peg_tip_frame(self) -> Pose:
        return self.geometry.peg_tip_frame

    @property
    def peg_grasp_frame(self) -> Pose:
        return self.geometry.peg_grasp_frame


def params_hash(cfg: PihAssetsCfg) -> str:
    payload = json.dumps(
        {"version": GENERATOR_VERSION, "cfg": cfg.to_dict()}, sort_keys=True
    )
    return hashlib.sha256(payload.encode()).hexdigest()[:16]


def _assets(cfg: PihAssetsCfg, directory: Path) -> PihAssets:
    return PihAssets(
        cfg=cfg,
        geometry=build_geometry(cfg),
        directory=directory,
        peg_urdf=directory / "peg.urdf",
        board_urdf=directory / "board.urdf",
        peg_stl=directory / "peg.stl",
        board_stl=directory / "board.stl",
    )


def generate_pih_assets(
    cfg: PihAssetsCfg | None = None, cache_dir: str | Path | None = None
) -> PihAssets:
    """Returns the assets for `cfg` (nominal when None), writing them only if not cached yet."""
    cfg = cfg or PihAssetsCfg.from_yaml()
    root = Path(cache_dir or os.environ.get(CACHE_ENV_VAR) or DEFAULT_CACHE_DIR)
    assets = _assets(cfg, root / f"pih_{params_hash(cfg)}")
    if not assets.directory.is_dir():
        root.mkdir(parents=True, exist_ok=True)
        # write next to the target and rename, so a concurrent reader never sees a half-written directory
        tmp = Path(tempfile.mkdtemp(dir=root, prefix=".tmp_"))
        try:
            _write_files(assets, tmp)
            try:
                tmp.rename(assets.directory)
            except OSError:
                # lost a race to another writer: their files are identical
                if not assets.directory.is_dir():
                    raise
        finally:
            shutil.rmtree(tmp, ignore_errors=True)
    return assets


def export_pih_assets(cfg: PihAssetsCfg | None, out_dir: str | Path) -> PihAssets:
    """Writes the assets for `cfg` (nominal when None) into `out_dir`, overwriting the files there."""
    cfg = cfg or PihAssetsCfg.from_yaml()
    assets = _assets(cfg, Path(out_dir))
    assets.directory.mkdir(parents=True, exist_ok=True)
    _write_files(assets, assets.directory)
    return assets


def _write_files(assets: PihAssets, directory: Path) -> None:
    """peg/board URDF and STL plus `params.yaml` into `directory`, named as in `assets`."""
    geom, cfg = assets.geometry, assets.cfg
    export_stl(geom.peg_boxes, directory / assets.peg_stl.name)
    export_stl(geom.board_boxes, directory / assets.board_stl.name)
    board_mass = PLA_DENSITY * sum(b.volume for b in geom.board_boxes)
    write_urdf(
        directory / assets.peg_urdf.name,
        "peg",
        geom.peg_boxes,
        cfg.peg.mass,
        assets.peg_stl.name,
        _PEG_RGBA,
    )
    write_urdf(
        directory / assets.board_urdf.name,
        "board",
        geom.board_boxes,
        board_mass,
        assets.board_stl.name,
        _BOARD_RGBA,
    )
    with open(directory / "params.yaml", "w") as f:
        yaml.safe_dump(
            {"generator_version": GENERATOR_VERSION, **_plain(cfg.to_dict())},
            f,
            sort_keys=False,
        )


def _plain(value):
    """Tuples to lists, so the YAML stays readable without Python tags."""
    if isinstance(value, dict):
        return {k: _plain(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_plain(v) for v in value]
    return value
