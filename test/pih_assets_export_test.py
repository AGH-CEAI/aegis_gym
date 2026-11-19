import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np
import pytest

from aegis_gym.aux.pih_assets import (
    Box,
    PihAssetsCfg,
    boxes_to_mesh,
    export_pih_assets,
    export_stl,
    generate_pih_assets,
    mass_properties,
    params_hash,
)

_TWO_BY_TWO = {
    "peg": {"legs": {"grid": [2, 2], "length": 0.03}},
    "board": {"slot_pitch": 0.1},
}


@pytest.fixture(params=[None, _TWO_BY_TWO], ids=["1x1", "2x2"])
def assets(request, tmp_path):
    return generate_pih_assets(
        PihAssetsCfg.from_yaml(overrides=request.param), cache_dir=tmp_path
    )


def _collision_boxes(urdf_path) -> list[tuple[list[float], list[float]]]:
    root = ET.parse(urdf_path).getroot()
    out = []
    for col in root.iter("collision"):
        xyz = [float(v) for v in col.find("origin").get("xyz").split()]
        size = [float(v) for v in col.find("geometry/box").get("size").split()]
        out.append((xyz, size))
    return out


def test_files_written(assets):
    for path in (assets.peg_urdf, assets.board_urdf, assets.peg_stl, assets.board_stl):
        assert path.is_file()
    assert (assets.directory / "params.yaml").is_file()


@pytest.mark.parametrize("part", ["peg", "board"])
def test_urdf_boxes_match_geometry(assets, part):
    boxes = getattr(assets.geometry, f"{part}_boxes")
    parsed = _collision_boxes(getattr(assets, f"{part}_urdf"))
    assert len(parsed) == len(boxes)
    for (xyz, size), box in zip(parsed, boxes, strict=True):
        assert xyz == pytest.approx(box.center, abs=1e-12)
        assert size == pytest.approx(box.size, abs=1e-12)


def test_urdf_keeps_sub_millimetre_clearance(tmp_path):
    """Hole width read back from the URDF still gives the clearance to well under a micrometre."""
    assets = generate_pih_assets(
        PihAssetsCfg.from_yaml(overrides={"board": {"clearances": [0.0002]}}), tmp_path
    )
    boxes = _collision_boxes(assets.board_urdf)
    edges = sorted(
        {
            round(xyz[0] + sign * size[0] / 2, 12)
            for xyz, size in boxes
            for sign in (-1, 1)
        }
    )
    # x edges: board edge, hole edge, hole edge, board edge
    assert len(edges) == 4
    assert edges[2] - edges[1] - assets.cfg.peg.side == pytest.approx(0.0002, abs=1e-9)


def test_urdf_visual_mesh_is_relative(assets):
    mesh = ET.parse(assets.peg_urdf).getroot().find("link/visual/geometry/mesh")
    assert mesh.get("filename") == assets.peg_stl.name


@pytest.mark.parametrize("part", ["peg", "board"])
def test_stl_is_watertight_union(assets, part):
    import trimesh

    mesh = trimesh.load(getattr(assets, f"{part}_stl"))
    boxes = getattr(assets.geometry, f"{part}_boxes")
    assert mesh.is_watertight
    assert mesh.is_winding_consistent
    # STL stores float32, so ~1e-7 relative; collisions come from the exact URDF boxes, not from this mesh
    # a positive volume also means outward normals
    assert mesh.volume == pytest.approx(sum(b.volume for b in boxes), rel=1e-6)
    lows, highs = zip(*(b.bounds for b in boxes), strict=True)
    assert mesh.bounds[0] == pytest.approx(np.min(lows, axis=0), abs=1e-8)
    assert mesh.bounds[1] == pytest.approx(np.max(highs, axis=0), abs=1e-8)


def test_stl_in_millimetres(tmp_path):
    mesh = export_stl(
        [Box(center=(0, 0, 0.04), size=(0.025, 0.025, 0.08))],
        tmp_path / "peg_mm.stl",
        scale=1000,
    )
    assert mesh.extents == pytest.approx([25, 25, 80])


def test_touching_boxes_merge_into_one_shell():
    mesh = boxes_to_mesh([Box((0, 0, 0.5), (1, 1, 1)), Box((0, 0, 1.5), (1, 1, 1))])
    assert mesh.is_watertight
    assert mesh.volume == pytest.approx(2.0)
    assert len(mesh.split(only_watertight=False)) == 1


def test_single_box_inertia():
    box = Box(center=(0.1, 0.0, 0.2), size=(0.02, 0.04, 0.06))
    com, inertia = mass_properties([box], mass=0.5)
    assert com == pytest.approx(box.center)
    expected = (
        0.5 / 12 * np.diag([0.04**2 + 0.06**2, 0.02**2 + 0.06**2, 0.02**2 + 0.04**2])
    )
    assert inertia == pytest.approx(expected)


def test_split_box_inertia_equals_whole():
    whole = Box(center=(0, 0, 0.5), size=(0.2, 0.3, 1.0))
    halves = [Box((0, 0, 0.25), (0.2, 0.3, 0.5)), Box((0, 0, 0.75), (0.2, 0.3, 0.5))]
    for got, exp in zip(
        mass_properties(halves, 2.0), mass_properties([whole], 2.0), strict=True
    ):
        assert got == pytest.approx(exp)


def test_cache_reused(tmp_path):
    cfg = PihAssetsCfg.from_yaml()
    first = generate_pih_assets(cfg, tmp_path)
    mtime = first.peg_urdf.stat().st_mtime_ns
    second = generate_pih_assets(cfg, tmp_path)
    assert second.directory == first.directory
    assert second.peg_urdf.stat().st_mtime_ns == mtime
    assert not list(tmp_path.glob(".tmp_*"))


def test_new_params_new_directory(tmp_path):
    a = generate_pih_assets(PihAssetsCfg.from_yaml(), tmp_path)
    b = generate_pih_assets(
        PihAssetsCfg.from_yaml(overrides={"board": {"clearances": [0.002]}}), tmp_path
    )
    assert a.directory != b.directory
    assert params_hash(a.cfg) == params_hash(PihAssetsCfg.from_yaml())


def test_cache_dir_from_env(tmp_path, monkeypatch):
    monkeypatch.setenv("AEGIS_GYM_PIH_CACHE", str(tmp_path))
    assert generate_pih_assets().directory.parent == tmp_path


def test_hole_frames(assets):
    frames = assets.hole_frames
    assert [f.clearance for f in frames] == list(assets.cfg.board.clearances)
    assert all(len(f.pose) == 7 for f in frames)


_ASSETS_DIR = (
    Path(__file__).resolve().parents[1] / "aegis_gym" / "assets" / "peg_in_hole"
)
_ASSET_FILES = ("peg.urdf", "peg.stl", "board.urdf", "board.stl", "params.yaml")


@pytest.mark.parametrize("name", _ASSET_FILES)
def test_committed_assets_are_up_to_date(tmp_path, name):
    """The files the envs load match nominal.yaml: after a change run `python3 -m scripts.peg_in_hole.generate_assets`."""
    export_pih_assets(None, tmp_path)
    assert (_ASSETS_DIR / name).read_bytes() == (tmp_path / name).read_bytes()


def test_committed_board_has_one_1mm_hole():
    boxes = _collision_boxes(_ASSETS_DIR / "board.urdf")
    walls = [
        (xyz, size) for xyz, size in boxes if size[2] > 0.01
    ]  # the floor is 5 mm thick
    # the hole is the gap between the walls around the origin, along x and along y
    gap_x = min(x - s[0] / 2 for (x, _, _), s in walls if x > 0) - max(
        x + s[0] / 2 for (x, _, _), s in walls if x < 0
    )
    assert gap_x == pytest.approx(0.026, abs=1e-9)
