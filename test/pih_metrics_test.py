import math

import numpy as np
import pytest

from aegis_gym.aux.pih_assets import Box
from aegis_gym.aux.pih_assets.metrics import (
    max_penetration,
    quat_to_matrix,
    union_depth,
    world_vertices,
)


def _pose(x=0.0, y=0.0, z=0.0, tilt_deg=0.0):
    half = math.radians(tilt_deg) / 2
    return (x, y, z, math.cos(half), 0.0, math.sin(half), 0.0)  # tilt about y


_PLATE = (Box(center=(0.0, 0.0, -0.005), size=(0.1, 0.1, 0.01)),)  # top face at z = 0
_CUBE = (
    Box(center=(0.0, 0.0, 0.01), size=(0.02, 0.02, 0.02)),
)  # bottom face at its frame


def test_penetration_zero_when_apart_or_touching():
    assert max_penetration(_CUBE, _pose(z=0.001), _PLATE, _pose()) == 0.0
    assert max_penetration(_CUBE, _pose(z=0.0), _PLATE, _pose()) == pytest.approx(
        0.0, abs=1e-12
    )


def test_penetration_of_a_sunk_cube():
    assert max_penetration(_CUBE, _pose(z=-0.001), _PLATE, _pose()) == pytest.approx(
        0.001
    )


def test_penetration_of_a_tilted_corner():
    # tilted 30 deg about y, the lowest corner (x = +10 mm, z = 0) drops 10 mm * sin(30) = 5 mm below the frame
    lift = 0.01 * math.sin(math.radians(30)) - 0.0005
    depth = max_penetration(_CUBE, _pose(z=lift, tilt_deg=30), _PLATE, _pose())
    assert depth == pytest.approx(0.0005, abs=1e-9)


def test_penetration_of_a_plate_edge_into_a_peg():
    """A rim corner inside the peg's side is caught from the other set's vertices."""
    peg = (Box(center=(0.0, 0.0, 0.0), size=(0.02, 0.02, 0.02)),)
    rim = (
        Box(center=(0.0195, 0.0195, 0.0), size=(0.02, 0.02, 0.005)),
    )  # corner 0.5 mm in, both ways
    assert max_penetration(peg, _pose(), rim, _pose()) == pytest.approx(
        0.0005, abs=1e-9
    )


def test_penetration_is_symmetric():
    a = max_penetration(_CUBE, _pose(z=-0.002, tilt_deg=7), _PLATE, _pose())
    b = max_penetration(_PLATE, _pose(), _CUBE, _pose(z=-0.002, tilt_deg=7))
    assert a == pytest.approx(b)


def test_quat_to_matrix_rotates_about_y():
    r = quat_to_matrix(_pose(tilt_deg=90)[3:])
    assert r @ np.array([1.0, 0.0, 0.0]) == pytest.approx([0.0, 0.0, -1.0], abs=1e-12)


def test_world_vertices():
    v = world_vertices(_CUBE, _pose(x=1.0, z=2.0))
    assert v.shape == (8, 3)
    assert v.min(axis=0) == pytest.approx([0.99, -0.01, 2.0])
    assert v.max(axis=0) == pytest.approx([1.01, 0.01, 2.02])


def test_hole_floor_next_to_a_wall_is_measured_from_the_floor_top():
    """A corner sunk 2 mm into the hole floor, 0.25 mm from the wall: the floor's side face is inside the board."""
    wall = Box(
        center=(-0.02, 0.0, -0.015), size=(0.02, 0.1, 0.03)
    )  # x in [-0.03, -0.01], full height
    floor = Box(
        center=(0.0, 0.0, -0.0275), size=(0.02, 0.1, 0.005)
    )  # x in [-0.01, 0.01], top at -0.025
    cube = (
        Box(center=(0.0, 0.0, 0.00975), size=(0.0195, 0.0195, 0.0195)),
    )  # 0.25 mm from the wall
    depth = max_penetration(cube, _pose(z=-0.025 - 0.002), (wall, floor), _pose())
    assert depth == pytest.approx(0.002, abs=1e-9)


def test_union_depth_runs_through_adjacent_boxes():
    boxes = (
        Box(center=(0.005, 0.0, 0.0), size=(0.01, 0.01, 0.01)),
        Box(center=(0.015, 0.0, 0.0), size=(0.01, 0.01, 0.01)),
    )  # one 20 x 10 x 10 mm solid
    depth = union_depth(np.array([[0.009, 0.0, 0.0], [0.03, 0.0, 0.0]]), boxes)
    assert depth == pytest.approx(
        [0.005, 0.0]
    )  # shortest way out is across y/z (5 mm), not x


def test_batch_matches_single_poses():
    from aegis_gym.aux.pih_assets.metrics import max_penetration_batch

    poses = [
        _pose(z=-0.002, tilt_deg=7),
        _pose(x=0.03, z=0.001),
        _pose(z=0.002, tilt_deg=30),
        _pose(y=-0.01),
    ]
    batch = max_penetration_batch(_CUBE, np.array(poses), _PLATE, _pose(z=0.0005))
    single = [max_penetration(_CUBE, p, _PLATE, _pose(z=0.0005)) for p in poses]
    assert batch == pytest.approx(single, abs=1e-12)
    assert batch.max() > 0  # the cases do overlap


def test_corner_inside_a_hole_corner_is_shallow():
    """0.1 mm and 0.2 mm past both walls at the corner of a hole: the way out is diagonal, back into the hole."""
    walls = (
        Box(center=(-0.015, 0.0, -0.015), size=(0.01, 0.04, 0.03)),  # x < -0.01
        Box(
            center=(0.0, -0.015, -0.015), size=(0.02, 0.01, 0.03)
        ),  # y < -0.01, between the x walls
    )  # the hole corner is at (-0.01, -0.01); the hole is the empty quadrant x > -0.01, y > -0.01
    depth = union_depth(np.array([[-0.0101, -0.0102, -0.015]]), walls)
    assert depth == pytest.approx([math.hypot(0.0001, 0.0002)], abs=1e-9)
