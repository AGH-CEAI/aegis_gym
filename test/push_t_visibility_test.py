import math

import pytest
import torch as th

from aegis_gym.envs.push_t_visibility import (
    TEE_RECTANGLES,
    place_tee_points,
    project_points,
    steps_since_visible,
    tee_top_points,
    visible_fraction,
)


def _camera_above(height: float = 1.0) -> th.Tensor:
    """[1, 4, 4] a camera at `height` above the origin, looking down (-z), the image's up along +y."""
    transform = th.eye(4)
    transform[2, 3] = height
    return transform.unsqueeze(0)


def test_tee_top_points_are_inside_the_outline_with_the_margin():
    points = tee_top_points(spacing=0.01, margin=0.005)
    inside = th.zeros(len(points), dtype=th.bool)
    for x_min, x_max, y_min, y_max in TEE_RECTANGLES:
        inside |= (
            (points[:, 0] >= x_min + 0.005 - 1e-6)
            & (points[:, 0] <= x_max - 0.005 + 1e-6)
            & (points[:, 1] >= y_min + 0.005 - 1e-6)
            & (points[:, 1] <= y_max - 0.005 + 1e-6)
        )
    assert inside.all()
    # the crossbar 0.19 x 0.04 and the stem 0.04 x 0.14 inner areas at 1 cm: ~100 + 75 points
    assert 150 < len(points) < 200
    assert len(th.unique(points, dim=0)) == len(points)


def test_place_tee_points_rotates_about_the_tee_origin():
    points = th.tensor([[0.1, 0.0], [0.0, -0.1]])
    placed = place_tee_points(
        points,
        xy=th.tensor([[0.5, 0.2]]),
        yaw=th.tensor([math.pi / 2]),
        z=th.tensor([0.04]),
    )
    assert th.allclose(
        placed[0], th.tensor([[0.5, 0.3, 0.04], [0.6, 0.2, 0.04]]), atol=1e-6
    )


def test_projection_in_the_opengl_camera_convention():
    points = th.tensor(
        [[[0.0, 0.0, 0.0], [0.1, 0.0, 0.0], [0.0, 0.1, 0.0], [0.0, 0.0, 2.0]]]
    )
    pixels, in_front = project_points(
        _camera_above(), points, fov_deg=90.0, width=64, height=48
    )
    focal = 0.5 * 48 / math.tan(math.radians(45))
    assert pixels[0, 0].tolist() == pytest.approx([32.0, 24.0])
    # +x to the right, +y up in the image (smaller rows)
    assert pixels[0, 1].tolist() == pytest.approx([32.0 + 0.1 * focal, 24.0])
    assert pixels[0, 2].tolist() == pytest.approx([32.0, 24.0 - 0.1 * focal])
    assert in_front[0].tolist() == [True, True, True, False]


def _segmentation_of(
    pixels: th.Tensor, width: int, height: int, label: int
) -> th.Tensor:
    segmentation = th.zeros(1, height, width, dtype=th.long)
    column, row = pixels[0, :, 0].floor().long(), pixels[0, :, 1].floor().long()
    segmentation[0, row, column] = label
    return segmentation


def test_visible_fraction_is_the_tee_share_among_tee_and_occluder_points():
    tee, robot, table = 3, 7, 1
    points = place_tee_points(
        tee_top_points(), xy=th.zeros(1, 2), yaw=th.zeros(1), z=th.zeros(1)
    )
    pixels, in_front = project_points(
        _camera_above(), points, 40.0, width=256, height=256
    )
    segmentation = _segmentation_of(pixels, 256, 256, tee)
    tee_labels, occluder_labels = th.tensor([tee]), th.tensor([robot])

    full = visible_fraction(segmentation, pixels, in_front, tee_labels, occluder_labels)
    assert full.tolist() == pytest.approx([1.0])

    # the robot covers the image's left half (x < 0): the points with x >= 0 stay visible
    segmentation[0, :, :128] = th.where(segmentation[0, :, :128] == tee, robot, table)
    half = visible_fraction(segmentation, pixels, in_front, tee_labels, occluder_labels)
    expected = (points[0, :, 0] >= 0).float().mean()
    assert half.tolist() == pytest.approx([expected.item()], abs=1e-6)

    # the table at the tee's pixels (outside the tee and the occluders): no estimate
    nothing = visible_fraction(
        th.full_like(segmentation, table), pixels, in_front, tee_labels, occluder_labels
    )
    assert math.isnan(nothing.item())


def test_points_out_of_the_image_or_behind_the_camera_are_ignored():
    pixels = th.tensor([[[-1.0, 0.0], [0.5, 0.5], [0.5, 0.5]]])
    in_front = th.tensor([[True, True, False]])
    segmentation = th.full((1, 2, 2), 3)
    fraction = visible_fraction(
        segmentation, pixels, in_front, th.tensor([3]), th.tensor([7])
    )
    assert fraction.tolist() == [1.0]


def test_steps_since_visible():
    fractions = th.tensor([[0.9, 0.1], [0.2, 0.1], [float("nan"), 0.8], [0.6, 0.3]])
    assert steps_since_visible(fractions, threshold=0.5).tolist() == [
        [0, 1],
        [1, 2],
        [2, 0],
        [0, 1],
    ]
