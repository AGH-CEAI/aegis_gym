"""
The visible share of the tee's top face in a camera image, from a segmentation image: sample points on
the top face, projected into the camera, read back from the segmentation. For the memory students'
analysis (the tee pose estimate vs the time since the tee was last visible).
"""

import math

import torch as th

# The tee's outline in its frame [m] (the frame of `env.object.get_pose()`, `T_shape.urdf`'s collision
# boxes): the crossbar and the stem. The frame's origin is on the bottom face.
TEE_RECTANGLES: tuple[tuple[float, float, float, float], ...] = (
    (-0.1, 0.1, -0.025, 0.025),  # x min, x max, y min, y max
    (-0.025, 0.025, -0.175, -0.025),
)
TEE_HEIGHT = 0.04


def tee_top_points(spacing: float = 0.01, margin: float = 0.005) -> th.Tensor:
    """
    [N, 2] a grid on the tee's top face, `margin` inside its outline (the outline's pixels mix the
    tee with the table in a low resolution image).
    """
    points = []
    for x_min, x_max, y_min, y_max in TEE_RECTANGLES:
        xs = th.arange(x_min + margin, x_max - margin + 1e-9, spacing)
        ys = th.arange(y_min + margin, y_max - margin + 1e-9, spacing)
        points.append(th.cartesian_prod(xs, ys))
    return th.unique(th.cat(points).round(decimals=6), dim=0)


def place_tee_points(
    points: th.Tensor, xy: th.Tensor, yaw: th.Tensor, z: th.Tensor
) -> th.Tensor:
    """[B, N, 3] the tee frame's `points` [N, 2] at the planar poses `xy` [B, 2], `yaw` [B], height `z` [B]."""
    cos, sin = th.cos(yaw), th.sin(yaw)
    rotation = th.stack([th.stack([cos, -sin], -1), th.stack([sin, cos], -1)], -2)
    planar = points.to(xy) @ rotation.transpose(-1, -2) + xy.unsqueeze(1)
    return th.cat([planar, z.view(-1, 1, 1).expand(-1, len(points), 1)], dim=-1)


def project_points(
    camera_transform: th.Tensor,
    points: th.Tensor,
    fov_deg: float,
    width: int,
    height: int,
) -> tuple[th.Tensor, th.Tensor]:
    """
    The pixel coordinates [B, N, 2] (column, row) of the world `points` [B, N, 3] in the cameras with
    the camera -> world `camera_transform` [B, 4, 4] in the OpenGL convention (x right, y up, looking
    along -z; Genesis' `Camera.get_transform()`) and the vertical `fov_deg`; and [B, N] whether the
    point is in front of the camera.
    """
    rotation = camera_transform[:, :3, :3]
    position = camera_transform[:, :3, 3]
    local = (points - position.unsqueeze(1)) @ rotation
    x, y, z = local.unbind(-1)
    depth = -z
    in_front = depth > 1e-6
    depth = th.where(in_front, depth, th.ones_like(depth))
    focal = 0.5 * height / math.tan(math.radians(0.5 * fov_deg))
    column = 0.5 * width + focal * x / depth
    row = 0.5 * height - focal * y / depth
    return th.stack([column, row], dim=-1), in_front


def visible_fraction(
    segmentation: th.Tensor,
    pixels: th.Tensor,
    in_front: th.Tensor,
    tee_labels: th.Tensor,
    occluder_labels: th.Tensor,
) -> th.Tensor:
    """
    [B] the share of the tee's sample points seen as the tee among those seen as the tee or an
    occluder (NaN when none), from the `segmentation` [B, H, W] labels at the `pixels` [B, N, 2] of
    the points `in_front` [B, N]. Points on other labels (the table at the outline, out of the image)
    don't count.
    """
    num_envs, height, width = segmentation.shape
    column = pixels[..., 0].floor().long()
    row = pixels[..., 1].floor().long()
    inside = in_front & (column >= 0) & (column < width) & (row >= 0) & (row < height)
    flat = row.clamp(0, height - 1) * width + column.clamp(0, width - 1)
    labels = segmentation.reshape(num_envs, -1).gather(1, flat)
    is_tee = inside & th.isin(labels, tee_labels.to(labels))
    is_occluder = inside & th.isin(labels, occluder_labels.to(labels))
    seen = (is_tee | is_occluder).sum(-1)
    fraction = is_tee.sum(-1).float() / seen.clamp(min=1)
    return th.where(seen > 0, fraction, th.full_like(fraction, float("nan")))


def steps_since_visible(fractions: th.Tensor, threshold: float = 0.5) -> th.Tensor:
    """
    [T, B] the steps since the tee was last visible (`fractions` [T, B] >= `threshold`; 0 while
    visible), counted from the episode's first step if it wasn't visible yet. NaN counts as hidden.
    """
    visible = fractions >= threshold
    counts = th.zeros_like(fractions, dtype=th.long)
    running = th.zeros(fractions.shape[1], dtype=th.long, device=fractions.device)
    for t in range(fractions.shape[0]):
        running = th.where(visible[t], th.zeros_like(running), running + 1)
        counts[t] = running
    return counts
