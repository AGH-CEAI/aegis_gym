"""
Contact metrics on the generated boxes, independent of the simulator.

`max_penetration` estimates how deep two rigid bodies, each a union of boxes axis-aligned in its own frame,
interpenetrate: the deepest corner of either body inside the other. The depth of a point is its Euclidean distance
to the empty space of the other body (`union_depth`), so faces shared by adjacent boxes (a hole floor next to the
hole walls) are not surface, and a corner just inside the inner edge of a hole wall is shallow. It catches a corner
or an edge end digging into a face (what a peg does when it hits the rim of a hole), not two edges crossing
mid-length, so it is a lower bound of the true depth.
"""

import functools
import itertools

import numpy as np

from .geometry import Box

_CORNERS = np.array(list(itertools.product((-1.0, 1.0), repeat=3)))  # (8, 3)
_EPS = 1e-9  # [m] a point closer than this to the surface counts as outside
_FAR = 1e3  # [m] the extent of the padding cells around a body


def quat_to_matrix(quat) -> np.ndarray:
    """Rotation matrix of a wxyz quaternion."""
    w, x, y, z = np.asarray(quat, dtype=float) / np.linalg.norm(quat)
    return np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
            [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
            [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)],
        ]
    )


def _bounds(boxes: tuple[Box, ...]) -> tuple[np.ndarray, np.ndarray]:
    """(n, 3) lower and upper corners of the boxes in their body frame."""
    centres = np.array([b.center for b in boxes], dtype=float)
    half = np.array([b.size for b in boxes], dtype=float) / 2
    return centres - half, centres + half


def _corners(boxes: tuple[Box, ...]) -> np.ndarray:
    """(8 * n, 3) corners of the boxes in their body frame."""
    lo, hi = _bounds(boxes)
    centres, half = (lo + hi) / 2, (hi - lo) / 2
    return (centres[:, None, :] + _CORNERS[None] * half[:, None, :]).reshape(-1, 3)


def world_vertices(boxes: tuple[Box, ...], pose) -> np.ndarray:
    """Corners (8 per box) of a body at `pose` (x, y, z, qw, qx, qy, qz) in the world, shape (8 * n, 3)."""
    return _corners(boxes) @ quat_to_matrix(pose[3:]).T + np.asarray(
        pose[:3], dtype=float
    )


@functools.lru_cache(maxsize=32)
def _free_cells(boxes: tuple[Box, ...]) -> tuple[np.ndarray, np.ndarray]:
    """
    (m, 3) lower and upper corners of the empty space around and inside the union of `boxes`: the empty cells of
    the rectilinear grid through all box faces, padded with unbounded cells outside it.
    """
    lo, hi = _bounds(boxes)
    axes = []
    for i in range(3):
        values = np.unique(np.round(np.concatenate([lo[:, i], hi[:, i]]), 12))
        axes.append(np.concatenate([[-_FAR], values, [_FAR]]))
    centres = np.stack(
        np.meshgrid(*[(a[:-1] + a[1:]) / 2 for a in axes], indexing="ij"), axis=-1
    ).reshape(-1, 3)
    solid = np.all((centres[:, None, :] > lo) & (centres[:, None, :] < hi), axis=2).any(
        axis=1
    )
    index = np.stack(
        np.meshgrid(*[np.arange(len(a) - 1) for a in axes], indexing="ij"), axis=-1
    ).reshape(-1, 3)
    free = index[~solid]
    cell_lo = np.stack([axes[i][free[:, i]] for i in range(3)], axis=1)
    cell_hi = np.stack([axes[i][free[:, i] + 1] for i in range(3)], axis=1)
    return cell_lo, cell_hi


def union_depth(points: np.ndarray, boxes: tuple[Box, ...]) -> np.ndarray:
    """
    Depth of every point (body frame) inside the union of `boxes`: the Euclidean distance to the nearest point
    outside it (to the nearest empty grid cell); 0 for points outside or on the surface.
    """
    lo, hi = _bounds(boxes)
    inside = np.all(
        (points[:, None, :] > lo + _EPS) & (points[:, None, :] < hi - _EPS), axis=2
    ).any(axis=1)
    depth = np.zeros(len(points))
    if inside.any():
        cell_lo, cell_hi = _free_cells(tuple(boxes))
        p = points[inside][:, None, :]
        gap = np.maximum(
            np.maximum(cell_lo - p, p - cell_hi), 0.0
        )  # (P, m, 3) point to cell, per axis
        depth[inside] = np.sqrt((gap**2).sum(axis=2)).min(axis=1)
    return depth


def max_penetration(
    boxes_a: tuple[Box, ...], pose_a, boxes_b: tuple[Box, ...], pose_b
) -> float:
    """Deepest penetration [m] between two rigid box unions at the given poses (0 when apart or just touching)."""
    rot_a, rot_b = quat_to_matrix(pose_a[3:]), quat_to_matrix(pose_b[3:])
    pos_a, pos_b = (
        np.asarray(pose_a[:3], dtype=float),
        np.asarray(pose_b[:3], dtype=float),
    )
    # corners of each body expressed in the other body's frame
    a_in_b = ((_corners(boxes_a) @ rot_a.T + pos_a) - pos_b) @ rot_b
    b_in_a = ((_corners(boxes_b) @ rot_b.T + pos_b) - pos_a) @ rot_a
    return float(
        max(union_depth(a_in_b, boxes_b).max(), union_depth(b_in_a, boxes_a).max())
    )


def quats_to_matrices(quats: np.ndarray) -> np.ndarray:
    """(n, 3, 3) rotation matrices of (n, 4) wxyz quaternions."""
    q = np.asarray(quats, dtype=float)
    w, x, y, z = (q / np.linalg.norm(q, axis=1, keepdims=True)).T
    return np.stack(
        [
            np.stack(
                [1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
                axis=1,
            ),
            np.stack(
                [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
                axis=1,
            ),
            np.stack(
                [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)],
                axis=1,
            ),
        ],
        axis=1,
    )


def max_penetration_batch(
    boxes_a: tuple[Box, ...], poses_a: np.ndarray, boxes_b: tuple[Box, ...], pose_b
) -> np.ndarray:
    """`max_penetration` of body A at each of (n, 7) `poses_a` against body B at one pose, shape (n,)."""
    poses_a = np.asarray(poses_a, dtype=float)
    rot_a, pos_a = quats_to_matrices(poses_a[:, 3:]), poses_a[:, :3]
    rot_b, pos_b = quat_to_matrix(pose_b[3:]), np.asarray(pose_b[:3], dtype=float)
    n = len(poses_a)
    corners_a, corners_b = _corners(boxes_a), _corners(boxes_b)
    # A's corners in B's frame: (R_a c + p_a - p_b) R_b
    a_world = np.einsum("njk,pk->npj", rot_a, corners_a) + pos_a[:, None]
    a_in_b = (a_world - pos_b) @ rot_b
    depth_b = union_depth(a_in_b.reshape(-1, 3), boxes_b).reshape(n, -1).max(axis=1)
    # B's corners in each A frame: (R_b c + p_b - p_a) R_a
    b_world = corners_b @ rot_b.T + pos_b
    b_in_a = np.einsum("npj,njk->npk", b_world[None] - pos_a[:, None], rot_a)
    depth_a = union_depth(b_in_a.reshape(-1, 3), boxes_a).reshape(n, -1).max(axis=1)
    return np.maximum(depth_a, depth_b)
