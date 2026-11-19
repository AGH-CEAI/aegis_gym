"""
Watertight STL of a union of axis-aligned boxes, without a boolean engine.

The boxes' coordinates define a global rectilinear grid; a cell is solid when its centre lies in any box, and every
face between a solid and an empty cell becomes two triangles. Faces use only grid vertices, so
there are no T-junctions and the result is closed (as long as no two solid cells touch along an edge only).
"""

from pathlib import Path

import numpy as np
import trimesh

from .geometry import Box

_EPS = 1e-12


def boxes_to_mesh(boxes: tuple[Box, ...] | list[Box]) -> trimesh.Trimesh:
    lows = np.array([b.bounds[0] for b in boxes])
    highs = np.array([b.bounds[1] for b in boxes])
    axes = [_unique_sorted(np.concatenate([lows[:, i], highs[:, i]])) for i in range(3)]
    centres = np.meshgrid(*[(a[:-1] + a[1:]) / 2 for a in axes], indexing="ij")
    pts = np.stack(centres, axis=-1)[..., None, :]  # (nx, ny, nz, 1, 3)
    # (nx, ny, nz): cell centre inside any box
    solid = np.any(np.all((pts > lows) & (pts < highs), axis=-1), axis=-1)

    padded = np.pad(solid, 1)
    vertices: list[np.ndarray] = []
    faces: list[list[int]] = []
    for axis in range(3):
        # a boundary face on plane k of `axis` lies between cells k-1 and k
        lo = np.take(padded, range(padded.shape[axis] - 1), axis=axis)
        hi = np.take(padded, range(1, padded.shape[axis]), axis=axis)
        others = [a for a in range(3) if a != axis]
        sl = [slice(1, -1)] * 3
        sl[axis] = slice(None)
        lo, hi = lo[tuple(sl)], hi[tuple(sl)]
        for idx in zip(*np.nonzero(lo != hi), strict=True):
            k, u, v = idx[axis], idx[others[0]], idx[others[1]]
            quad = []
            for du, dv in ((0, 0), (1, 0), (1, 1), (0, 1)):
                p = np.empty(3)
                p[axis] = axes[axis][k]
                p[others[0]] = axes[others[0]][u + du]
                p[others[1]] = axes[others[1]][v + dv]
                quad.append(p)
            base = len(vertices)
            vertices.extend(quad)
            faces += [[base, base + 1, base + 2], [base, base + 2, base + 3]]

    mesh = trimesh.Trimesh(
        vertices=np.array(vertices), faces=np.array(faces), process=True
    )
    mesh.merge_vertices()
    trimesh.repair.fix_normals(mesh)  # consistent winding, outward normals
    return mesh


def export_stl(
    boxes: tuple[Box, ...] | list[Box], path: str | Path, scale: float = 1.0
) -> trimesh.Trimesh:
    """Writes the union of `boxes`; `scale=1000` gives millimetres for slicers."""
    mesh = boxes_to_mesh(boxes)
    if scale != 1.0:
        mesh.apply_scale(scale)
    mesh.export(str(path))
    return mesh


def _unique_sorted(values: np.ndarray) -> np.ndarray:
    values = np.sort(values)
    keep = np.concatenate([[True], np.diff(values) > _EPS])
    return values[keep]
