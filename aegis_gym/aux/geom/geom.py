import numpy as np
import torch as th

# Implementation taken from https://github.com/Genesis-Embodied-AI/genesis-world/blob/main/genesis/utils/geom.py


@th.jit.script
def transform_by_quat(v, quat, out: th.Tensor | None = None):
    q_w, q_x, q_y, q_z = quat[..., :1], quat[..., 1:2], quat[..., 2:3], quat[..., 3:]
    q_ww, q_wx, q_wy, q_wz = q_w * q_w, q_w * q_x, q_w * q_y, q_w * q_z
    q_xx, q_xy, q_xz = q_x * q_x, q_x * q_y, q_x * q_z
    q_yy, q_yz = q_y * q_y, q_y * q_z
    q_zz = q_z**2

    vs = v / (q_ww + q_xx + q_yy + q_zz)
    v_x, v_y, v_z = vs[..., :1], vs[..., 1:2], vs[..., 2:]

    if out is None:
        out = th.empty(vs.shape, dtype=vs.dtype, device=vs.device)
    u_x, u_y, u_z = out[..., :1], out[..., 1:2], out[..., 2:]

    u_x.copy_(
        v_x * (q_xx + q_ww - q_yy - q_zz)
        + v_y * (2.0 * q_xy - 2.0 * q_wz)
        + v_z * (2.0 * q_xz + 2.0 * q_wy)
    )
    u_y.copy_(
        v_x * (2.0 * q_wz + 2.0 * q_xy)
        + v_y * (q_ww - q_xx + q_yy - q_zz)
        + v_z * (2.0 * q_yz - 2.0 * q_wx)
    )
    u_z.copy_(
        v_x * (2.0 * q_xz - 2.0 * q_wy)
        + v_y * (2.0 * q_wx + 2.0 * q_yz)
        + v_z * (q_ww - q_xx - q_yy + q_zz)
    )

    return out


@th.jit.script
def transform_quat_by_quat(u: th.Tensor, v: th.Tensor) -> th.Tensor:
    w1, x1, y1, z1 = u[..., 0], u[..., 1], u[..., 2], u[..., 3]
    w2, x2, y2, z2 = v[..., 0], v[..., 1], v[..., 2], v[..., 3]
    ww = (z1 + x1) * (x2 + y2)
    yy = (w1 - y1) * (w2 + z2)
    zz = (w1 + y1) * (w2 - z2)
    xx = ww + yy + zz
    qq = 0.5 * (xx + (z1 - x1) * (x2 - y2))

    out = th.empty(qq.shape + (4,), dtype=qq.dtype, device=qq.device)
    out[..., 0] = qq - ww + (z1 - y1) * (y2 - z2)
    out[..., 1] = qq - xx + (x1 + w1) * (x2 + w2)
    out[..., 2] = qq - yy + (w1 - x1) * (y2 + z2)
    out[..., 3] = qq - zz + (z1 + y1) * (w2 - x2)

    out /= th.linalg.vector_norm(out, ord=2, dim=-1, keepdim=True)
    return out


def quat_to_rotvec_error(q_target: th.Tensor, q_current: th.Tensor) -> th.Tensor:
    """
    Orientation error between two [N, 4] (w, x, y, z) quaternions, expressed as
    an [N, 3] rotation vector (axis * angle) in the world frame, i.e. the rotation
    that brings `q_current` onto `q_target` along the shortest path.
    """
    tw, tx, ty, tz = q_target.unbind(dim=-1)
    # conjugate of the current orientation
    cw, cx, cy, cz = (q_current * q_current.new_tensor([1, -1, -1, -1])).unbind(-1)
    # Hamilton product: q_err = q_target * conj(q_current)
    ew = tw * cw - tx * cx - ty * cy - tz * cz
    ex = tw * cx + tx * cw + ty * cz - tz * cy
    ey = tw * cy - tx * cz + ty * cw + tz * cx
    ez = tw * cz + tx * cy - ty * cx + tz * cw
    vec = th.stack([ex, ey, ez], dim=-1)

    # resolve the double-cover to take the shortest rotation
    sign = th.where(ew < 0, -1.0, 1.0).unsqueeze(-1)
    ew, vec = ew * sign.squeeze(-1), vec * sign

    vec_norm = th.linalg.vector_norm(vec, dim=-1)
    angle = 2 * th.atan2(vec_norm, ew)
    return vec * (angle / th.clamp(vec_norm, min=1e-8)).unsqueeze(-1)


def quat_to_z_euler(quats: th.Tensor) -> th.Tensor:
    """
    Recovers the yaw angle from a quaternion representing a pure Z-axis rotation,
    correcting for the quaternion's double-cover sign ambiguity.
    """
    signs = th.ones_like(quats[:, -1])
    signs[quats[:, -1] < 0] = -1.0
    qw = th.clamp(quats[:, 0] * signs, min=-1.0, max=1.0)
    return 2 * th.acos(qw)


def quat_to_zrot(quats: th.Tensor, device: th.device) -> th.Tensor:
    """Converts a pure Z-axis rotation quaternion into a 3x3 homogeneous 2D
    rotation matrix."""
    alphas = quat_to_z_euler(quats)
    n = quats.shape[0]
    rot = th.zeros(n, 3, 3, device=device, dtype=th.float32)
    rot[:, 2, 2] = 1.0
    cos_a, sin_a = th.cos(alphas), th.sin(alphas)
    rot[:, 0, 0] = cos_a
    rot[:, 1, 1] = cos_a
    rot[:, 0, 1] = -sin_a
    rot[:, 1, 0] = sin_a
    return rot


def check_points_in_polygon(points: np.ndarray, polygon: np.ndarray) -> np.ndarray:
    """
    Ray-casting point-in-polygon test, vectorized over points.
    """
    x, y = points[:, 0], points[:, 1]
    px, py = polygon[:, 0], polygon[:, 1]
    n = len(polygon)
    inside = np.zeros(len(points), dtype=bool)
    j = n - 1
    for i in range(n):
        pxi, pyi, pxj, pyj = px[i], py[i], px[j], py[j]
        cond = ((pyi > y) != (pyj > y)) & (
            x < (pxj - pxi) * (y - pyi) / (pyj - pyi + 1e-30) + pxi
        )
        inside ^= cond
        j = i
    return inside
