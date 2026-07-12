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
