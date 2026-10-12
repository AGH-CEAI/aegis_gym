"""Helpers shared by the F/T sensor scripts in `scripts/ft_sensor/`."""

import torch as th

from aegis_gym.aux.geom import transform_by_quat
from aegis_gym.aux.logging import get_logger
from aegis_gym.envs.manipulator import BaseManipulator

logger = get_logger("ft_sensor")


def step(scene) -> None:
    """One environment step. `pre_step()` paces the loop on the real robot (it blocks
    until a full policy period has elapsed) and is a no-op in simulation."""
    scene.pre_step()
    scene.step()


def fmt_wrench(wrench: th.Tensor) -> str:
    return "[" + " ".join(f"{v:8.3f}" for v in wrench.tolist()) + "]"


def quat_to_matrix(quat: th.Tensor) -> th.Tensor:
    """Rotation matrix of a WXYZ quaternion, columns being the local axes in world."""
    eye = th.eye(3, dtype=quat.dtype, device=quat.device)
    return transform_by_quat(eye, quat.unsqueeze(0).expand(3, 4)).T


def sensor_quat(manipulator: BaseManipulator) -> th.Tensor:
    """Orientation of the frame the wrench is reported in, env 0.

    In simulation that is `get_ft_frame_quat()`. The real bridge does not publish the
    sensor link, but per the URDF the only rotation between `tool_mount_link` and the
    TCP is the +90 deg about Z of `adapter_from_sensor_end_joint`, which is exactly the
    sensor's output offset -- so there the TCP orientation is the sensor frame.
    """
    getter = getattr(manipulator, "get_ft_frame_quat", manipulator.get_tcp_orientation)
    return getter()[0]


def is_modelled(manipulator: BaseManipulator) -> bool:
    """Whether this backend exposes the simulation-only modelling internals.

    The real robot reports one measurement over the bridge and nothing else: there is
    no gravity term to inspect, no untared reading, and no sensor-link handle. Every
    block guarded by this is a sim-side cross-check, not part of the test itself.
    """
    return hasattr(manipulator, "get_ft_wrench_raw")


def log_frame(manipulator: BaseManipulator, label: str) -> None:
    """Where the frame's own axes point, in world. Without this the wrench cannot be
    read: 'rotate 90 deg about X' means one thing in world axes and another in the
    tool's, and at the home pose the two differ by ~90 deg."""
    rot = quat_to_matrix(sensor_quat(manipulator))
    axes = "  ".join(
        f"{ax}->[{' '.join(f'{v:6.3f}' for v in rot[:, i].tolist())}]"
        for i, ax in enumerate("xyz")
    )
    logger.info(f"  {label} frame (sensor output frame): {axes}")


def rotation_since(
    quat_ref: th.Tensor, quat_now: th.Tensor
) -> tuple[float, list[float]]:
    """Angle (deg) and world axis of the rotation taking `quat_ref` to `quat_now`."""
    rel = quat_to_matrix(quat_now) @ quat_to_matrix(quat_ref).T
    cos = th.clamp((th.diagonal(rel).sum() - 1.0) / 2.0, -1.0, 1.0)
    angle = th.rad2deg(th.acos(cos))
    axis = th.tensor(
        [rel[2, 1] - rel[1, 2], rel[0, 2] - rel[2, 0], rel[1, 0] - rel[0, 1]],
        device=rel.device,
    )
    norm = th.linalg.norm(axis)
    if float(norm) > 1e-6:
        axis = axis / norm
    return float(angle), [round(v, 3) for v in axis.tolist()]
