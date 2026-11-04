"""
Batched model of the real A1 admittance chain, advanced once per policy step.

Real chain: gRPC StepTCP -> reference limiter (aegis_grpc `AdmittanceReference`) -> JTC segment ->
`admittance_controller` (filtered F/T, M x'' + D x' + K x = F on the translation axes) -> UR.
As in ROS 2 Humble's `admittance_rule_impl.hpp`, the spring displacement x is FK(measured joints)
- FK(reference), i.e. the integrated offset plus the arm's tracking error, so the controller also
integrates the tracking lag (`humble_spring`; anti-windup by clamping that error and the offset).
Here the limiter is reproduced exactly; the spring is integrated in substeps over the policy step
with the force held from the step start (Genesis exposes no per-substep hook), which stands in for
part of the real loop delay. The defaults are the values deployed on the robot:
`aegis_control/config/controllers/admittance.yaml` and `aegis_grpc/launch/start_server.launch.py`.
"""

import math
from dataclasses import dataclass

import torch as th

from aegis_gym.aux.geom import (
    quat_to_rotvec_error,
    rotvec_to_quat,
    transform_quat_by_quat,
)


@dataclass(frozen=True)
class AdmittanceParams:
    mass: float = 5.0  # [kg]
    stiffness: float = 500.0  # [N/m]
    damping_ratio: float = 1.0
    filter_coefficient: float = (
        0.1  # exponential filter of the wrench, per controller cycle
    )
    controller_dt: float = 0.002  # [s] controller_manager period
    integration_dt: float = 0.004  # [s] substep of the spring integration
    max_step_lin: float = 0.004  # [m] per call
    max_step_rot: float = 0.02  # [rad] per call
    max_offset_lin: float = 0.02  # [m] reference to measured pose
    max_offset_rot: float = 0.1  # [rad]
    # Spring displacement from the measured pose, as the Humble controller computes it.
    humble_spring: bool = True
    # Anti-windup: a blocked arm (contact, joint limit) must not grow the offset without bound.
    max_tracking_error: float = 0.005  # [m] of the error entering the spring
    max_spring_offset: float = 0.02  # [m], the reference limiter's max_offset_lin
    # "Dark energy": an empirical term for what the model does not explain. Without `humble_spring`
    # (tool inertia in the F/T + the plain M/D/K) ~4/5 of the real overshoot was missing; fitted then
    # on a1_square_payload.csv (2026-11-04) as 0.050 s / 0.18 s: after a velocity change the target
    # keeps this much of the previous velocity, fading with this time constant. 0 switches it off.
    velocity_memory_s: float = 0.0  # [s]
    velocity_memory_fade_s: float = 0.18  # [s]

    @property
    def damping(self) -> float:
        return 2.0 * self.damping_ratio * math.sqrt(self.mass * self.stiffness)


def _clip_norm(v: th.Tensor, max_norm: float) -> tuple[th.Tensor, th.Tensor]:
    norm = th.linalg.vector_norm(v, dim=-1, keepdim=True)
    over = norm > max_norm
    scale = th.where(over, max_norm / th.clamp(norm, min=1e-12), 1.0)
    return v * scale, over.squeeze(-1)


class SimAdmittance:
    def __init__(
        self, num_envs: int, device: th.device, params: AdmittanceParams | None = None
    ):
        self.params = params or AdmittanceParams()
        self.active = th.zeros(num_envs, dtype=th.bool, device=device)
        self.ref_pos = th.zeros(num_envs, 3, device=device)
        self.ref_quat = th.zeros(num_envs, 4, device=device)
        self.ref_quat[:, 0] = 1.0
        self.offset = th.zeros(num_envs, 3, device=device)
        self.offset_vel = th.zeros(num_envs, 3, device=device)
        self.force = th.zeros(num_envs, 3, device=device)
        self.vel_memory = th.zeros(num_envs, 3, device=device)

    def deactivate(self, envs_idx: th.Tensor | None = None) -> None:
        if envs_idx is None:
            self.active[:] = False
        else:
            self.active[envs_idx] = False

    def activate(
        self, meas_pos: th.Tensor, meas_quat: th.Tensor, force: th.Tensor
    ) -> None:
        """Starts the inactive envs at the measured pose with a relaxed spring, like Enable on the robot."""
        new = ~self.active
        if not bool(new.any()):
            return
        self.ref_pos[new] = meas_pos[new]
        self.ref_quat[new] = meas_quat[new]
        self.offset[new] = 0.0
        self.offset_vel[new] = 0.0
        self.force[new] = force[new]
        self.vel_memory[new] = 0.0
        self.active[new] = True

    def step_reference(
        self,
        translation: th.Tensor,
        rotation: th.Tensor,
        meas_pos: th.Tensor,
        meas_quat: th.Tensor,
    ) -> tuple[th.Tensor, th.Tensor, th.Tensor]:
        """`AdmittanceReference::Step`: rate limit, integrate from the previous reference, bound
        the offset from the measured pose. Returns (clipped [N], applied translation, rotation)."""
        p = self.params
        dp, clipped_lin = _clip_norm(translation, p.max_step_lin)
        dr, clipped_rot = _clip_norm(rotation, p.max_step_rot)
        clipped = clipped_lin | clipped_rot

        pos = self.ref_pos + dp
        quat = transform_quat_by_quat(rotvec_to_quat(dr), self.ref_quat)

        lin_offset, clipped_off = _clip_norm(pos - meas_pos, p.max_offset_lin)
        pos = meas_pos + lin_offset
        rot_offset, clipped_rot_off = _clip_norm(
            quat_to_rotvec_error(quat, meas_quat), p.max_offset_rot
        )
        quat = th.where(
            clipped_rot_off.unsqueeze(-1),
            transform_quat_by_quat(rotvec_to_quat(rot_offset), meas_quat),
            quat,
        )
        clipped = clipped | clipped_off | clipped_rot_off

        applied_dp = pos - self.ref_pos
        applied_dr = quat_to_rotvec_error(quat, self.ref_quat)
        self.ref_pos = pos
        self.ref_quat = quat
        return clipped, applied_dp, applied_dr

    def step_spring(
        self, force: th.Tensor, dt: float, tracking_error: th.Tensor | None = None
    ) -> None:
        """Integrates the translational spring over `dt` with `force` [N, 3] (world) held constant.

        `tracking_error` [N, 3] is measured minus commanded TCP position at the step start; with
        `humble_spring` it is added to the offset to form the spring displacement, held over the step.
        """
        p = self.params
        n_sub = max(1, round(dt / p.integration_dt))
        h = dt / n_sub
        alpha = 1.0 - (1.0 - p.filter_coefficient) ** (h / p.controller_dt)
        error = th.zeros_like(self.offset)
        if p.humble_spring and tracking_error is not None:
            error, _ = _clip_norm(tracking_error, p.max_tracking_error)
        for _ in range(n_sub):
            self.force = self.force + alpha * (force - self.force)
            displacement = self.offset + error
            acc = (
                self.force - p.damping * self.offset_vel - p.stiffness * displacement
            ) / p.mass
            self.offset_vel = self.offset_vel + h * acc
            self.offset = self.offset + h * self.offset_vel
            self._limit_offset()

    def _limit_offset(self) -> None:
        """Clamps the offset and removes the outward velocity (back-calculation anti-windup)."""
        self.offset, clamped = _clip_norm(self.offset, self.params.max_spring_offset)
        if not bool(clamped.any()):
            return
        direction = self.offset / th.clamp(
            th.linalg.vector_norm(self.offset, dim=-1, keepdim=True), min=1e-12
        )
        outward = th.clamp((self.offset_vel * direction).sum(-1, keepdim=True), min=0.0)
        self.offset_vel = th.where(
            clamped.unsqueeze(-1),
            self.offset_vel - outward * direction,
            self.offset_vel,
        )

    def step_velocity_memory(self, lin_vel: th.Tensor, dt: float) -> th.Tensor:
        """Translation [N, 3] added to the target: the empirical "dark energy" term (see the params).

        Most likely origin: Humble's admittance spring position is FK(measured joints) - FK(reference),
        so the controller integrates the UR's own tracking error on top of the tool's inertial load;
        every velocity change of the reference then deflects the arm along the old velocity (start:
        extra lag, stop: overshoot). The tool inertia alone, now in the simulated F/T, gives ~1/5.
        """
        p = self.params
        offset = p.velocity_memory_s * (self.vel_memory - lin_vel)
        self.vel_memory = self.vel_memory + (
            1.0 - math.exp(-dt / p.velocity_memory_fade_s)
        ) * (lin_vel - self.vel_memory)
        return offset

    def target_pos(self) -> th.Tensor:
        return self.ref_pos + self.offset
