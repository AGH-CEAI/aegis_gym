from dataclasses import dataclass
from typing import Literal

from .base_cfg import BaseCfg


@dataclass(slots=True)
class RobotCfg(BaseCfg):
    ee_link_name: str
    gripper_link_names: list[str]
    default_arm_dof: list[float]
    default_gripper_dof: list[float]
    ik_method: Literal["gs_ikv", "dls_ikv"]
    urdf_id_cell: str
    urdf_id_cell_collision: str
    urdf_id_no_cell: str
    # Where the simulated contact part of the F/T reading comes from.
    #   "contact"  -- the solver's own contact forces. Measured, not inferred.
    #   "jacobian" -- actuator torque through a Jacobian. Kept for comparison only:
    #                 it reads the arm's own effort as well as the contact, so while
    #                 the arm is moving its tangential channel is unusable (measured
    #                 8x high and drifting against the solver's steady truth).
    fts_wrench_source: Literal["contact", "jacobian"]

    fts_payload_mass: float | None
    fts_payload_com: list[float] | None

    # Largest following error a joint may hold, in radians. The simulated servo
    # integrates its setpoint like the real one, so without a bound it would reach any
    # contact force at all (611 N inside 40 s when measured). This is the analogue of
    # the cell's force limit, and the two are proportional -- measured against a rigid
    # surface at kp = 3500 Nm/rad:
    #     0.005 rad -> 40 N     0.02 rad -> 162 N     0.05 rad -> 384 N
    #     0.010 rad -> 79 N     0.03 rad -> 237 N
    # The constant is about 7.9 kN/rad, but it scales with the lever, so treat it as the
    # order of magnitude rather than a calibration. Match it to the force limit set in
    # the real cell's safety configuration.
    servo_follow_error_max_rad: float

    # Multiplies the arm's joint kp (kv follows as its square root, preserving the
    # damping ratio). This is the sim-to-real knob for contact force *dynamics*: force
    # is K_eff * following error, so it scales the force-against-lag slope directly.
    # Measure K_eff with `ft_sensor_torque_probe` in both worlds and set this to
    # K_eff_real / K_eff_sim. Note the joints also have a +/-87 Nm force range, which
    # caps the achievable force independently once kp * error exceeds it.
    servo_stiffness_scale: float

    servo_command_delay_s: float
