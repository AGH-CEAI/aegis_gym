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
    urdf_id_cell_no_leds: str
    urdf_id_cell_no_leds_collision: str
    urdf_id_no_cell: str

    fts_payload_mass: float | None
    fts_payload_com: list[float] | None

    # Max joint following error [rad]; acts as the cell's force limit.
    # Measured at kp=3500: 0.005 -> 40 N, 0.01 -> 79 N, 0.02 -> 162 N, 0.05 -> 384 N.
    servo_follow_error_max_rad: float

    # Scales arm kp (kv by sqrt). Set to K_eff_real / K_eff_sim from `ft_stiffness_probe.py`.
    servo_stiffness_scale: float

    # Delay between a velocity command and the arm acting on it [s].
    servo_command_delay_s: float
