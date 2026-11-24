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

    ft_payload_mass: float | None
    ft_payload_com: list[float] | None

    # A1 reference limits; must match the gRPC server launch arguments of the same name
    admittance_max_step_lin: float = 0.004  # [m] per policy step
    admittance_max_offset_lin: float = 0.02  # [m]
    admittance_max_accel_lin: float = 0.0  # [m/s^2], 0 = off
