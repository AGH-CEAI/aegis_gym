from dataclasses import dataclass
from .base_cfg import BaseCfg


@dataclass(slots=True)
class RobotCfg(BaseCfg):
    ee_link_name: str
    gripper_link_names: list[str]
    default_arm_dof: list[float]
    default_gripper_dof: list[float]
    ik_method: str  # TODO set a literal
    urdf_id_cell: str
    urdf_id_cell_collision: str
    urdf_id_no_cell: str
