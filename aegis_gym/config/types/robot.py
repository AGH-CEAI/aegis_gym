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
