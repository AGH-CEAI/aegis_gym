import numpy as np
import torch as th
import genesis as gs

from ...scene.robot_commander_interface import RobotCommanderInterface

gsEntity = gs.engine.entities.base_entity.Entity


class RobotCommanderSimGenesis(RobotCommanderInterface):
    def __init__(
        self, gs_robot: gsEntity, motor_dofs: tuple[str], device: str = "cuda"
    ):
        super().__init__()
        self.robot = gs_robot
        self.device = device
        self.motor_dofs = motor_dofs

        # TODO Take HOME position from SRDF file.
        cfg = {
            "default_joint_angles": {
                "shoulder_pan_joint": 0.0,
                "shoulder_lift_joint": -2.10,
                "elbow_joint": 2.10,
                "wrist_1_joint": -1.57,
                "wrist_2_joint": -1.57,
                "wrist_3_joint": 0.0,
                # "robotiq_hande_left_finger_joint": 0.025,
                # "robotiq_hande_right_finger_joint": 0.025,
            },
        }
        self.dof_home = th.tensor(
            [cfg["default_joint_angles"][name] for name in cfg["dof_names"]],
            device=device,
        )

    # TODO consider changing interface types to th.Tensor
    # def get_joint_positions(self) -> gs.Tensor:
    def get_joint_positions(self) -> np.ndarray:
        return self.robot.get_dofs_position(self.motor_dofs).cpu().numpy()

    # TODO consider changing interface types to th.Tensor
    # def get_joint_velocities(self) -> gs.Tensor:
    def get_joint_velocities(self) -> np.ndarray:
        return self.robot.get_dofs_velocity(self.motor_dofs).cpu().numpy()

    # TODO consider changing interface types to th.Tensor
    # def get_tcp_position(self) -> gs.Tensor:
    def get_tcp_position(self) -> np.ndarray:
        return self.robot.get_links_pos()[7, :].cpu().numpy()

    def control_dofs_position(
        self, target_pos: th.Tensor, max_vel: float = 0.3, max_accel: float = 0.3
    ) -> None:
        # TODO use max cel and accel
        self.robot.control_dofs_position(target_pos, self.motor_dofs)

    def move_to_home(self) -> None:
        # TODO (issue#X) Do we need trajectory to home in simulation?
        # self.control_dofs_position(self.dof_home)
        self._teleport_to_home()

    def _teleport_to_home(self) -> None:
        self.robot.set_dofs_position(
            position=self.dof_home,
            dofs_idx_local=self.motor_dofs,
            zero_velocity=True,
        )
        self.robot.zero_all_dofs_velocity()
