from abc import ABC, abstractmethod
import torch as th


class RobotCommanderInterface(ABC):
    def __init__(self, device: str):
        self.device = device

        # TODO(issue#6) Take HOME position from the aegis_ros' SRDF file
        self.dof_home_dict = {
            "shoulder_pan_joint": 0.0,
            "shoulder_lift_joint": -2.09,
            "elbow_joint": 2.09,
            "wrist_1_joint": -1.57,
            "wrist_2_joint": -1.57,
            "wrist_3_joint": 0.0,
            # "robotiq_hande_left_finger_joint": 0.025,
            # "robotiq_hande_right_finger_joint": 0.025,
        }
        self.dof_home = th.tensor(
            [val for val in self.dof_home_dict.values()],
            device=self.device,
        )

    @abstractmethod
    def get_joint_positions(self) -> th.Tensor:
        raise NotImplementedError

    @abstractmethod
    def get_joint_velocities(self) -> th.Tensor:
        raise NotImplementedError

    @abstractmethod
    def get_tcp_position(self) -> th.Tensor:
        raise NotImplementedError

    @abstractmethod
    def get_tcp_orientation(self) -> th.Tensor:
        raise NotImplementedError

    @abstractmethod
    def get_tcp_pose(self) -> th.Tensor:
        raise NotImplementedError

    @abstractmethod
    def control_dofs_position(
        self, target_pos: th.Tensor, max_vel: float = 0.3, max_accel: float = 0.3
    ) -> None:
        raise NotImplementedError
    
    def control_dofs_position_servo(
        self, target_pos: th.Tensor, max_vel: float = 0.3, max_accel: float = 0.3
    ) -> None:
        raise NotImplementedError

    @abstractmethod
    def control_tcp_position(
        self,
        target_pos: th.Tensor,
        target_ori: th.Tensor,
        max_vel: float = 0.3,
        max_accel: float = 0.3,
    ) -> None:
        raise NotImplementedError
    
    @abstractmethod
    def control_tcp_position_servo(
        self,
        target_pos: th.Tensor,
        target_ori: th.Tensor,
        max_vel: float = 0.3,
        max_accel: float = 0.3,
    ) -> None:
        raise NotImplementedError

    @abstractmethod
    def move_to_home(self) -> None:
        raise NotImplementedError
