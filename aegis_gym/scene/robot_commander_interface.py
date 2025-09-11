from abc import ABC, abstractmethod
import torch as th


class RobotCommanderInterface(ABC):
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

    @abstractmethod
    def move_to_home(self) -> None:
        raise NotImplementedError
