from abc import ABC, abstractmethod
import numpy as np


class RobotCommanderInterface(ABC):
    @abstractmethod
    def get_joint_positions(self) -> np.ndarray:
        raise NotImplementedError

    @abstractmethod
    def get_joint_velocities(self) -> np.ndarray:
        raise NotImplementedError

    @abstractmethod
    def get_tcp_position(self) -> np.ndarray:
        raise NotImplementedError

    @abstractmethod
    def control_dofs_position(
        self, target_pos: np.ndarray, max_vel: float = 0.3, max_accel: float = 0.3
    ) -> None:
        raise NotImplementedError

    @abstractmethod
    def move_to_home(self) -> None:
        raise NotImplementedError
