from abc import ABC, abstractmethod
import numpy as np


class RobotCommanderInterface(ABC):
    @abstractmethod
    def get_joint_positions(self) -> np.ndarray:
        pass

    @abstractmethod
    def get_joint_velocities(self) -> np.ndarray:
        pass

    @abstractmethod
    def get_tcp_position(self) -> np.ndarray:
        pass

    @abstractmethod
    def control_dofs_position(
        self, target_pos: np.ndarray, max_vel: float = 0.3, max_accel: float = 0.3
    ) -> None:
        pass

    @abstractmethod
    def move_to_home(self) -> None:
        pass

    @abstractmethod
    def publish_target_pos(self, pos: np.ndarray) -> None:
        pass

    @abstractmethod
    def shutdown(self) -> None:
        pass
