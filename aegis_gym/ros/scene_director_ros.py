from typing import Optional

import rclpy

try:
    from aegis_director.robot_director import RobotDirector
except ImportError:
    print(
        "Failed to import aegis_director. Double check if you have sourced the AGH-CEAI/aegis_ros project."
    )
    raise ImportError

from ..scene import SceneDirectorInterface, EntityType, SceneEntity
from .robot_commander_ros import RobotCommanderROS
from .scene_entities_ros import EntityTypeROS


class SceneDirectorROS(SceneDirectorInterface):
    _instance: Optional["SceneDirectorROS"] = None

    def __new__(cls) -> "SceneDirectorROS":
        if cls._instance is None:
            cls._instance = super(SceneDirectorROS, cls).__new__(cls)
        return cls._instance

    def __del__(self) -> None:
        self.shutdown()

    def __init__(self) -> None:
        if hasattr(self, "_initialized") and self._initialized:
            return
        super().__init__()
        rclpy.init()
        self.robot_director = RobotDirector(synchronous=True)
        self._scene_node = rclpy.create_node("scene_director")
        self._initialized = True

    def shutdown(self) -> None:
        rclpy.shutdown()

    def get_robot_commander(self) -> RobotCommanderROS:
        return RobotCommanderROS(self.robot_director)

    def add_entity(self, entity: EntityType) -> SceneEntity:
        return EntityTypeROS[entity](self._node)

    def build(self) -> None:
        pass

    def step(self) -> None:
        pass
