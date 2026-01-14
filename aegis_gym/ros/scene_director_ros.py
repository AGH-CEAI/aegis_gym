import asyncio
from typing import Optional

import rclpy


try:
    from aegis_director.robot_director import RobotDirector
except ImportError:
    print(
        "Failed to import aegis_director. Double check if you have sourced the AGH-CEAI/aegis_ros project."
    )
    RobotDirector = None
if RobotDirector is None:
    raise ImportError

try:
    from aegis_grpc_client import AegisRobotClient
except ImportError:
    print(
        "Failed to import aegis_grpc_client. "
        "Double check if you have installed the `aegis_grpc_client` and `proto_aegis_grpc` packages."
    )
    raise ImportError

from ..scene import SceneDirectorInterface, EntityType, SceneEntity
from .robot_commander_ros import RobotCommanderROS
from .scene_entities_ros import EntityTypeROS


class SceneDirectorROS(SceneDirectorInterface):
    _instance: Optional["SceneDirectorROS"] = None

    def __new__(cls, *args, **kwargs) -> "SceneDirectorROS":
        if cls._instance is None:
            cls._instance = super(SceneDirectorROS, cls).__new__(cls)
        return cls._instance

    def __del__(self) -> None:
        if self.robot_client.is_connected:
            asyncio.run(self.robot_client.disconnect())
        self.shutdown()

    def __init__(self, device: str = "cuda", enable_scene_camera: bool = False) -> None:
        if hasattr(self, "_initialized") and self._initialized:
            return
        super().__init__(device)
        rclpy.init()
        self.robot_director = RobotDirector(synchronous=True)
        self._scene_node = rclpy.create_node("scene_director")

        self.robot_client = AegisRobotClient(server_address="127.0.0.1:50051")
        asyncio.run(self.robot_client.connect())

        self._initialized = True

    def shutdown(self) -> None:
        rclpy.shutdown()

    def get_robot_commander(self) -> RobotCommanderROS:
        return RobotCommanderROS(self.robot_director, self.robot_client, self.device)

    def add_entity(self, entity: EntityType) -> SceneEntity:
        return EntityTypeROS[entity](self._scene_node)

    def build(self) -> None:
        pass

    def step(self) -> None:
        pass
