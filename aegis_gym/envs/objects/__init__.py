from aegis_gym.aux.logging import get_logger

from .base_objects import (
    BaseBox,
    BaseMesh,
    BaseObject,
    BaseURDF,
    ObjectProperties,
    ObjectType,
)
from .base_objects_factory import BaseObjectsFactory

logger = get_logger(__name__)

try:
    from .sim.genesis_objects import GenesisBox, GenesisMesh, GenesisURDF
    from .sim.genesis_objects_factory import GenesisObjectsFactory
except (ImportError, TypeError) as e:
    GenesisBox, GenesisMesh, GenesisURDF = None, None, None
    GenesisObjectsFactory = None
    logger.warning(f"Couldn't import genesis_objects: {e}")

try:
    from .real.ros_grpc_objects import RosGrpcBox
    from .real.ros_grpc_objects_factory import RosGrpcObjectsFactory
except (ImportError, TypeError) as e:
    RosGrpcBox = None
    RosGrpcObjectsFactory = None
    logger.warning(f"Couldn't import ros_grpc_objects: {e}")

__all__ = [
    "BaseBox",
    "BaseMesh",
    "BaseObject",
    "BaseObjectsFactory",
    "BaseURDF",
    "GenesisBox",
    "GenesisMesh",
    "GenesisObjectsFactory",
    "GenesisURDF",
    "ObjectProperties",
    "ObjectType",
    "RosGrpcBox",
    "RosGrpcObjectsFactory",
]
