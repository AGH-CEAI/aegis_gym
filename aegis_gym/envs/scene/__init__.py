from aegis_gym.aux.logging import get_logger

from .base_scene import BaseScene
from .sim.genesis_scene import GenesisScene

logger = get_logger(__name__)

try:
    from .real.ros_grpc_scene import RosGrcpScene
except ImportError as e:
    RosGrpcScene = None
    logger.error(f"Import error: Couldn't import GraspEnvRos: {e}")

__all__ = [
    "BaseScene",
    "GenesisScene",
    "RosGrcpScene",
]
