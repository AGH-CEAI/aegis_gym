from aegis_gym.aux.logging import get_logger

from .base_manipulator import BaseManipulator, CameraModality

logger = get_logger(__name__)

try:
    from .sim.genesis_manipulator import GenesisManipulator
except (ImportError, TypeError) as e:
    GenesisManipulator = None
    logger.warning(f"Couldn't import GenesisManipulator: {e}")

try:
    from .real.ros_grpc_manipulator import RosGrpcManipulator
except (ImportError, TypeError) as e:
    RosGrpcManipulator = None
    logger.warning(f"Couldn't import RosGrpcManipulator: {e}")

__all__ = [
    "BaseManipulator",
    "CameraModality",
    "GenesisManipulator",
    "RosGrpcManipulator",
]
