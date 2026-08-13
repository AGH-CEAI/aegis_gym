from .base_manipulator import BaseManipulator, CameraModality
from .sim.genesis_manipulator import GenesisManipulator

try:
    from .real.ros_grpc_manipulator import RosGrpcManipulator
except (ImportError, TypeError) as e:
    RosGrpcManipulator = None
    print(f"[ImportError] Couldn't import RosGrpcManipulator: {e}")

__all__ = [
    "BaseManipulator",
    "CameraModality",
    "GenesisManipulator",
    "RosGrpcManipulator",
]
