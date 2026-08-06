from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version("aegis_gym")
except PackageNotFoundError:
    __version__ = "unknown"

from .register_envs import register_envs
from .scene.scene_director_factory import SceneDirectorType

ENV_IDS = register_envs(SceneDirectorType.ROS)
