from importlib.metadata import version, PackageNotFoundError

try:
    __version__ = version("aegis_gym")
except PackageNotFoundError:
    __version__ = "unknown"

from .scene.scene_director_factory import SceneDirectorType
from .register_envs import register_envs


ENV_IDS = register_envs(SceneDirectorType.ROS)
