import os
import re
import subprocess
import tempfile
import warnings
from pathlib import Path

# Source: https://github.com/ament/ament_index/blob/humble/ament_index_python/ament_index_python/constants.py
RESOURCE_INDEX_SUBFOLDER = "share/ament_index/resource_index"
AMENT_PREFIX_PATH_ENV_VAR = "AMENT_PREFIX_PATH"


def generate_aegis_urdf(show_cell: bool = True) -> Path:
    pkg_share = Path(_get_package_share_directory("aegis_description"))
    xacro_path = pkg_share / "urdf" / "aegis.urdf.xacro"
    _, urdf_path = tempfile.mkstemp(suffix=".urdf", prefix="aegis_urdf_", dir="/tmp")

    if show_cell:
        xacro_args = ["disable_cell_collision:=true", "disable_cell:=false"]
    else:
        xacro_args = ["disable_cell:=true"]

    urdf_with_uris = subprocess.run(
        ["xacro", str(xacro_path)] + xacro_args,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=True,
    ).stdout
    Path(urdf_path).write_bytes(_resolve_packages_paths(urdf_with_uris))

    return Path(urdf_path)


# Definition taken from ROS 2 ament_index_python
# https://github.com/ament/ament_index/blob/humble/ament_index_python/ament_index_python/packages.py#L65
def _get_package_share_directory(package_name: str) -> Path:
    """
    Return the share directory of the given ROS 2 package.
    """
    path = os.path.join(_get_package_prefix(package_name), "share", package_name)
    if not os.path.exists(path):
        warnings.warn(
            f"Share directory for {package_name} ({path}) does not exist.", stacklevel=2
        )
    return path


# Definition taken from ROS 2 ament_index_python
# https://github.com/ament/ament_index/blob/humble/ament_index_python/ament_index_python/packages.py#L39
def _get_package_prefix(package_name: str):
    # This regex checks for a valid package name as defined by REP-127 including the recommended
    #  exemptions. See https://ros.org/reps/rep-0127.html#name
    if re.fullmatch("[a-zA-Z0-9][a-zA-Z0-9_-]+", package_name, re.ASCII) is None:
        raise ValueError(f"'{package_name}' is not a valid package name")
    try:
        package_prefix = _get_resource_prefix_path("packages", package_name)
    except LookupError:
        raise ValueError(
            f"package '{package_name}' not found, searching: {_get_search_paths()}"
        )
    return package_prefix


# Definition taken from ROS 2 ament_index_python
# https://github.com/ament/ament_index/blob/humble/ament_index_python/ament_index_python/resources.py#L52
def _get_resource_prefix_path(resource_type: str, resource_name: str) -> str:
    assert resource_type, "The resource type must not be empty"
    assert resource_name, "The resource name must not be empty"

    if _name_is_invalid(resource_type):
        raise ValueError(f"Resource type '{resource_type}' is invalid")
    if _name_is_invalid(resource_name):
        raise ValueError(f"Resource name '{resource_name}' is invalid")
    for path in _get_search_paths():
        resource_path = os.path.join(
            path, RESOURCE_INDEX_SUBFOLDER, resource_type, resource_name
        )
        if os.path.isfile(resource_path):
            return path
    raise LookupError(
        f"Could not find the resource '{resource_name}' of type '{resource_type}'"
    )


# Definition taken from ROS 2 ament_index_python
# https://github.com/ament/ament_index/blob/humble/ament_index_python/ament_index_python/resources.py#L34
def _name_is_invalid(resource_name):
    return ("/" in resource_name) or ("\\" in resource_name)


# Definition taken from ROS 2 ament_index_python
# https://github.com/ament/ament_index/blob/humble/ament_index_python/ament_index_python/search_paths.py#L20
def _get_search_paths() -> list[str]:
    ament_prefix_path = os.environ.get(AMENT_PREFIX_PATH_ENV_VAR)
    if not ament_prefix_path:
        raise EnvironmentError(
            f"Environment variable '{AMENT_PREFIX_PATH_ENV_VAR}' is not set or empty.\n"
            "Even without installed ROS, please source the build project: `source ros_ws/install/local_setup.sh`.\n"
        )

    paths = ament_prefix_path.split(os.pathsep)
    return [p for p in paths if p and os.path.exists(p)]


def _resolve_packages_paths(urdf: bytes) -> bytes:
    urdf_str = urdf.decode("utf-8")
    pattern = r"package://([a-zA-Z0-9_]+)/"
    matches = re.findall(pattern, urdf_str)
    for match in matches:
        package_path = _get_package_share_directory(match)
        urdf_str = urdf_str.replace(f"package://{match}/", f"{package_path}/")
    return urdf_str.encode("utf-8")
