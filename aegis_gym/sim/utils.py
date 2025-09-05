import re
import subprocess
import tempfile
from pathlib import Path

from ament_index_python.packages import get_package_share_directory


def generate_aegis_urdf() -> Path:
    pkg_share = Path(get_package_share_directory("aegis_description"))
    xacro_path = pkg_share / "urdf" / "aegis.urdf.xacro"
    fd, urdf_path = tempfile.mkstemp(suffix=".urdf", prefix="aegis_urdf_", dir="/tmp")

    urdf_with_uris = subprocess.run(
        ["xacro", str(xacro_path)],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=True,
    ).stdout
    Path(urdf_path).write_bytes(_resolve_packages_paths(urdf_with_uris))

    return Path(urdf_path)


def _resolve_packages_paths(urdf: bytes) -> bytes:
    urdf_str = urdf.decode("utf-8")
    pattern = r"package://([a-zA-Z0-9_]+)/"
    matches = re.findall(pattern, urdf_str)
    for match in matches:
        package_path = get_package_share_directory(match)
        urdf_str = urdf_str.replace(f"package://{match}/", f"{package_path}/")
    return urdf_str.encode("utf-8")
