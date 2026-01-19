import re
import subprocess
import tempfile
from importlib import metadata
from natsort import natsorted
from pathlib import Path
from typing import Any, Callable

import genesis as gs
from ament_index_python.packages import get_package_share_directory
from rsl_rl.runners import OnPolicyRunner

from behavior_cloning import BehaviorCloning


def generate_aegis_urdf(show_cell: bool = True) -> Path:
    pkg_share = Path(get_package_share_directory("aegis_description"))
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


def _resolve_packages_paths(urdf: bytes) -> bytes:
    urdf_str = urdf.decode("utf-8")
    pattern = r"package://([a-zA-Z0-9_]+)/"
    matches = re.findall(pattern, urdf_str)
    for match in matches:
        package_path = get_package_share_directory(match)
        urdf_str = urdf_str.replace(f"package://{match}/", f"{package_path}/")
    return urdf_str.encode("utf-8")


def check_rsl_rl_version() -> None:
    """Ensure correct RSL-RL library is installed."""
    try:
        try:
            if metadata.version("rsl-rl"):
                raise ImportError
        except metadata.PackageNotFoundError:
            if metadata.version("rsl-rl-lib") != "3.3.0":
                raise ImportError
    except (metadata.PackageNotFoundError, ImportError) as e:
        raise ImportError(
            "Please uninstall 'rsl_rl' and install 'rsl-rl-lib==3.3.0'."
        ) from e


def load_teacher_policy(env: Any, rl_train_cfg: dict, exp_name: str) -> Callable:
    """Load teacher policy."""
    log_dir = Path("logs") / f"{exp_name + '_' + 'rl'}"
    assert log_dir.exists(), f"Log directory {log_dir} does not exist"
    checkpoint_files = [
        f for f in log_dir.iterdir() if re.match(r"model_\d+\.pt", f.name)
    ]
    try:
        *_, last_ckpt = natsorted(checkpoint_files)
    except ValueError as e:
        raise FileNotFoundError(f"No checkpoint files found in {log_dir}") from e
    assert last_ckpt is not None, f"No checkpoint found in {log_dir}"
    runner = OnPolicyRunner(env, rl_train_cfg, log_dir, device=gs.device)
    runner.load(last_ckpt)
    print(f"Loaded teacher policy from checkpoint {last_ckpt} from {log_dir}")
    teacher_policy = runner.get_inference_policy(device=gs.device)
    return teacher_policy


def load_rl_policy(env: Any, train_cfg: dict, log_dir: Path) -> Callable:
    """Load reinforcement learning policy."""
    runner = OnPolicyRunner(env, train_cfg, log_dir, device=gs.device)

    # Find the latest checkpoint
    checkpoint_files = [
        f for f in log_dir.iterdir() if re.match(r"model_\d+\.pt", f.name)
    ]
    if not checkpoint_files:
        raise FileNotFoundError(f"No checkpoint files found in {log_dir}")

    try:
        *_, last_ckpt = natsorted(checkpoint_files)
    except ValueError as e:
        raise FileNotFoundError(f"No checkpoint files found in {log_dir}") from e
    runner.load(last_ckpt)
    print(f"Loaded RL checkpoint from {last_ckpt}")

    return runner.get_inference_policy(device=gs.device)


def load_bc_policy(env: Any, bc_cfg: dict, log_dir: Path) -> Callable:
    """Load behavior cloning policy."""
    # Create behavior cloning instance
    bc_runner = BehaviorCloning(env, bc_cfg, None, device=gs.device)

    # Find the latest checkpoint
    checkpoint_files = [
        f for f in log_dir.iterdir() if re.match(r"checkpoint_\d+\.pt", f.name)
    ]
    if not checkpoint_files:
        raise FileNotFoundError(f"No checkpoint files found in {log_dir}")

    try:
        *_, last_ckpt = natsorted(checkpoint_files)
    except ValueError as e:
        raise FileNotFoundError(f"No checkpoint files found in {log_dir}") from e
    print(f"Loaded BC checkpoint from {last_ckpt}")
    bc_runner.load(last_ckpt)

    return bc_runner._policy
