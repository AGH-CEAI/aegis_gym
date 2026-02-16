import re
from pathlib import Path
from typing import Any, Callable

from natsort import natsorted

import torch as th
from rsl_rl.runners import OnPolicyRunner

from behavior_cloning import BehaviorCloning


def load_teacher_policy(
    env: Any, rl_train_cfg: dict, exp_name: str, device: th.device
) -> Callable:
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
    runner = OnPolicyRunner(env, rl_train_cfg, log_dir, device=device)
    runner.load(last_ckpt)
    print(f"Loaded teacher policy from checkpoint {last_ckpt} from {log_dir}")
    teacher_policy = runner.get_inference_policy(device=device)
    return teacher_policy


def load_rl_policy(
    env: Any, train_cfg: dict, log_dir: Path, device: th.device
) -> Callable:
    """Load reinforcement learning policy."""
    runner = OnPolicyRunner(env, train_cfg, log_dir, device=device)

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

    return runner.get_inference_policy(device=device)


def load_bc_policy(
    env: Any, bc_cfg: dict, log_dir: Path, device: th.device
) -> Callable:
    """Load behavior cloning policy."""
    # Create behavior cloning instance
    bc_runner = BehaviorCloning(env, bc_cfg, None, log_dir, device=device)

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
