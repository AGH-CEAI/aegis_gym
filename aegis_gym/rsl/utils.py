import re
from pathlib import Path
from typing import Any, Callable, Optional

import torch as th
from clearml import Task
from natsort import natsorted
from rsl_rl.runners import OnPolicyRunner

from behavior_cloning import BehaviorCloning


def load_teacher_policy(
    env: Any,
    rl_train_cfg: dict,
    device: th.device,
    exp_name: Optional[str] = None,
    log_dir: Optional[Path] = None,
    clearml_task_id: Optional[str] = None,
    clearml_model_id: Optional[str] = None,
    clearml_artifact_name: str = "model",
) -> Callable:
    if clearml_model_id is not None:
        from clearml import Model

        clearml_model = Model(model_id=clearml_model_id)
        last_ckpt = Path(clearml_model.get_weights(raise_on_error=True))
        print(f"[Policy Loader] Loaded from ClearML model {clearml_model_id}")

    elif clearml_task_id is not None:
        last_ckpt = Path(
            get_latest_clearml_checkpoint(clearml_task_id, clearml_artifact_name)
        )
        print(f"[Policy Loader] Loaded from ClearML task {clearml_task_id}")

    else:
        if log_dir is None and exp_name is None:
            raise ValueError(
                "Couldn't figure out the path to load the pre-trained policy. Provide log_dir or exp_name or ClearML's model_id or task_id."
            )
        resolved_log_dir = log_dir or Path("logs") / f"{exp_name}_rl"
        assert resolved_log_dir.exists(), (
            f"Log directory {resolved_log_dir} does not exist"
        )
        checkpoint_files = [
            f for f in resolved_log_dir.iterdir() if re.match(r"model_\d+\.pt", f.name)
        ]
        try:
            *_, last_ckpt = natsorted(checkpoint_files)
        except ValueError as e:
            raise FileNotFoundError(
                f"No checkpoint files found in {resolved_log_dir}"
            ) from e
        print(f"[Policy Loader] Loaded from local checkpoint {last_ckpt}")

    runner = OnPolicyRunner(env, rl_train_cfg, last_ckpt.parent, device=device)
    runner.load(last_ckpt)
    return runner.get_inference_policy(device=device)


def get_latest_clearml_checkpoint(task_id: str, artifact_prefix: str) -> str:
    """
    List all artifacts matching a prefix pattern (e.g. 'model_100',
    'model_checkpoint_200') and return the local path of the most recent one.
    """
    print(
        f"[Policy Loader] Loading the latest checkpoint from ClearML task id: {task_id}"
    )
    task = Task.get_task(task_id=task_id)

    # Filter artifacts whose name matches the pattern: prefix_<number>
    pattern = re.compile(rf"^{re.escape(artifact_prefix)}_(\d+)$")

    matched = []
    for name in task.artifacts:
        m = pattern.match(name)
        if m:
            iteration = int(m.group(1))
            matched.append((iteration, name))

    if not matched:
        raise FileNotFoundError(
            f"No artifacts matching '{artifact_prefix}_<N>' found in task {task_id}. "
            f"Available artifacts: {list(task.artifacts.keys())}"
        )

    # Pick the one with the highest iteration number
    matched.sort(key=lambda x: x[0])
    for iteration, name in matched:
        print(f"[Policy Loader] Found checkpoint: {name} (iter {iteration})")

    latest_iter, latest_name = matched[-1]
    print(f"[Policy Loader] Selecting latest: {latest_name} (iter {latest_iter})")

    local_path = task.artifacts[latest_name].get_local_copy()
    if local_path is None:
        raise FileNotFoundError(f"Failed to download artifact '{latest_name}'")

    return local_path


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
    print(f"[Policy Loader] Loaded RL checkpoint from {last_ckpt}")

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
    print(f"[Policy Loader] Loaded BC checkpoint from {last_ckpt}")
    bc_runner.load(last_ckpt)

    return bc_runner._policy
