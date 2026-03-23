import re
from pathlib import Path
from typing import Any, Callable, Optional

import torch as th
from clearml import Task, Model
from natsort import natsorted
from rsl_rl.runners import OnPolicyRunner

from behavior_cloning import BehaviorCloning


def load_rl_policy(
    env: Any,
    rl_cfg: dict,
    device: th.device,
    exp_name: Optional[str] = None,
    log_dir: Optional[Path] = None,
    clearml_task_id: Optional[str] = None,
    clearml_model_id: Optional[str] = None,
    clearml_artifact_name: str = "model",
) -> Callable:
    print("[Policy Loader] Resolving RL checkpoint")
    last_ckpt = resolve_checkpoint(
        exp_name=exp_name,
        log_dir=log_dir,
        clearml_task_id=clearml_task_id,
        clearml_model_id=clearml_model_id,
        clearml_artifact_name=clearml_artifact_name,
    )
    print(f"[Policy Loader] Resolved RL checkpoint path: {last_ckpt}")

    runner = OnPolicyRunner(env, rl_cfg, log_dir, device=device)
    runner.load(last_ckpt)
    print("[Policy Loader] Loaded RL checkpoint")
    return runner.get_inference_policy(device=device)


def load_bc_policy(
    env: Any,
    bc_cfg: dict,
    device: th.device,
    exp_name: Optional[str] = None,
    log_dir: Optional[Path] = None,
    clearml_task_id: Optional[str] = None,
    clearml_model_id: Optional[str] = None,
    clearml_artifact_name: str = "model",
) -> Callable:
    print("[Policy Loader] Resolving BC checkpoint")
    last_ckpt = resolve_checkpoint(
        exp_name=exp_name,
        log_dir=log_dir,
        clearml_task_id=clearml_task_id,
        clearml_model_id=clearml_model_id,
        clearml_artifact_name=clearml_artifact_name,
    )
    print(f"[Policy Loader] Resolved BC checkpoint path: {last_ckpt}")

    bc_runner = BehaviorCloning(env, bc_cfg, None, log_dir, device=device)
    bc_runner.load(last_ckpt)
    print("[Policy Loader] Loaded BC checkpoint")
    return bc_runner._policy


def resolve_checkpoint(
    exp_name: Optional[str] = None,
    log_dir: Optional[Path] = None,
    clearml_task_id: Optional[str] = None,
    clearml_model_id: Optional[str] = None,
    clearml_artifact_name: str = "model",
    local_checkpoint_pattern: str = r"model_\d+\.pt",
) -> Path:
    print("[Policy Loader] Resolving method and path to load the policy model")

    if clearml_model_id is not None:
        print(f"[Policy Loader] Loading from ClearML model {clearml_model_id}")
        clearml_model = Model(model_id=clearml_model_id)
        ckpt = Path(clearml_model.get_weights(raise_on_error=True))
        print(f"[Policy Loader] Resolved ClearML model {clearml_model_id} to {ckpt}")
        return ckpt

    if clearml_task_id is not None:
        print(f"[Policy Loader] Loading from ClearML task {clearml_task_id}")
        ckpt = Path(
            get_latest_clearml_checkpoint(clearml_task_id, clearml_artifact_name)
        )
        print(f"[Policy Loader] Resolved ClearML task {clearml_task_id} to {ckpt}")
        return ckpt

    print("[Policy Loader] Loading from local file system")
    if log_dir is None and exp_name is None:
        raise ValueError(
            "Cannot resolve a checkpoint: provide log_dir, exp_name, "
            "clearml_model_id, or clearml_task_id."
        )
    resolved_log_dir = log_dir or Path("logs") / f"{exp_name}_rl"
    ckpt = resolve_latest_local_checkpoint(resolved_log_dir, local_checkpoint_pattern)
    print(f"[Policy Loader] Resolved local checkpoint → {ckpt}")
    return ckpt


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


def resolve_latest_local_checkpoint(log_dir: Path, r_pattern: str) -> Path:
    if not log_dir.exists():
        raise FileNotFoundError(f"Log directory {log_dir} does not exist")

    checkpoint_files = [f for f in log_dir.iterdir() if re.match(r_pattern, f.name)]
    if not checkpoint_files:
        raise FileNotFoundError(
            f"No checkpoint files matching '{r_pattern}' found in {log_dir}"
        )

    *_, last_ckpt = natsorted(checkpoint_files)
    return last_ckpt
