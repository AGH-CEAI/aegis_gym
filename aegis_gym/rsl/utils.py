import re
from pathlib import Path
from typing import Any, Optional

import torch as th
import torch.nn as nn
from clearml import Task, Model, InputModel
from natsort import natsorted
from rsl_rl.runners import OnPolicyRunner

from behavior_cloning import BehaviorCloning
from config.types import DebugCfg, Checkpoint, RLCfg, BCCfg


def load_rl_policy(
    env: Any,
    rl_cfg: RLCfg,
    device: th.device,
    load_cfg_from_clearml: bool = True,
    exp_name: Optional[str] = None,
    log_dir: Optional[Path] = None,
    clearml_task_id: Optional[str] = None,
    clearml_model_id: Optional[str] = None,
    clearml_artifact_name: str = "model",
    enable_logging: bool = True,
) -> nn.Module:
    print("[Policy Loader] Resolving RL checkpoint")
    last_ckpt = resolve_checkpoint(
        exp_name=exp_name,
        log_dir=log_dir,
        clearml_task_id=clearml_task_id,
        clearml_model_id=clearml_model_id,
        clearml_artifact_name=clearml_artifact_name,
        local_checkpoint_pattern=r"model_\d+\.pt",
    )
    print(f"[Policy Loader] Resolved RL checkpoint path: {last_ckpt}")
    if load_cfg_from_clearml:
        if clearml_task_id is None and clearml_model_id is not None:
            clearml_task_id = InputModel(model_id=clearml_model_id).task
        if clearml_task_id is None:
            raise ValueError(
                "Cannot load RL config from ClearML: provide either clearml_task_id or clearml_model_id"
            )
        task = Task.get_task(task_id=clearml_task_id)

        # TODO: somehow migrate this feature to the ConfigManager
        cfg_from_clearml = task.get_configuration_object_as_dict("rl_cfg")
        if cfg_from_clearml:
            # TODO this is wrong: we could not apply patches from ConfigManager
            # if ANY kind of extra modificiation is performed, the ConfigManager should be involved
            rl_cfg = RLCfg.from_dict(cfg_from_clearml)
            print(
                f"[Policy Loader] Overwritten the RL config by the configuration from task: {clearml_task_id}"
            )
        else:
            print(
                f"[Policy Loader] Failed to obtain the RL config from task {clearml_task_id}. Proceeding with the current one"
            )
    else:
        print("[Policy Loader] Keeping the current RL config")

    runner = OnPolicyRunner(
        env=env,
        train_cfg=rl_cfg.as_dict(),
        log_dir=str(log_dir) if enable_logging else None,
        device=str(device),
    )
    runner.load(str(last_ckpt))
    print("[Policy Loader] Loaded RL checkpoint")
    return runner.get_inference_policy(device=str(device))


def load_bc_policy(
    env: Any,
    bc_cfg: BCCfg,
    debug_cfg: DebugCfg,
    device: th.device,
    load_cfg_from_clearml: bool = True,
    exp_name: Optional[str] = None,
    log_dir: Optional[Path] = None,
    clearml_task_id: Optional[str] = None,
    clearml_model_id: Optional[str] = None,
    clearml_artifact_name: str = "model",
) -> nn.Module:
    print("[Policy Loader] Resolving BC checkpoint")
    last_ckpt = resolve_checkpoint(
        exp_name=exp_name,
        log_dir=log_dir,
        clearml_task_id=clearml_task_id,
        clearml_model_id=clearml_model_id,
        clearml_artifact_name=clearml_artifact_name,
        local_checkpoint_pattern=r"checkpoint_\d+\.pt",
    )
    print(f"[Policy Loader] Resolved BC checkpoint path: {last_ckpt}")
    if load_cfg_from_clearml:
        if clearml_task_id is None and clearml_model_id is not None:
            clearml_task_id = InputModel(model_id=clearml_model_id).task
        if clearml_task_id is None:
            raise ValueError(
                "Cannot load BC config from ClearML: provide either clearml_task_id or clearml_model_id"
            )
        task = Task.get_task(task_id=clearml_task_id)
        cfg_from_clearml = task.get_configuration_object_as_dict("bc_cfg")
        if cfg_from_clearml:
            bc_cfg = BCCfg.from_dict(cfg_from_clearml)
            print(
                f"[Policy Loader] Overwritten the BC config by the configuration from task: {clearml_task_id}"
            )
        else:
            print(
                f"[Policy Loader] Failed to obtain the BC config from task {clearml_task_id}. Proceeding with the current one"
            )
    else:
        print("[Policy Loader] Keeping the current BC config")

    bc_runner = BehaviorCloning(
        env=env,
        bc_cfg=bc_cfg,
        debug_cfg=debug_cfg,
        teacher=None,
        log_dir=log_dir,
        device=device,
    )
    bc_runner.load(str(last_ckpt))
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


def get_latest_clearml_checkpoint(
    clearml_task_id: str, clearml_artifact_name: str
) -> str:
    """
    List all artifacts matching a prefix pattern (e.g. 'model_100',
    'model_checkpoint_200') and return the local path of the most recent one.
    """
    print(
        f"[Policy Loader] Loading the latest checkpoint from ClearML task ID: {clearml_task_id}"
    )
    task = Task.get_task(task_id=clearml_task_id)

    # Filter artifacts whose name matches the pattern: prefix_<number>
    pattern = re.compile(rf"^{re.escape(clearml_artifact_name)}_(\d+)$")

    matched = []
    for name in task.artifacts:
        m = pattern.match(name)
        if m:
            iteration = int(m.group(1))
            matched.append((iteration, name))

    if not matched:
        raise FileNotFoundError(
            f"No artifacts matching '{clearml_artifact_name}_<N>' found in task {clearml_task_id}. "
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


def get_bc_checkpoints(
    log_dir: Optional[Path] = None,
    clearml_task_id: Optional[str] = None,
    clearml_model_id: Optional[str] = None,
    clearml_artifact_name: str = "model",
) -> list[Checkpoint]:
    """
    Returns sorted list of all BC checkpoints.
    """
    if clearml_model_id is not None:
        print(f"[Policy Loader] Loading from ClearML model {clearml_model_id}")
        clearml_model = Model(model_id=clearml_model_id)
        ckpt = Path(clearml_model.get_weights(raise_on_error=True))
        print(f"[Policy Loader] Resolved ClearML model {clearml_model_id} to {ckpt}")
        return [Checkpoint(0, ckpt)]

    if clearml_task_id is not None:
        print(
            f"[Policy Loader] Loading all BC checkpoints from ClearML task ID: {clearml_task_id}"
        )
        task = Task.get_task(task_id=clearml_task_id)
        pattern = re.compile(rf"^{re.escape(clearml_artifact_name)}_(\d+)$")

        matched: set[Checkpoint] = set()
        for name in task.artifacts:
            m = pattern.match(name)
            if not m:
                continue
            chk_iter = int(m.group(1))
            local_path = task.artifacts[name].get_local_copy()
            if local_path is None:
                raise FileNotFoundError(f"Failed to download artifact '{name}'")
            matched.add(Checkpoint(chk_iter, Path(local_path)))

        if not matched:
            raise FileNotFoundError(
                f"No artifacts matching '{clearml_artifact_name}_<N>' found in task {clearml_task_id} "
                f"Available artifacts: {list(task.artifacts.keys())}"
            )

        results = sorted(matched)
        for ckpt in results:
            print(
                f"[Policy Loader] Found checkpoint: {ckpt.path.name} (iter {ckpt.step})"
            )
        return results

    if log_dir is not None:
        print(
            f"[Policy Loader] Loading all BC checkpoints from local filesystem: {log_dir}"
        )
        if not log_dir.exists():
            raise FileNotFoundError(f"Log directory {log_dir} does not exist")
        pattern = re.compile(r"checkpoint_(\d+)\.pt")
        matched: set[Checkpoint] = set()
        for file_path in log_dir.iterdir():
            m = pattern.match(file_path.name)
            if not m:
                continue
            matched.add(Checkpoint(int(m.group(1)), file_path))
        if not matched:
            raise FileNotFoundError(
                f"No BC checkpoint files matching 'checkpoint_<N>.pt' found in {log_dir}"
            )
        results = sorted(matched)
        for ckpt in results:
            print(
                f"[Policy Loader] Found checkpoint: {ckpt.path.name} (iter {ckpt.step})"
            )
        return results

    raise ValueError("Cannot resolve a checkpoint")
