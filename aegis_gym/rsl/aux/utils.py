import re
from collections.abc import Callable
from pathlib import Path
from typing import Any

from clearml import InputModel, Model, Task
from config import LaunchArgs, get_logger
from config.types import Algorithm, Checkpoint, ExpConfig
from envs import BaseEnv
from natsort import natsorted
from runners import BehaviorCloningRunner, OnPolicyRunner
from torch import nn


def load_policy(env: BaseEnv, cfg: ExpConfig, alg: Algorithm | None = None) -> Callable:
    args: LaunchArgs = cfg.args

    algorithm = alg or cfg.args.algorithm

    policy_args = {
        "env": env,
        "cfg": cfg,
        "load_cfg_from_clearml": not args.enforce_current_config,
        "exp_name": args.experiment_name,
        "clearml_artifact_name": "model",
    }

    # TODO(issue#120) generalize loading config from ClearML
    if algorithm == Algorithm.RL:
        policy_args["clearml_task_id"] = args.load_rl_task_id
        policy_args["clearml_model_id"] = args.load_rl_model_id
        return load_rl_policy(**policy_args)
    if algorithm == Algorithm.BC:
        policy_args["clearml_task_id"] = args.load_bc_task_id
        policy_args["clearml_model_id"] = args.load_bc_model_id
        policy = load_bc_policy(**policy_args)
        policy.eval()
        return policy
    raise ValueError("Unknown learning method")


def load_rl_policy(
    env: Any,
    cfg: ExpConfig,
    load_cfg_from_clearml: bool = True,
    exp_name: str | None = None,
    clearml_task_id: str | None = None,
    clearml_model_id: str | None = None,
    clearml_artifact_name: str = "model",
) -> nn.Module:
    logger = get_logger("Policy Loader")
    logger.info("Resolving RL checkpoint")
    log_dir = cfg.logger_cfg.local_log_dir
    last_ckpt = resolve_checkpoint(
        exp_name=exp_name,
        log_dir=log_dir,
        clearml_task_id=clearml_task_id,
        clearml_model_id=clearml_model_id,
        clearml_artifact_name=clearml_artifact_name,
        local_checkpoint_pattern=r"model_\d+\.pt",
    )
    logger.info(f"Resolved RL checkpoint path: {last_ckpt}")
    if load_cfg_from_clearml:
        if clearml_task_id is None and clearml_model_id is not None:
            clearml_task_id = InputModel(model_id=clearml_model_id).task
        if clearml_task_id is None:
            raise ValueError(
                "Cannot load RL config from ClearML: provide either clearml_task_id or clearml_model_id"
            )
        task = Task.get_task(task_id=clearml_task_id)

        # TODO(issue#120) somehow migrate this feature to the ConfigManager
        cfg_from_clearml = task.get_configuration_object_as_dict("rl_cfg")
        if cfg_from_clearml:
            # TODO(issue#120) this is wrong: we can not apply patches from ConfigManager
            # if ANY kind of extra modificiation is performed, the ConfigManager should be involved
            # TODO(issu#120) For the loaded policy models, get config from the ClearML task/model.
            logger.warning(
                f"WARNING: There is no current option to overwrite the RL config by the configuration from task: {clearml_task_id}."
            )
        else:
            logger.info(
                f"Failed to obtain the RL config from task {clearml_task_id}. Proceeding with the current one"
            )
    else:
        logger.info("Keeping the current RL config")

    runner = OnPolicyRunner(
        env=env,
        cfg=cfg,
    )
    runner.load(last_ckpt)
    logger.info("Loaded RL checkpoint")
    return runner.get_inference_policy(device=cfg.get_device())


def load_bc_policy(
    env: Any,
    cfg: ExpConfig,
    load_cfg_from_clearml: bool = True,
    exp_name: str | None = None,
    log_dir: Path | None = None,
    clearml_task_id: str | None = None,
    clearml_model_id: str | None = None,
    clearml_artifact_name: str = "model",
) -> nn.Module:
    logger = get_logger("Policy Loader")
    logger.info("Resolving BC checkpoint")
    last_ckpt = resolve_checkpoint(
        exp_name=exp_name,
        log_dir=log_dir,
        clearml_task_id=clearml_task_id,
        clearml_model_id=clearml_model_id,
        clearml_artifact_name=clearml_artifact_name,
        local_checkpoint_pattern=r"checkpoint_\d+\.pt",
    )
    logger.info(f"Resolved BC checkpoint path: {last_ckpt}")
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
            # TODO(issu#120) For the loaded policy models, get config from the ClearML task/model.
            logger.warning(
                f"WARNING: There is no current option to overwrite the BC config by the configuration from task: {clearml_task_id}."
            )
        else:
            logger.info(
                f"Failed to obtain the BC config from task {clearml_task_id}. Proceeding with the current one"
            )
    else:
        logger.info("Keeping the current BC config")

    bc_runner = BehaviorCloningRunner(
        env=env,
        cfg=cfg,
        teacher=None,
    )
    bc_runner.load(last_ckpt)
    logger.info("Loaded BC checkpoint")
    return bc_runner.get_inference_policy(device=cfg.get_device())


def resolve_checkpoint(
    exp_name: str | None = None,
    log_dir: Path | None = None,
    clearml_task_id: str | None = None,
    clearml_model_id: str | None = None,
    clearml_artifact_name: str = "model",
    local_checkpoint_pattern: str = r"model_\d+\.pt",
) -> Path:
    logger = get_logger("Policy Loader")
    logger.info("Resolving method and path to load the policy model")

    if clearml_model_id is not None:
        logger.info(f"Loading from ClearML model {clearml_model_id}")
        clearml_model = Model(model_id=clearml_model_id)
        ckpt = Path(clearml_model.get_weights(raise_on_error=True))
        logger.info(f"Resolved ClearML model {clearml_model_id} to {ckpt}")
        return ckpt

    if clearml_task_id is not None:
        logger.info(f"Loading from ClearML task {clearml_task_id}")
        ckpt = Path(
            get_latest_clearml_checkpoint(clearml_task_id, clearml_artifact_name)
        )
        logger.info(f"Resolved ClearML task {clearml_task_id} to {ckpt}")
        return ckpt

    logger.info("Loading from local file system")
    if log_dir is None and exp_name is None:
        raise ValueError(
            "Cannot resolve a checkpoint: provide log_dir, exp_name, "
            "clearml_model_id, or clearml_task_id."
        )
    resolved_log_dir = log_dir or Path("logs") / f"{exp_name}_rl"
    ckpt = resolve_latest_local_checkpoint(resolved_log_dir, local_checkpoint_pattern)
    logger.info(f"Resolved local checkpoint → {ckpt}")
    return ckpt


def get_latest_clearml_checkpoint(
    clearml_task_id: str, clearml_artifact_name: str
) -> str:
    """
    List all artifacts matching a prefix pattern (e.g. 'model_100',
    'model_checkpoint_200') and return the local path of the most recent one.
    """
    logger = get_logger("Policy Loader")
    logger.info(
        f"Loading the latest checkpoint from ClearML task ID: {clearml_task_id}"
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
        logger.info(f"Found checkpoint: {name} (iter {iteration})")

    latest_iter, latest_name = matched[-1]
    logger.info(f"Selecting latest: {latest_name} (iter {latest_iter})")

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
    log_dir: Path | None = None,
    clearml_task_id: str | None = None,
    clearml_model_id: str | None = None,
    clearml_artifact_name: str = "model",
) -> list[Checkpoint]:
    """
    Returns sorted list of all BC checkpoints.
    """
    logger = get_logger("Policy Loader")
    if clearml_model_id is not None:
        logger.info(f"Loading from ClearML model {clearml_model_id}")
        clearml_model = Model(model_id=clearml_model_id)
        ckpt = Path(clearml_model.get_weights(raise_on_error=True))
        logger.info(f"Resolved ClearML model {clearml_model_id} to {ckpt}")
        return [Checkpoint(0, ckpt)]

    if clearml_task_id is not None:
        logger.info(
            f"Loading all BC checkpoints from ClearML task ID: {clearml_task_id}"
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
            logger.info(f"Found checkpoint: {ckpt.path.name} (iter {ckpt.step})")
        return results

    if log_dir is not None:
        logger.info(f"Loading all BC checkpoints from local filesystem: {log_dir}")
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
            logger.info(f"Found checkpoint: {ckpt.path.name} (iter {ckpt.step})")
        return results

    raise ValueError("Cannot resolve a checkpoint")
