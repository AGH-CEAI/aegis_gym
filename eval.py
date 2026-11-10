import copy
import time
from collections.abc import Callable
from pathlib import Path

import cv2  # noqa: F401
import torch as th
from clearml import Task
from tqdm import tqdm

from aegis_gym.aux.logging import get_logger, setup_logger
from aegis_gym.aux.utils import (
    list_clearml_checkpoint_artifacts,
    load_policy,
    without_clearml_load_binding,
)
from aegis_gym.config import (
    ConfigManager,
    LaunchArgs,
    parse_arguments,
)
from aegis_gym.config.types import (
    IMAGE_MODALITIES,
    POLICY_PREVIEW_SEEDS,
    Algorithm,
    DebugCfg,
    ExpConfig,
    Modality,
)
from aegis_gym.envs import BaseEnv
from aegis_gym.envs.wrappers import ObsPreviewEnvWrapper, VisionAugEnvWrapper
from aegis_gym.runners import BehaviorCloningRunner, OnPolicyRunner
from aegis_gym.runners.policy_preview import (
    InferencePolicy,
    PolicyPreviewRecorder,
    get_preview_target_task,
    make_visual_bc_policy,
    preview_iteration,
)
from train import (
    FINAL_EVAL_TITLE,
    create_env,
    create_policy_evaluator,
    create_policy_preview_recorder,
    evaluate_bc_student,
    init_clearml_task,
)


def main():
    logger = get_logger("Eval")
    # Set PyTorch default dtype to float32 for better performance
    th.set_default_dtype(th.float32)

    args: LaunchArgs = parse_arguments()
    if args.resume:
        logger.warning(
            "`--resume` has no effect in the evaluation script. Continuing in 5 s..."
        )
        time.sleep(5)
    # checked before creating the evaluation task
    preview_target = get_preview_target_task(args) if args.preview_checkpoints else None
    eval_target = get_eval_target_task(args) if args.evaluate_checkpoints else None
    # The ClearML task must exists for connecting configuration
    task = init_clearml_task(
        project_name=args.project_name,
        algorithm=args.algorithm,
        control=args.control_type,
        exp_name=args.experiment_name,
    )
    device = th.device("cuda" if th.cuda.is_available() else "cpu")
    ConfigManager.setup_config(
        argv=args, device=device, task=task, source_task_id=get_source_task_id(args)
    )
    cfg: ExpConfig = ConfigManager.get_config()

    env: BaseEnv = create_env(cfg)

    if cfg.debug_cfg.enabled and (
        cfg.debug_cfg.enable_vis_preview or cfg.debug_cfg.enable_record_obs
    ):
        logger.info(">>> Wrapping env with ObsPreviewEnvWrapper")
        env = ObsPreviewEnvWrapper(env=env, cfg_debug=cfg.debug_cfg)

    logger.info(
        f"The episode length is defined as {cfg.env_cfg.episode_length_s} s, which corresponds to {cfg.env_cfg.max_steps}"
    )
    logger.info("Setup done")

    if preview_target is not None:
        preview_checkpoints(env=env, cfg=cfg, target=preview_target)
        logger.info("Finished evaluation script")
        return

    if eval_target is not None:
        evaluate_checkpoints(env=env, cfg=cfg, target=eval_target)
        logger.info("Finished evaluation script")
        return

    with th.no_grad():
        if is_checkpoints_sweep_required(
            args, record=cfg.logger_cfg.enable_policy_preview
        ):
            raise NotImplementedError("[Eval] Deprecated feature")

        eval_policy_single(env=env, cfg=cfg, clearml_task=task)

    logger.info("Finished evaluation script")


def get_source_task_id(args: LaunchArgs) -> str | None:
    """ClearML task the evaluated policy was trained in."""
    if args.algorithm == Algorithm.RL:
        return ConfigManager.resolve_source_task_id(
            task_id=args.load_rl_task_id, model_id=args.load_rl_model_id
        )
    return ConfigManager.resolve_source_task_id(
        task_id=args.load_bc_task_id, model_id=args.load_bc_model_id
    )


def preview_checkpoints(env: BaseEnv, cfg: ExpConfig, target: Task) -> None:
    """
    Records the policy preview of each `model_<N>` checkpoint of the `target` training task and
    reports it to its Debug Samples (and the `policy_preview/*` scalars), at the iteration the
    training would have reported it at (see `preview_iteration()`).
    """
    logger = get_logger("Eval")
    algorithm = cfg.args.algorithm
    if env.num_envs < len(POLICY_PREVIEW_SEEDS):
        logger.warning(
            f"Only {env.num_envs} envs, the preview shows the first {env.num_envs} of the "
            f"{len(POLICY_PREVIEW_SEEDS)} seeds (set `-B {len(POLICY_PREVIEW_SEEDS)}`)."
        )
    recorder = PolicyPreviewRecorder(
        env=env,
        out_dir=Path(cfg.logger_cfg.local_log_dir) / "policy_preview",
        report_task=target,
    )
    if not recorder.is_available():
        raise RuntimeError("The policy preview isn't available for this environment.")

    load_checkpoint = make_checkpoint_loader(env=env, cfg=cfg)
    checkpoints = list_clearml_checkpoint_artifacts(target)
    logger.info(
        f"Previewing {len(checkpoints)} checkpoints of the task {target.id}: "
        + ", ".join(name for _, name in checkpoints)
    )
    for checkpoint_iter, name in checkpoints:
        local_path = target.artifacts[name].get_local_copy()
        if local_path is None:
            raise FileNotFoundError(f"Failed to download the artifact '{name}'")
        policy = load_checkpoint(Path(local_path))
        recorder.record(
            policy=policy, iteration=preview_iteration(algorithm, checkpoint_iter)
        )
        # sent per checkpoint, the previews reported so far are kept if interrupted
        target.flush(wait_for_uploads=True)
    logger.info(f"Previews reported to: {target.get_output_log_web_page()}")


def get_eval_target_task(args: LaunchArgs) -> Task:
    """The ClearML training task whose checkpoints are evaluated (`--evaluate-checkpoints`)."""
    is_rl = args.algorithm == Algorithm.RL
    task_id = args.load_rl_task_id if is_rl else args.load_bc_task_id
    if task_id is None:
        flag = "--load-rl-task-id" if is_rl else "--load-bc-task-id"
        raise ValueError(
            f"`--evaluate-checkpoints` requires the training task, see `{flag}`."
        )
    task = Task.get_task(task_id=task_id)
    if task.get_status() == "published":
        raise ValueError(
            f"The ClearML task {task_id} is published, its results can't be extended."
        )
    return task


def evaluate_checkpoints(env: BaseEnv, cfg: ExpConfig, target: Task) -> None:
    """
    The success rate of the `target` task's best (`model_best`) and last checkpoint on the
    held-out seeds, reported to the `target` task as `policy_eval_final_best` / `_last` (with the
    domain randomization: `..._dr<seed>`), also as its single values.
    """
    logger = get_logger("Eval")
    args = cfg.args
    cfg_log = cfg.logger_cfg
    if args.eval_episodes is not None:
        cfg_log.policy_eval_final_episodes = args.eval_episodes
    elif cfg_log.policy_eval_final_episodes <= 0:
        cfg_log.policy_eval_final_episodes = 1000
    cfg_log.policy_eval_dr_seed = args.eval_dr_seed

    is_bc = args.algorithm == Algorithm.BC
    if is_bc and args.eval_dr_seed is not None and cfg.dr_cfg.enabled:
        env = VisionAugEnvWrapper(
            env=env, cfg_image_aug=cfg.dr_cfg.image_aug, cfg_env=cfg.env_cfg
        )
    evaluator = create_policy_evaluator(
        env=env,
        cfg=cfg,
        final=True,
        report_task=target,
        # per task: the evaluations of several tasks in one experiment don't overwrite each other
        out_dir=Path(cfg_log.local_log_dir) / "policy_eval" / target.id,
    )
    if evaluator is None:
        raise RuntimeError("The policy evaluation isn't available for this setup.")

    prefix = "checkpoint" if is_bc else "model"
    checkpoints = list_clearml_checkpoint_artifacts(
        target, clearml_artifact_name=prefix
    )
    named = {"last": checkpoints[-1][1]} if checkpoints else {}
    if "model_best" in target.artifacts:
        named["best"] = "model_best"
    if not named:
        raise FileNotFoundError(
            f"The ClearML task {target.id} has no `model_best` or `{prefix}_<N>` checkpoints, "
            f"available artifacts: {list(target.artifacts)}"
        )

    load_checkpoint = make_checkpoint_runner_loader(env=env, cfg=cfg)
    suffix = evaluator.title_suffix
    # by iteration, so the best checkpoint being the last one is evaluated once
    to_evaluate: dict[int, tuple[Callable, list[str]]] = {}
    for kind, name in named.items():
        local_path = target.artifacts[name].get_local_copy()
        if local_path is None:
            raise FileNotFoundError(f"Failed to download the artifact '{name}'")
        policy, iteration = load_checkpoint(Path(local_path))
        logger.info(f"The {kind} checkpoint `{name}` is of the iteration {iteration}")
        # a copy, the runner's policy is replaced by the next checkpoint
        _, titles = to_evaluate.setdefault(iteration, (copy.deepcopy(policy), []))
        titles.append(f"{FINAL_EVAL_TITLE}_{kind}{suffix}")

    for iteration, (policy, titles) in to_evaluate.items():
        logger.info(f">>> Evaluating the iteration {iteration} as {', '.join(titles)}")
        if is_bc:
            evaluate_bc_student(
                evaluator=evaluator,
                env=env,
                student=policy,
                cfg=cfg,
                iteration=iteration,
                titles=titles,
                single_values=True,
            )
        else:
            evaluator.evaluate(
                policy=policy, iteration=iteration, titles=titles, single_values=True
            )
        target.flush(wait_for_uploads=True)
    logger.info(f"Evaluation reported to: {target.get_output_log_web_page()}")


def make_checkpoint_runner_loader(
    env: BaseEnv, cfg: ExpConfig
) -> Callable[[Path], tuple[Callable, int]]:
    """Loads a checkpoint into a single runner: (its policy, the checkpoint's iteration)."""
    device = cfg.get_device()
    if cfg.args.algorithm == Algorithm.RL:
        rl_runner = OnPolicyRunner(env=env, cfg=cfg)

        def load_rl(path: Path) -> tuple[Callable, int]:
            with without_clearml_load_binding():
                rl_runner.load(path)
            return rl_runner.get_inference_policy(device=device), rl_runner.current_iter

        return load_rl

    bc_runner = BehaviorCloningRunner(env=env, cfg=cfg, teacher=None)

    def load_bc(path: Path) -> tuple[Callable, int]:
        with without_clearml_load_binding():
            bc_runner.load(path)
        return bc_runner.get_inference_policy(device=device), bc_runner.current_iter

    return load_bc


def make_checkpoint_loader(
    env: BaseEnv, cfg: ExpConfig
) -> Callable[[Path], InferencePolicy]:
    """
    Loads the checkpoints into a single runner, as the preview's policy. They aren't connected to
    the evaluation task as its input models, these are the target task's own checkpoints.
    """
    device = cfg.get_device()
    if cfg.args.algorithm == Algorithm.RL:
        rl_runner = OnPolicyRunner(env=env, cfg=cfg)

        def load_rl(path: Path) -> InferencePolicy:
            with without_clearml_load_binding():
                rl_runner.load(path)
            return rl_runner.get_inference_policy(device=device)

        return load_rl

    bc_runner = BehaviorCloningRunner(env=env, cfg=cfg, teacher=None)

    def load_bc(path: Path) -> InferencePolicy:
        with without_clearml_load_binding():
            bc_runner.load(path)
        student = bc_runner.get_inference_policy(device=device).eval()
        return make_visual_bc_policy(
            env=env, bc_policy=student, state_obs=cfg.bc_cfg.policy.state_obs
        )

    return load_bc


def is_checkpoints_sweep_required(args: LaunchArgs, record: bool) -> bool:
    logger = get_logger("Eval")
    sweep = (args.algorithm == Algorithm.BC) and (
        args.bc_all_checkpoints or args.bc_eval_every is not None
    )
    if args.algorithm == Algorithm.RL and (
        args.bc_all_checkpoints or args.bc_eval_every is not None
    ):
        logger.warning(
            "WARNING: multi-checkpoint sweep are only supported for BC; ignoring for RL"
        )
    if sweep and record:
        logger.warning("WARNING: record is ignored during multi-checkpoint sweep")
    return sweep


def eval_policy_single(
    env: BaseEnv,
    cfg: ExpConfig,
    clearml_task: Task,
) -> None:
    args = cfg.args
    device = cfg.get_device()
    max_steps = int(cfg.env_cfg.max_steps)

    # TODO(issue#101): Design arguments and config manager for policy loading
    policy = load_policy(env=env, cfg=cfg)
    obs, _ = env.reset()
    metrics = run_eval(
        env=env,
        policy=policy,
        stage=args.algorithm,
        max_steps=max_steps,
        obs=obs,
        device=device,
        debug_cfg=cfg.debug_cfg,
    )
    log_metrics(clearml_task, metrics)

    preview = create_policy_preview_recorder(env=env, cfg=cfg)
    if preview is None:
        return
    if args.algorithm == Algorithm.BC:
        policy = make_visual_bc_policy(
            env=env, bc_policy=policy, state_obs=cfg.bc_cfg.policy.state_obs
        )
    preview.record(policy=policy, iteration=0)


# TODO(issue#100): Unify policy model types under a common base class or type alias
def run_eval(
    env: BaseEnv,
    policy: Callable,
    stage: Algorithm,
    max_steps: int,
    obs: th.Tensor,
    device: th.device,
    debug_cfg: DebugCfg,
) -> dict[str, float]:
    logger = get_logger("Eval")
    total_rewards = th.zeros(env.num_envs, device=device)
    episode_lengths = th.zeros(env.num_envs, device=device)

    start_time = time.perf_counter()
    total_inference_time = 0.0

    # TODO(issue#135) This shouldn't be defined in the eval code!
    def get_obs_vis() -> th.Tensor:
        obs = env.get_modality_observations(modalities=IMAGE_MODALITIES)
        return th.cat([obs[m] for m in IMAGE_MODALITIES], dim=1).float()

    vis_debug_params = {}
    if debug_cfg.enabled:
        vis_debug_params = debug_cfg.as_dict()
        vis_debug_params.pop("enabled")

    for _ in tqdm(range(max_steps), desc="Evaluation", unit="step"):
        match stage:
            case Algorithm.RL:
                actions = policy(obs)
            case Algorithm.BC:
                rgb_obs = get_obs_vis()
                tcp_pose = env.get_modality_observation(Modality.TCP_POSE)
                actions = policy(rgb_obs, tcp_pose)

        obs, rews, _dones, _infos = env.step(actions)

        total_rewards += rews
        episode_lengths += 1

    logger.info("Finished model inference")

    end_time = time.perf_counter()
    total_inference_time += end_time - start_time

    mean_reward = total_rewards.mean().item()
    mean_episode_length = episode_lengths.mean().item()
    mean_inference_time = total_inference_time / max_steps
    fps = 1.0 / mean_inference_time

    return {
        "mean_reward": mean_reward,
        "mean_episode_length": mean_episode_length,
        "mean_inference_time_s": mean_inference_time,
        "policy_fps": fps,
    }


def log_metrics(task: Task, metrics: dict[str, float], step: int = 0) -> None:
    print_logger = get_logger("Eval")
    info_str = (
        f"Mean reward: {metrics['mean_reward']:.6f}\n"
        f"Mean episode length: {metrics['mean_episode_length']:.0f}\n"
        f"Mean inference time: {metrics['mean_inference_time_s']:.6f}\n"
        f"FPS: {metrics['policy_fps']:.2f}"
    )
    print_logger.info(info_str)

    logger = task.get_logger()
    logger.report_scalar("Eval/mean_reward", "series", metrics["mean_reward"], step)
    logger.report_scalar(
        "Eval/mean_episode_length", "series", metrics["mean_episode_length"], step
    )
    logger.report_scalar(
        "Perf/mean_inference_time_s", "series", metrics["mean_inference_time_s"], step
    )
    logger.report_scalar("Perf/fps", "series", metrics["policy_fps"], step)


if __name__ == "__main__":
    setup_logger("INFO")
    logger = get_logger("Train")

    try:
        main()
    except KeyboardInterrupt:
        logger.info("\n\n\n[Eval] > Exiting (invoked by user)")
