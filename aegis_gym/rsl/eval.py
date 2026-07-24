import time
from collections.abc import Callable

import torch as th
from aux import load_policy
from clearml import Task
from config import ConfigManager, LaunchArgs, get_logger, parse_arguments, setup_logger
from config.types import (
    IMAGE_MODALITIES,
    Algorithm,
    DebugCfg,
    ExpConfig,
    Modality,
)
from envs import BaseEnv
from envs.wrappers import ObsPreviewEnvWrapper
from tqdm import tqdm
from train import create_env, init_clearml_task


def main():
    logger = get_logger("Eval")
    # Set PyTorch default dtype to float32 for better performance
    th.set_default_dtype(th.float32)

    args: LaunchArgs = parse_arguments()
    # The ClearML task must exists for connecting configuration
    task = init_clearml_task(
        project_name=args.project_name,
        algorithm=args.algorithm,
        control=args.control_type,
        exp_name=args.experiment_name,
    )
    device = th.device("cuda" if th.cuda.is_available() else "cpu")
    ConfigManager.setup_config(argv=args, device=device, task=task)
    cfg: ExpConfig = ConfigManager.get_config()

    env: BaseEnv = create_env(cfg)

    if cfg.debug_cfg.enabled and (
        cfg.debug_cfg.enable_vis_preview or cfg.debug_cfg.enable_record_obs
    ):
        logger.ingo(">>> Wrapping env with ObsPreviewEnvWrapper")
        env = ObsPreviewEnvWrapper(env=env, cfg_debug=cfg.debug_cfg)

    logger.info(
        f"The episode length is defined as {cfg.env_cfg.episode_length_s} s, which corresponds to {cfg.env_cfg.max_steps}"
    )
    logger.info("Setup done")

    with th.no_grad():
        if is_checkpoints_sweep_required(args):
            raise NotImplementedError("[Eval] Deprecated feature")

        eval_policy_single(env=env, cfg=cfg, clearml_task=task)

    logger.info("Finished evaluation script")


def is_checkpoints_sweep_required(args: LaunchArgs) -> bool:
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
    if sweep and args.enable_recording:
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

        obs, rews, dones, infos = env.step(actions)

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
