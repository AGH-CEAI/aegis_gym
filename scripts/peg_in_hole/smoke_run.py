"""
Peg-in-hole smoke run: plays episodes with the oracle (`PegInHoleEnv.scripted_action()`) or a
random policy and logs the episode metrics, to check the env, the assets and the contacts in
the simulation without any training. With the oracle most episodes end in a success.

Run from the repo root, e.g.:
    python -m scripts.peg_in_hole.smoke_run --env peg_in_hole \
        -c aegis_gym/config/presets/peg_in_hole.yaml -B 4 --vis --no-record -e pih_smoke
    python -m scripts.peg_in_hole.smoke_run ... --policy random --episodes 5
"""

import argparse

import torch as th

from aegis_gym.aux.logging import get_logger, setup_logger
from aegis_gym.config import ConfigManager, parse_arguments
from aegis_gym.envs import PegInHoleEnv
from train import create_env, init_clearml_task

LOG_EVERY = 25  # steps

logger = get_logger("pih_smoke")


def _extra_args(p: argparse.ArgumentParser) -> None:
    p.add_argument("--policy", choices=["scripted", "random"], default="scripted")
    p.add_argument("--episodes", type=int, default=3, help="episodes per env")


def main() -> None:
    th.set_default_dtype(th.float32)
    args = parse_arguments(extra_argparser=_extra_args)
    task = init_clearml_task(
        project_name=args.project_name,
        algorithm=args.algorithm,
        control=args.control_type,
        exp_name=args.experiment_name,
    )
    device = th.device("cuda" if th.cuda.is_available() else "cpu")
    ConfigManager.setup_config(argv=args, device=device, task=task)
    cfg = ConfigManager.get_config()
    policy, episodes = args._args_raw.policy, args._args_raw.episodes

    env = create_env(cfg)
    env = getattr(env, "unwrapped", env)
    if not isinstance(env, PegInHoleEnv):
        raise TypeError("Run it with `--env peg_in_hole`.")
    env.reset()

    finished = th.zeros(env.num_envs, dtype=th.long, device=device)
    successes = 0
    step_no = 0
    while (finished < episodes).any():
        if policy == "scripted":
            actions = env.scripted_action()
        else:
            actions = th.rand(env.num_envs, env.num_actions, device=device) * 2 - 1
        _, reward, dones, extras = env.step(actions)
        step_no += 1

        if step_no % LOG_EVERY == 0:
            logger.info(
                f"step {step_no:5d} | env 0: xy {1000 * env.xy_err[0]:5.2f} mm, "
                f"depth {1000 * env.depth[0]:6.2f} mm, "
                f"yaw {th.rad2deg(env.yaw_err[0]):6.2f} deg, "
                f"|F| {env.force_norm[0]:6.2f} N, reward {reward[0]:+.4f}"
            )
        if dones.any():
            successes += int((extras["success"] & dones).sum())
            finished += dones.long()
            ended = {
                k: v.mean().item() if isinstance(v, th.Tensor) else v
                for k, v in extras["episode"].items()
            }
            logger.info(f"step {step_no:5d} | {int(dones.sum())} ended: {ended}")

    total = int(finished.sum())
    logger.info(f"Done: {successes}/{total} successful episodes ({policy} policy).")


if __name__ == "__main__":
    setup_logger("INFO")
    try:
        main()
    except KeyboardInterrupt:
        logger.info("Stopped by user")
