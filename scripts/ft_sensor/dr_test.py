"""
F/T domain randomization check (simulation only): the episode is reset a few times;
in every episode the TCP moves over the table, hovers (the noise and the bias are
measured there, with nothing touching), then touches the table, presses and lifts.
The readings are compared against the DR config (`domain_randomization.yaml`).

Run from the repo root, e.g.:
    python -m scripts.ft_sensor.dr_test --env reacher -B 4 --vis --plotjuggler

Expected:
    - bias: constant within an episode, resampled on every reset, within +-range,
      different for every env;
    - clean = hover mean - bias: the same in every episode and env (only the weight,
      |F| ~12.52 N);
    - noise: std per axis ~ the configured std (slightly less, truncated at 3 sigma),
      |noise| never above the configured limit;
    - touch: a clear contact in every episode, the peak dF similar between episodes
      (the PD gains DR may change it a bit).
In PlotJuggler `ft_sensor/force/*` jitters by the noise, jumps by the bias on every
reset of env 0 and shows the contact peak.
"""

import torch as th

from aegis_gym.aux.logging import get_logger, setup_logger
from aegis_gym.config import ConfigManager, parse_arguments
from train import create_env, init_clearml_task

N_EPISODES = 5
SETTLE_S = 0.5
SAMPLE_S = 2.0  # hover sampling for the noise and bias statistics
# Touch (robot base frame, metres). The table spans x ~0.32..0.87 m, top at z ~0.0 m.
HOVER_POS = (0.55, 0.0, 0.10)
TABLE_TOP_Z = 0.0
APPROACH_SPEED = 0.05  # m/s
PRESS_SPEED = 0.01  # m/s
CONTACT_THRESHOLD_N = 2.0
PRESS_HOLD_S = 1.0
AXES = ("Fx", "Fy", "Fz", "Tx", "Ty", "Tz")

logger = get_logger("ft_dr")


def fmt(v: th.Tensor, prec: int = 4) -> str:
    return "[" + " ".join(f"{x:+.{prec}f}" for x in v.tolist()) + "]"


def main() -> None:
    th.set_default_dtype(th.float32)
    args = parse_arguments()
    task = init_clearml_task(
        project_name=args.project_name,
        algorithm=args.algorithm,
        control=args.control_type,
        exp_name=args.experiment_name or "ft_dr_test",
    )
    device = th.device("cuda" if th.cuda.is_available() else "cpu")
    ConfigManager.setup_config(argv=args, device=device, task=task)
    cfg = ConfigManager.get_config()

    dr = cfg.dr_cfg
    noise_cfg, bias_cfg = dr.ft_sensor_noise, dr.ft_sensor_bias
    logger.info(
        f"DR enabled={dr.enabled} | noise enabled={noise_cfg.enabled} | "
        f"bias enabled={bias_cfg.enabled}"
    )
    noise_std = th.tensor([*noise_cfg.force_std, *noise_cfg.torque_std], device=device)
    noise_limit = th.tensor(
        [*noise_cfg.force_limit, *noise_cfg.torque_limit], device=device
    )
    bias_range = th.tensor(
        [*bias_cfg.force_range, *bias_cfg.torque_range], device=device
    )

    env = create_env(cfg)
    env = getattr(env, "unwrapped", env)
    manip = env.manipulator
    scene = env._scene
    dt = env.get_policy_dt()

    def step(lin_vel: th.Tensor | None = None) -> None:
        action = th.zeros(env.num_envs, 6, device=device)
        if lin_vel is not None:
            action[:, :3] = lin_vel / manip.max_linear_speed
        scene.pre_step()
        manip.ctrl_apply_vel_action(action, open_gripper=None)
        scene.step()

    def tcp() -> th.Tensor:
        return manip.get_tcp_position()[0]

    def move_to(target: th.Tensor) -> None:
        for _ in range(round(15.0 / dt)):
            err = target - tcp()
            if err.norm() < 0.005:
                break
            vel = err * 2.0
            if vel.norm() > APPROACH_SPEED:
                vel = vel / vel.norm() * APPROACH_SPEED
            step(vel)
        for _ in range(round(SETTLE_S / dt)):
            step()

    hover = th.tensor(HOVER_POS, device=device)
    down_fast = th.tensor([0.0, 0.0, -APPROACH_SPEED], device=device)
    down_slow = th.tensor([0.0, 0.0, -PRESS_SPEED], device=device)
    up = th.tensor([0.0, 0.0, APPROACH_SPEED], device=device)

    all_noise, biases, cleans, peaks, contact_zs = [], [], [], [], []
    for ep in range(N_EPISODES):
        env.reset()  # resamples the bias (reset_idx -> randomize_domain)
        manip.ctrl_go_to_home()
        for _ in range(round(SETTLE_S / dt)):
            step()
        move_to(hover)

        # --- hover: noise and bias statistics ---
        bias_at_start = manip.get_ft_residual_bias()
        samples = []
        for _ in range(round(SAMPLE_S / dt)):
            step()
            samples.append(manip.get_ft_wrench().clone())
        samples = th.stack(samples)  # [steps, num_envs, 6]
        mean = samples.mean(dim=0)  # [num_envs, 6]

        # --- touch: descend, press, lift ---
        contact_z = None
        for _ in range(round(20.0 / dt)):
            z = tcp()[2].item()
            step(down_fast if z > TABLE_TOP_Z + 0.03 else down_slow)
            if (
                manip.get_ft_wrench()[0, :3] - mean[0, :3]
            ).norm() > CONTACT_THRESHOLD_N:
                contact_z = z
                break
        peak = th.zeros(6, device=device)
        for _ in range(round(PRESS_HOLD_S / dt)):
            step(down_slow)
            d = manip.get_ft_wrench()[0] - mean[0]
            if d[:3].norm() > peak[:3].norm():
                peak = d.clone()
        for _ in range(round(1.0 / dt)):
            step(up)

        bias = manip.get_ft_residual_bias()
        clean = mean - bias
        all_noise.append(samples - mean)
        biases.append(bias)
        cleans.append(clean)
        peaks.append(peak)
        contact_zs.append(contact_z)

        logger.info(f"--- episode {ep} ---")
        logger.info(f"  bias env0        {fmt(bias[0])}")
        logger.info(f"  hover mean env0  {fmt(mean[0])}")
        logger.info(f"  clean env0       {fmt(clean[0])}   (mean - bias)")
        logger.info(f"  noise std env0   {fmt(samples[:, 0].std(dim=0), 5)}")
        logger.info(
            f"  bias unchanged within the episode: {th.equal(bias_at_start, bias)}"
        )
        if env.num_envs > 1:
            logger.info(f"  bias Fz per env  {fmt(bias[:, 2], 3)}")
        logger.info(
            f"  touch: contact at z={contact_z}  peak dW env0 {fmt(peak, 2)}"
            f"  |dF|={peak[:3].norm():.2f} N"
        )

    noise = th.cat(all_noise).reshape(-1, 6)
    biases = th.stack(biases)  # [episodes, num_envs, 6]
    cleans = th.stack(cleans)

    logger.info("=== Summary ===")
    logger.info(
        "  axis |  noise std  cfg std | max|noise|  cfg limit | max|bias|  cfg range"
    )
    for i, axis in enumerate(AXES):
        logger.info(
            f"  {axis:>4} | {noise[:, i].std():9.5f} {noise_std[i]:8.5f} |"
            f" {noise[:, i].abs().max():10.5f} {noise_limit[i]:10.5f} |"
            f" {biases[..., i].abs().max():9.5f} {bias_range[i]:10.5f}"
        )
    logger.info(
        f"  noise within limit: {bool((noise.abs() <= noise_limit + 1e-6).all())}"
    )
    logger.info(
        f"  bias within range:  {bool((biases.abs() <= bias_range + 1e-6).all())}"
    )
    logger.info(
        f"  bias differs between episodes (env0): "
        f"{not th.allclose(biases[0, 0], biases[-1, 0])}"
    )
    spread = (cleans - cleans[0, 0]).abs().amax(dim=(0, 1))
    logger.info(
        f"  clean wrench spread over episodes and envs: {fmt(spread)} "
        "(should be ~0, only the noise of the mean)"
    )
    peak_f = [p[:3].norm().item() for p in peaks]
    logger.info(
        f"  touch |dF| per episode: {[round(x, 2) for x in peak_f]} N, "
        f"contacts: {sum(z is not None for z in contact_zs)}/{N_EPISODES}"
    )


if __name__ == "__main__":
    setup_logger("INFO")
    try:
        main()
    except KeyboardInterrupt:
        logger.info("Stopped by user")
