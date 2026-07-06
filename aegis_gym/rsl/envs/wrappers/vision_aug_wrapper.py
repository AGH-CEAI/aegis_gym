from typing import Optional, Collection

import torch as th
import torch.nn.functional as F
from tensordict import TensorDict

from .base_wrapper import BaseEnvWrapper
from ..base_env import BaseEnv, StepReturn, Modality, IMAGE_MODALITIES, ResetReturn

from config.types import ImageAugCfg, EnvCfg


class VisionAugEnvWrapper(BaseEnvWrapper):
    """
    The wrapper to augment visual observations from the environment.
    """

    def __init__(self, env: BaseEnv, cfg_image_aug: ImageAugCfg, cfg_env: EnvCfg):
        super().__init__(env=env)
        self._cfg = cfg_image_aug
        self.image_resolution = cfg_env.image_resolution
        self.device = env.device
        self._aug_profile: dict[str, th.Tensor] = self._init_aug_profile()

    def get_observations(self) -> TensorDict:
        obs = self.get_modality_observations()
        obs = self._augment(obs)
        agent_obs = self._build_agent_observations(obs)
        return self._format_rslrl_observations(agent_obs)

    def get_modality_observations(
        self, modalities: Optional[Collection[Modality]] = None
    ) -> TensorDict:
        obs = self._env.get_modality_observations(modalities)
        # Warning: env wrappers doesn't use cache - every call to the _augument() probably will result in different results.
        return self._augment(obs)

    def _augment(self, obs: TensorDict) -> TensorDict:
        image_keys = [m for m in obs.keys() if Modality(m) in IMAGE_MODALITIES]
        if not image_keys:
            return obs

        images = obs.select(*image_keys)
        augmented = images.apply(self._apply_image_augmentation)
        obs.update(augmented)
        return obs

    def reset(self) -> ResetReturn:
        res = self._env.reset()
        # TODO(issue#121) Previously it was called in the reset_idx()
        self._resample_aug_profile(th.arange(self.num_envs, device=self.device))
        return res

    def step(self, actions: th.Tensor) -> StepReturn:
        res = self._env.step(actions)
        return res

    def _init_aug_profile(self) -> dict[str, th.Tensor]:
        if not self._cfg.enabled or not self._cfg.per_episode_aug:
            return {}
        N = self.num_envs
        return {
            "brightness_jitter": th.zeros(N, device=self.device),
            "contrast_jitter": th.zeros(N, device=self.device),
            "gaussian_noise_std": th.zeros(N, device=self.device),
            "gamma_range": th.zeros(N, device=self.device),
            "blur_active": th.zeros(N, dtype=th.bool, device=self.device),
            "channel_jitter": th.zeros(N, 3, device=self.device),
            "cutout_active": th.zeros(N, dtype=th.bool, device=self.device),
            "cutout_y": th.zeros(N, dtype=th.long, device=self.device),
            "cutout_x": th.zeros(N, dtype=th.long, device=self.device),
            "cutout_h": th.zeros(N, dtype=th.long, device=self.device),
            "cutout_w": th.zeros(N, dtype=th.long, device=self.device),
        }

    def _resample_aug_profile(self, envs_idx: th.Tensor) -> None:
        if not self._aug_profile:
            return
        n = len(envs_idx)
        dev = self.device

        def sample(max_val: float) -> th.Tensor:
            active = th.rand(n, device=dev) < 0.5
            return active.float() * (th.rand(n, device=dev) * max_val)

        self._aug_profile["brightness_jitter"][envs_idx] = sample(
            self._cfg.brightness_jitter
        )
        self._aug_profile["contrast_jitter"][envs_idx] = sample(
            self._cfg.contrast_jitter
        )
        self._aug_profile["gaussian_noise_std"][envs_idx] = sample(
            self._cfg.gaussian_noise_std
        )
        self._aug_profile["gamma_range"][envs_idx] = sample(self._cfg.gamma_range)
        self._aug_profile["blur_active"][envs_idx] = (
            th.rand(n, device=dev) < self._cfg.blur_prob
        )

        ch = self._cfg.channel_jitter
        active = (th.rand(n, device=dev) < 0.5).float().unsqueeze(1)
        self._aug_profile["channel_jitter"][envs_idx] = (
            active * (th.rand(n, 3, device=dev) * 2.0 - 1.0) * ch
        )

        cutout_cfg = self._cfg.cutout
        prob = cutout_cfg.prob
        min_sz = cutout_cfg.min_size
        max_sz = cutout_cfg.max_size
        H, W = self.image_resolution

        active = th.rand(n, device=dev) < prob
        self._aug_profile["cutout_active"][envs_idx] = active

        hs = th.randint(min_sz, max_sz + 1, (n,), device=dev)
        ws = th.randint(min_sz, max_sz + 1, (n,), device=dev)
        ys = (th.rand(n, device=dev) * (H - hs).clamp(min=1)).long()
        xs = (th.rand(n, device=dev) * (W - ws).clamp(min=1)).long()

        self._aug_profile["cutout_h"][envs_idx] = hs
        self._aug_profile["cutout_w"][envs_idx] = ws
        self._aug_profile["cutout_y"][envs_idx] = ys
        self._aug_profile["cutout_x"][envs_idx] = xs

    def _apply_image_augmentation(self, im: th.Tensor) -> th.Tensor:
        aug = self._cfg
        if not aug.enabled:
            return im

        N, C, H, W = im.shape
        device = self.device
        prof = self._aug_profile

        def sample_magnitude(key: str, shape: tuple) -> th.Tensor:
            """Magnitude in [0, max_val]; sign is re-sampled each frame for variety."""
            mag = (
                prof[key].view(shape)
                if prof
                else th.rand(shape, device=device) * getattr(aug, key)
            )
            return mag * (th.rand(shape, device=device) * 2.0 - 1.0)

        def sample_signed(key: str, shape: tuple) -> th.Tensor:
            """Already a signed delta in the profile (channel_jitter)."""
            if prof:
                return prof[key].view(shape)
            return (th.rand(shape, device=device) * 2.0 - 1.0) * getattr(aug, key)

        # -- Brightness --
        b_delta = sample_magnitude("brightness_jitter", (N, 1, 1, 1))
        if b_delta.abs().any():
            im = im * (1.0 + b_delta)

        # -- Per-channel jitter (signed delta, no re-sampling) --
        ch_delta = sample_signed("channel_jitter", (N, C, 1, 1))
        if ch_delta.abs().any():
            im = im * (1.0 + ch_delta)

        # -- Contrast --
        c_delta = sample_magnitude("contrast_jitter", (N, 1, 1, 1))
        if c_delta.abs().any():
            mean = im.mean(dim=(1, 2, 3), keepdim=True)
            im = (im - mean) * (1.0 + c_delta) + mean

        # -- Gaussian noise (magnitude only, always additive) --
        noise_std = sample_magnitude("gaussian_noise_std", (N, 1, 1, 1)).abs()
        if noise_std.any():
            im = im + th.randn_like(im) * noise_std

        # -- Gamma --
        g_delta = sample_magnitude("gamma_range", (N, 1, 1, 1))
        if g_delta.abs().any():
            im = im.clamp(1e-6, 1.0).pow(1.0 + g_delta)

        # -- Gaussian blur (per-env) --
        blur_active = (
            prof["blur_active"] if prof else th.rand(N, device=device) < aug.blur_prob
        )
        if blur_active.any():
            blurred = self._apply_gaussian_blur(
                im, kernel_size=aug.blur_kernel_size, sigma=aug.blur_sigma
            )
            im = th.where(blur_active.view(N, 1, 1, 1), blurred, im)

        # -- Cutout --
        if aug.cutout.prob > 0.0 or (prof and prof["cutout_active"].any()):
            im = self._apply_cutout(im)

        return im.clamp(0.0, 1.0)

    def _apply_gaussian_blur(
        self,
        im: th.Tensor,
        kernel_size: int,
        sigma: float,
    ) -> th.Tensor:
        x = (
            th.arange(kernel_size, dtype=th.float32, device=self.device)
            - kernel_size // 2
        )
        k1d = th.exp(-(x**2) / (2.0 * sigma**2))
        k1d = k1d / k1d.sum()
        k2d = k1d.unsqueeze(0) * k1d.unsqueeze(1)
        C = im.shape[1]
        kernel = k2d.unsqueeze(0).unsqueeze(0).expand(C, 1, kernel_size, kernel_size)
        pad = kernel_size // 2
        return F.conv2d(im, kernel, padding=pad, groups=C)

    def _apply_cutout(self, im: th.Tensor) -> th.Tensor:
        N, C, H, W = im.shape
        device = im.device
        prof = self._aug_profile

        # --- Resolve box coordinates (profile replay or fresh sample) ---
        if prof and "cutout_active" in prof:
            # Replay per-episode profile (already on correct device)
            active = prof["cutout_active"]  # (N,) bool
            hs = prof["cutout_h"]  # (N,) long
            ws = prof["cutout_w"]  # (N,) long
            ys = prof["cutout_y"]  # (N,) long
            xs = prof["cutout_x"]  # (N,) long
        else:
            if not prof:
                return im  # DR disabled / no per-episode aug
            cutout_cfg = self._cfg.cutout
            active = th.rand(N, device=device) < cutout_cfg.prob
            hs = th.randint(
                cutout_cfg.min_size, cutout_cfg.max_size + 1, (N,), device=device
            )
            ws = th.randint(
                cutout_cfg.min_size, cutout_cfg.max_size + 1, (N,), device=device
            )
            ys = (th.rand(N, device=device) * (H - hs).clamp(min=1)).long()
            xs = (th.rand(N, device=device) * (W - ws).clamp(min=1)).long()

        if not active.any():
            return im

        # --- Build vectorized binary mask: 0 = zeroed region ---
        row_idx = th.arange(H, device=device).view(1, H, 1)  # (1, H, 1)
        col_idx = th.arange(W, device=device).view(1, 1, W)  # (1, 1, W)

        in_box = (
            (row_idx >= ys.view(N, 1, 1))
            & (row_idx < (ys + hs).view(N, 1, 1))
            & (col_idx >= xs.view(N, 1, 1))
            & (col_idx < (xs + ws).view(N, 1, 1))
        )  # (N, H, W)

        mask = ~(in_box & active.view(N, 1, 1))  # (N, H, W)
        return im * mask.unsqueeze(1)  # broadcast over C
