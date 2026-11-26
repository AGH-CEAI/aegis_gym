from collections.abc import Collection

import torch as th
import torch.nn.functional as F
from tensordict import TensorDict

from aegis_gym.config.types import IMAGE_MODALITIES, EnvCfg, ImageAugCfg, Modality

from ..base_env import BaseEnv, ResetReturn, StepReturn
from .base_wrapper import BaseEnvWrapper


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
        self._episode_steps = th.zeros(self.num_envs, dtype=th.long, device=self.device)
        self._last_frames: dict[
            int, th.Tensor
        ] = {}  # per camera, for the `freeze` dropout

    def get_observations(self) -> TensorDict:
        obs = self.get_modality_observations()
        obs = self._augment(obs)
        agent_obs = self._build_agent_observations(obs)
        return self._format_rslrl_observations(agent_obs)

    def get_modality_observations(
        self, modalities: Collection[Modality] | None = None
    ) -> TensorDict:
        obs = self._env.get_modality_observations(modalities)
        # Warning: env wrappers doesn't use cache - every call to the _augument() probably will result in different results.
        return self._augment(obs)

    def _augment(self, obs: TensorDict) -> TensorDict:
        image_keys = [m for m in obs.keys() if Modality(m) in IMAGE_MODALITIES]  # noqa: SIM118
        if not image_keys:
            return obs

        for key in image_keys:
            camera = IMAGE_MODALITIES.index(Modality(key))
            obs[key] = self._apply_image_augmentation(obs[key], camera)
        return obs

    def reset(self) -> ResetReturn:
        res = self._env.reset()
        # TODO(issue#121) Previously it was called in the reset_idx()
        self.resample_augmentation()
        return res

    def resample_augmentation(self) -> None:
        """Draws new per episode augmentation profiles of all the envs."""
        self._resample_aug_profile(th.arange(self.num_envs, device=self.device))

    def apply_domain_randomization(self, envs_idx: th.Tensor) -> None:
        """The scene's randomizations, then new augmentation profiles of the `envs_idx`."""
        self._env.apply_domain_randomization(envs_idx)
        self._resample_aug_profile(envs_idx)

    def step(self, actions: th.Tensor) -> StepReturn:
        res = self._env.step(actions)
        # the env resets its ended episodes itself (its own `apply_domain_randomization()`, not this one)
        self._episode_steps += 1
        ended = res[2].bool().nonzero().flatten()
        if len(ended):
            self._resample_aug_profile(ended)
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
            **self._sample_cutouts(N),
            **self._sample_camera_dropout(N),
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

        for key, value in (
            self._sample_cutouts(n) | self._sample_camera_dropout(n)
        ).items():
            self._aug_profile[key][envs_idx] = value
        self._episode_steps[envs_idx] = 0

    def _sample_cutouts(self, n: int) -> dict[str, th.Tensor]:
        """
        The cut-out boxes of `n` envs: [n, P, K] (P cameras with `per_camera`, else 1; K = `count`)
        activity, corners and sizes [px], [n, P, K, 3] fill colours.
        """
        cfg = self._cfg.cutout
        dev = self.device
        H, W = self.image_resolution
        shape = (n, len(IMAGE_MODALITIES) if cfg.per_camera else 1, cfg.count)
        num_boxes = th.randint(1, cfg.count + 1, shape[:2] + (1,), device=dev)
        active = (th.rand(shape[:2] + (1,), device=dev) < cfg.prob) & (
            th.arange(cfg.count, device=dev) < num_boxes
        )
        hs = th.randint(cfg.min_size, cfg.max_size + 1, shape, device=dev).clamp(max=H)
        ws = th.randint(cfg.min_size, cfg.max_size + 1, shape, device=dev).clamp(max=W)
        return {
            "cutout_active": active,
            "cutout_h": hs,
            "cutout_w": ws,
            "cutout_y": (th.rand(shape, device=dev) * (H - hs + 1)).long(),
            "cutout_x": (th.rand(shape, device=dev) * (W - ws + 1)).long(),
            "cutout_color": th.rand(shape + (3,), device=dev),
        }

    def _sample_camera_dropout(self, n: int) -> dict[str, th.Tensor]:
        """The camera drops of `n` envs: [n, P] (P cameras) activity, start and length [steps], [n, P, 3] fill colours."""
        cfg = self._cfg.camera_dropout
        dev = self.device
        shape = (n, len(IMAGE_MODALITIES))
        may_drop = th.tensor(
            [not cfg.cameras or m.value in cfg.cameras for m in IMAGE_MODALITIES],
            device=dev,
        )
        return {
            "dropout_active": (th.rand(shape, device=dev) < cfg.prob) & may_drop,
            "dropout_start": th.randint(0, cfg.max_start_steps + 1, shape, device=dev),
            "dropout_steps": th.randint(
                cfg.min_steps, cfg.max_steps + 1, shape, device=dev
            ),
            "dropout_color": th.rand(shape + (3,), device=dev),
        }

    def _apply_camera_dropout(self, im: th.Tensor, camera: int) -> th.Tensor:
        prof = self._aug_profile
        start, steps = (
            prof["dropout_start"][:, camera],
            prof["dropout_steps"][:, camera],
        )
        t = self._episode_steps
        dropped = (
            prof["dropout_active"][:, camera]
            & (t >= start)
            & ((steps <= 0) | (t < start + steps))
        )
        fill = self._cfg.camera_dropout.fill
        if fill == "freeze":
            last = self._last_frames.get(camera)
            if last is None or last.shape != im.shape:
                last = self._last_frames[camera] = im.clone()
            last.copy_(th.where(dropped.view(-1, 1, 1, 1), last, im))
            return last.clone()
        if not dropped.any():
            return im
        if fill == "black":
            replacement = th.zeros_like(im)
        elif fill == "color":
            replacement = prof["dropout_color"][
                :, camera, : im.shape[1], None, None
            ].expand_as(im)
        elif fill == "noise":
            replacement = th.rand_like(im)
        else:
            raise ValueError(
                f"Unknown camera dropout fill '{fill}', available: black, color, noise, freeze"
            )
        return th.where(dropped.view(-1, 1, 1, 1), replacement, im)

    def _apply_image_augmentation(self, im: th.Tensor, camera: int = 0) -> th.Tensor:
        aug = self._cfg
        if not aug.enabled:
            return im

        N, C, _H, _W = im.shape
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
        if aug.cutout.prob > 0.0:
            im = self._apply_cutout(im, camera)

        if prof and aug.camera_dropout.prob > 0.0:
            im = self._apply_camera_dropout(im, camera)

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

    def _apply_cutout(self, im: th.Tensor, camera: int) -> th.Tensor:
        N, C, H, W = im.shape
        boxes = self._aug_profile if self._aug_profile else self._sample_cutouts(N)
        p = camera if self._cfg.cutout.per_camera else 0
        active = boxes["cutout_active"][:, p]  # [N, K]
        if not active.any():
            return im
        ys, xs = boxes["cutout_y"][:, p], boxes["cutout_x"][:, p]
        hs, ws = boxes["cutout_h"][:, p], boxes["cutout_w"][:, p]
        rows = th.arange(H, device=im.device).view(1, 1, H, 1)
        cols = th.arange(W, device=im.device).view(1, 1, 1, W)
        in_box = (
            (rows >= ys[..., None, None])
            & (rows < (ys + hs)[..., None, None])
            & (cols >= xs[..., None, None])
            & (cols < (xs + ws)[..., None, None])
            & active[..., None, None]
        )  # [N, K, H, W]
        fill = self._cfg.cutout.fill
        if fill == "black":
            return im * ~in_box.any(dim=1, keepdim=True)
        if fill == "noise":
            return th.where(in_box.any(dim=1, keepdim=True), th.rand_like(im), im)
        if fill == "color":
            colors = boxes["cutout_color"][:, p]  # [N, K, 3]
            for k in range(in_box.shape[1]):  # later boxes on top
                im = th.where(in_box[:, k : k + 1], colors[:, k, :C, None, None], im)
            return im
        raise ValueError(
            f"Unknown cut-out fill '{fill}', available: black, color, noise"
        )
