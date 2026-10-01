import math
from collections.abc import Callable
from pathlib import Path
from typing import ClassVar

import cv2
import numpy as np
import torch as th
from clearml import Task
from tensordict import TensorDict
from tqdm import tqdm

from aegis_gym.aux.logging import get_logger
from aegis_gym.config.types import IMAGE_MODALITIES, POLICY_PREVIEW_SEEDS, Modality
from aegis_gym.envs import BaseEnv
from aegis_gym.envs.wrappers import camera_obs_to_image

InferencePolicy = Callable[[TensorDict], th.Tensor]


class PolicyPreviewRecorder:
    """
    Records the policy acting from the fixed `POLICY_PREVIEW_SEEDS` initial states into a single
    grid video (one tile per seed) and reports it to the ClearML "Debug Samples" of the task.
    The episodes run on the first environments of the given env; all the environments are reset
    afterwards, so it can be used between the training iterations.

    With the visual observations the tiles show the cameras as the network gets them (not
    augmented), otherwise the scene's overview preview camera.
    """

    TITLE = "policy_preview"
    _CAMERA_SIDE = 160
    _GRID_COLS = 5
    _HOLD_LAST_FRAME_S = 1.0
    _OUTCOME_COLORS: ClassVar[dict[str, tuple[int, int, int]]] = {  # RGB
        "success": (40, 200, 40),
        "time_outs": (230, 190, 30),
    }
    _OUTCOME_COLOR_OTHER = (220, 50, 50)

    def __init__(self, env: BaseEnv, out_dir: Path):
        self._env = env.unwrapped
        self._out_dir = Path(out_dir)
        self._logger = get_logger("PolicyPreview")
        self._num_envs = min(env.num_envs, len(POLICY_PREVIEW_SEEDS))
        self._seeds = POLICY_PREVIEW_SEEDS[: self._num_envs]
        self._camera_modalities: list[Modality] = (
            [m for m in IMAGE_MODALITIES if m in self._env.available_modalities]
            if self._env.has_camera_observations()
            else []
        )
        self._series = "camera_inputs" if self._camera_modalities else "preview"

    def is_available(self) -> bool:
        return bool(self._camera_modalities) or self._env.is_preview_available()

    def record(self, policy: InferencePolicy, iteration: int) -> dict[str, float]:
        """Records the preview, reports it to ClearML and returns its metrics."""
        if not self.is_available():
            raise RuntimeError("The policy preview isn't available, see `--record`.")
        self._out_dir.mkdir(parents=True, exist_ok=True)
        video_path = self._out_dir / f"policy_preview_{iteration:06d}.webm"
        self._logger.info(
            f"Recording the policy preview of {self._num_envs} seeds into {video_path}"
        )

        with th.inference_mode():
            with self._env.nominal_domain():
                metrics = self._run_episodes(policy=policy, video_path=video_path)
            self._env.reset()
        getattr(self._env, "extras", {}).pop("episode", None)

        self._report(video_path=video_path, metrics=metrics, iteration=iteration)
        return metrics

    def _run_episodes(
        self, policy: InferencePolicy, video_path: Path
    ) -> dict[str, float]:
        env = self._env
        n = self._num_envs
        env.reset()
        obs = env.reset_seeded(
            envs_idx=th.arange(n, device=env.device), seeds=self._seeds
        )

        policy_dt = env.get_policy_dt()
        fps = max(1, round(1 / policy_dt))
        max_steps = int(env.max_episode_length) + 1

        active = np.ones(n, dtype=bool)
        outcomes: list[str | None] = [None] * n
        returns = np.zeros(n)
        lengths = np.zeros(n, dtype=int)
        frames = self._render_frames()

        progress = tqdm(
            total=max_steps, desc=f"Policy preview ({n} seeds)", unit="step"
        )
        with progress, _open_video_writer(video_path, fps=fps) as writer:
            writer.append_data(
                self._compose(frames, outcomes, returns, lengths, policy_dt)
            )
            for _ in range(max_steps):
                actions = policy(obs)
                obs, rewards, dones, extras = env.step(actions)

                rewards_np = rewards[:n].cpu().numpy()
                dones_np = dones[:n].cpu().numpy().astype(bool)
                returns[active] += rewards_np[active]
                lengths[active] += 1

                new_frames = self._render_frames()
                for i in np.flatnonzero(active & dones_np):
                    outcomes[i] = self._classify_outcome(extras, i)
                active &= ~dones_np
                frames[active] = new_frames[active]

                progress.update()
                progress.set_postfix(
                    done=f"{n - active.sum()}/{n}",
                    success=sum(o == "success" for o in outcomes),
                )

                writer.append_data(
                    self._compose(frames, outcomes, returns, lengths, policy_dt)
                )
                if not active.any():
                    break

            last = self._compose(frames, outcomes, returns, lengths, policy_dt)
            for _ in range(round(self._HOLD_LAST_FRAME_S * fps)):
                writer.append_data(last)

        return {
            "success_rate": float(np.mean([o == "success" for o in outcomes])),
            "mean_return": float(returns.mean()),
            "mean_episode_length_s": float(lengths.mean() * policy_dt),
        }

    def _render_frames(self) -> np.ndarray:
        """[n, H, W, 3] uint8 RGB frames of the preview envs."""
        if not self._camera_modalities:
            return self._env.render_preview().cpu().numpy()

        obs = self._env.get_modality_observations(self._camera_modalities)
        frames = []
        for i in range(self._num_envs):
            images = []
            for modality in self._camera_modalities:
                img = camera_obs_to_image(
                    obs[modality.value][i],
                    max_side=self._CAMERA_SIDE,
                    interpolation=cv2.INTER_NEAREST,
                )
                name = _camera_label(modality)
                (text_w, _), _ = cv2.getTextSize(
                    name, cv2.FONT_HERSHEY_SIMPLEX, 0.35, 1
                )
                _put_text(img, name, (img.shape[1] - text_w - 6, 14), scale=0.35)
                images.append(img)
            frames.append(np.hstack(images))
        return np.stack(frames)

    @staticmethod
    def _classify_outcome(extras: dict, env_idx: int) -> str:
        """Name of the flag in `extras` that ended the episode of the `env_idx`."""

        def is_set(key: str) -> bool:
            value = extras.get(key)
            return isinstance(value, th.Tensor) and bool(value[env_idx])

        for key in ("success", "time_outs"):
            if is_set(key):
                return key
        for key, value in extras.items():
            if isinstance(value, th.Tensor) and value.dtype == th.bool and is_set(key):
                return key
        return "terminated"

    def _compose(
        self,
        frames: np.ndarray,
        outcomes: list[str | None],
        returns: np.ndarray,
        lengths: np.ndarray,
        policy_dt: float,
    ) -> np.ndarray:
        """Arranges the [n, H, W, 3] frames into a labelled grid image."""
        n, height, width, _ = frames.shape
        cols = min(self._GRID_COLS, n)
        rows = math.ceil(n / cols)
        grid = np.zeros((rows * height, cols * width, 3), dtype=np.uint8)
        for i in range(n):
            tile = np.ascontiguousarray(frames[i])
            self._draw_labels(
                tile, self._seeds[i], outcomes[i], returns[i], lengths[i] * policy_dt
            )
            row, col = divmod(i, cols)
            grid[row * height : (row + 1) * height, col * width : (col + 1) * width] = (
                tile
            )
        return grid

    def _draw_labels(
        self,
        tile: np.ndarray,
        seed: int,
        outcome: str | None,
        ret: float,
        time_s: float,
    ) -> None:
        height, width, _ = tile.shape
        _put_text(tile, f"seed {seed}", (8, 20), scale=0.5)
        _put_text(tile, f"t={time_s:4.1f}s  R={ret:7.2f}", (8, height - 10), scale=0.45)
        if outcome is None:
            return
        color = self._OUTCOME_COLORS.get(outcome, self._OUTCOME_COLOR_OTHER)
        label = "TIMEOUT" if outcome == "time_outs" else outcome.upper()
        cv2.rectangle(tile, (0, 0), (width - 1, height - 1), color, 4)
        _put_text(tile, label, (8, 44), scale=0.6, color=color, thickness=2)

    def _report(
        self, video_path: Path, metrics: dict[str, float], iteration: int
    ) -> None:
        self._logger.info(
            f"Policy preview (iteration {iteration}): "
            + ", ".join(f"{k}={v:.3f}" for k, v in metrics.items())
        )
        task = Task.current_task()
        if task is None:
            self._logger.warning("No ClearML task, the preview is only saved locally.")
            return
        clearml_logger = task.get_logger()
        clearml_logger.report_media(
            title=self.TITLE,
            series=self._series,
            iteration=iteration,
            local_path=str(video_path),
        )
        for name, value in metrics.items():
            clearml_logger.report_scalar(
                title=self.TITLE, series=name, value=value, iteration=iteration
            )


def _camera_label(modality: Modality) -> str:
    """Short camera name, e.g. `tool_left`."""
    return modality.name.lower().removeprefix("camera_").removesuffix("_rgb")


def make_visual_bc_policy(env: BaseEnv, bc_policy: Callable) -> InferencePolicy:
    """Adapts the BC student, acting on (camera images, TCP pose), to the preview's policy call."""
    env = env.unwrapped

    def policy(_obs: TensorDict) -> th.Tensor:
        images = env.get_modality_observations(modalities=IMAGE_MODALITIES)
        rgb = th.cat([images[m] for m in IMAGE_MODALITIES], dim=1).float()
        return bc_policy(rgb, env.manipulator.get_tcp_pose().float())

    return policy


def _put_text(
    img: np.ndarray,
    text: str,
    org: tuple[int, int],
    scale: float,
    color: tuple[int, int, int] = (255, 255, 255),
    thickness: int = 1,
) -> None:
    """Text with a dark outline, readable on any background."""
    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(img, text, org, font, scale, (0, 0, 0), thickness + 2, cv2.LINE_AA)
    cv2.putText(img, text, org, font, scale, color, thickness, cv2.LINE_AA)


# constant quality mode with the realtime speed, a 30 s preview encodes in ~1 s
_VP9_PARAMS = [
    *("-b:v", "0"),
    *("-crf", "40"),
    *("-deadline", "realtime"),
    *("-cpu-used", "8"),
    *("-row-mt", "1"),
]


def _open_video_writer(path: Path, fps: int):
    """
    WebM (VP9) video writer: about half the size of H.264 (MP4), and it plays natively in every
    browser, also in the Linux builds without the proprietary codecs.
    """
    # imageio's ffmpeg comes with the simulation dependencies
    import imageio.v2 as imageio

    return imageio.get_writer(
        str(path),
        format="FFMPEG",
        fps=fps,
        codec="libvpx-vp9",
        pixelformat="yuv420p",
        macro_block_size=16,
        ffmpeg_params=_VP9_PARAMS,
        ffmpeg_log_level="error",
    )
