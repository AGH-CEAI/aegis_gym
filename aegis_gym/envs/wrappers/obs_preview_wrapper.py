from datetime import datetime
from pathlib import Path
from typing import Optional

import cv2
import numpy as np

import torch as th
from tensordict import TensorDict

from config.types import DebugCfg, Modality, IMAGE_MODALITIES
from .base_wrapper import BaseEnvWrapper
from ..base_env import BaseEnv, StepReturn, ResetReturn


class ObsPreviewEnvWrapper(BaseEnvWrapper):
    """
    The debug wrapper to preview observations (e.g. visual) from the environment.
    """

    def __init__(self, env: BaseEnv, cfg_debug: DebugCfg):
        super().__init__(env=env)
        self._cfg = cfg_debug
        self._modalities = self._env.available_modalities
        self._record_dir: Optional[Path] = None
        self._frame_count: int = 0

        if not self._cfg.enabled:
            raise ValueError(
                "Activate the debugging mode to use the ObsPreviewEnvWrapper (`--debug-enable`)."
            )

    def step(self, actions: th.Tensor) -> StepReturn:
        res = self._env.step(actions)
        mm_obs = self._env.get_modality_observations(self._modalities)
        self._preview(mm_obs)
        return res

    def reset(self) -> ResetReturn:
        res = self._env.reset()
        mm_obs = self._env.get_modality_observations(self._modalities)
        self._preview(mm_obs)
        return res

    def _preview(self, multimodal_obs: TensorDict) -> None:
        if not (self._cfg.enable_vis_preview or self._cfg.enable_record_obs):
            return
        self._show_visual_obs(multimodal_obs)

    def _show_visual_obs(self, mm_obs: TensorDict) -> None:
        present_keys = frozenset(mm_obs.keys())
        ordered_image_modalities = [m for m in IMAGE_MODALITIES if m in present_keys]
        if not ordered_image_modalities:
            return

        images = mm_obs.select(*(m for m in ordered_image_modalities))
        preview = self._format_visual_obs(
            mm_obs=images, ordered_modalities=ordered_image_modalities, normalize=True
        )

        if self._cfg.enable_vis_preview:
            cv2.imshow("[DEBUG] Visual observation preview", preview)
            cv2.waitKey(1)

        if self._cfg.enable_record_obs:
            self._write_frame(preview)

    def _write_frame(self, frame: np.ndarray) -> None:
        if self._record_dir is None:
            self._record_dir = self._create_record_dir()
        out_path = self._record_dir / f"frame_{self._frame_count:06d}.png"
        cv2.imwrite(str(out_path), frame)
        self._frame_count += 1

    def _create_record_dir(self) -> Path:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        record_dir = self._cfg.record_dir / f"obs_preview_{timestamp}"
        record_dir.mkdir(parents=True, exist_ok=True)
        print(
            f"[ObsPreviewEnvWrapper] Recording observation preview frames to {record_dir}"
        )
        return record_dir

    def _format_visual_obs(
        self,
        mm_obs: TensorDict,
        ordered_modalities: list[Modality],
        normalize: bool,
    ) -> np.ndarray:
        max_side = self._cfg.vis_preview_side
        num_envs = min(mm_obs.batch_size[0], self._cfg.vis_preview_max_envs)
        GRID_LINE_COLOR = (80, 80, 80)  # BGR, subtle gray

        rows: list[np.ndarray] = []
        for env_idx in range(num_envs):
            row_images: list[np.ndarray] = []
            for col_idx, modality in enumerate(ordered_modalities):
                img = (
                    mm_obs[modality.value][env_idx].permute(1, 2, 0).cpu().numpy()
                )  # CHW -> HWC
                img = (
                    (img * 255).astype(np.uint8) if normalize else img.astype(np.uint8)
                )
                img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

                height, width = img.shape[:2]
                scale = min(max_side / width, max_side / height)
                img = cv2.resize(
                    img,
                    (int(width * scale), int(height * scale)),
                    interpolation=cv2.INTER_AREA,
                )
                if env_idx == 0:
                    cv2.putText(
                        img,
                        modality.value,
                        org=(10, 30),
                        fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                        fontScale=0.5,
                        color=(0, 255, 0),
                        thickness=2,
                    )
                if col_idx == 0:
                    img_height = img.shape[0]
                    cv2.putText(
                        img,
                        f"env {env_idx}",
                        org=(10, img_height - 10),
                        fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                        fontScale=0.5,
                        color=(0, 255, 0),
                        thickness=2,
                    )

                # 1px separator on right and bottom edges (skip last col/row to avoid a double-thick outer border)
                right = 1 if col_idx < len(ordered_modalities) - 1 else 0
                bottom = 1 if env_idx < num_envs - 1 else 0
                img = cv2.copyMakeBorder(
                    img,
                    top=0,
                    bottom=bottom,
                    left=0,
                    right=right,
                    borderType=cv2.BORDER_CONSTANT,
                    value=GRID_LINE_COLOR,
                )
                row_images.append(img)
            rows.append(np.hstack(row_images))

        return np.vstack(rows)
