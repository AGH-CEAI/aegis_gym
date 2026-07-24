import time
from collections import deque
from pathlib import Path
from typing import Any, Literal

import numpy as np
import torch as th
import torch.nn.functional as F
import torchvision.utils as vutils
from rsl_rl.utils.logger import Logger
from torch import nn

from aegis_gym.config import ConfigManager
from aegis_gym.config.types import (
    IMAGE_MODALITIES,
    BCCfg,
    CamerasSetup,
    ExpConfig,
)
from aegis_gym.envs import BaseEnv, BaseManipulator

from .base_runner import BasePolicyRunner
from .bc_policy import ExperienceBuffer, Policy, ResetLastLayerTarget


class BehaviorCloningRunner(BasePolicyRunner):
    """Multi-task behavior cloning with action prediction and object pose estimation"""

    def __init__(
        self,
        env: BaseEnv,
        cfg: ExpConfig,
        teacher: nn.Module | None,
    ):
        super().__init__(env=env, cfg=cfg)

        # HACK(issue#111) overwrite RLCfg with BCCfg
        self.cfg_train: BCCfg = cfg.bc_cfg
        self._log_dir = cfg.logger_cfg.local_log_dir
        self._teacher = teacher
        self._extract_config()

        # TODO(issue#111) simplify config
        rsl_rl_bc_cfg = cfg.bc_cfg.as_dict()
        rsl_rl_bc_cfg.update(cfg.logger_cfg.as_dict())
        self.logger = Logger(
            log_dir=str(self._log_dir),
            cfg=rsl_rl_bc_cfg,
            env_cfg=env.get_cfg_as_dict(),
            num_envs=env.num_envs,
            is_distributed=False,
            gpu_world_size=1,
            gpu_global_rank=0,
            device=str(self.device),
        )

        # Training state
        self._current_iter = 0
        self._rewbuffer = deque(maxlen=100)
        self._cur_reward_sum = th.zeros(
            self.env.num_envs, dtype=th.float, device=self.device
        )
        self._best_model_reward: float = float("-inf")
        self._best_model_iter: int = -1

        self._setup_policy()
        print("\n=== POLICY ===")
        print(self._policy)

    def _extract_config(self) -> None:
        self._num_steps_per_env = self.cfg_train.num_steps_per_env
        self._use_teacher_mixing = self.cfg_train.use_teacher_mixing

        encoder_type = self.cfg_train.policy.encoder_type
        self._enable_recon = encoder_type == "autoencoder"

        self._save_recons = self.cfg_train.save_recons
        self._save_recon_freq = self.cfg_train.save_recon_freq
        self._reset_last_layer_cfg = self._resolve_reset_last_layer_cfg(
            interval=self.cfg_train.reset_last_layer_weights_interval,
            part=self.cfg_train.reset_last_layer_weights_part,
        )

    def _setup_policy(self) -> None:
        match self.env.get_cameras_setup():
            case CamerasSetup.DEFAULT:
                num_cameras = 3
            case CamerasSetup.SCENE_DUAL:
                num_cameras = 2
            case _:
                raise ValueError(f"Wrong camera setup: {self.env.get_cameras_setup()}")

        rgb_shape = (num_cameras * 3, self.env.image_height, self.env.image_width)
        action_dim = self.env.num_actions

        # TODO(issue#111) simplify config
        cfg = ConfigManager.get_config()
        im_h, im_w = cfg.env_cfg.image_resolution

        # Multi-task policy with action and pose heads
        self._policy = Policy(
            cfg_policy=self.cfg_train.policy,
            action_dim=action_dim,
            num_cameras=num_cameras,
            image_height=im_h,
            image_width=im_w,
            device=self.device,
        )

        # Initialize optimizer
        self._optimizer = th.optim.Adam(
            self._policy.parameters(), lr=self.cfg_train.learning_rate
        )

        # Experience buffer with pose data
        self._buffer = ExperienceBuffer(
            num_envs=self.env.num_envs,
            max_size=self.cfg_train.buffer_size,
            img_shape=rgb_shape,
            state_dim=self.cfg_train.policy.action_head_state_obs_dim,
            action_dim=action_dim,
            device=str(self.device),
            dtype=self._policy.dtype,
        )

    def learn(
        self, num_learning_iterations: int, init_at_random_ep_len: bool = False
    ) -> None:
        self._buffer.clear()

        for it in range(num_learning_iterations):
            # Collect experience
            start_time = time.time()
            self._collect_with_rl_teacher()
            end_time = time.time()
            forward_time = end_time - start_time

            # Training steps for both action and pose prediction
            total_action_loss = 0.0
            total_pose_loss = 0.0
            total_recon_loss = 0.0
            num_batches = 0
            last_recons = None
            last_batch = None

            start_time = time.time()
            generator = self._buffer.get_batches(
                self.cfg_train.num_mini_batches, self.cfg_train.num_epochs
            )

            for batch in generator:
                # Forward pass for both action and pose prediction
                pred_action = self._policy(batch["rgb_obs"], batch["robot_pose"])
                pred_poses = self._policy.predict_pose(batch["rgb_obs"])

                # Compute action prediction loss
                action_loss = F.mse_loss(pred_action, batch["actions"])

                # Compute pose estimation loss (position + orientation)
                pose_loss = th.tensor(0.0, device=self.device)
                for pred_pose in pred_poses:
                    pose_loss += self._compute_pose_loss(
                        pred_pose, batch["object_poses"]
                    )

                recon_loss, recons = self._compute_recon_loss(batch)
                if recons is not None:
                    last_recons = recons
                    last_batch = batch

                # Combined loss with weights
                total_loss = action_loss + pose_loss + recon_loss

                # Backward pass
                self._optimizer.zero_grad()
                total_loss.backward()
                th.nn.utils.clip_grad_norm_(
                    self._policy.parameters(), self.cfg_train.max_grad_norm
                )
                self._optimizer.step()

                total_action_loss += action_loss
                total_pose_loss += pose_loss
                total_recon_loss += recon_loss
                num_batches += 1

            self._save_reconstructions(batch=last_batch, recons=last_recons, it=it)

            end_time = time.time()
            backward_time = end_time - start_time

            # Compute average losses
            if num_batches == 0:
                raise ValueError("No batches collected")

            if self.logger is not None:
                self._log_metrics(
                    it=it,
                    avg_action_loss=total_action_loss / num_batches,
                    avg_pose_loss=total_pose_loss / num_batches,
                    avg_recon_loss=total_recon_loss / num_batches,
                    current_lr=self._optimizer.param_groups[0]["lr"],
                    fps=(self._num_steps_per_env * self.env.num_envs) / (forward_time),
                    forward_time=forward_time,
                    backward_time=backward_time,
                )

            # Save checkpoints periodically
            if self.logger is not None and (it + 1) % self.cfg_train.save_freq == 0:
                ckpt_path = Path(self._log_dir) / f"checkpoint_{it + 1:04d}.pt"
                self.save(ckpt_path)
                if self.logger is not None:
                    self.logger.save_model(path=str(ckpt_path), it=it + 1)

            # Save best model based on mean reward
            if self.logger is not None and len(self._rewbuffer) > 0:
                self._update_best_model(it, float(np.mean(self._rewbuffer)))

            # The last layer reset should be done AFTER saving a checkpoint
            self._maybe_reset_last_layer_weights(it)

        if self.logger is not None and self._best_model_iter >= 0:
            print(
                f"\nBest model:\n"
                f" Iteration = {self._best_model_iter}\n"
                f" Mean reward = {self._best_model_reward:.2f}\n"
            )

    def _update_best_model(self, it: int, mean_reward: float) -> None:
        skip = self.cfg_train.best_model_skip_iters
        if it < skip or mean_reward <= self._best_model_reward:
            return

        self._best_model_reward = mean_reward
        self._best_model_iter = it + 1

        path = Path(self._log_dir) / "checkpoint_best.pt"
        self.save(path)
        self.logger.save_model(
            path=str(path),
            it=self._best_model_iter,
            custom_name="model_best",
        )

        print(
            f"New best model!\n Iteration = {self._best_model_iter}\n Mean reward = {mean_reward:.2f}"
        )

    def _compute_pose_loss(
        self, pred_poses: th.Tensor, target_poses: th.Tensor
    ) -> th.Tensor:
        """Compute pose loss with separate position and orientation components."""
        # Split into position and orientation
        pred_pos = pred_poses[:, :3]
        pred_quat = pred_poses[:, 3:7]
        target_pos = target_poses[:, :3]
        target_quat = target_poses[:, 3:7]

        # Position loss (MSE)
        pos_loss = F.mse_loss(pred_pos, target_pos)

        # Orientation loss (quaternion distance)
        # Normalize quaternions
        pred_quat = F.normalize(pred_quat, p=2, dim=1)
        target_quat = F.normalize(target_quat, p=2, dim=1)

        # Quaternion distance: 1 - |dot(q1, q2)|
        # Note: we use this as a proxy for the actual distance between two quaternions
        # because the impact of the orientation loss (auxiliary task) is not significant
        # compared to the action loss (main task)
        quat_dot = th.sum(pred_quat * target_quat, dim=1)
        quat_loss = th.mean(1.0 - th.abs(quat_dot))

        return pos_loss + quat_loss

    def _collect_with_rl_teacher(self) -> None:
        """Collect experience from environment using stereo rgb images and object poses."""
        # Get state observation
        obs = self.env.get_observations()

        # TODO(issue#134) observation should be created only in the environment code!
        def get_obs_vis() -> th.Tensor:
            obs = self.env.get_modality_observations(modalities=IMAGE_MODALITIES)

            return th.cat([obs[m] for m in IMAGE_MODALITIES], dim=1).float()

        with th.inference_mode():
            for _ in range(self._num_steps_per_env):
                rgb_obs = get_obs_vis()

                # Get teacher action
                teacher_action = self._teacher(obs).detach()

                # Get end-effector position
                robot: BaseManipulator = self.env.manipulator
                ee_pose = robot.get_tcp_pose()

                # Get object pose in camera frame
                # object_pose_camera = self._get_object_pose_in_camera_frame()
                object_pose = self.env.object.get_pose()

                # Store in buffer
                self._buffer.add(rgb_obs, ee_pose, object_pose, teacher_action)

                # Step environment with student action
                student_action = self._policy(rgb_obs.float(), ee_pose.float())

                action = student_action
                if self._use_teacher_mixing:
                    # Simple Dagger: use student action if its difference with teacher action is less than 0.5
                    action_diff = th.norm(student_action - teacher_action, dim=-1)
                    condition = (
                        (action_diff < 1.0).unsqueeze(-1).expand_as(student_action)
                    )
                    action = th.where(condition, student_action, teacher_action)

                next_obs, reward, done, _ = self.env.step(action)
                self._cur_reward_sum += reward

                obs = next_obs
                new_ids = (done > 0).nonzero(as_tuple=False)
                self._rewbuffer.extend(
                    self._cur_reward_sum[new_ids][:, 0].cpu().numpy().tolist()
                )
                self._cur_reward_sum[new_ids] = 0

    def _compute_recon_loss(
        self, batch: dict[str, th.Tensor]
    ) -> tuple[th.Tensor, tuple[th.Tensor, ...] | None]:
        if not self._enable_recon:
            return th.tensor(0.0, device=self.device), None

        enc = self._policy.get_encoder()
        recons = enc.reconstruct(batch["rgb_obs"])

        recon_loss = th.tensor(0.0, device=self.device)

        for c in range(self._policy.num_cameras):
            target = batch["rgb_obs"][:, c * 3 : (c + 1) * 3]
            recon_loss += F.mse_loss(recons[c], target)

        recon_loss /= self._policy.num_cameras

        return recon_loss, recons

    def _save_reconstructions(
        self,
        batch: dict[str, th.Tensor],
        recons: tuple[th.Tensor, ...],
        it: int,
    ) -> None:
        recon_enabled = (
            self._enable_recon
            and self._save_recons
            and (it + 1) % self._save_recon_freq == 0
            and recons is not None
            and batch is not None
        )

        if not recon_enabled:
            return

        Path("reconstructions").mkdir(exist_ok=True)

        batch_idx = 0
        for c in range(self._policy.num_cameras):
            orig = batch["rgb_obs"][batch_idx, c * 3 : (c + 1) * 3].detach().cpu()
            recon = recons[c][batch_idx].detach().cpu()

            # TODO(issue#81): Save reconstructed images in ClearML
            vutils.save_image(orig, f"reconstructions/orig_iter{it + 1:04d}_c{c}.png")
            vutils.save_image(recon, f"reconstructions/recon_iter{it + 1:04d}_c{c}.png")

    def _resolve_reset_last_layer_cfg(
        self, interval: int, part: Literal["actor", "critic", "both"]
    ) -> dict:
        if interval == 0:
            return {"enabled": False, "part": "both", "interval": None}

        target = ResetLastLayerTarget.from_value(part)

        if interval is None or interval <= 0:
            return {"enabled": False, "target": target, "interval": None}

        return {"enabled": True, "target": target, "interval": interval}

    def _maybe_reset_last_layer_weights(self, it: int) -> None:
        if not self._reset_last_layer_cfg["enabled"]:
            return

        interval = self._reset_last_layer_cfg["interval"]
        target = self._reset_last_layer_cfg["target"]
        if interval is None:
            return

        if (it + 1) % interval == 0:
            if not hasattr(self._policy, "reset_last_layer_weights"):
                raise AttributeError("Policy does not support reset_last_layer_weights")
            self._policy.reset_last_layer_weights(target.value)

    def _log_metrics(
        self,
        it: int,
        avg_action_loss: th.Tensor,
        avg_pose_loss: th.Tensor,
        avg_recon_loss: th.Tensor,
        current_lr: float,
        fps: float,
        forward_time: float,
        backward_time: float,
    ) -> None:
        total_loss = avg_action_loss + avg_pose_loss + avg_recon_loss

        self.logger.writer.add_scalar("Loss/action", avg_action_loss.item(), it)
        self.logger.writer.add_scalar("Loss/pose", avg_pose_loss.item(), it)
        self.logger.writer.add_scalar("Loss/recon", avg_recon_loss.item(), it)
        self.logger.writer.add_scalar("Loss/total", total_loss.item(), it)
        self.logger.writer.add_scalar("Train/learning_rate", current_lr, it)
        self.logger.writer.add_scalar("Train/buffer_size", self._buffer.size, it)
        self.logger.writer.add_scalar("Perf/fps", fps, it)
        self.logger.writer.add_scalar("Perf/forward_time", forward_time, it)
        self.logger.writer.add_scalar("Perf/backward_time", backward_time, it)

        mean_reward = None
        if len(self._rewbuffer) > 0:
            mean_reward = float(np.mean(self._rewbuffer))
            self.logger.writer.add_scalar("Reward/mean", mean_reward, it)

        print("--------------------------------")
        info_str = (
            f" | Iteration:     {it + 1:04d}\n"
            f" | Action Loss:   {avg_action_loss:.6f}\n"
            f" | Pose Loss:     {avg_pose_loss:.6f}\n"
            f" | Total Loss:    {total_loss:.6f}\n"
            f" | Learning Rate: {current_lr:.6f}\n"
            f" | Forward Time:  {forward_time:.2f}s\n"
            f" | Backward Time: {backward_time:.2f}s\n"
            f" | FPS:           {int(fps)}"
        )

        if mean_reward is not None:
            info_str += f"\n | Mean Reward:   {mean_reward:.4f}"

        print(info_str)

    def save(self, path: Path, infos: dict | None = None) -> None:
        """Save model checkpoint in given `path`."""
        checkpoint = {
            "model_state_dict": self._policy.state_dict(),
            "optimizer_state_dict": self._optimizer.state_dict(),
            "current_iter": self._current_iter,
            "config": self.cfg_train,
        }
        th.save(checkpoint, path)
        print(f"Model saved to {path}")

    def load(self, path: Path) -> None:
        """Load model checkpoint."""
        checkpoint = th.load(path, map_location=self.device, weights_only=False)
        self._policy.load_state_dict(checkpoint["model_state_dict"])
        self._optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        self.current_iter = checkpoint["current_iter"]
        print(f"Model loaded from {path}")

    def get_inference_policy(self, device: th.device) -> Any:
        return self._policy.to(device)

    def export_policy(self, path: Path, filename: str = "policy.pt") -> None:
        raise NotImplementedError()
