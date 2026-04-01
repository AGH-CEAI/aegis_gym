import os
import time
from collections import deque
from collections.abc import Iterator
from pathlib import Path
from typing import Callable, Optional

import numpy as np
import torch as th
import torch.nn as nn
import torch.nn.functional as F
import torchvision.utils as vutils
from clearml import Task
from rsl_rl.utils.logger import Logger


class BehaviorCloning:
    """Multi-task behavior cloning with action prediction and object pose estimation"""

    def __init__(
        self, env, cfg: dict, teacher: nn.Module, log_dir: Path, device: str = "cpu"
    ):
        self._env = env
        self._cfg = cfg
        self._device = device
        self._teacher = teacher
        self._num_steps_per_env = cfg["num_steps_per_env"]

        # ClearML allows only one active Task per process.
        # Since RSL-RL creates its own ClearML Task during RL training,
        # we explicitly close any existing Task here to allow Behavior Cloning
        # to create and log to a new, separate ClearML Task.
        if Task.current_task():
            Task.current_task().close()

        self.logger = None
        if log_dir is not None:
            self.logger = Logger(
                log_dir=str(log_dir),
                cfg=cfg,
                env_cfg=getattr(env, "cfg", {}),
                num_envs=env.num_envs,
                is_distributed=False,
                gpu_world_size=1,
                gpu_global_rank=0,
                device=device,
            )

        if env.camera_setup == "default":
            num_cameras = 3
        elif env.camera_setup == "scene_dual":
            num_cameras = 2
        else:
            raise ValueError(f"Unknown camera_setup: {env.camera_setup}")

        cfg["policy"]["num_cameras"] = num_cameras
        rgb_shape = (num_cameras * 3, env.image_height, env.image_width)
        action_dim = env.num_actions

        # Multi-task policy with action and pose heads
        self._policy = Policy(cfg["policy"], action_dim).to(device)

        # Initialize optimizer
        self._optimizer = th.optim.Adam(
            self._policy.parameters(), lr=cfg["learning_rate"]
        )

        # Experience buffer with pose data
        self._buffer = ExperienceBuffer(
            num_envs=env.num_envs,
            max_size=self._cfg["buffer_size"],
            img_shape=rgb_shape,
            state_dim=self._cfg["policy"]["action_head"]["state_obs_dim"],
            action_dim=action_dim,
            device=device,
            dtype=self._policy.dtype,
        )

        # Training state
        self._current_iter = 0
        self._rewbuffer = deque(maxlen=100)
        self._cur_reward_sum = th.zeros(
            self._env.num_envs, dtype=th.float, device=self._device
        )

    def learn(self, num_learning_iterations: int) -> None:
        self._buffer.clear()
        encoder_type = self._cfg["policy"]["type"]
        use_autoencoder = encoder_type == "autoencoder"
        save_recons = self._cfg.get("save_recons", False)
        save_recon_freq = self._cfg.get("save_recon_freq", 100)

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
                self._cfg.get("num_mini_batches", 4), self._cfg["num_epochs"]
            )

            for batch in generator:
                # Forward pass for both action and pose prediction
                pred_action = self._policy(batch["rgb_obs"], batch["robot_pose"])
                pred_poses = self._policy.predict_pose(batch["rgb_obs"])

                # Compute action prediction loss
                action_loss = F.mse_loss(pred_action, batch["actions"])

                # Compute pose estimation loss (position + orientation)
                pose_loss = th.tensor(0.0, device=self._device)
                for pred_pose in pred_poses:
                    pose_loss += self._compute_pose_loss(
                        pred_pose, batch["object_poses"]
                    )

                recon_loss = th.tensor(0.0, device=self._device)
                if use_autoencoder:
                    enc = self._policy.get_encoder()
                    recons = enc.reconstruct(batch["rgb_obs"])
                    for c in range(self._policy.num_cameras):
                        target = batch["rgb_obs"][:, c * 3 : (c + 1) * 3]
                        recon_loss += F.mse_loss(recons[c], target)
                    recon_loss /= self._policy.num_cameras

                    last_recons = recons
                    last_batch = batch

                # Combined loss with weights
                total_loss = action_loss + pose_loss + recon_loss

                # Backward pass
                self._optimizer.zero_grad()
                total_loss.backward()
                th.nn.utils.clip_grad_norm_(
                    self._policy.parameters(), self._cfg["max_grad_norm"]
                )
                self._optimizer.step()

                total_action_loss += action_loss
                total_pose_loss += pose_loss
                total_recon_loss += recon_loss
                num_batches += 1

            if (
                use_autoencoder
                and save_recons
                and (it + 1) % save_recon_freq == 0
                and last_recons is not None
            ):
                self._save_reconstructions(last_batch, last_recons, it)

            end_time = time.time()
            backward_time = end_time - start_time

            # Compute average losses
            if num_batches == 0:
                raise ValueError("No batches collected")
            else:
                avg_action_loss = total_action_loss / num_batches
                avg_pose_loss = total_pose_loss / num_batches
                avg_recon_loss = total_recon_loss / num_batches

            fps = (self._num_steps_per_env * self._env.num_envs) / (forward_time)
            # Logging
            if self.logger is not None and (it + 1) % self._cfg["log_freq"] == 0:
                current_lr = self._optimizer.param_groups[0]["lr"]

                self._log_metrics(
                    it=it,
                    avg_action_loss=avg_action_loss,
                    avg_pose_loss=avg_pose_loss,
                    avg_recon_loss=avg_recon_loss,
                    current_lr=current_lr,
                    fps=fps,
                    forward_time=forward_time,
                    backward_time=backward_time,
                )

            # Save checkpoints periodically
            if self.logger is not None and (it + 1) % self._cfg["save_freq"] == 0:
                ckpt_path = os.path.join(
                    self.logger.log_dir, f"checkpoint_{it + 1:04d}.pt"
                )
                self.save(ckpt_path)
                if self.logger is not None:
                    self.logger.save_model(ckpt_path, it + 1)

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
        obs = self._env.get_observations()
        with th.inference_mode():
            for _ in range(self._num_steps_per_env):
                rgb_obs = self._env.get_observations_vis(normalize=True)

                # Get teacher action
                teacher_action = self._teacher(obs).detach()

                # Get end-effector position
                ee_pose = self._env.robot.ee_pose

                # Get object pose in camera frame
                # object_pose_camera = self._get_object_pose_in_camera_frame()
                object_pose = th.cat(
                    [
                        self._env.object.get_pos(),
                        self._env.object.get_quat(),
                    ],
                    dim=-1,
                )

                # Store in buffer
                self._buffer.add(rgb_obs, ee_pose, object_pose, teacher_action)

                # Step environment with student action
                student_action = self._policy(rgb_obs.float(), ee_pose.float())

                # Simple Dagger: use student action if its difference with teacher action is less than 0.5
                action_diff = th.norm(student_action - teacher_action, dim=-1)
                condition = (action_diff < 1.0).unsqueeze(-1).expand_as(student_action)
                action = th.where(condition, student_action, teacher_action)

                next_obs, reward, done, _ = self._env.step(action)
                self._cur_reward_sum += reward

                obs = next_obs
                new_ids = (done > 0).nonzero(as_tuple=False)
                self._rewbuffer.extend(
                    self._cur_reward_sum[new_ids][:, 0].cpu().numpy().tolist()
                )
                self._cur_reward_sum[new_ids] = 0

    def _save_reconstructions(self, batch, recons, it):
        Path("reconstructions").mkdir(exist_ok=True)

        b = 0
        for c in range(self._policy.num_cameras):
            orig = batch["rgb_obs"][b, c * 3 : (c + 1) * 3].detach().cpu()
            recon = recons[c][b].detach().cpu()

            vutils.save_image(orig, f"reconstructions/orig_iter{it + 1:04d}_c{c}.png")
            vutils.save_image(recon, f"reconstructions/recon_iter{it + 1:04d}_c{c}.png")

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

    def save(self, path: str) -> None:
        """Save model checkpoint."""
        checkpoint = {
            "model_state_dict": self._policy.state_dict(),
            "optimizer_state_dict": self._optimizer.state_dict(),
            "current_iter": self._current_iter,
            "config": self._cfg,
        }
        th.save(checkpoint, path)
        print(f"Model saved to {path}")

    def load(self, path: str) -> None:
        """Load model checkpoint."""
        checkpoint = th.load(path, map_location=self._device, weights_only=False)
        self._policy.load_state_dict(checkpoint["model_state_dict"])
        self._optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        self.current_iter = checkpoint["current_iter"]
        print(f"Model loaded from {path}")

    def load_finetuned_model(self, path: str) -> None:
        """Load a fine-tuned model checkpoint."""
        checkpoint = th.load(path, map_location=self._device, weights_only=False)
        self._policy.load_state_dict(checkpoint["model_state_dict"])
        self._optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        self._current_iter = checkpoint["current_iter"]
        print(f"Fine-tuned model loaded from {path}")


class ExperienceBuffer:
    """A first-in-first-out buffer for experience replay."""

    def __init__(
        self,
        num_envs: int,
        max_size: int,
        img_shape: tuple[int, int, int],
        state_dim: int,
        action_dim: int,
        device: str = "cpu",
        dtype: Optional[th.dtype] = None,
    ):
        self._num_envs = num_envs
        self._max_size = max_size
        self._img_shape = img_shape
        self._state_dim = state_dim
        self._action_dim = action_dim
        self._device = device
        self._ptr = 0
        self._size = 0

        # Buffers for data
        self._rgb_obs = th.empty(
            max_size, num_envs, *img_shape, dtype=dtype, device=device
        )
        self._robot_pose = th.empty(
            max_size, num_envs, state_dim, dtype=dtype, device=device
        )
        self._object_poses = th.empty(max_size, num_envs, 7, dtype=dtype, device=device)
        self._actions = th.empty(
            max_size, num_envs, action_dim, dtype=dtype, device=device
        )

    def add(
        self,
        rgb_obs: th.Tensor,
        robot_pose: th.Tensor,
        object_poses: th.Tensor,
        actions: th.Tensor,
    ) -> None:
        """Add experience to buffer."""
        self._ptr = (self._ptr + 1) % self._max_size
        self._rgb_obs[self._ptr] = rgb_obs
        self._robot_pose[self._ptr] = robot_pose
        self._object_poses[self._ptr] = object_poses
        self._actions[self._ptr] = actions
        self._size = min(self._size + 1, self._max_size)

    def get_batches(
        self, num_mini_batches: int, num_epochs: int
    ) -> Iterator[dict[str, th.Tensor]]:
        """Generate batches for training."""
        # calculate the size of each mini-batch
        batch_size = self._size // num_mini_batches
        for _ in range(num_epochs):
            indices = th.randperm(self._size)
            for batch_idx in range(0, self._size, batch_size):
                batch_indices = indices[batch_idx : batch_idx + batch_size]

                # Yield a mini-batch of data
                yield {
                    "rgb_obs": self._rgb_obs[batch_indices].reshape(
                        -1, *self._img_shape
                    ),
                    "robot_pose": self._robot_pose[batch_indices].reshape(
                        -1, self._state_dim
                    ),
                    "object_poses": self._object_poses[batch_indices].reshape(-1, 7),
                    "actions": self._actions[batch_indices].reshape(
                        -1, self._action_dim
                    ),
                }

    def clear(self) -> None:
        """Clear the buffer."""
        self._rgb_obs.zero_()
        self._robot_pose.zero_()
        self._object_poses.zero_()
        self._actions.zero_()
        self._ptr = 0
        self._size = 0

    def is_full(self) -> bool:
        """Check if buffer is full."""
        return self._size == self._max_size

    @property
    def size(self) -> int:
        """Get buffer size."""
        return self._size


class Policy(nn.Module):
    def __init__(self, config: dict, action_dim: int):
        super().__init__()
        self.num_cameras = config["num_cameras"]

        self.vision_encoder = self._build_vision_encoder(config)

        vision_dim = self.vision_encoder.output_dim

        self.feature_fusion = nn.Sequential(
            nn.Linear(vision_dim * self.num_cameras, vision_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
        )

        action_cfg = config["action_head"]
        self.state_obs_dim = action_cfg.get("state_obs_dim")
        mlp_input_dim = vision_dim
        if self.state_obs_dim is not None:
            mlp_input_dim += self.state_obs_dim

        self.action_head = self._build_mlp(
            input_dim=mlp_input_dim,
            hidden_dims=action_cfg["hidden_dims"],
            output_dim=action_dim,
        )

        self.pose_head = self._build_mlp(
            input_dim=vision_dim,
            hidden_dims=config["pose_head"]["hidden_dims"],
            output_dim=7,
        )

    def forward(
        self, rgb_obs: th.Tensor, state_obs: Optional[th.Tensor] = None
    ) -> th.Tensor:
        features = self.vision_encoder(rgb_obs)
        fused = self.feature_fusion(th.cat(features, dim=-1))
        if state_obs is not None and self.state_obs_dim is not None:
            fused = th.cat([fused, state_obs], dim=-1)
        return self.action_head(fused)

    def predict_pose(self, rgb_obs: th.Tensor) -> tuple[th.Tensor, ...]:
        features = self.vision_encoder(rgb_obs)
        return tuple(self.pose_head(f) for f in features)

    def get_encoder(self):
        return self.vision_encoder

    @property
    def dtype(self) -> th.dtype:
        """Get the dtype of the policy's parameters."""
        return next(self.parameters()).dtype

    def _build_vision_encoder(self, config: dict) -> "VisionEncoder":
        encoder_type = config["type"]
        vision_cfg = config["vision_encoder"]

        def cnn_builder() -> nn.Sequential:
            return self._build_cnn(vision_cfg, encoder_type)

        # TODO(issue#73): Extract vision encoder construction into a separate function
        match encoder_type:
            case "shared_cnn":
                return SharedCNNEncoder(
                    self.num_cameras,
                    cnn_builder=cnn_builder,
                    vision_cfg=vision_cfg,
                )
            case "per_camera_cnn":
                return PerCameraCNNEncoder(
                    self.num_cameras,
                    cnn_builder=cnn_builder,
                    vision_cfg=vision_cfg,
                )
            case "autoencoder":
                return AutoencoderCNNEncoder(
                    self.num_cameras,
                    cnn_builder=cnn_builder,
                    vision_cfg=vision_cfg,
                )
            case _:
                raise ValueError(f"Unknown vision encoder type: {encoder_type}")

    @staticmethod
    def _build_cnn(config: dict, encoder_type: str) -> nn.Sequential:
        layers = []
        for c in config["conv_layers"]:
            layers.append(
                nn.Conv2d(
                    c["in_channels"],
                    c["out_channels"],
                    kernel_size=c["kernel_size"],
                    stride=c["stride"],
                    padding=c["padding"],
                )
            )
            layers.append(nn.BatchNorm2d(c["out_channels"]))
            layers.append(nn.ReLU())

        if encoder_type != "autoencoder":
            if config.get("pooling") == "adaptive_avg":
                layers.append(
                    nn.AdaptiveAvgPool2d((config["pool_size"], config["pool_size"]))
                )
            # TODO(issue#79): Remove hardcoded dimensions related to autoencoder and make them configurable
            elif config.get("pooling") == "linear":
                layers.append(nn.Flatten())
                layers.append(nn.Linear(32 * 16 * 16, 32 * 4 * 4))
                layers.append(nn.ReLU())

        return nn.Sequential(*layers)

    @staticmethod
    def _build_mlp(
        input_dim: int, hidden_dims: list[int], output_dim: int
    ) -> nn.Sequential:
        # TODO(issue#71): Investigate indexing and micro-optimizations in vision encoder forward pass
        layers = []
        for h in hidden_dims:
            layers.append(nn.Linear(input_dim, h))
            layers.append(nn.ReLU())
            input_dim = h
        layers.append(nn.Linear(input_dim, output_dim))
        return nn.Sequential(*layers)


class VisionEncoder(nn.Module):
    def __init__(self, num_cameras: int):
        super().__init__()
        self.num_cameras = num_cameras
        self.output_dim = None

    def forward(self, rgb_obs: th.Tensor) -> tuple[th.Tensor, ...]:
        raise NotImplementedError


class SharedCNNEncoder(VisionEncoder):
    def __init__(self, num_cameras: int, cnn_builder: Callable, vision_cfg: dict):
        super().__init__(num_cameras)
        self.encoder = cnn_builder()

        last_channels = vision_cfg["conv_layers"][-1]["out_channels"]
        pool_size = vision_cfg["pool_size"]

        self.output_dim = last_channels * pool_size * pool_size

    # TODO(issue#71): Investigate indexing and micro-optimizations in vision encoder forward pass
    def forward(self, rgb_obs: th.Tensor) -> tuple[th.Tensor, ...]:
        features = []
        for i in range(self.num_cameras):
            cam_rgb = rgb_obs[:, i * 3 : (i + 1) * 3]
            feat = self.encoder(cam_rgb).flatten(start_dim=1)
            features.append(feat)
        return tuple(features)


class PerCameraCNNEncoder(VisionEncoder):
    def __init__(self, num_cameras: int, cnn_builder: Callable, vision_cfg: dict):
        super().__init__(num_cameras)
        self.encoders = nn.ModuleList([cnn_builder() for _ in range(num_cameras)])

        last_channels = vision_cfg["conv_layers"][-1]["out_channels"]
        pool_size = vision_cfg["pool_size"]

        self.output_dim = last_channels * pool_size * pool_size

    # TODO(issue#71): Investigate indexing and micro-optimizations in vision encoder forward pass
    def forward(self, rgb_obs: th.Tensor) -> tuple[th.Tensor, ...]:
        features = []
        for i in range(self.num_cameras):
            cam_rgb = rgb_obs[:, i * 3 : (i + 1) * 3]
            feat = self.encoders[i](cam_rgb).flatten(start_dim=1)
            features.append(feat)
        return tuple(features)


# TODO(issue#79): Remove hardcoded dimensions related to autoencoder and make them configurable
class AutoencoderCNNEncoder(VisionEncoder):
    def __init__(self, num_cameras: int, cnn_builder: Callable, vision_cfg: dict):
        super().__init__(num_cameras)

        self.encoder = cnn_builder()
        self.decoder = self.build_decoder()

        self.feature_channels = vision_cfg["conv_layers"][-1]["out_channels"]
        self.feature_size = 16

        self.flatten_dim = self.feature_channels * self.feature_size * self.feature_size
        self.latent_dim = 512

        self.to_latent = nn.Linear(self.flatten_dim, self.latent_dim)
        self.from_latent = nn.Linear(self.latent_dim, self.flatten_dim)

        self.output_dim = self.latent_dim

    def build_decoder(self):
        return nn.Sequential(
            nn.ConvTranspose2d(
                32, 16, kernel_size=3, stride=2, padding=1, output_padding=1
            ),
            nn.ReLU(),
            nn.ConvTranspose2d(
                16, 8, kernel_size=3, stride=2, padding=1, output_padding=1
            ),
            nn.ReLU(),
            nn.Conv2d(8, 3, kernel_size=3, stride=1, padding=1),
            nn.Sigmoid(),
        )

    def forward(self, rgb_obs: th.Tensor) -> tuple[th.Tensor, ...]:
        features = [None] * self.num_cameras

        for i in range(self.num_cameras):
            cam_rgb = rgb_obs[:, i * 3 : (i + 1) * 3]

            fmap = self.encoder(cam_rgb)
            flat = fmap.flatten(start_dim=1)

            latent = self.to_latent(flat)

            features[i] = latent

        return tuple(features)

    def reconstruct(self, rgb_obs: th.Tensor) -> tuple[th.Tensor, ...]:
        recons = [None] * self.num_cameras

        for i in range(self.num_cameras):
            cam_rgb = rgb_obs[:, i * 3 : (i + 1) * 3]

            fmap = self.encoder(cam_rgb)
            flat = fmap.flatten(start_dim=1)

            latent = self.to_latent(flat)

            flat_recon = self.from_latent(latent)
            fmap_recon = flat_recon.view(
                -1, self.feature_channels, self.feature_size, self.feature_size
            )

            recons[i] = self.decoder(fmap_recon)

        return tuple(recons)
