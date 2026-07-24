from typing import Optional, Iterator

from strenum import StrEnum

import torch as th
import torch.nn as nn

from aegis_gym.config.types import PolicyBCCfg, FusionCfg, VisionEncoderCfg

from .bc_encoders import (
    AutoencoderCNNEncoder,
    BaseVisionEncoder,
    ConcatenatedCNNEncoder,
    PerCameraCNNEncoder,
    SharedCNNEncoder,
)
from .bc_fusions import (
    BaseFusionModule,
    LinearFusion,
    SpatialAttentionFusion,
    VectorAttentionFusion,
)


class ResetLastLayerTarget(StrEnum):
    ACTION = "action"
    POSE = "pose"
    ALL = "all"

    @classmethod
    def from_value(cls, value: str) -> "ResetLastLayerTarget":
        if value == "both":
            return cls(cls.ALL)
        try:
            return cls(value)
        except ValueError as exc:
            raise ValueError(
                "reset_last_layer_weights.part must be 'action', 'pose', or 'all'"
            ) from exc

    def get_heads(self) -> tuple[str, ...]:
        if self is ResetLastLayerTarget.ACTION:
            return ("action_head",)
        if self is ResetLastLayerTarget.POSE:
            return ("pose_head",)
        return ("action_head", "pose_head")


class Policy(nn.Module):
    def __init__(
        self,
        cfg_policy: PolicyBCCfg,
        action_dim: int,
        num_cameras: int,
        image_height: int,
        image_width: int,
        device: th.device,
    ):
        super().__init__()

        self.device = device
        self._cfg = cfg_policy
        self.num_cameras = num_cameras
        self.encoder_type = self._cfg.encoder_type
        self.fusion_type = self._cfg.fusion_type
        self.use_pose_head = self._cfg.use_pose_head

        print(f"Encoder type: {self.encoder_type}")
        print(f"Fusion type: {self.fusion_type}")
        print(f"Use pose head: {self.use_pose_head}")

        self.vision_encoder = self._build_vision_encoder().to(device)
        self.feature_fusion = self._build_fusion(
            image_height=image_height, image_width=image_width
        )

        vision_obs_dim = self.feature_fusion.output_dim
        state_obs_dim = self._cfg.action_head_state_obs_dim

        action_input_dim = vision_obs_dim + state_obs_dim

        print(f"Vision obs dim: {vision_obs_dim}")
        print(f"State obs dim: {state_obs_dim}")

        print(f"Action head input dim: {action_input_dim}")
        self.action_head = self._build_mlp(
            input_dim=action_input_dim,
            hidden_dims=self._cfg.action_head_hidden_dims,
            output_dim=action_dim,
        )

        self.pose_head: nn.Sequential | None = None
        if self.use_pose_head:
            pose_input_dim = self.feature_fusion.pose_input_dim
            print(f"Pose head input dim: {pose_input_dim}")
            self.pose_head = self._build_mlp(
                input_dim=pose_input_dim,
                hidden_dims=self._cfg.pose_head_hidden_dims,
                output_dim=7,
            )
        self.to(device)

    def forward(
        self, rgb_obs: th.Tensor, state_obs: Optional[th.Tensor] = None
    ) -> th.Tensor:
        features = self.vision_encoder(rgb_obs)
        fused = self.feature_fusion(features)
        fused = th.cat([fused, state_obs], dim=-1)
        return self.action_head(fused)

    def predict_pose(self, rgb_obs: th.Tensor) -> tuple[th.Tensor, ...]:
        if not self.use_pose_head:
            return tuple()
        features = self.vision_encoder(rgb_obs)
        pose_feats = self.feature_fusion.prepare_pose_features(features)
        return tuple(self.pose_head(f) for f in pose_feats)

    def get_encoder(self):
        return self.vision_encoder

    @property
    def dtype(self) -> th.dtype:
        """Get the dtype of the policy's parameters."""
        return next(self.parameters()).dtype

    def _build_vision_encoder(self) -> BaseVisionEncoder:

        vision_cfg = self._cfg.vision_encoder
        if self.fusion_type == "attention_spatial":
            vision_cfg = self._cfg.vision_encoder_spatial

        def cnn_builder() -> nn.Sequential:
            return self._build_cnn(vision_cfg)

        # TODO(issue#73): Extract vision encoder construction into a separate function
        encoder_classes: dict[str, BaseVisionEncoder] = {
            "concatenated_cnn": ConcatenatedCNNEncoder,
            "shared_cnn": SharedCNNEncoder,
            "per_camera_cnn": PerCameraCNNEncoder,
            "autoencoder": AutoencoderCNNEncoder,
        }
        try:
            return encoder_classes[self.encoder_type](
                num_cameras=self.num_cameras,
                cnn_builder=cnn_builder,
                vision_cfg=vision_cfg,
            )
        except IndexError:
            raise ValueError(f"Unknown vision encoder type: {self.encoder_type}")

    def _build_fusion(self, image_height: int, image_width: int) -> BaseFusionModule:
        c, h, w = self.vision_encoder.infer_output_shape(
            image_height=image_height,
            image_width=image_width,
        )

        match self.fusion_type:
            case "linear":
                fusion_cfg: FusionCfg = self._cfg.linear_fusion

                # Dry-run the encoder to find how many tensors it actually returns
                dummy = th.zeros(
                    1,
                    self.num_cameras * 3,
                    image_height,
                    image_width,
                    device=self.device,
                )
                with th.no_grad():
                    num_feature_tensors = len(self.vision_encoder(dummy))

                return LinearFusion(
                    # vision_dim=fusion_cfg.get("fusion_output_dim", 512),
                    vision_dim=fusion_cfg.fusion_output_dim,
                    num_cameras=self.num_cameras,
                    in_channels=c,
                    image_height=h,
                    image_width=w,
                    pool_size=fusion_cfg.pool_size,
                    num_feature_tensors=num_feature_tensors,
                )
            case "attention_vector":
                fusion_cfg = self._cfg.attention_vector_fusion
                return VectorAttentionFusion(
                    vision_dim=fusion_cfg.fusion_output_dim,
                    num_cameras=self.num_cameras,
                    in_channels=c,
                    num_heads=fusion_cfg.num_heads,
                    pool_size=fusion_cfg.pool_size,
                )
            case "attention_spatial":
                fusion_cfg = self._cfg.attention_spatial_fusion
                return SpatialAttentionFusion(
                    vision_dim=fusion_cfg.fusion_output_dim,
                    num_cameras=self.num_cameras,
                    in_channels=c,
                    image_height=h,
                    image_width=w,
                    num_heads=fusion_cfg.num_heads,
                )
            case _:
                raise ValueError(f"Unknown fusion_type: {self.fusion_type}")

    @staticmethod
    def _build_cnn(cfg: VisionEncoderCfg) -> nn.Sequential:
        layers = []
        for c in cfg.conv_layers:
            layers.append(
                nn.Conv2d(
                    c.in_channels,
                    c.out_channels,
                    kernel_size=c.kernel_size,
                    stride=c.stride,
                    padding=c.padding,
                )
            )
            layers.append(nn.BatchNorm2d(c.out_channels))
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

    def reset_last_layer_weights(self, part: str = "all") -> None:
        if isinstance(part, ResetLastLayerTarget):
            target = part
        else:
            target = ResetLastLayerTarget.from_value(part)

        if target in {
            ResetLastLayerTarget.ACTION,
            ResetLastLayerTarget.ALL,
        }:
            self._reset_last_layer_weights_sequential(self.action_head)

        if self.pose_head is not None and target in {
            ResetLastLayerTarget.POSE,
            ResetLastLayerTarget.ALL,
        }:
            self._reset_last_layer_weights_sequential(self.pose_head)

    @staticmethod
    def _reset_last_layer_weights_sequential(module: nn.Sequential) -> None:
        for layer in reversed(module):
            if isinstance(layer, nn.Linear):
                nn.init.orthogonal_(layer.weight)
                if layer.bias is not None:
                    nn.init.zeros_(layer.bias)
                return
        raise RuntimeError("No Linear layer found to reset in the given module.")


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
