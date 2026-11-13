import math
from collections.abc import Callable, Iterator
from typing import Any

import torch as th
from strenum import StrEnum
from torch import nn

from aegis_gym.aux.geom import quat_to_z_euler, transform_by_quat
from aegis_gym.aux.logging import get_logger
from aegis_gym.config.types import FusionCfg, MemoryCfg, PolicyBCCfg, VisionEncoderCfg

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


def _reference_offset(env) -> th.Tensor:
    manipulator = env.manipulator
    return manipulator.get_tcp_reference()[:, :2] - manipulator.get_tcp_pose()[:, :2]


def _contact_force(env) -> th.Tensor:
    """
    The tared F/T force in the world XY, scaled to the A1 spring deflection [m] by the env's
    `force_obs_scale`: the env's own force observation (`observe_force`).
    """
    manipulator = env.manipulator
    force = transform_by_quat(
        manipulator.get_ft_wrench()[:, :3], manipulator.get_tcp_pose()[:, 3:]
    )
    return force[:, :2] * env.force_obs_scale


def _planar_pose(xy: th.Tensor, yaw: th.Tensor) -> th.Tensor:
    return th.cat([xy, th.sin(yaw).unsqueeze(-1), th.cos(yaw).unsqueeze(-1)], dim=-1)


def _tee_pose(env) -> th.Tensor:
    pose = env.object.get_pose()
    return _planar_pose(pose[:, :2], quat_to_z_euler(pose[:, 3:]))


def _goal_pose(env) -> th.Tensor:
    return _planar_pose(env.goal_pose[:, :2], env.goal_yaw)


# the latent pose decoder's x, y in dm: on the scale of the sin / cos of the yaw
LATENT_POSE_XY_SCALE = 10.0


def latent_pose_target(object_poses: th.Tensor) -> th.Tensor:
    """[N, 4] the latent pose decoder's target from the [N, 7] (x, y, z, quat) tee poses."""
    xy = object_poses[:, :2] * LATENT_POSE_XY_SCALE
    return _planar_pose(xy, quat_to_z_euler(object_poses[:, 3:7]))


def latent_pose_errors(
    pred: th.Tensor, object_poses: th.Tensor
) -> tuple[th.Tensor, th.Tensor]:
    """[N] the position [m] and yaw [rad] errors of the latent pose decoder's [N, 4] predictions."""
    target = latent_pose_target(object_poses)
    pos_err = th.linalg.vector_norm(pred[:, :2] - target[:, :2], dim=-1)
    yaw_err = th.atan2(pred[:, 2], pred[:, 3]) - th.atan2(target[:, 2], target[:, 3])
    yaw_err = th.atan2(th.sin(yaw_err), th.cos(yaw_err)).abs()
    return pos_err / LATENT_POSE_XY_SCALE, yaw_err


# The student's non-image inputs by name. `teacher_obs` is privileged (the true tee pose): a
# sim-only sanity check of the distillation, not deployable. `tee_pose` (privileged too) and
# `goal_pose` are absolute, in the world frame: x, y, sin / cos yaw (hypothesis 2 of the handoff).
# `contact_force` is deployable (the F/T sensor over the gRPC bridge).
STUDENT_STATE_OBS: dict[str, Callable] = {
    "tcp_pose": lambda env: env.manipulator.get_tcp_pose(),
    "reference_offset": _reference_offset,
    "contact_force": _contact_force,
    "teacher_obs": lambda env: env.get_observations()["policy"],
    "tee_pose": _tee_pose,
    "goal_pose": _goal_pose,
}


def build_student_state(env, names: list[str]) -> th.Tensor:
    """[num_envs, D] the `names` inputs of `STUDENT_STATE_OBS`, concatenated."""
    unknown = set(names) - STUDENT_STATE_OBS.keys()
    if unknown:
        raise ValueError(
            f"Unknown student state_obs {sorted(unknown)}; use {list(STUDENT_STATE_OBS)}"
        )
    # unwrapped: the wrappers' observation calls augment (and resample) the images
    env = env.unwrapped
    return th.cat([STUDENT_STATE_OBS[n](env).float() for n in names], dim=-1)


MemoryState = tuple[th.Tensor, ...]
"""The student's memory state: batch-first tensors (see `enc_mem_dec.memory.MemoryState`)."""


def build_memory(cfg: MemoryCfg, input_dim: int) -> nn.Module | None:
    """The `enc_mem_dec` memory of `cfg`, None for "none" (enc_mem_dec is needed otherwise only)."""
    if cfg.type == "none":
        if cfg.residual:
            raise ValueError("memory.residual needs a memory, the type is 'none'")
        return None
    try:
        from enc_mem_dec.memory import make_memory
    except ImportError as e:
        raise ImportError(
            f"memory.type '{cfg.type}' needs the enc_mem_dec package (the worker image "
            "geonosis:5000/aegis_gym_emd)"
        ) from e
    kwargs: dict[str, Any] = {}
    if cfg.type in ("gru", "lstm"):
        kwargs = {"hidden_dim": cfg.hidden_dim, "num_layers": cfg.num_layers}
    elif cfg.type == "frame_stack":
        kwargs = {"num_frames": cfg.num_frames}
        if cfg.frame_stride != 1:  # older enc_mem_dec images have no stride
            kwargs["stride"] = cfg.frame_stride
    elif cfg.type == "attention_window":
        kwargs = {
            "window": cfg.window,
            "model_dim": cfg.model_dim,
            "num_heads": cfg.num_heads,
        }
    elif cfg.type == "hopfield":
        kwargs = {
            "window": cfg.window,
            "model_dim": cfg.model_dim,
            "update_steps": cfg.update_steps,
        }
    # "identity": the memoryless student trained on the chunks (the control of the memory students)
    name = "none" if cfg.type == "identity" else cfg.type
    return make_memory(name, input_dim=input_dim, residual=cfg.residual, **kwargs)


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
        state_obs_dim: int,
        num_cameras: int,
        image_height: int,
        image_width: int,
        device: th.device,
    ):
        super().__init__()

        logger = get_logger("Policy")

        self.device = device
        self._cfg = cfg_policy
        self.num_cameras = num_cameras
        self.encoder_type = self._cfg.encoder_type
        self.fusion_type = self._cfg.fusion_type
        self.use_pose_head = self._cfg.use_pose_head

        logger.info(f"Encoder type: {self.encoder_type}")
        logger.info(f"Fusion type: {self.fusion_type}")
        logger.info(f"Use pose head: {self.use_pose_head}")

        self.vision_encoder = self._build_vision_encoder().to(device)
        self.feature_fusion = self._build_fusion(
            image_height=image_height, image_width=image_width
        )

        vision_obs_dim = self.feature_fusion.output_dim
        configured_dim = self._cfg.action_head_state_obs_dim
        if configured_dim is not None and configured_dim != state_obs_dim:
            raise ValueError(
                f"action_head_state_obs_dim {configured_dim} doesn't match state_obs "
                f"{self._cfg.state_obs} ({state_obs_dim}); set it to {state_obs_dim} or null"
            )

        features_dim = vision_obs_dim + state_obs_dim
        self.memory = build_memory(self._cfg.memory, input_dim=features_dim)
        action_input_dim = (
            features_dim if self.memory is None else self.memory.output_dim
        )

        logger.info(f"Vision obs dim: {vision_obs_dim}")
        logger.info(f"State obs dim: {state_obs_dim}")
        logger.info(
            f"Memory: {self._cfg.memory.type if self.memory is None else self.memory.name}"
        )

        logger.info(f"Action head input dim: {action_input_dim}")
        self.action_head = self._build_mlp(
            input_dim=action_input_dim,
            hidden_dims=self._cfg.action_head_hidden_dims,
            output_dim=action_dim,
        )

        self.latent_pose_head: nn.Sequential | None = None
        if self._cfg.latent_pose_loss_weight > 0:
            self.latent_pose_head = self._build_mlp(
                input_dim=action_input_dim,
                hidden_dims=self._cfg.latent_pose_hidden_dims,
                output_dim=4,
            )
            logger.info(
                f"Latent pose decoder, loss weight {self._cfg.latent_pose_loss_weight}"
            )

        self.pose_head: nn.Sequential | None = None
        if self.use_pose_head:
            pose_input_dim = self.feature_fusion.pose_input_dim
            logger.info(f"Pose head input dim: {pose_input_dim}")
            self.pose_head = self._build_mlp(
                input_dim=pose_input_dim,
                hidden_dims=self._cfg.pose_head_hidden_dims,
                output_dim=7,
            )
        self.to(device)

    @property
    def has_memory(self) -> bool:
        return self.memory is not None

    def forward(
        self, rgb_obs: th.Tensor, state_obs: th.Tensor | None = None
    ) -> th.Tensor:
        """The memoryless student's actions; with a memory see `step()` and `sequence()`."""
        return self.forward_with_latent(rgb_obs, state_obs)[0]

    def forward_with_latent(
        self, rgb_obs: th.Tensor, state_obs: th.Tensor
    ) -> tuple[th.Tensor, th.Tensor]:
        """The memoryless student's actions and the action head's input."""
        if self.has_memory:
            raise RuntimeError("The student has a memory: use step() or sequence().")
        latent = self._features(rgb_obs, state_obs)
        return self.action_head(latent), latent

    def predict_latent_pose(self, latent: th.Tensor) -> th.Tensor | None:
        """[..., 4] the tee's planar pose from the action head's input, None without the decoder."""
        if self.latent_pose_head is None:
            return None
        return self.latent_pose_head(latent)

    def initial_memory_state(self, batch_size: int) -> MemoryState:
        # the parameters' device: `self.device` is the one at the construction (before `.to()`)
        device = next(self.parameters()).device
        return self.memory.initial_state(batch_size, device)

    def step(
        self,
        rgb_obs: th.Tensor,
        state_obs: th.Tensor,
        memory_state: MemoryState,
        episode_starts: th.Tensor,
        return_latent: bool = False,
    ) -> tuple[th.Tensor, MemoryState] | tuple[th.Tensor, MemoryState, th.Tensor]:
        """
        One step of the student with a memory: the actions and the next memory state, reset first
        for the `episode_starts` [batch] (bool); with `return_latent`, also the action head's input.
        """
        memory_state = self.memory.reset(memory_state, episode_starts)
        features = self._features(rgb_obs, state_obs)
        y, memory_state = self.memory.step(features, memory_state)
        if return_latent:
            return self.action_head(y), memory_state, y
        return self.action_head(y), memory_state

    def sequence(
        self,
        rgb_obs: th.Tensor,
        state_obs: th.Tensor,
        episode_starts: th.Tensor,
        burn_in: int,
        return_latents: bool = False,
    ) -> th.Tensor | tuple[th.Tensor, th.Tensor]:
        """
        The actions [time - burn_in, batch, action_dim] on chunks [time, batch, ...] of episode
        steps, from the initial memory state: the first `burn_in` steps only warm the state up
        (without gradients), the state is reset at the `episode_starts` [time, batch]. With
        `return_latents`, also the action head's inputs [time - burn_in, batch, ...].
        """
        batch_size = rgb_obs.shape[1]
        state = self.initial_memory_state(batch_size)
        if burn_in > 0:
            with th.no_grad():
                features = self._chunk_features(rgb_obs[:burn_in], state_obs[:burn_in])
                _, state = self.memory.sequence(
                    features, state, episode_starts[:burn_in]
                )
        features = self._chunk_features(rgb_obs[burn_in:], state_obs[burn_in:])
        y, _ = self.memory.sequence(features, state, episode_starts[burn_in:])
        actions = self.action_head(y)
        return (actions, y) if return_latents else actions

    def _features(self, rgb_obs: th.Tensor, state_obs: th.Tensor) -> th.Tensor:
        fused = self.feature_fusion(self.vision_encoder(rgb_obs))
        return th.cat([fused, state_obs], dim=-1)

    def _chunk_features(self, rgb_obs: th.Tensor, state_obs: th.Tensor) -> th.Tensor:
        num_steps, batch_size = rgb_obs.shape[:2]
        features = self._features(rgb_obs.flatten(0, 1), state_obs.flatten(0, 1))
        return features.view(num_steps, batch_size, -1)

    def predict_pose(self, rgb_obs: th.Tensor) -> tuple[th.Tensor, ...]:
        if not self.use_pose_head:
            return ()
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
    """
    A first-in-first-out buffer for experience replay: a ring over the steps, each of all the envs,
    so every env's column is in time order (the episode starts are stored for the memory students).
    """

    def __init__(
        self,
        num_envs: int,
        max_size: int,
        img_shape: tuple[int, int, int],
        state_dim: int,
        action_dim: int,
        device: str = "cpu",
        dtype: th.dtype | None = None,
        image_dtype: th.dtype | None = None,
    ):
        self._num_envs = num_envs
        self._max_size = max_size
        self._img_shape = img_shape
        self._state_dim = state_dim
        self._action_dim = action_dim
        self._device = device
        self._ptr = 0
        self._size = 0
        self._dtype = dtype or th.get_default_dtype()
        self._image_dtype = image_dtype or self._dtype

        # Buffers for data
        self._rgb_obs = th.empty(
            max_size, num_envs, *img_shape, dtype=self._image_dtype, device=device
        )
        self._robot_pose = th.empty(
            max_size, num_envs, state_dim, dtype=dtype, device=device
        )
        self._object_poses = th.empty(max_size, num_envs, 7, dtype=dtype, device=device)
        self._actions = th.empty(
            max_size, num_envs, action_dim, dtype=dtype, device=device
        )
        self._episode_starts = th.zeros(
            max_size, num_envs, dtype=th.bool, device=device
        )

    def add(
        self,
        rgb_obs: th.Tensor,
        robot_pose: th.Tensor,
        object_poses: th.Tensor,
        actions: th.Tensor,
        episode_starts: th.Tensor | None = None,
    ) -> None:
        """Add experience to buffer; `episode_starts` [num_envs] marks the first steps of episodes."""
        self._ptr = (self._ptr + 1) % self._max_size
        self._rgb_obs[self._ptr] = self._encode_images(rgb_obs)
        self._robot_pose[self._ptr] = robot_pose
        self._object_poses[self._ptr] = object_poses
        self._actions[self._ptr] = actions
        if episode_starts is None:
            self._episode_starts[self._ptr] = False
        else:
            self._episode_starts[self._ptr] = episode_starts
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
                    "rgb_obs": self._decode_images(
                        self._rgb_obs[batch_indices].reshape(-1, *self._img_shape)
                    ),
                    "robot_pose": self._robot_pose[batch_indices].reshape(
                        -1, self._state_dim
                    ),
                    "object_poses": self._object_poses[batch_indices].reshape(-1, 7),
                    "actions": self._actions[batch_indices].reshape(
                        -1, self._action_dim
                    ),
                }

    def get_sequence_batches(
        self,
        num_mini_batches: int,
        num_epochs: int,
        sequence_length: int,
        burn_in: int,
    ) -> Iterator[dict[str, Any]]:
        """
        Mini-batches of chunks in time order, `[burn_in + sequence_length, num_chunks, ...]` per
        key, plus the `burn_in` actually used: while the buffer holds fewer steps, the chunks and
        then the burn-in get shorter. Each epoch tiles every env's column with non-overlapping
        trained parts at a random phase; the burn-in steps precede them (overlapping the previous
        chunk's). A chunk never crosses the ring's write position (between the newest and the
        oldest step).
        """
        length = min(sequence_length, self._size)
        burn_in = min(burn_in, self._size - length)
        # the physical index of the oldest step
        oldest = (self._ptr - self._size + 1) % self._max_size
        steps = th.arange(burn_in + length, device=self._device)
        for _ in range(num_epochs):
            phase = int(th.randint(0, length, (1,)))
            # the logical starts of the trained parts, every one with a full burn-in before it
            first = burn_in + (phase if burn_in + phase + length <= self._size else 0)
            starts = th.arange(
                first, self._size - length + 1, length, device=self._device
            )
            chunks = th.cartesian_prod(
                starts, th.arange(self._num_envs, device=self._device)
            )
            chunks = chunks.view(-1, 2)[th.randperm(len(chunks), device=self._device)]
            batch_size = max(1, math.ceil(len(chunks) / num_mini_batches))
            for begin in range(0, len(chunks), batch_size):
                batch = chunks[begin : begin + batch_size]
                time_idx = (
                    oldest + batch[:, 0].unsqueeze(0) - burn_in + steps.unsqueeze(1)
                ) % self._max_size
                env_idx = batch[:, 1].unsqueeze(0)
                yield {
                    "rgb_obs": self._decode_images(self._rgb_obs[time_idx, env_idx]),
                    "robot_pose": self._robot_pose[time_idx, env_idx],
                    "object_poses": self._object_poses[time_idx, env_idx],
                    "actions": self._actions[time_idx, env_idx],
                    "episode_starts": self._episode_starts[time_idx, env_idx],
                    "burn_in": burn_in,
                }

    def _encode_images(self, images: th.Tensor) -> th.Tensor:
        if self._image_dtype != th.uint8:
            return images
        return images.clamp(0.0, 1.0).mul(255.0).round_().to(th.uint8)

    def _decode_images(self, images: th.Tensor) -> th.Tensor:
        if self._image_dtype != th.uint8:
            return images
        return images.to(self._dtype).div_(255.0)

    @staticmethod
    def estimate_bytes(
        num_envs: int,
        max_size: int,
        img_shape: tuple[int, int, int],
        state_dim: int,
        action_dim: int,
        dtype: th.dtype,
        image_dtype: th.dtype,
    ) -> int:
        """Memory of the buffer tensors allocated in `__init__`."""
        steps = max_size * num_envs
        image_bytes = (
            math.prod(img_shape) * th.empty(0, dtype=image_dtype).element_size()
        )
        other_bytes = (state_dim + 7 + action_dim) * th.empty(
            0, dtype=dtype
        ).element_size()
        episode_start_bytes = 1
        return steps * (image_bytes + other_bytes + episode_start_bytes)

    def clear(self) -> None:
        """Clear the buffer."""
        self._rgb_obs.zero_()
        self._robot_pose.zero_()
        self._object_poses.zero_()
        self._actions.zero_()
        self._episode_starts.zero_()
        self._ptr = 0
        self._size = 0

    @property
    def img_shape(self) -> tuple[int, int, int]:
        return self._img_shape

    @property
    def state_dim(self) -> int:
        return self._state_dim

    def is_full(self) -> bool:
        """Check if buffer is full."""
        return self._size == self._max_size

    @property
    def size(self) -> int:
        """Get buffer size."""
        return self._size
