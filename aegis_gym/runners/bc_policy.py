import math
from collections.abc import Callable, Iterator
from typing import Any, NamedTuple

import torch as th
from strenum import StrEnum
from torch import nn

from aegis_gym.aux.geom import planar_pose, quat_to_z_euler
from aegis_gym.aux.logging import get_logger
from aegis_gym.config.types import (
    FusionCfg,
    MemoryCfg,
    MultimodalCfg,
    PolicyBCCfg,
    VisionEncoderCfg,
)

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

# the latent pose decoder's x, y in dm: on the scale of the sin / cos of the yaw
LATENT_POSE_XY_SCALE = 10.0


LATENT_POSE_FRAMES = ("world", "goal")


def goal_planar_pose(env) -> th.Tensor:
    """[num_envs, 3] the goal's x, y [m] and yaw [rad] in the world frame."""
    return th.cat([env.goal_pose[:, :2], env.goal_yaw.reshape(-1, 1)], dim=-1)


def latent_pose_target(
    object_poses: th.Tensor, goal_poses: th.Tensor | None = None
) -> th.Tensor:
    """
    [N, 4] the latent pose decoder's target from the [N, 7] (x, y, z, quat) tee poses: in the world
    frame, or with the [N, 3] (x, y, yaw) `goal_poses` in the goal's frame (the tee relative to the goal).
    """
    xy = object_poses[:, :2]
    yaw = quat_to_z_euler(object_poses[:, 3:7])
    if goal_poses is not None:
        goal_yaw = goal_poses[:, 2]
        cos, sin = th.cos(goal_yaw), th.sin(goal_yaw)
        offset = xy - goal_poses[:, :2]
        xy = th.stack(
            [
                cos * offset[:, 0] + sin * offset[:, 1],
                -sin * offset[:, 0] + cos * offset[:, 1],
            ],
            dim=-1,
        )
        yaw = yaw - goal_yaw
    return planar_pose(xy * LATENT_POSE_XY_SCALE, yaw)


def latent_pose_errors(
    pred: th.Tensor, object_poses: th.Tensor, goal_poses: th.Tensor | None = None
) -> tuple[th.Tensor, th.Tensor]:
    """
    [N] the position [m] and yaw [rad] errors of the latent pose decoder's [N, 4] predictions (in the
    goal's frame with `goal_poses`, see `latent_pose_target`).
    """
    target = latent_pose_target(object_poses, goal_poses)
    pos_err = th.linalg.vector_norm(pred[:, :2] - target[:, :2], dim=-1)
    yaw_err = th.atan2(pred[:, 2], pred[:, 3]) - th.atan2(target[:, 2], target[:, 3])
    yaw_err = th.atan2(th.sin(yaw_err), th.cos(yaw_err)).abs()
    return pos_err / LATENT_POSE_XY_SCALE, yaw_err


MemoryState = tuple[th.Tensor, ...]
"""The student's memory state: batch-first tensors (see `enc_mem_dec.memory.MemoryState`)."""

DYNAMICS_EXPERT = "dynamics"


class PriorMemoryState(NamedTuple):
    """
    The memory state with the world model's prior expert: the memory's own, the last step's joint
    latent [batch, latent] and executed action [batch, action] and whether they exist [batch].
    """

    memory: MemoryState
    latent: th.Tensor
    action: th.Tensor
    valid: th.Tensor


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


def camera_modality(index: int) -> str:
    """The multimodal fusion's name of camera `index` (the order of the stacked RGB channels)."""
    return f"camera_{index}"


def build_multimodal(
    cfg: MultimodalCfg,
    num_cameras: int,
    cnn_builder: Callable[[], nn.Module],
    cnn_channels: int,
    state_obs_dims: dict[str, int],
) -> nn.Module:
    """
    The `enc_mem_dec` `MultimodalFusion` of `cfg`: an encoder per camera (`cnn_builder()` with
    `cnn_channels` output channels, pooled) and one per state input of `state_obs_dims` (in order).
    """
    try:
        from enc_mem_dec.multimodal import (
            ImageModalityEncoder,
            MultimodalFusion,
            StateModalityEncoder,
        )
    except ImportError as e:
        raise ImportError(
            f"multimodal.fusion '{cfg.fusion}' needs the enc_mem_dec package with `multimodal` "
            "(the worker image geonosis:5000/aegis_gym_emd)"
        ) from e
    encoders: dict[str, nn.Module] = {
        camera_modality(i): ImageModalityEncoder(
            cnn_builder(),
            backbone_channels=cnn_channels,
            output_dim=cfg.feature_dim,
            pooling=cfg.image_pooling,
        )
        for i in range(num_cameras)
    }
    for name, dim in state_obs_dims.items():
        encoders[name] = StateModalityEncoder(
            dim, output_dim=cfg.feature_dim, hidden_dims=cfg.state_hidden_dims
        )
    extra = (
        {"extra_experts": [DYNAMICS_EXPERT]}
        if cfg.dynamics.enabled and cfg.dynamics.prior_expert
        else {}
    )
    return MultimodalFusion(
        encoders,
        fusion=cfg.fusion,
        latent_dim=cfg.latent_dim,
        prior_expert=cfg.prior_expert,
        gate_hidden_dim=cfg.gate_hidden_dim,
        sample=cfg.sample,
        **extra,
    )


def _stack_fusions(outs: list) -> Any:
    """The `FusionOutput`s of the steps [batch, ...] -> one of [time * batch, ...] (time-major)."""
    first = outs[0]

    def cat(values: list[th.Tensor]) -> th.Tensor:
        return th.cat(values, dim=0)

    gaussian = type(first.experts)
    return type(first)(
        latent=cat([o.latent for o in outs]),
        features={k: cat([o.features[k] for o in outs]) for k in first.features},
        experts=gaussian(
            cat([o.experts.mean for o in outs]), cat([o.experts.log_var for o in outs])
        ),
        joint=gaussian(
            cat([o.joint.mean for o in outs]), cat([o.joint.log_var for o in outs])
        ),
        log_weights=(
            None if first.log_weights is None else cat([o.log_weights for o in outs])
        ),
    )


def _observation_posterior(out: Any, num_modalities: int) -> Any:
    """The (g)PoE of the `num_modalities` modality experts of a `FusionOutput` alone (its gate weights renormalized)."""
    from enc_mem_dec.multimodal import Gaussian, product_of_experts

    experts = Gaussian(
        out.experts.mean[..., :num_modalities, :],
        out.experts.log_var[..., :num_modalities, :],
    )
    log_weights = None
    if out.log_weights is not None:
        log_weights = th.log_softmax(out.log_weights[..., :num_modalities, :], dim=-2)
    return product_of_experts(experts, log_weights)


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
        state_obs_dims: dict[str, int] | None = None,
    ):
        """
        `state_obs_dims`: the dimension of each `state_obs` input (their sum is `state_obs_dim`); needed
        by the multimodal fusion with more than one state input.
        """
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

        configured_dim = self._cfg.action_head_state_obs_dim
        if configured_dim is not None and configured_dim != state_obs_dim:
            raise ValueError(
                f"action_head_state_obs_dim {configured_dim} doesn't match state_obs "
                f"{self._cfg.state_obs} ({state_obs_dim}); set it to {state_obs_dim} or null"
            )

        self.vision_encoder: BaseVisionEncoder | None = None
        self.feature_fusion: BaseFusionModule | None = None
        self.multimodal: nn.Module | None = None
        # the multimodal fusion's output of the last `_features()` call (its experts' KL, weights)
        self.last_fusion = None
        self._last_presence: dict[str, th.Tensor] | None = None  # its modality dropout
        if self._cfg.multimodal.fusion != "none":
            self.multimodal = self._build_multimodal(state_obs_dim, state_obs_dims)
            features_dim = self.multimodal.output_dim
            vision_obs_dim = None
            logger.info(
                f"Multimodal fusion: {self._cfg.multimodal.fusion} of {self.multimodal.encoders.names}"
            )
        else:
            self.vision_encoder = self._build_vision_encoder().to(device)
            self.feature_fusion = self._build_fusion(
                image_height=image_height, image_width=image_width
            )
            vision_obs_dim = self.feature_fusion.output_dim
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
        self.latent_pose_frame = self._cfg.latent_pose_frame
        if self.latent_pose_frame not in LATENT_POSE_FRAMES:
            raise ValueError(
                f"latent_pose_frame '{self.latent_pose_frame}' not in {LATENT_POSE_FRAMES}"
            )
        if self._cfg.latent_pose_loss_weight > 0:
            self.latent_pose_head = self._build_mlp(
                input_dim=action_input_dim,
                hidden_dims=self._cfg.latent_pose_hidden_dims,
                output_dim=4,
            )
            logger.info(
                f"Latent pose decoder, loss weight {self._cfg.latent_pose_loss_weight}, "
                f"{self.latent_pose_frame} frame"
            )

        self.dynamics: nn.Module | None = None
        cfg_dyn = self._cfg.multimodal.dynamics
        if cfg_dyn.enabled:
            if (
                self.multimodal is None
                or self._cfg.multimodal.fusion == "concat"
                or self.memory is None
            ):
                raise ValueError(
                    "multimodal.dynamics needs the poe / gpoe fusion and a memory (chunks)"
                )
            from enc_mem_dec.multimodal import MultimodalDynamics

            self.dynamics = MultimodalDynamics(
                self.multimodal.encoders.names,
                latent_dim=self._cfg.multimodal.latent_dim,
                action_dim=action_dim,
                hidden_dims=cfg_dyn.hidden_dims,
                reward=cfg_dyn.reward_weight > 0,
            )
            logger.info(
                f"Latent world model: KL weight {cfg_dyn.kl_weight}, reward weight "
                f"{cfg_dyn.reward_weight}, detached policy {cfg_dyn.detach_policy}"
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
        """
        [..., 4] the tee's planar pose (in `latent_pose_frame`) from the action head's input, None
        without the decoder.
        """
        if self.latent_pose_head is None:
            return None
        return self.latent_pose_head(latent)

    @property
    def has_prior_expert(self) -> bool:
        """The world model's joint transition is a gPoE expert (`dynamics.prior_expert`)."""
        return self.dynamics is not None and self._cfg.multimodal.dynamics.prior_expert

    def initial_memory_state(self, batch_size: int) -> MemoryState | PriorMemoryState:
        # the parameters' device: `self.device` is the one at the construction (before `.to()`)
        device = next(self.parameters()).device
        state = self.memory.initial_state(batch_size, device)
        if not self.has_prior_expert:
            return state
        latent_dim = self._cfg.multimodal.latent_dim
        action_dim = self.action_head[-1].out_features
        return PriorMemoryState(
            state,
            th.zeros(batch_size, latent_dim, device=device),
            th.zeros(batch_size, action_dim, device=device),
            th.zeros(batch_size, dtype=th.bool, device=device),
        )

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
        if self.has_prior_expert:
            return self._prior_step(
                rgb_obs, state_obs, memory_state, episode_starts, return_latent
            )
        memory_state = self.memory.reset(memory_state, episode_starts)
        features = self._features(rgb_obs, state_obs)
        y, memory_state = self.memory.step(features, memory_state)
        if return_latent:
            return self.action_head(y), memory_state, y
        return self.action_head(y), memory_state

    def _prior_step(
        self,
        rgb_obs: th.Tensor,
        state_obs: th.Tensor,
        prior_state: PriorMemoryState,
        episode_starts: th.Tensor,
        return_latent: bool,
    ) -> (
        tuple[th.Tensor, PriorMemoryState]
        | tuple[th.Tensor, PriorMemoryState, th.Tensor]
    ):
        """`step()` with the prior expert; the next prior uses the student's own action (clamped)."""
        memory_state = self.memory.reset(prior_state.memory, episode_starts)
        features = self._fuse_with_prior(
            self._encode(rgb_obs, state_obs),
            prior_state.latent,
            prior_state.action,
            prior_state.valid & ~episode_starts,
            self._modality_presence(rgb_obs.shape[0], rgb_obs.device),
        )
        self.last_fusion = self._last_step_fusion
        y, memory_state = self.memory.step(features, memory_state)
        actions = self.action_head(y)
        next_state = PriorMemoryState(
            memory_state,
            features,
            actions.detach().clamp(-1.0, 1.0),
            th.ones_like(prior_state.valid),
        )
        if return_latent:
            return actions, next_state, y
        return actions, next_state

    def sequence(
        self,
        rgb_obs: th.Tensor,
        state_obs: th.Tensor,
        episode_starts: th.Tensor,
        burn_in: int,
        return_latents: bool = False,
        executed_actions: th.Tensor | None = None,
    ) -> th.Tensor | tuple[th.Tensor, th.Tensor]:
        """
        The actions [time - burn_in, batch, action_dim] on chunks [time, batch, ...] of episode
        steps, from the initial memory state: the first `burn_in` steps only warm the state up
        (without gradients), the state is reset at the `episode_starts` [time, batch]. With
        `return_latents`, also the action head's inputs [time - burn_in, batch, ...]. The prior expert
        needs the `executed_actions` [time, batch, action_dim].
        """
        batch_size = rgb_obs.shape[1]
        if self.has_prior_expert:
            if executed_actions is None:
                raise ValueError(
                    "The prior expert needs the executed actions of the chunks."
                )
            return self._prior_sequence(
                rgb_obs,
                state_obs,
                episode_starts,
                burn_in,
                executed_actions,
                return_latents,
            )
        state = self.initial_memory_state(batch_size)
        if burn_in > 0:
            with th.no_grad():
                features = self._chunk_features(rgb_obs[:burn_in], state_obs[:burn_in])
                _, state = self.memory.sequence(
                    features, state, episode_starts[:burn_in]
                )
        features = self._chunk_features(rgb_obs[burn_in:], state_obs[burn_in:])
        if self.dynamics is not None and self._cfg.multimodal.dynamics.detach_policy:
            features = features.detach()
        y, _ = self.memory.sequence(features, state, episode_starts[burn_in:])
        actions = self.action_head(y)
        return (actions, y) if return_latents else actions

    def _features(self, rgb_obs: th.Tensor, state_obs: th.Tensor) -> th.Tensor:
        if self.multimodal is not None:
            return self._multimodal_features(rgb_obs, state_obs)
        fused = self.feature_fusion(self.vision_encoder(rgb_obs))
        return th.cat([fused, state_obs], dim=-1)

    def _modality_presence(
        self, batch_size: int, device: th.device
    ) -> dict[str, th.Tensor] | None:
        """The training step's modality dropout: [batch_size] bool per modality, None when off."""
        cfg = self._cfg.multimodal
        if cfg.modality_dropout <= 0.0 or not self.training or not th.is_grad_enabled():
            return None
        names = self.multimodal.encoders.names
        droppable = [
            n
            for n in names
            if not cfg.modality_dropout_names or n in cfg.modality_dropout_names
        ]
        present = (
            th.rand(batch_size, len(droppable), device=device) >= cfg.modality_dropout
        )
        if len(droppable) == len(names):
            # at least one modality stays: a random one of those all dropped
            none = ~present.any(dim=1)
            keep = th.randint(len(droppable), (batch_size,), device=device)
            present[none, keep[none]] = True
        return dict(zip(droppable, present.unbind(dim=1)))

    def _prior_sequence(
        self,
        rgb_obs: th.Tensor,
        state_obs: th.Tensor,
        episode_starts: th.Tensor,
        burn_in: int,
        executed_actions: th.Tensor,
        return_latents: bool,
    ) -> th.Tensor | tuple[th.Tensor, th.Tensor]:
        """`sequence()` with the prior expert: the fusion step by step (the encoders once per part)."""
        batch_size = rgb_obs.shape[1]
        prior = self.initial_memory_state(batch_size)
        state, latent, action, valid = prior
        if burn_in > 0:
            with th.no_grad():
                features, latent, valid = self._fuse_chunk_with_prior(
                    rgb_obs[:burn_in],
                    state_obs[:burn_in],
                    episode_starts[:burn_in],
                    executed_actions[:burn_in],
                    latent,
                    action,
                    valid,
                    present=None,
                )
                _, state = self.memory.sequence(
                    features, state, episode_starts[:burn_in]
                )
                action = executed_actions[burn_in - 1]
        present = self._modality_presence(batch_size, rgb_obs.device)
        features, _, _ = self._fuse_chunk_with_prior(
            rgb_obs[burn_in:],
            state_obs[burn_in:],
            episode_starts[burn_in:],
            executed_actions[burn_in:],
            latent,
            action,
            valid,
            present,
        )
        if self.dynamics is not None and self._cfg.multimodal.dynamics.detach_policy:
            features = features.detach()
        y, _ = self.memory.sequence(features, state, episode_starts[burn_in:])
        actions = self.action_head(y)
        return (actions, y) if return_latents else actions

    def _fuse_chunk_with_prior(
        self,
        rgb_obs: th.Tensor,
        state_obs: th.Tensor,
        episode_starts: th.Tensor,
        executed_actions: th.Tensor,
        latent: th.Tensor,
        action: th.Tensor,
        valid: th.Tensor,
        present: dict[str, th.Tensor] | None,
    ) -> tuple[th.Tensor, th.Tensor, th.Tensor]:
        """
        The fused latents [time, batch, latent] of a chunk part, step by step from the carried last
        `latent`, `action` and `valid`; `last_fusion` holds all its steps (flattened as the other
        chunk paths). Returns the latents, the last latent and validity.
        """
        num_steps, batch_size = rgb_obs.shape[:2]
        if self.training and th.is_grad_enabled():
            p = self._cfg.multimodal.dynamics.prior_dropout
            keep_prior = th.rand(batch_size, device=rgb_obs.device) >= p
        else:
            keep_prior = None
        encoded = self._encode(rgb_obs.flatten(0, 1), state_obs.flatten(0, 1))
        encoded = {k: v.view(num_steps, batch_size, -1) for k, v in encoded.items()}
        self._last_presence = (
            {k: v.expand(num_steps, -1).flatten() for k, v in present.items()}
            if present is not None
            else None
        )
        latents, outs = [], []
        for t in range(num_steps):
            valid = valid & ~episode_starts[t]
            latent = self._fuse_with_prior(
                {k: v[t] for k, v in encoded.items()},
                latent,
                action,
                valid if keep_prior is None else valid & keep_prior,
                present,
            )
            latents.append(latent)
            outs.append(self._last_step_fusion)
            action, valid = executed_actions[t], th.ones_like(valid)
        self.last_fusion = _stack_fusions(outs)
        return th.stack(latents), latent, valid

    def _encode(self, rgb_obs: th.Tensor, state_obs: th.Tensor) -> dict[str, th.Tensor]:
        """The multimodal encoders' features of a batch."""
        return self.multimodal.encoders(self._multimodal_inputs(rgb_obs, state_obs))

    def _fuse_with_prior(
        self,
        encoded: dict[str, th.Tensor],
        last_latent: th.Tensor,
        last_action: th.Tensor,
        has_last: th.Tensor,
        present: dict[str, th.Tensor] | None,
    ) -> th.Tensor:
        """One step's fusion with the joint transition's prediction as the `dynamics` expert."""
        prior = self.dynamics.predict_joint(last_latent, last_action)
        out = self.multimodal.fuse(
            encoded,
            present=(present or {}) | {DYNAMICS_EXPERT: has_last},
            extras={DYNAMICS_EXPERT: prior},
        )
        self._last_step_fusion = out
        if out.joint is not None and not th.is_grad_enabled():
            return out.joint.mean
        return out.latent

    def _multimodal_inputs(
        self, rgb_obs: th.Tensor, state_obs: th.Tensor
    ) -> dict[str, th.Tensor]:
        inputs = {
            camera_modality(i): camera
            for i, camera in enumerate(rgb_obs.chunk(self.num_cameras, dim=1))
        }
        inputs |= dict(
            zip(
                self._state_obs_dims,
                state_obs.split(list(self._state_obs_dims.values()), dim=-1),
            )
        )
        return inputs

    def _multimodal_features(
        self,
        rgb_obs: th.Tensor,
        state_obs: th.Tensor,
        present: dict[str, th.Tensor] | None = None,
    ) -> th.Tensor:
        inputs = {
            camera_modality(i): camera
            for i, camera in enumerate(rgb_obs.chunk(self.num_cameras, dim=1))
        }
        inputs |= dict(
            zip(
                self._state_obs_dims,
                state_obs.split(list(self._state_obs_dims.values()), dim=-1),
            )
        )
        if present is None:
            present = self._modality_presence(rgb_obs.shape[0], rgb_obs.device)
        self._last_presence = present
        out = (
            self.multimodal(inputs, present=present)
            if present is not None
            else self.multimodal(inputs)
        )
        self.last_fusion = out
        # acting (collection, preview, evaluation runs without gradients): the joint's mean, a
        # deterministic student; the sample only in the training step
        if out.joint is not None and not th.is_grad_enabled():
            return out.joint.mean
        return out.latent

    def world_model_losses(
        self, executed_actions: th.Tensor, rewards: th.Tensor, episode_starts: th.Tensor
    ) -> dict[str, th.Tensor]:
        """
        The latent world model's losses (`MultimodalDynamics.losses()`: `dynamics_kl`, `reward_nll`) on
        the last `sequence()` call's trained steps: their `executed_actions` [time, batch, action_dim],
        `rewards` and `episode_starts` [time, batch]; a transition into an episode start isn't counted,
        nor the experts of a dropped modality. {} without the world model.
        """
        out = self.last_fusion
        if self.dynamics is None or out is None or out.experts is None:
            return {}
        num_steps, batch_size = executed_actions.shape[:2]
        names = self.multimodal.encoders.names
        M = len(names)
        experts = out.experts
        mean = experts.mean[..., :M, :].reshape(num_steps, batch_size, M, -1)
        log_var = experts.log_var[..., :M, :].reshape(num_steps, batch_size, M, -1)
        joint = out.joint
        if self.has_prior_expert:
            # the target is the posterior of the observations alone (the paper's q(z | x)): with the prior
            # expert in it, the KL is minimized by ignoring the observations (a posterior collapse)
            joint = _observation_posterior(out, M)
        joint_mean = joint.mean.reshape(num_steps, batch_size, -1)
        joint_log_var = joint.log_var.reshape(num_steps, batch_size, -1)
        latent = out.latent.reshape(num_steps, batch_size, -1)
        expert_latents = (
            mean + (0.5 * log_var).exp() * th.randn_like(mean)
            if self.training
            else mean
        )
        expert_valid = None
        if getattr(self, "_last_presence", None):
            present = th.stack(
                [
                    self._last_presence.get(
                        n, th.ones_like(out.latent[:, 0], dtype=th.bool)
                    )
                    for n in names
                ],
                dim=-1,
            ).reshape(num_steps, batch_size, M)
            expert_valid = present[:-1] & present[1:]
        from enc_mem_dec.multimodal import Gaussian

        return self.dynamics.losses(
            expert_latents=expert_latents[:-1],
            joint_latent=latent[:-1],
            next_experts=Gaussian(mean[1:], log_var[1:]),
            next_joint=Gaussian(joint_mean[1:], joint_log_var[1:]),
            action=executed_actions[:-1],
            reward=rewards[:-1] if self.dynamics.rewards is not None else None,
            valid=~episode_starts[1:],
            expert_valid=expert_valid,
        )

    def expert_kl(self) -> th.Tensor | None:
        """The experts' KL to N(0, I) of the last features, None without experts."""
        if self.last_fusion is None or self.last_fusion.experts is None:
            return None
        return self.last_fusion.expert_kl()

    def fusion_stats(self) -> dict[str, float]:
        """
        Of the last features, per modality: `share_<name>`, its share of the joint mean (the
        normalized weighted precision, mean over the batch and the latent dims) and with gpoe
        `weight_<name>`, its gate weight.
        """
        out = self.last_fusion
        if out is None or out.experts is None:
            return {}
        names = (
            self.multimodal.encoders.names
            + list(getattr(self.multimodal, "extra_experts", []))
            + (["prior"] if self._cfg.multimodal.prior_expert else [])
        )
        log_precision = -out.experts.log_var.detach()
        if out.log_weights is not None:
            log_precision = log_precision + out.log_weights.detach()
        shares = th.softmax(log_precision, dim=-2).flatten(0, -3).mean(dim=(0, -1))
        stats = {f"share_{n}": v.item() for n, v in zip(names, shares)}
        if out.weights is not None:
            weights = out.weights.detach().flatten(0, -3).mean(dim=(0, -1))
            stats |= {f"weight_{n}": v.item() for n, v in zip(names, weights)}
        return stats

    def _chunk_features(self, rgb_obs: th.Tensor, state_obs: th.Tensor) -> th.Tensor:
        num_steps, batch_size = rgb_obs.shape[:2]
        if self.multimodal is not None:
            # one modality dropout per chunk, the same on all its steps
            present = self._modality_presence(batch_size, rgb_obs.device)
            if present is not None:
                present = {
                    k: v.expand(num_steps, -1).flatten() for k, v in present.items()
                }
            features = self._multimodal_features(
                rgb_obs.flatten(0, 1), state_obs.flatten(0, 1), present
            )
        else:
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

    def _vision_cfg(self) -> VisionEncoderCfg:
        if self.fusion_type == "attention_spatial":
            return self._cfg.vision_encoder_spatial
        return self._cfg.vision_encoder

    def _build_multimodal(
        self, state_obs_dim: int, state_obs_dims: dict[str, int] | None
    ) -> nn.Module:
        if self.use_pose_head:
            raise ValueError(
                "use_pose_head reads the vision encoder's features: off with multimodal.fusion"
            )
        if self.encoder_type == "autoencoder":
            raise ValueError(
                "the autoencoder encoder doesn't go with multimodal.fusion"
            )
        names = list(self._cfg.state_obs)
        if state_obs_dims is None:
            if len(names) != 1:
                raise ValueError(
                    f"multimodal.fusion with the state_obs {names} needs state_obs_dims"
                )
            state_obs_dims = {names[0]: state_obs_dim}
        if (
            list(state_obs_dims) != names
            or sum(state_obs_dims.values()) != state_obs_dim
        ):
            raise ValueError(
                f"state_obs_dims {state_obs_dims} don't match state_obs {names} ({state_obs_dim})"
            )
        self._state_obs_dims = dict(state_obs_dims)
        vision_cfg = self._vision_cfg()
        return build_multimodal(
            self._cfg.multimodal,
            num_cameras=self.num_cameras,
            cnn_builder=lambda: self._build_cnn(vision_cfg),
            cnn_channels=vision_cfg.conv_layers[-1].out_channels,
            state_obs_dims=self._state_obs_dims,
        )

    def _build_vision_encoder(self) -> BaseVisionEncoder:

        vision_cfg = self._vision_cfg()

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
        # the goals' x, y, yaw (for the latent pose decoder in the goal's frame)
        self._goal_poses = th.zeros(max_size, num_envs, 3, dtype=dtype, device=device)
        self._actions = th.empty(
            max_size, num_envs, action_dim, dtype=dtype, device=device
        )
        self._episode_starts = th.zeros(
            max_size, num_envs, dtype=th.bool, device=device
        )
        # what followed each step (`set_last_transition()`): the executed action and the reward
        self._executed_actions = th.zeros(
            max_size, num_envs, action_dim, dtype=dtype, device=device
        )
        self._rewards = th.zeros(max_size, num_envs, dtype=dtype, device=device)

    def add(
        self,
        rgb_obs: th.Tensor,
        robot_pose: th.Tensor,
        object_poses: th.Tensor,
        actions: th.Tensor,
        episode_starts: th.Tensor | None = None,
        goal_poses: th.Tensor | None = None,
    ) -> None:
        """
        Add experience to buffer; `episode_starts` [num_envs] marks the first steps of episodes,
        `goal_poses` [num_envs, 3] the goals' x, y, yaw (zeros without).
        """
        self._ptr = (self._ptr + 1) % self._max_size
        self._rgb_obs[self._ptr] = self._encode_images(rgb_obs)
        self._robot_pose[self._ptr] = robot_pose
        self._object_poses[self._ptr] = object_poses
        if goal_poses is None:
            self._goal_poses[self._ptr] = 0.0
        else:
            self._goal_poses[self._ptr] = goal_poses
        self._actions[self._ptr] = actions
        if episode_starts is None:
            self._episode_starts[self._ptr] = False
        else:
            self._episode_starts[self._ptr] = episode_starts
        self._executed_actions[self._ptr] = 0.0
        self._rewards[self._ptr] = 0.0
        self._size = min(self._size + 1, self._max_size)

    def set_last_transition(
        self, executed_actions: th.Tensor, rewards: th.Tensor
    ) -> None:
        """The action [num_envs, action_dim] executed in the last added step and its reward [num_envs]."""
        self._executed_actions[self._ptr] = executed_actions
        self._rewards[self._ptr] = rewards

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
                # flatten, not reshape(-1, D): ambiguous for the empty (D = 0) state
                yield {
                    "rgb_obs": self._decode_images(
                        self._rgb_obs[batch_indices].flatten(0, 1)
                    ),
                    "robot_pose": self._robot_pose[batch_indices].flatten(0, 1),
                    "object_poses": self._object_poses[batch_indices].flatten(0, 1),
                    "goal_poses": self._goal_poses[batch_indices].flatten(0, 1),
                    "actions": self._actions[batch_indices].flatten(0, 1),
                    "executed_actions": self._executed_actions[batch_indices].flatten(
                        0, 1
                    ),
                    "rewards": self._rewards[batch_indices].flatten(0, 1),
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
                    "goal_poses": self._goal_poses[time_idx, env_idx],
                    "actions": self._actions[time_idx, env_idx],
                    "episode_starts": self._episode_starts[time_idx, env_idx],
                    "executed_actions": self._executed_actions[time_idx, env_idx],
                    "rewards": self._rewards[time_idx, env_idx],
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
        # the state, object pose, goal x / y / yaw, label, executed action, reward
        other_bytes = (state_dim + 7 + 3 + 2 * action_dim + 1) * th.empty(
            0, dtype=dtype
        ).element_size()
        episode_start_bytes = 1
        return steps * (image_bytes + other_bytes + episode_start_bytes)

    def clear(self) -> None:
        """Clear the buffer."""
        self._rgb_obs.zero_()
        self._robot_pose.zero_()
        self._object_poses.zero_()
        self._goal_poses.zero_()
        self._actions.zero_()
        self._episode_starts.zero_()
        self._executed_actions.zero_()
        self._rewards.zero_()
        self._ptr = 0
        self._size = 0

    @property
    def img_shape(self) -> tuple[int, int, int]:
        return self._img_shape

    @property
    def state_dim(self) -> int:
        return self._state_dim

    @property
    def action_dim(self) -> int:
        return self._action_dim

    def is_full(self) -> bool:
        """Check if buffer is full."""
        return self._size == self._max_size

    @property
    def size(self) -> int:
        """Get buffer size."""
        return self._size
