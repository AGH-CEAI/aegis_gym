from dataclasses import dataclass, field
from typing import Literal

from .base_cfg import BaseCfg
from .rl import AlgorithmCfg


@dataclass(slots=True)
class CNNLayerCfg(BaseCfg):
    in_channels: int
    out_channels: int
    kernel_size: int
    stride: int
    padding: int


@dataclass(slots=True)
class VisionEncoderCfg(BaseCfg):
    conv_layers: list[CNNLayerCfg]


@dataclass(slots=True)
class FusionCfg(BaseCfg):
    fusion_output_dim: int | None = None
    pool_size: int | None = None
    num_heads: int | None = None


@dataclass(slots=True)
class MemoryCfg(BaseCfg):
    # the enc_mem_dec memory between the fused image features (+ state_obs) and the action head:
    # "none" (memoryless, needs no enc_mem_dec), "frame_stack", "gru", "lstm", "attention_window";
    # "identity": memoryless too, but trained on the chunks as the memories (their control)
    type: str = "none"
    # the current step's features appended to the memory's output
    residual: bool = False
    hidden_dim: int = 256  # gru, lstm
    num_layers: int = 1  # gru, lstm
    num_frames: int = 4  # frame_stack
    frame_stride: int = 1  # frame_stack: the steps between the stacked frames
    window: int = 16  # attention_window: the steps attended, the current one included
    model_dim: int = 128  # attention_window
    num_heads: int = 4  # attention_window
    # trained on chunks of the buffer: `burn_in` steps warm the state up (no gradients), the loss
    # is on the next `sequence_length` steps
    sequence_length: int = 32
    burn_in: int = 16


@dataclass(slots=True)
class PolicyBCCfg(BaseCfg):
    encoder_type: str
    fusion_type: str
    use_pose_head: bool
    vision_encoder: VisionEncoderCfg
    vision_encoder_spatial: VisionEncoderCfg
    linear_fusion: FusionCfg
    attention_vector_fusion: FusionCfg
    attention_spatial_fusion: FusionCfg
    # None: taken from `state_obs`; a number is checked against it
    action_head_state_obs_dim: int | None
    action_head_hidden_dims: list[int]
    pose_head_hidden_dims: list[int]
    # the student's non-image inputs, concatenated in this order (see `STUDENT_STATE_OBS`)
    state_obs: list[str] = field(default_factory=lambda: ["tcp_pose"])
    memory: MemoryCfg = field(default_factory=MemoryCfg)
    # a decoder of the tee's planar pose (x, y in dm, sin / cos yaw) from the action head's input (the
    # memory's output, the features without a memory), its loss weighted by this (0: no decoder)
    latent_pose_loss_weight: float = 0.0
    latent_pose_hidden_dims: list[int] = field(default_factory=lambda: [128])


@dataclass(slots=True)
class BCCfg(BaseCfg):
    algorithm: AlgorithmCfg
    best_model_skip_iters: int
    buffer_size: int
    eval_freq: int
    learning_rate: float
    max_grad_norm: float
    num_epochs: int
    num_mini_batches: int
    num_steps_per_env: int
    reset_last_layer_weights_interval: int
    reset_last_layer_weights_part: Literal["actor", "critic", "both"]
    save_freq: int
    save_recon_freq: int
    save_recons: bool
    use_teacher_mixing: bool
    use_cameras: bool
    policy: PolicyBCCfg
    # train on the teacher actions clipped to the env's [-1, 1], i.e. on what is executed
    clip_teacher_actions: bool = False
    # "uint8" stores the [0, 1] images in 1/255 steps, a quarter of the float32 memory
    image_buffer_dtype: str = "float32"
