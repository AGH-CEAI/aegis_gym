from dataclasses import dataclass
from pathlib import Path
from typing import Literal, Optional
from .base_cfg import BaseCfg

@dataclass(slots=True)
class CNNLayerCfg(BaseCfg):
    in_channels: int
    out_channels: int
    kernel_size: int
    stride: int
    padding: int

@dataclass(slots=True)
class FusionCfg(BaseCfg):
    fusion_output_dim: Optional[int]
    pool_size: Optional[int]
    num_heads: Optional[int]

@dataclass(slots=True)
class PolicyBCCfg(BaseCfg):
    encoder_type: str
    fusion_type: str
    use_pose_head: bool
    vision_encoder: dict[str, list[CNNLayerCfg]]
    vision_encoder_spatial: dict[str, list[CNNLayerCfg]]
    linear_fusion: FusionCfg
    attention_vector_fusion: FusionCfg
    attention_spatial_fusion: FusionCfg
    action_head_state_obs_dim: int
    action_head_hidden_dims: list[int, ...]
    pose_head_hidden_dims: list[int, ..]

@dataclass(slots=True)
class BCCfg(BaseCfg):
    algorithm_rnd_cfg: Optional[dict]
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
    policy: PolicyBCCfg
