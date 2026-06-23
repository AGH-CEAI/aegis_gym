from dataclasses import dataclass
from typing import Optional, Literal

from .base_cfg import BaseCfg


@dataclass(slots=True)
class AlgorithmCfg(BaseCfg):
    class_name: str
    learning_rate: float
    num_learning_epochs: int
    num_mini_batches: int
    schedule: str
    value_loss_coef: float
    clip_param: float
    use_clipped_value_loss: bool
    desired_kl: float
    entropy_coef: float
    gamma: float
    lam: float
    max_grad_norm: float
    normalize_advantage_per_mini_batch: bool
    rnd_cfg: Optional[dict]
    symmetry_cfg: Optional[dict]


@dataclass(slots=True)
class PolicyCfg(BaseCfg):
    class_name: str
    activation: str
    actor_obs_normalization: bool
    critic_obs_normalization: bool
    init_noise_std: float
    actor_hidden_dims: list[int, ...]
    critic_hidden_dims: list[int, ...]
    noise_std_type: Literal["scalar", "log"]
    state_dependent_std: bool
    detach_actor_grad: bool


@dataclass(slots=True)
class RLCfg(BaseCfg):
    class_name: str
    num_steps_per_env: int
    seed: int
    obs_group_policy: list[str, ...]
    obs_group_critic: list[str, ...]
    save_interval: int
    best_model_skip_iters: int
    experiment_name: str
    run_name: str
    algorithm: AlgorithmCfg
    reset_last_layer_weights_interval: int
    reset_last_layer_weights_part: Literal["actor", "critic", "both"]
    init_member_classes: list
    policy: PolicyCfg
