from dataclasses import dataclass, field

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
    rnd_cfg: dict | None
    symmetry_cfg: dict | None


@dataclass(slots=True)
class ActorCfg(BaseCfg):
    class_name: str
    hidden_dims: list[int]
    activation: str
    obs_normalization: bool
    detach_actor_grad: bool = False
    # `GaussianDistribution` (`init_std`, `std_type`) or `HeteroscedasticGaussianDistribution`
    distribution_cfg: dict = field(default_factory=dict)


@dataclass(slots=True)
class CriticCfg(BaseCfg):
    class_name: str
    hidden_dims: list[int]
    activation: str
    obs_normalization: bool


@dataclass(slots=True)
class RLCfg(BaseCfg):
    class_name: str
    num_steps_per_env: int
    max_iterations: int
    seed: int
    use_cameras: bool
    obs_groups: dict
    save_interval: int
    best_model_skip_iters: int
    experiment_name: str
    run_name: str
    algorithm: AlgorithmCfg
    # reset_last_layer_weights_interval: int
    # reset_last_layer_weights_part: Literal["actor", "critic", "both"]
    reset_last_layer_weights: dict
    init_member_classes: dict
    actor: ActorCfg
    critic: CriticCfg
