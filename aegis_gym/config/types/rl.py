from dataclasses import dataclass
from typing import Literal

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
class DistributionCfg(BaseCfg):
    """How the actions are sampled around the actor output."""

    # "GaussianDistribution" keeps one standard deviation per action, learned and independent of
    # the state. "HeteroscedasticGaussianDistribution" makes the actor output it as well, so it
    # varies with the state.
    class_name: str
    # Standard deviation at the start of the training, in action units.
    init_std: float
    # "scalar" stores the standard deviation as it is, "log" stores its logarithm, which keeps it
    # positive and makes the updates multiplicative. `init_std` is in action units either way.
    std_type: Literal["scalar", "log"]


@dataclass(slots=True)
class ActorCfg(BaseCfg):
    class_name: str
    hidden_dims: list[int]
    activation: str
    obs_normalization: bool
    distribution_cfg: DistributionCfg


@dataclass(slots=True)
class CriticCfg(BaseCfg):
    class_name: str
    hidden_dims: list[int]
    activation: str
    obs_normalization: bool


@dataclass(slots=True)
class PolicyCfg(BaseCfg):
    actor: ActorCfg
    critic: CriticCfg
    reset_last_layer_weights: dict
    detach_actor_grad: bool


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
    init_member_classes: dict
    policy: PolicyCfg
