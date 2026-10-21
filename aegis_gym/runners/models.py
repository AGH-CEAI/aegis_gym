import torch as th
from rsl_rl.models import MLPModel
from rsl_rl.modules import HiddenState
from tensordict import TensorDict
from torch import nn


class AegisMLPModel(MLPModel):
    def __init__(self, *args, detach_actor_grad: bool = False, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.detach_actor_grad = detach_actor_grad

    def get_latent(
        self,
        obs: TensorDict,
        masks: th.Tensor | None = None,
        hidden_state: HiddenState = None,
    ) -> th.Tensor:
        latent = super().get_latent(obs, masks, hidden_state)
        return latent.detach() if self.detach_actor_grad else latent

    def reset_last_layer_weights(self, scale: float = 1.0) -> None:
        for module in reversed(self.mlp):
            if isinstance(module, nn.Linear):
                nn.init.orthogonal_(module.weight, gain=scale)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
                return
        raise RuntimeError("Model does not contain any linear layers to reset.")
