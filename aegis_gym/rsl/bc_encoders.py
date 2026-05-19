from typing import Callable

import torch as th
import torch.nn as nn


class BaseVisionEncoder(nn.Module):
    def __init__(self, num_cameras: int):
        super().__init__()
        self.num_cameras = num_cameras

    def forward(self, rgb_obs: th.Tensor) -> tuple[th.Tensor, ...]:
        raise NotImplementedError

    # TODO(issue#89): Integrate jaxtyping for tensor shape annotations
    # TODO(issue#90): Explore PyTorch meta device for shape inference
    def infer_output_shape(
        self,
        image_height: int = 64,
        image_width: int = 64,
        channels: int = 3,
    ) -> tuple[int, int, int]:
        model_device = next(self.parameters()).device
        dummy = th.zeros((1, channels, image_height, image_width), device=model_device)
        with th.no_grad():
            out = self._single_forward(dummy)  # (1, C, H, W)
        _, c, h, w = out.shape
        return c, h, w

    def _single_forward(self, x: th.Tensor) -> th.Tensor:
        # Method for calculating the output shape
        raise NotImplementedError


class ConcatenatedCNNEncoder(BaseVisionEncoder):
    def __init__(self, num_cameras: int, cnn_builder: Callable, vision_cfg: dict):
        super().__init__(num_cameras)
        self.encoder = cnn_builder()
        self._patch_first_conv(num_cameras)

    def _patch_first_conv(self, num_cameras: int) -> None:
        """Replace the first Conv2d to accept num_cameras*3 input channels."""
        for i, layer in enumerate(self.encoder):
            if isinstance(layer, nn.Conv2d):
                old = layer
                new_conv = nn.Conv2d(
                    in_channels=num_cameras * old.in_channels,
                    out_channels=old.out_channels,
                    kernel_size=old.kernel_size,
                    stride=old.stride,
                    padding=old.padding,
                    bias=old.bias is not None,
                )
                self.encoder[i] = new_conv
                return  # only patch the first one

    def infer_output_shape(self, image_height=64, image_width=64, channels=3):
        model_device = next(self.parameters()).device
        dummy = th.zeros(
            (1, self.num_cameras * channels, image_height, image_width),
            device=model_device,
        )
        with th.no_grad():
            out = self._single_forward(dummy)
        _, c, h, w = out.shape
        return c, h, w

    def _single_forward(self, x: th.Tensor) -> th.Tensor:
        return self.encoder(x)

    def forward(self, rgb_obs: th.Tensor) -> tuple[th.Tensor, ...]:
        # rgb_obs is already (B, num_cameras*3, H, W) — pass directly
        feat = self.encoder(rgb_obs)
        return (feat,)  # single tensor wrapped in tuple for API consistency


class SharedCNNEncoder(BaseVisionEncoder):
    def __init__(self, num_cameras: int, cnn_builder: Callable, vision_cfg: dict):
        super().__init__(num_cameras)
        self.encoder = cnn_builder()

    def _single_forward(self, x: th.Tensor) -> th.Tensor:
        return self.encoder(x)

    def forward(self, rgb_obs: th.Tensor) -> tuple[th.Tensor, ...]:
        B = rgb_obs.shape[0]

        # (B, num_cameras*3, H, W) -> (B, num_cameras, 3, H, W)
        x = rgb_obs.view(B, self.num_cameras, 3, *rgb_obs.shape[2:])

        # (B, num_cameras, 3, H, W) -> (B*num_cameras, 3, H, W)
        x = x.flatten(0, 1)

        # single batched forward pass
        feats = self.encoder(x)  # (B*num_cameras, feat_dim, ...)

        # (B*num_cameras, ...) -> (B, num_cameras, ...) -> tuple of (B, ...) per camera
        feats = feats.unflatten(0, (B, self.num_cameras))
        return tuple(feats.unbind(dim=1))


class PerCameraCNNEncoder(BaseVisionEncoder):
    def __init__(self, num_cameras: int, cnn_builder: Callable, vision_cfg: dict):
        super().__init__(num_cameras)
        self.encoders = nn.ModuleList([cnn_builder() for _ in range(num_cameras)])

    def _single_forward(self, x: th.Tensor) -> th.Tensor:
        # All encoders share the same architecture, so any one can be used for shape inference
        return self.encoders[0](x)

    def forward(self, rgb_obs: th.Tensor) -> tuple[th.Tensor, ...]:
        # splits evenly along channel dim
        cam_inputs = rgb_obs.chunk(self.num_cameras, dim=1)
        return tuple(enc(cam) for enc, cam in zip(self.encoders, cam_inputs))


class AutoencoderCNNEncoder(BaseVisionEncoder):
    def __init__(self, num_cameras: int, cnn_builder: Callable, vision_cfg: dict):
        super().__init__(num_cameras)

        self.encoder = cnn_builder()

        self.feature_channels = vision_cfg["conv_layers"][-1]["out_channels"]
        self.feature_size = 16
        self.flatten_dim = self.feature_channels * self.feature_size * self.feature_size
        self.latent_dim = 512

        self.to_latent = nn.Linear(self.flatten_dim, self.latent_dim)
        self.from_latent = nn.Linear(self.latent_dim, self.flatten_dim)
        self.feature_size = self.feature_size
        self.decoder = self._build_decoder(vision_cfg["conv_layers"])

    def _single_forward(self, x: th.Tensor) -> th.Tensor:
        fmap = self.encoder(x)  # (1, 32, 16, 16)
        latent = self.to_latent(fmap.flatten(start_dim=1))  # (1, 512)
        return latent.unsqueeze(-1).unsqueeze(-1)  # (1, 512, 1, 1)

    def forward(self, rgb_obs: th.Tensor) -> tuple[th.Tensor, ...]:
        B = rgb_obs.shape[0]
        x = rgb_obs.view(B, self.num_cameras, 3, *rgb_obs.shape[2:]).flatten(0, 1)
        latent = self._encode(x)
        latent = latent.unsqueeze(-1).unsqueeze(-1)  # (B*num_cameras, 512, 1, 1)
        return tuple(latent.unflatten(0, (B, self.num_cameras)).unbind(dim=1))

    def reconstruct(self, rgb_obs: th.Tensor) -> tuple[th.Tensor, ...]:
        B = rgb_obs.shape[0]
        x = rgb_obs.view(B, self.num_cameras, 3, *rgb_obs.shape[2:]).flatten(0, 1)
        latent = self._encode(x)
        recons = self._decode(latent)
        return tuple(recons.unflatten(0, (B, self.num_cameras)).unbind(dim=1))

    def _encode(self, x: th.Tensor) -> th.Tensor:
        """Returns latent for a pre-sliced input."""
        fmap = self.encoder(x)
        return self.to_latent(fmap.flatten(1))

    def _decode(self, x_latent: th.Tensor) -> th.Tensor:
        """Returns reconstruction for a latent input."""
        flat_recon = self.from_latent(x_latent)
        fmap_recon = flat_recon.view(
            -1, self.feature_channels, self.feature_size, self.feature_size
        )
        return self.decoder(fmap_recon)

    def _build_decoder(self, cfg_conv_layers: list[dict]) -> nn.Sequential:
        decoder_layers: list[nn.Module] = []

        # The configuration is provided for the encoder, for decoder we need to reverse it
        for layer in reversed(cfg_conv_layers):
            in_ch = layer["out_channels"]
            out_ch = layer["in_channels"]
            stride = layer["stride"]

            if stride > 1:
                decoder_layers.append(
                    nn.ConvTranspose2d(
                        in_ch,
                        out_ch,
                        kernel_size=3,
                        stride=stride,
                        padding=1,
                        output_padding=stride - 1,
                    )
                )
            else:
                decoder_layers.append(
                    nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=1, padding=1)
                )

            decoder_layers.append(nn.ReLU())

        # The decoder layer differs in configuration
        decoder_layers[-1] = nn.Sigmoid()

        return nn.Sequential(*decoder_layers)
