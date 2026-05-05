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
        device: str = "cpu",
    ) -> tuple[int, int, int]:
        dummy = th.zeros((1, channels, image_height, image_width), device=device)
        with th.no_grad():
            out = self._single_forward(dummy)  # (1, C, H, W)
        _, c, h, w = out.shape
        return c, h, w

    def _single_forward(self, x: th.Tensor) -> th.Tensor:
        # Method for calculating the output shape
        raise NotImplementedError


class SharedCNNEncoder(BaseVisionEncoder):
    def __init__(self, num_cameras: int, cnn_builder: Callable, vision_cfg: dict):
        super().__init__(num_cameras)
        self.encoder = cnn_builder()

    def _single_forward(self, x: th.Tensor) -> th.Tensor:
        return self.encoder(x)

    # TODO(issue#71): Investigate indexing and micro-optimizations in vision encoder forward pass
    def forward(self, rgb_obs: th.Tensor) -> tuple[th.Tensor, ...]:
        features = []
        for i in range(self.num_cameras):
            cam_rgb = rgb_obs[:, i * 3 : (i + 1) * 3]
            feat = self.encoder(cam_rgb)
            features.append(feat)
        return tuple(features)


class PerCameraCNNEncoder(BaseVisionEncoder):
    def __init__(self, num_cameras: int, cnn_builder: Callable, vision_cfg: dict):
        super().__init__(num_cameras)
        self.encoders = nn.ModuleList([cnn_builder() for _ in range(num_cameras)])

    def _single_forward(self, x: th.Tensor) -> th.Tensor:
        # All encoders share the same architecture, so any one can be used for shape inference
        return self.encoders[0](x)

    # TODO(issue#71): Investigate indexing and micro-optimizations in vision encoder forward pass
    def forward(self, rgb_obs: th.Tensor) -> tuple[th.Tensor, ...]:
        features = []
        for i in range(self.num_cameras):
            cam_rgb = rgb_obs[:, i * 3 : (i + 1) * 3]
            feat = self.encoders[i](cam_rgb)
            features.append(feat)
        return tuple(features)


# TODO(issue#79): Remove hardcoded dimensions related to autoencoder and make them configurable
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
        self.feature_channels = self.feature_channels
        self.feature_size = self.feature_size
        self.decoder = self._build_decoder()

    def _single_forward(self, x: th.Tensor) -> th.Tensor:
        fmap = self.encoder(x)  # (1, 32, 16, 16)
        latent = self.to_latent(fmap.flatten(start_dim=1))  # (1, 512)
        return latent.unsqueeze(-1).unsqueeze(-1)  # (1, 512, 1, 1)

    def forward(self, rgb_obs: th.Tensor) -> tuple[th.Tensor, ...]:
        features = []
        for i in range(self.num_cameras):
            cam_rgb = rgb_obs[:, i * 3 : (i + 1) * 3]
            fmap = self.encoder(cam_rgb)  # (B, 32, 16, 16)
            latent = self.to_latent(fmap.flatten(start_dim=1))  # (B, 512)
            features.append(latent.unsqueeze(-1).unsqueeze(-1))  # (B, 512, 1, 1)
        return tuple(features)

    # TODO(issue#71): Investigate indexing and micro-optimizations in vision encoder forward pass
    def reconstruct(self, rgb_obs: th.Tensor) -> tuple[th.Tensor, ...]:
        recons = []
        for i in range(self.num_cameras):
            cam_rgb = rgb_obs[:, i * 3 : (i + 1) * 3]
            fmap = self.encoder(cam_rgb)  # (B, 32, 16, 16)
            flat = fmap.flatten(start_dim=1)
            latent = self.to_latent(flat)  # (B, 512)
            flat_recon = self.from_latent(latent)  # (B, 8192)
            fmap_recon = flat_recon.view(
                -1, self.feature_channels, self.feature_size, self.feature_size
            )  # (B, 32, 16, 16)
            recons.append(self.decoder(fmap_recon))  # (B, 3, 64, 64)
        return tuple(recons)

    def _build_decoder(self) -> nn.Sequential:
        return nn.Sequential(
            # (B, 32, 16, 16)
            nn.ConvTranspose2d(
                32, 16, kernel_size=3, stride=2, padding=1, output_padding=1
            ),
            nn.ReLU(),
            # (B, 16, 32, 32)
            nn.ConvTranspose2d(
                16, 8, kernel_size=3, stride=2, padding=1, output_padding=1
            ),
            nn.ReLU(),
            # (B, 8, 64, 64)
            nn.Conv2d(8, 3, kernel_size=3, stride=1, padding=1),
            nn.Sigmoid(),
            # (B, 3, 64, 64)
        )
