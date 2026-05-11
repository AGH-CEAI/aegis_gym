import torch as th
import torch.nn as nn
import torch.nn.functional as F


class BaseFusionModule(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, features: tuple) -> th.Tensor:
        raise NotImplementedError

    def prepare_pose_features(self, features: tuple) -> tuple:
        raise NotImplementedError


class LinearFusion(BaseFusionModule):
    def __init__(
        self,
        vision_dim: int,
        num_cameras: int,
        in_channels: int,
        image_height: int,
        image_width: int,
        pool_size: int,
        num_feature_tensors: int = None,
    ):
        super().__init__()
        self.pool_size = pool_size
        self.is_latent_vector = image_height == 1 and image_width == 1

        # How many tensors the encoder actually returns (may differ from num_cameras)
        n = num_feature_tensors if num_feature_tensors is not None else num_cameras

        feature_dim_cam = in_channels * pool_size * pool_size
        self.pose_proj = None
        if self.is_latent_vector:
            feature_dim_cam = in_channels
            self.pose_proj = nn.Linear(in_channels, vision_dim)

        self.net = nn.Sequential(
            nn.Linear(feature_dim_cam * n, vision_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
        )
        self.output_dim = vision_dim
        self.pose_input_dim = feature_dim_cam

    def _to_feature_vector(self, feat: th.Tensor) -> th.Tensor:
        if self.is_latent_vector:
            f = feat.flatten(start_dim=1)
        else:
            f = F.adaptive_avg_pool2d(feat, (self.pool_size, self.pool_size))
            f = f.flatten(start_dim=1)
        return f

    def forward(self, features: tuple) -> th.Tensor:
        feat_vecs = [self._to_feature_vector(f) for f in features]
        fused_feat_vec = th.cat(feat_vecs, dim=-1)
        return self.net(fused_feat_vec)

    def prepare_pose_features(self, features: tuple) -> tuple:
        feat_vecs = [self._to_feature_vector(f) for f in features]
        if self.pose_proj is None:
            return tuple(feat_vecs)
        return tuple(self.pose_proj(f) for f in feat_vecs)


class VectorAttentionFusion(BaseFusionModule):
    def __init__(
        self,
        vision_dim: int,  # 512
        num_cameras: int,  # 3
        in_channels: int,  # 32
        num_heads: int,  # 4
        pool_size: int,  # 4
    ):
        super().__init__()
        self.pool_size = pool_size
        flat_per_cam = in_channels * pool_size * pool_size

        self.proj_in = nn.Linear(flat_per_cam, vision_dim)
        self.cam_embed = nn.Parameter(th.randn(1, num_cameras, vision_dim) * 0.02)
        self.cls_token = nn.Parameter(th.randn(1, 1, vision_dim) * 0.02)
        self.norm = nn.LayerNorm(vision_dim)
        self.attn = nn.MultiheadAttention(
            vision_dim, num_heads, batch_first=True, dropout=0.0
        )
        self.mlp = nn.Sequential(nn.Linear(vision_dim, vision_dim), nn.GELU())

        self.output_dim = vision_dim
        self.pose_input_dim = flat_per_cam

    def _to_feature_vector(self, feat: th.Tensor) -> th.Tensor:
        f = F.adaptive_avg_pool2d(feat, (self.pool_size, self.pool_size))
        return f.flatten(start_dim=1)

    def forward(self, features: tuple) -> th.Tensor:
        feat_vecs = [self.proj_in(self._to_feature_vector(f)) for f in features]

        x = th.stack(feat_vecs, dim=1)
        x = x + self.cam_embed

        B = x.shape[0]
        cls = self.cls_token.expand(B, -1, -1)
        x = th.cat([cls, x], dim=1)

        x_norm = self.norm(x)
        attn_out, _ = self.attn(x_norm, x_norm, x_norm)
        x = x + attn_out

        cls_out = x[:, 0]
        return self.mlp(cls_out)

    def prepare_pose_features(self, features: tuple) -> tuple:
        return tuple(self._to_feature_vector(f) for f in features)


class SpatialAttentionFusion(BaseFusionModule):
    def __init__(
        self,
        vision_dim: int,  # 256
        num_cameras: int,  # 3
        in_channels: int,  # 64
        image_height: int,  # 8
        image_width: int,  # 8
        num_heads: int,  # 4
    ):
        super().__init__()
        num_patches = image_height * image_width

        self.patch_proj = nn.Linear(in_channels, vision_dim)
        self.pos_embed = nn.Parameter(th.randn(1, num_patches, vision_dim) * 0.02)
        self.cam_embed = nn.Parameter(th.randn(1, num_cameras, 1, vision_dim) * 0.02)
        self.cls_token = nn.Parameter(th.randn(1, 1, vision_dim) * 0.02)

        self.norm1 = nn.LayerNorm(vision_dim)
        self.attn = nn.MultiheadAttention(
            vision_dim, num_heads, batch_first=True, dropout=0.0
        )
        self.norm2 = nn.LayerNorm(vision_dim)
        self.mlp = nn.Sequential(nn.Linear(vision_dim, vision_dim), nn.GELU())

        self.spatial_pool = SpatialAttentionPooling(in_channels)

        self.output_dim = vision_dim
        self.pose_input_dim = in_channels

    # TODO(issue#71): Investigate indexing and micro-optimizations in vision encoder forward pass
    def forward(self, features: tuple) -> th.Tensor:
        B = features[0].shape[0]
        patch_tokens_per_cam = []

        for i, feat in enumerate(features):
            patches = feat.flatten(2).transpose(1, 2)
            patch_tokens = self.patch_proj(patches)
            patch_tokens = patch_tokens + self.pos_embed
            patch_tokens = patch_tokens + self.cam_embed[:, i]
            patch_tokens_per_cam.append(patch_tokens)

        x = th.cat(patch_tokens_per_cam, dim=1)

        cls = self.cls_token.expand(B, -1, -1)
        x = th.cat([cls, x], dim=1)

        x_norm = self.norm1(x)
        attn_out, _ = self.attn(x_norm, x_norm, x_norm)
        x = x + attn_out

        cls_out = x[:, 0]
        return self.mlp(self.norm2(cls_out))

    def prepare_pose_features(self, features: tuple) -> tuple:
        return tuple(self.spatial_pool(f) for f in features)


class SpatialAttentionPooling(nn.Module):
    def __init__(self, in_channels: int):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, 1, kernel_size=1)

    def forward(self, x: th.Tensor) -> th.Tensor:
        B, C, H, W = x.shape
        weights = self.conv(x)
        weights = weights.view(B, 1, H * W)
        weights = th.softmax(weights, dim=-1)
        feats = x.view(B, C, H * W)
        return (feats * weights).sum(dim=-1)
