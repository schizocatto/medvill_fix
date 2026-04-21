from __future__ import annotations
import torch
import torch.nn as nn
import torchvision.models as tv_models
from einops import rearrange
from typing import Optional


class ResNetImageEncoder(nn.Module):
    """ResNet-50 backbone; returns spatial feature vectors (B, N, C)."""

    def __init__(self, pool_type: str = "max"):
        super().__init__()
        backbone = tv_models.resnet50(weights=tv_models.ResNet50_Weights.IMAGENET1K_V2)
        # Drop the global pool and FC layers; keep up to layer4
        self.encoder = nn.Sequential(*list(backbone.children())[:-2])
        self.out_dim = 2048

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feats = self.encoder(x)                        # (B, 2048, H', W')
        B, C, H, W = feats.shape
        return feats.view(B, C, H * W).transpose(1, 2)  # (B, H'*W', 2048)


class PatchEmbedding(nn.Module):
    """ViT-style patch embedding; returns (B, num_patches, embed_dim)."""

    def __init__(
        self,
        img_size: int = 512,
        patch_size: int = 16,
        in_channels: int = 3,
        embed_dim: int = 768,
    ):
        super().__init__()
        self.num_patches = (img_size // patch_size) ** 2
        self.proj = nn.Conv2d(
            in_channels, embed_dim, kernel_size=patch_size, stride=patch_size
        )
        self.out_dim = embed_dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.proj(x)                            # (B, embed_dim, H/P, W/P)
        return rearrange(x, "b c h w -> b (h w) c")  # (B, num_patches, embed_dim)


class ImageBertEmbeddings(nn.Module):
    """Project image features → hidden_size, add positional & type embeddings."""

    def __init__(
        self,
        num_image_embeds: int,
        img_hidden_sz: int,
        hidden_size: int = 768,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.num_image_embeds = num_image_embeds
        self.proj = nn.Linear(img_hidden_sz, hidden_size)
        self.pos_embeddings = nn.Embedding(num_image_embeds + 1, hidden_size)
        self.token_type_embeddings = nn.Embedding(2, hidden_size)
        self.norm = nn.LayerNorm(hidden_size, eps=1e-12)
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        img_features: torch.Tensor,
        token_type_ids: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        B, N, _ = img_features.shape

        # Randomly sample if more patches than required
        if N > self.num_image_embeds:
            idx = torch.randperm(N, device=img_features.device)[: self.num_image_embeds]
            img_features = img_features[:, idx, :]
        elif N < self.num_image_embeds:
            # Repeat-pad to fill slots (rare edge case for ViT)
            pad = self.num_image_embeds - N
            img_features = torch.cat(
                [img_features, img_features[:, :pad, :]], dim=1
            )

        projected = self.proj(img_features)  # (B, num_image_embeds, hidden_size)

        positions = torch.arange(
            projected.size(1), dtype=torch.long, device=projected.device
        ).unsqueeze(0).expand(B, -1)

        if token_type_ids is None:
            token_type_ids = torch.zeros(
                B, projected.size(1), dtype=torch.long, device=projected.device
            )

        emb = projected + self.pos_embeddings(positions) + self.token_type_embeddings(token_type_ids)
        return self.dropout(self.norm(emb))
