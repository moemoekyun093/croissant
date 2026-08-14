"""
Embeds scalar numeric values into R^k using sinusoidal features (the same
family as transformer positional embeddings / diffusion timestep
embeddings), followed by a small learnable projection.

Raw numeric cell values can span wildly different scales (age: 30,
revenue: 5_000_000). A sign-log transform is applied before the
sinusoidal features so the embedding is sensitive to relative rather than
absolute magnitude -- without this, large values dominate the embedding
and small ones collapse together.
"""

import math

import torch
import torch.nn as nn


def sign_log_transform(x: torch.Tensor) -> torch.Tensor:
    """sign(x) * log(1 + |x|) -- compresses dynamic range, keeps sign."""
    return torch.sign(x) * torch.log1p(torch.abs(x))


def sinusoidal_features(values: torch.Tensor, dim: int) -> torch.Tensor:
    """
    values: [N] float tensor
    returns: [N, dim]

    Standard sin/cos frequency embedding, as used for diffusion timestep
    embeddings and transformer positional encodings.
    """

    half_dim = dim // 2

    freqs = torch.exp(
        -math.log(10000.0)
        * torch.arange(half_dim, device=values.device, dtype=values.dtype)
        / max(half_dim - 1, 1)
    )

    args = values[:, None] * freqs[None, :]

    embedding = torch.cat([torch.sin(args), torch.cos(args)], dim=-1)

    if dim % 2 == 1:
        embedding = torch.cat(
            [embedding, torch.zeros_like(embedding[:, :1])], dim=-1
        )

    return embedding


class NumericEmbedder(nn.Module):
    def __init__(self, output_dim: int, sinusoidal_dim: int = 128):
        super().__init__()

        self.sinusoidal_dim = sinusoidal_dim

        self.proj = nn.Sequential(
            nn.Linear(sinusoidal_dim, output_dim),
            nn.SiLU(),
            nn.Linear(output_dim, output_dim),
        )

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        """
        values: [N] float tensor of raw numeric cell values
        returns: [N, output_dim]
        """

        transformed = sign_log_transform(values)
        features = sinusoidal_features(transformed, self.sinusoidal_dim)
        return self.proj(features)