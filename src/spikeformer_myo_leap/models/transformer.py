"""Transformer baseline for EMG-to-pose regression."""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
from torch import Tensor, nn


class SinusoidalPositionalEncoding(nn.Module):
    """Standard sinusoidal positional encoding for sequence models."""

    def __init__(self, embed_dim: int, max_len: int = 8192) -> None:
        super().__init__()
        position = torch.arange(0, max_len, dtype=torch.float32).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, embed_dim, 2, dtype=torch.float32) * (-math.log(10000.0) / embed_dim))
        pe = torch.zeros(max_len, embed_dim, dtype=torch.float32)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term[: pe[:, 1::2].shape[1]])
        self.register_buffer("pe", pe, persistent=False)

    def forward(self, x: Tensor) -> Tensor:
        """Add positional encoding to ``[B, T, E]`` embeddings."""

        return x + self.pe[: x.size(1)].unsqueeze(0).to(dtype=x.dtype, device=x.device)


@dataclass(frozen=True)
class TransformerModelDefaults:
    """Reference default hyperparameters for the Transformer regressor."""

    input_dim: int = 8
    embed_dim: int = 64
    num_layers: int = 4
    heads: int = 4
    ff_mult: int = 2
    dropout: float = 0.1
    output_dim: int = 63


class EMGTransformerWindowRegressor(nn.Module):
    """Transformer encoder baseline for windowed EMG regression."""

    def __init__(
        self,
        input_dim: int = 8,
        embed_dim: int = 64,
        num_layers: int = 4,
        heads: int = 4,
        ff_mult: int = 2,
        dropout: float = 0.1,
        output_dim: int = 63,
    ) -> None:
        super().__init__()
        self.input_proj = nn.Linear(input_dim, embed_dim)
        self.positional_encoding = SinusoidalPositionalEncoding(embed_dim=embed_dim)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=heads,
            dim_feedforward=embed_dim * ff_mult,
            dropout=dropout,
            batch_first=True,
            activation="gelu",
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.head = nn.Linear(embed_dim, output_dim)

    def forward(self, x: Tensor) -> Tensor:
        """Map EMG windows ``[B, T, C]`` to pose vectors ``[B, O]``."""

        x = self.input_proj(x)
        x = self.positional_encoding(x)
        x = self.encoder(x)
        return self.head(x[:, -1, :])
