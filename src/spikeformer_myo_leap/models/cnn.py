"""CNN baseline for EMG-to-pose regression."""

from __future__ import annotations

from dataclasses import dataclass

from torch import Tensor, nn


@dataclass(frozen=True)
class CNNModelDefaults:
    """Reference default hyperparameters for the CNN regressor."""

    input_dim: int = 8
    embed_dim: int = 64
    num_blocks: int = 4
    output_dim: int = 63


class EMGCNNRegressor(nn.Module):
    """Temporal CNN regressor baseline."""

    def __init__(
        self,
        input_dim: int = 8,
        embed_dim: int = 64,
        num_blocks: int = 4,
        output_dim: int = 63,
    ) -> None:
        super().__init__()
        blocks = []
        for block_index in range(num_blocks):
            in_channels = input_dim if block_index == 0 else embed_dim
            blocks.append(
                nn.Sequential(
                    nn.Conv1d(in_channels, embed_dim, kernel_size=3, padding=1),
                    nn.BatchNorm1d(embed_dim),
                    nn.ReLU(inplace=True),
                    nn.MaxPool1d(kernel_size=2),
                )
            )
        self.blocks = nn.ModuleList(blocks)
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.head = nn.Linear(embed_dim, output_dim)

    def forward(self, x: Tensor) -> Tensor:
        """Map EMG windows ``[B, T, C]`` to pose vectors ``[B, O]``."""

        x = x.permute(0, 2, 1)
        for block in self.blocks:
            x = block(x)
        x = self.pool(x).squeeze(-1)
        return self.head(x)
