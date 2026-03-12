"""Spiking CNN baseline for EMG-to-pose regression."""

from __future__ import annotations

from dataclasses import dataclass

from torch import Tensor, nn

try:
    from spikingjelly.clock_driven import neuron
except ImportError as exc:  # pragma: no cover - depends on local environment
    raise ImportError(
        "spikingjelly is required for Spikeformer and spiking CNN models. "
        "Install it in the project environment before training."
    ) from exc


@dataclass(frozen=True)
class SpikingCNNModelDefaults:
    """Reference default hyperparameters for the spiking CNN regressor."""

    input_dim: int = 8
    embed_dim: int = 64
    num_blocks: int = 4
    output_dim: int = 63
    backend: str = "cupy"


class SpikingCNNRegressor(nn.Module):
    """Spiking temporal CNN regressor."""

    def __init__(
        self,
        input_dim: int = 8,
        embed_dim: int = 64,
        num_blocks: int = 4,
        output_dim: int = 63,
        backend: str = "cupy",
    ) -> None:
        super().__init__()
        conv_blocks = []
        bn_blocks = []
        lif_blocks = []
        for block_index in range(num_blocks):
            in_channels = input_dim if block_index == 0 else embed_dim
            conv_blocks.append(nn.Conv1d(in_channels, embed_dim, kernel_size=3, padding=1))
            bn_blocks.append(nn.BatchNorm1d(embed_dim))
            lif_blocks.append(neuron.MultiStepLIFNode(tau=2.0, detach_reset=True, backend=backend))
        self.conv_blocks = nn.ModuleList(conv_blocks)
        self.bn_blocks = nn.ModuleList(bn_blocks)
        self.lif_blocks = nn.ModuleList(lif_blocks)
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.head = nn.Linear(embed_dim, output_dim)

    def forward(self, x: Tensor) -> Tensor:
        """Map EMG windows ``[B, T, C]`` to pose vectors ``[B, O]``."""

        x = x.permute(0, 2, 1)
        for conv, bn, lif in zip(self.conv_blocks, self.bn_blocks, self.lif_blocks):
            x = conv(x)
            x = bn(x)
            x = lif(x.permute(2, 0, 1)).permute(1, 2, 0)
        x = self.pool(x).squeeze(-1)
        return self.head(x)
