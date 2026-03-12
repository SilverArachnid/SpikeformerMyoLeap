"""Spikeformer-based EMG-to-pose regression model."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor, nn

try:
    from spikingjelly.clock_driven import neuron
except ImportError as exc:  # pragma: no cover - depends on local environment
    raise ImportError(
        "spikingjelly is required for Spikeformer and spiking CNN models. "
        "Install it in the project environment before training."
    ) from exc


def get_backend_neuron(backend: str) -> type[nn.Module]:
    """Return the spiking neuron implementation for the requested backend."""

    if backend not in {"cupy", "torch"}:
        raise ValueError(f"Unsupported backend: {backend}. Expected 'cupy' or 'torch'.")
    return neuron.MultiStepLIFNode


def apply_lif_over_sequence(x: Tensor, lif: nn.Module) -> Tensor:
    """Apply a multi-step spiking neuron over the temporal axis of ``[B, C, T]``."""

    return lif(x.permute(2, 0, 1)).permute(1, 2, 0)


class SPS1D(nn.Module):
    """Two-stage spiking patch stem for 1D EMG signals."""

    def __init__(self, in_channels: int = 8, embed_dim: int = 64, backend: str = "cupy") -> None:
        super().__init__()
        lif_neuron = get_backend_neuron(backend)

        self.conv1 = nn.Conv1d(in_channels, embed_dim // 2, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm1d(embed_dim // 2)
        self.lif1 = lif_neuron(tau=2.0, detach_reset=True, backend=backend)

        self.conv2 = nn.Conv1d(embed_dim // 2, embed_dim, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm1d(embed_dim)
        self.lif2 = lif_neuron(tau=2.0, detach_reset=True, backend=backend)

    def forward(self, x: Tensor) -> Tensor:
        """Encode input EMG patches from ``[B, C, T]`` to ``[B, E, T]``."""

        x = self.conv1(x)
        x = self.bn1(x)
        x = apply_lif_over_sequence(x, self.lif1)
        x = self.conv2(x)
        x = self.bn2(x)
        x = apply_lif_over_sequence(x, self.lif2)
        return x


class ConditionalPositionalEncoding1D(nn.Module):
    """Depthwise local positional encoding for 1D EMG sequences."""

    def __init__(self, embed_dim: int, backend: str = "cupy") -> None:
        super().__init__()
        lif_neuron = get_backend_neuron(backend)
        self.conv = nn.Conv1d(embed_dim, embed_dim, kernel_size=3, padding=1, groups=embed_dim)
        self.bn = nn.BatchNorm1d(embed_dim)
        self.lif = lif_neuron(tau=2.0, detach_reset=True, backend=backend)

    def forward(self, x: Tensor) -> Tensor:
        """Apply local positional encoding to ``[B, E, T]`` features."""

        residual = x
        x = self.conv(x)
        x = self.bn(x)
        x = apply_lif_over_sequence(x, self.lif)
        return x + residual


class SpikeSelfAttentionBlock(nn.Module):
    """Spike-friendly multi-head self-attention block.

    The old implementation stored ``heads`` but never used it. This implementation
    performs an explicit head split over the embedding dimension, computes the
    spike-friendly ``K^T V`` summary per head, and applies it to per-head queries.
    """

    def __init__(self, embed_dim: int, heads: int, backend: str = "cupy") -> None:
        super().__init__()
        if embed_dim % heads != 0:
            raise ValueError(f"embed_dim ({embed_dim}) must be divisible by heads ({heads}).")

        lif_neuron = get_backend_neuron(backend)
        self.embed_dim = embed_dim
        self.heads = heads
        self.head_dim = embed_dim // heads

        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.k_proj = nn.Linear(embed_dim, embed_dim)
        self.v_proj = nn.Linear(embed_dim, embed_dim)
        self.out_proj = nn.Linear(embed_dim, embed_dim)

        self.q_bn = nn.BatchNorm1d(embed_dim)
        self.k_bn = nn.BatchNorm1d(embed_dim)
        self.v_bn = nn.BatchNorm1d(embed_dim)
        self.out_bn = nn.BatchNorm1d(embed_dim)

        self.q_lif = lif_neuron(tau=2.0, detach_reset=True, backend=backend)
        self.k_lif = lif_neuron(tau=2.0, detach_reset=True, backend=backend)
        self.v_lif = lif_neuron(tau=2.0, detach_reset=True, backend=backend)
        self.out_lif = lif_neuron(tau=2.0, detach_reset=True, backend=backend)

    def _project(self, x: Tensor, linear: nn.Linear, bn: nn.BatchNorm1d, lif: nn.Module) -> Tensor:
        """Project and spike-normalize ``[T, B, E]`` inputs."""

        x = linear(x)
        x = bn(x.permute(1, 2, 0)).permute(2, 0, 1)
        return lif(x)

    def _split_heads(self, x: Tensor) -> Tensor:
        """Reshape ``[T, B, E]`` into ``[T, B, H, D]``."""

        time_steps, batch_size, _ = x.shape
        return x.reshape(time_steps, batch_size, self.heads, self.head_dim)

    def forward(self, x: Tensor) -> Tensor:
        """Apply multi-head spike self-attention to ``[T, B, E]`` features."""

        q = self._split_heads(self._project(x, self.q_proj, self.q_bn, self.q_lif))
        k = self._split_heads(self._project(x, self.k_proj, self.k_bn, self.k_lif))
        v = self._split_heads(self._project(x, self.v_proj, self.v_bn, self.v_lif))

        kv_summary = torch.einsum("tbhd,tbhe->bhde", k, v)
        attended = torch.einsum("tbhd,bhde->tbhe", q, kv_summary)
        attended = attended.reshape(x.shape[0], x.shape[1], self.embed_dim)

        out = self.out_proj(attended)
        out = self.out_bn(out.permute(1, 2, 0)).permute(2, 0, 1)
        out = self.out_lif(out)
        return x + out


class SpikeformerBlock(nn.Module):
    """Spikeformer block with spike attention and a spiking MLP."""

    def __init__(self, embed_dim: int, heads: int, mlp_ratio: float = 2.0, backend: str = "cupy") -> None:
        super().__init__()
        hidden_dim = int(embed_dim * mlp_ratio)
        lif_neuron = get_backend_neuron(backend)

        self.attention = SpikeSelfAttentionBlock(embed_dim=embed_dim, heads=heads, backend=backend)
        self.fc1 = nn.Linear(embed_dim, hidden_dim)
        self.bn1 = nn.BatchNorm1d(hidden_dim)
        self.lif1 = lif_neuron(tau=2.0, detach_reset=True, backend=backend)

        self.fc2 = nn.Linear(hidden_dim, embed_dim)
        self.bn2 = nn.BatchNorm1d(embed_dim)
        self.lif2 = lif_neuron(tau=2.0, detach_reset=True, backend=backend)

    def forward(self, x: Tensor) -> Tensor:
        """Apply spike attention followed by a spiking MLP."""

        x = self.attention(x)
        time_steps, batch_size, channels = x.shape

        y = x.permute(1, 0, 2).reshape(batch_size * time_steps, channels)
        y = self.fc1(y)
        y = self.bn1(y.view(batch_size, time_steps, -1).permute(0, 2, 1))
        y = self.lif1(y.permute(2, 0, 1))

        y = self.fc2(y.permute(1, 0, 2).reshape(batch_size * time_steps, -1))
        y = self.bn2(y.view(batch_size, time_steps, -1).permute(0, 2, 1))
        y = self.lif2(y.permute(2, 0, 1))
        return x + y


@dataclass(frozen=True)
class SpikeformerModelDefaults:
    """Reference default hyperparameters for the Spikeformer regressor."""

    input_dim: int = 8
    embed_dim: int = 64
    num_blocks: int = 4
    heads: int = 4
    output_dim: int = 63
    mlp_ratio: float = 2.0
    backend: str = "cupy"


class EMGSpikeformerWindowRegressor(nn.Module):
    """Windowed Spikeformer regressor for EMG-to-pose prediction."""

    def __init__(
        self,
        input_dim: int = 8,
        embed_dim: int = 64,
        num_blocks: int = 4,
        heads: int = 4,
        output_dim: int = 63,
        mlp_ratio: float = 2.0,
        backend: str = "cupy",
    ) -> None:
        super().__init__()
        self.sps = SPS1D(in_channels=input_dim, embed_dim=embed_dim, backend=backend)
        self.cpe = ConditionalPositionalEncoding1D(embed_dim=embed_dim, backend=backend)
        self.blocks = nn.ModuleList(
            SpikeformerBlock(embed_dim=embed_dim, heads=heads, mlp_ratio=mlp_ratio, backend=backend)
            for _ in range(num_blocks)
        )
        self.head = nn.Linear(embed_dim, output_dim)

    def forward(self, x: Tensor) -> Tensor:
        """Map EMG windows ``[B, T, C]`` to pose vectors ``[B, O]``."""

        x = x.permute(0, 2, 1)
        x = self.sps(x)
        x = self.cpe(x)
        x = x.permute(2, 0, 1)

        for block in self.blocks:
            x = block(x)

        return self.head(x[-1])
