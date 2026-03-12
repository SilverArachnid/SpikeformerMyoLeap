"""CNN-LSTM baseline for EMG-to-pose regression."""

from __future__ import annotations

from dataclasses import dataclass

from torch import Tensor, nn
import torch.nn.functional as F


@dataclass(frozen=True)
class CNNLSTMModelDefaults:
    """Reference default hyperparameters for the CNN-LSTM regressor."""

    input_channels: int = 8
    hidden_dim: int = 128
    num_layers: int = 2
    dropout: float = 0.1
    output_dim: int = 63


class EMGCNNLSTMRegressor(nn.Module):
    """Temporal CNN-LSTM regressor for EMG windows."""

    def __init__(
        self,
        input_channels: int = 8,
        hidden_dim: int = 128,
        num_layers: int = 2,
        dropout: float = 0.1,
        output_dim: int = 63,
    ) -> None:
        super().__init__()
        self.conv1 = nn.Conv1d(input_channels, 32, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm1d(32)
        self.conv2 = nn.Conv1d(32, 64, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm1d(64)
        self.pool = nn.MaxPool1d(2)
        self.lstm = nn.LSTM(input_size=64, hidden_size=hidden_dim, num_layers=num_layers, batch_first=True)
        self.dropout = nn.Dropout(dropout)
        self.head = nn.Linear(hidden_dim, output_dim)

    def forward(self, x: Tensor) -> Tensor:
        """Map EMG windows ``[B, T, C]`` to pose vectors ``[B, O]``."""

        x = x.permute(0, 2, 1)
        x = self.pool(F.relu(self.bn1(self.conv1(x))))
        x = self.pool(F.relu(self.bn2(self.conv2(x))))
        x = x.permute(0, 2, 1)
        x, _ = self.lstm(x)
        x = self.dropout(x[:, -1, :])
        return self.head(x)
