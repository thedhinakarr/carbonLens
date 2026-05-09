"""
LSTM model for short-term EU ETS price pattern prediction.

Architecture (from PRD §10.2):
  - 2-layer LSTM with dropout (0.2)
  - Fully connected dense layer → output
  - Input: sliding window of daily prices + technical indicators
  - Output: next-day and next-week price prediction
"""

import torch
import torch.nn as nn


class CarbonLSTM(nn.Module):
    """LSTM for short-term carbon price forecasting."""

    def __init__(
        self,
        input_size: int,
        hidden_size: int = 128,
        num_layers: int = 2,
        dropout: float = 0.2,
        output_size: int = 1,
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout,
            batch_first=True,
        )
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x shape: (batch, seq_len, input_size)
        lstm_out, _ = self.lstm(x)
        # Take the last time step
        last_hidden = lstm_out[:, -1, :]
        return self.fc(last_hidden)
