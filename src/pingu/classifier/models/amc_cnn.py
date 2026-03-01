"""Convolutional Neural Network for Automatic Modulation Classification (AMC).

The architecture processes raw IQ time-series data (2-channel 1D input) through
a stack of Conv1d blocks and produces logits over the supported modulation types.
"""

from __future__ import annotations

import torch
import torch.nn as nn


class AMCCNN(nn.Module):
    """1-D CNN for automatic modulation classification.

    Input shape : ``(batch, 2, input_length)`` -- I and Q channels.
    Output shape: ``(batch, num_classes)``      -- raw logits.

    Architecture
    ------------
    Three convolutional stages (64 -> 128 -> 256 filters), each comprising
    Conv1d, BatchNorm1d, ReLU, and MaxPool1d. Followed by global average
    pooling, a fully-connected hidden layer, and a linear output head.

    Parameters
    ----------
    input_length : int
        Number of time-domain samples per IQ segment.
    num_classes : int
        Number of modulation classes (default 8).
    """

    # Default modulation class ordering consistent with pingu.types.ModulationType
    DEFAULT_CLASSES: list[str] = ["ssb", "cw", "am", "fsk2", "fsk4", "bpsk", "qpsk", "noise"]

    def __init__(self, input_length: int = 1024, num_classes: int = 8) -> None:
        super().__init__()
        self.input_length = input_length
        self.num_classes = num_classes

        # ---- Convolutional feature extractor ----
        self.features = nn.Sequential(
            # Block 1: 2 -> 64 filters
            nn.Conv1d(in_channels=2, out_channels=64, kernel_size=7, padding=3),
            nn.BatchNorm1d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(kernel_size=2),
            # Block 2: 64 -> 128 filters
            nn.Conv1d(in_channels=64, out_channels=128, kernel_size=5, padding=2),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(kernel_size=2),
            # Block 3: 128 -> 256 filters
            nn.Conv1d(in_channels=128, out_channels=256, kernel_size=3, padding=1),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(kernel_size=2),
        )

        # ---- Global average pooling ----
        self.global_pool = nn.AdaptiveAvgPool1d(1)

        # ---- Fully-connected classifier head ----
        self.classifier = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.3),
            nn.Linear(128, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Parameters
        ----------
        x : Tensor
            Input tensor of shape ``(batch, 2, input_length)``.

        Returns
        -------
        Tensor
            Logits of shape ``(batch, num_classes)``.
        """
        x = self.features(x)         # (B, 256, L')
        x = self.global_pool(x)      # (B, 256, 1)
        x = x.squeeze(-1)            # (B, 256)
        x = self.classifier(x)       # (B, num_classes)
        return x
