"""Shared dataclasses for the PINGU pipeline."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum

import numpy as np
from numpy.typing import NDArray


class ModulationType(Enum):
    """Supported modulation types."""
    SSB = "ssb"
    CW = "cw"
    AM = "am"
    FSK2 = "fsk2"
    FSK4 = "fsk4"
    BPSK = "bpsk"
    QPSK = "qpsk"
    NOISE = "noise"


@dataclass
class ReceiverConfig:
    """Configuration for a single receiver station."""
    id: str
    latitude: float          # degrees
    longitude: float         # degrees
    x: float = 0.0           # Cartesian x (m), computed from lat/lon or set directly
    y: float = 0.0           # Cartesian y (m)
    z: float = 0.0           # Cartesian z (m)
    clock_offset: float = 0.0  # Clock offset in seconds (for modeling imperfections)
    sample_rate: float = 48_000.0


@dataclass
class IQFrame:
    """A block of IQ samples from a single receiver."""
    receiver_id: str
    samples: NDArray[np.complex64]   # shape: (n_samples,)
    sample_rate: float               # Hz
    center_freq: float               # Hz
    timestamp: float                 # UNIX epoch seconds


@dataclass
class ChannelizedFrame:
    """Output of the channelizer: narrowband channels from one receiver."""
    receiver_id: str
    channels: NDArray[np.complex64]  # shape: (n_channels, n_samples)
    channel_freqs: NDArray[np.float64]  # center freq of each channel (Hz)
    channel_bw: float                # bandwidth per channel (Hz)
    sample_rate: float               # per-channel sample rate (Hz)
    timestamp: float


@dataclass
class Detection:
    """A detected signal in a specific channel."""
    receiver_id: str
    channel_index: int
    center_freq: float         # Hz
    bandwidth: float           # Hz
    snr_estimate: float        # dB
    modulation: ModulationType | None = None
    confidence: float = 0.0    # classification confidence [0, 1]
    timestamp: float = 0.0


@dataclass
class TDoAEstimate:
    """Time-difference-of-arrival between a pair of receivers."""
    receiver_i: str
    receiver_j: str
    delay: float              # seconds (τ_i - τ_j)
    variance: float           # seconds² (estimated uncertainty)
    correlation_peak: float   # peak value of cross-correlation [0, 1]
    center_freq: float        # Hz, frequency of the signal used
    timestamp: float = 0.0


@dataclass
class IntegratedTDoA:
    """Bayesian-integrated TDoA estimates across time."""
    delays: NDArray[np.float64]       # shape: (n_pairs,) — filtered TDoA vector
    covariance: NDArray[np.float64]   # shape: (n_pairs, n_pairs)
    pair_labels: list[tuple[str, str]]  # receiver ID pairs
    n_updates: int = 0
    timestamp: float = 0.0


@dataclass
class PositionEstimate:
    """Estimated transmitter position with uncertainty."""
    x: float                          # meters
    y: float                          # meters
    z: float = 0.0                    # meters
    latitude: float = 0.0            # degrees
    longitude: float = 0.0           # degrees
    covariance: NDArray[np.float64] = field(
        default_factory=lambda: np.eye(2) * 1e6
    )  # 2x2 position covariance (m²)
    residual: float = 0.0            # cost function residual
    confidence_radius_95: float = 0.0  # 95% confidence radius (m)
    timestamp: float = 0.0
