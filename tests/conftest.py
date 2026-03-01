"""Shared test fixtures for PINGU."""

import numpy as np
import pytest

from pingu.types import ReceiverConfig, IQFrame


@pytest.fixture
def rng():
    """Seeded random generator for reproducibility."""
    return np.random.default_rng(42)


@pytest.fixture
def sample_rate():
    return 48_000.0


@pytest.fixture
def pentagon_receivers() -> list[ReceiverConfig]:
    """5 receivers in a regular pentagon, ~100 km baseline."""
    R = 100_000.0  # 100 km radius
    receivers = []
    for i in range(5):
        angle = np.pi / 2 + 2 * np.pi * i / 5  # start at top
        receivers.append(
            ReceiverConfig(
                id=f"RX{i}",
                latitude=40.0 + 0.3 * np.sin(angle),
                longitude=-75.0 + 0.4 * np.cos(angle),
                x=R * np.cos(angle),
                y=R * np.sin(angle),
            )
        )
    return receivers


@pytest.fixture
def tone_1khz(sample_rate, rng) -> np.ndarray:
    """1 kHz complex tone, 0.1 s duration."""
    t = np.arange(int(0.1 * sample_rate)) / sample_rate
    return np.exp(2j * np.pi * 1000 * t).astype(np.complex64)


@pytest.fixture
def noise_signal(sample_rate, rng) -> np.ndarray:
    """Complex Gaussian noise, 0.1 s duration."""
    n = int(0.1 * sample_rate)
    return (rng.standard_normal(n) + 1j * rng.standard_normal(n)).astype(np.complex64) / np.sqrt(2)


@pytest.fixture
def sample_iq_frame(tone_1khz, sample_rate) -> IQFrame:
    """Sample IQ frame with a 1 kHz tone."""
    return IQFrame(
        receiver_id="RX0",
        samples=tone_1khz,
        sample_rate=sample_rate,
        center_freq=14.1e6,
        timestamp=0.0,
    )


@pytest.fixture
def tx_position() -> tuple[float, float]:
    """Known transmitter position (x, y) in meters."""
    return (30_000.0, -20_000.0)
