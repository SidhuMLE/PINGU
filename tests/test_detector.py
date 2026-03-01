"""Tests for the energy detector module."""

from __future__ import annotations

import numpy as np
import pytest

from pingu.detector.energy import EnergyDetector
from pingu.types import ChannelizedFrame


# ------------------------------------------------------------------
# Fixtures
# ------------------------------------------------------------------


@pytest.fixture
def detector() -> EnergyDetector:
    """Default energy detector with moderate Pfa."""
    return EnergyDetector(
        pfa=1e-3,
        block_size=256,
        guard_cells=2,
        reference_cells=8,
    )


@pytest.fixture
def sample_rate() -> float:
    return 48_000.0


def _make_frame(
    channels: np.ndarray,
    sample_rate: float = 48_000.0,
    n_channels: int | None = None,
) -> ChannelizedFrame:
    """Helper to wrap a channels array into a ChannelizedFrame."""
    if n_channels is None:
        n_channels = channels.shape[0]
    channel_bw = sample_rate / n_channels
    channel_freqs = np.arange(n_channels) * channel_bw
    return ChannelizedFrame(
        receiver_id="RX0",
        channels=channels,
        channel_freqs=channel_freqs,
        channel_bw=channel_bw,
        sample_rate=sample_rate,
        timestamp=0.0,
    )


# ------------------------------------------------------------------
# Tests
# ------------------------------------------------------------------


class TestEnergyDetector:
    """Tests for EnergyDetector."""

    def test_detect_tone_in_noise(self) -> None:
        """A signal burst in one channel should be detected by CFAR.

        CFAR operates within each channel across time blocks, so we inject
        a short burst (not a continuous tone) to ensure reference cells
        see only noise.
        """
        det = EnergyDetector(pfa=1e-3, block_size=128, guard_cells=1, reference_cells=8)
        rng = np.random.default_rng(123)
        n_channels = 8
        n_samples = 128 * 40  # 40 blocks per channel

        # All channels are noise
        noise_power = 1.0
        channels = (
            rng.standard_normal((n_channels, n_samples))
            + 1j * rng.standard_normal((n_channels, n_samples))
        ).astype(np.complex64) * np.sqrt(noise_power / 2)

        # Inject a strong burst into channel 3, blocks 20-22 only (~20 dB)
        tone_channel = 3
        tone_amplitude = 10.0
        for blk in range(20, 23):
            start = blk * 128
            end = start + 128
            t = np.arange(128) / 48_000.0
            tone = tone_amplitude * np.exp(2j * np.pi * 1000 * t).astype(np.complex64)
            channels[tone_channel, start:end] += tone

        frame = _make_frame(channels)
        detections = det.detect(frame)

        # At least the tone channel should be detected
        detected_channels = {d.channel_index for d in detections}
        assert tone_channel in detected_channels, (
            f"Expected channel {tone_channel} in detections, got {detected_channels}"
        )

    def test_no_detection_in_pure_noise(self, detector: EnergyDetector) -> None:
        """Pure noise with reasonable Pfa should yield no (or very few) detections."""
        rng = np.random.default_rng(456)
        n_channels = 16
        n_samples = 2048

        # Use a stricter Pfa for this test
        strict_detector = EnergyDetector(
            pfa=1e-6,
            block_size=256,
            guard_cells=2,
            reference_cells=8,
        )

        channels = (
            rng.standard_normal((n_channels, n_samples))
            + 1j * rng.standard_normal((n_channels, n_samples))
        ).astype(np.complex64) / np.sqrt(2)

        frame = _make_frame(channels)
        detections = strict_detector.detect(frame)

        # With Pfa = 1e-6 and 16 channels, false alarms should be extremely rare
        assert len(detections) <= 1, (
            f"Expected at most 1 false alarm, got {len(detections)}"
        )

    def test_snr_estimate_accuracy(self) -> None:
        """SNR estimate should be within a reasonable range of the true SNR."""
        det = EnergyDetector(pfa=1e-3, block_size=128, guard_cells=1, reference_cells=8)
        rng = np.random.default_rng(789)
        n_channels = 4
        n_blocks = 40
        n_samples = 128 * n_blocks

        true_snr_db = 15.0
        noise_power = 1.0

        channels = (
            rng.standard_normal((n_channels, n_samples))
            + 1j * rng.standard_normal((n_channels, n_samples))
        ).astype(np.complex64) * np.sqrt(noise_power / 2)

        # Inject a burst into channel 1 (blocks 18-22)
        signal_amplitude = np.sqrt(noise_power * 10 ** (true_snr_db / 10))
        for blk in range(18, 23):
            start = blk * 128
            end = start + 128
            t = np.arange(128) / 48_000.0
            sig = signal_amplitude * np.exp(2j * np.pi * 500 * t).astype(np.complex64)
            channels[1, start:end] += sig

        frame = _make_frame(channels, n_channels=n_channels)
        detections = det.detect(frame)

        # Find the detection for channel 1
        ch1_detections = [d for d in detections if d.channel_index == 1]
        assert len(ch1_detections) >= 1, f"Expected >=1 detection in channel 1, got {len(ch1_detections)}"

        estimated_snr = ch1_detections[0].snr_estimate
        # Allow generous tolerance — CFAR-based SNR estimates are inherently noisy
        assert estimated_snr > 5.0, (
            f"SNR estimate {estimated_snr:.1f} dB should be positive for a strong signal"
        )

    def test_threshold_computation(self) -> None:
        """Threshold should increase as Pfa decreases."""
        t1 = EnergyDetector._compute_threshold(pfa=1e-2, n_samples=256)
        t2 = EnergyDetector._compute_threshold(pfa=1e-4, n_samples=256)
        t3 = EnergyDetector._compute_threshold(pfa=1e-6, n_samples=256)
        assert t1 < t2 < t3, "Threshold should increase with stricter Pfa"

    def test_invalid_pfa_raises(self) -> None:
        """Invalid Pfa values should raise ValueError."""
        with pytest.raises(ValueError, match="pfa"):
            EnergyDetector(pfa=0.0)
        with pytest.raises(ValueError, match="pfa"):
            EnergyDetector(pfa=1.0)
        with pytest.raises(ValueError, match="pfa"):
            EnergyDetector(pfa=-0.1)
