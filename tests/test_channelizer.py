"""Tests for the channelizer module."""

from __future__ import annotations

import numpy as np
import pytest

from pingu.channelizer.fft import FFTChannelizer
from pingu.channelizer.polyphase import PolyphaseChannelizer
from pingu.types import ChannelizedFrame, IQFrame


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_tone_frame(
    freq_offset: float,
    sample_rate: float = 48_000.0,
    center_freq: float = 14.1e6,
    duration: float = 0.1,
) -> IQFrame:
    """Create an IQFrame containing a single complex tone.

    Args:
        freq_offset: Tone frequency offset from center_freq (Hz).
        sample_rate: Sample rate in Hz.
        center_freq: Center frequency in Hz.
        duration: Duration in seconds.

    Returns:
        IQFrame with the tone.
    """
    n_samples = int(duration * sample_rate)
    t = np.arange(n_samples) / sample_rate
    samples = np.exp(2j * np.pi * freq_offset * t).astype(np.complex64)
    return IQFrame(
        receiver_id="RX0",
        samples=samples,
        sample_rate=sample_rate,
        center_freq=center_freq,
        timestamp=0.0,
    )


def _make_noise_frame(
    sample_rate: float = 48_000.0,
    center_freq: float = 14.1e6,
    duration: float = 0.1,
    rng: np.random.Generator | None = None,
) -> IQFrame:
    """Create an IQFrame containing complex Gaussian noise."""
    if rng is None:
        rng = np.random.default_rng(123)
    n_samples = int(duration * sample_rate)
    noise = (
        rng.standard_normal(n_samples) + 1j * rng.standard_normal(n_samples)
    ).astype(np.complex64) / np.sqrt(2)
    return IQFrame(
        receiver_id="RX0",
        samples=noise,
        sample_rate=sample_rate,
        center_freq=center_freq,
        timestamp=0.0,
    )


# ---------------------------------------------------------------------------
# PolyphaseChannelizer tests
# ---------------------------------------------------------------------------

class TestPolyphaseChannelizer:
    """Tests for PolyphaseChannelizer."""

    def test_output_shape(self) -> None:
        """Channelized output should have shape (n_channels, n_samples)."""
        n_channels = 8
        ch = PolyphaseChannelizer(n_channels=n_channels)
        frame = _make_tone_frame(freq_offset=1000.0)

        result = ch.channelize(frame)

        assert isinstance(result, ChannelizedFrame)
        assert result.channels.shape[0] == n_channels
        # Per-channel sample count = ceil(n_input / M)
        expected_samples = int(np.ceil(len(frame.samples) / n_channels))
        assert result.channels.shape[1] == expected_samples

    def test_channel_freqs_length(self) -> None:
        """channel_freqs should have one entry per channel."""
        n_channels = 16
        ch = PolyphaseChannelizer(n_channels=n_channels)
        frame = _make_tone_frame(freq_offset=0.0)
        result = ch.channelize(frame)

        assert len(result.channel_freqs) == n_channels

    def test_tone_appears_in_correct_channel(self) -> None:
        """A tone at a known frequency offset should land in the expected channel.

        The polyphase FFT output is in standard FFT order:
        channel k maps to frequency offset k * (fs/M) for k=0..M/2-1
        and (k-M) * (fs/M) for k=M/2..M-1.
        """
        n_channels = 16
        sample_rate = 48_000.0
        channel_bw = sample_rate / n_channels  # 3000 Hz

        # +3000 Hz → FFT bin 1
        freq_offset = 3000.0
        expected_channel = round(freq_offset / channel_bw) % n_channels

        ch = PolyphaseChannelizer(n_channels=n_channels, overlap_factor=4)
        frame = _make_tone_frame(freq_offset=freq_offset, sample_rate=sample_rate)
        result = ch.channelize(frame)

        energy_per_channel = np.sum(np.abs(result.channels) ** 2, axis=1)
        peak_channel = int(np.argmax(energy_per_channel))

        assert peak_channel == expected_channel, (
            f"Expected tone in channel {expected_channel}, "
            f"but peak energy was in channel {peak_channel}"
        )

    def test_output_dtype_is_complex64(self) -> None:
        """Output channel samples should be complex64."""
        ch = PolyphaseChannelizer(n_channels=8)
        frame = _make_tone_frame(freq_offset=0.0)
        result = ch.channelize(frame)
        assert result.channels.dtype == np.complex64

    def test_properties(self) -> None:
        """n_channels and channel_bandwidth should reflect constructor args."""
        ch = PolyphaseChannelizer(n_channels=16, overlap_factor=8, window="blackman")
        assert ch.n_channels == 16
        assert ch.channel_bandwidth == pytest.approx(1.0 / 16)

    def test_invalid_n_channels(self) -> None:
        """n_channels < 2 should raise ValueError."""
        with pytest.raises(ValueError, match="n_channels"):
            PolyphaseChannelizer(n_channels=1)

    def test_invalid_overlap_factor(self) -> None:
        """overlap_factor < 1 should raise ValueError."""
        with pytest.raises(ValueError, match="overlap_factor"):
            PolyphaseChannelizer(n_channels=8, overlap_factor=0)

    @pytest.mark.parametrize("freq_offset,expected_channel", [
        (+2000.0, 1),   # Between ch0 (0 Hz) and ch1 (3000 Hz), closer to ch1
        (+8000.0, 3),   # Between ch2 (6000 Hz) and ch3 (9000 Hz), closer to ch3
        (-5000.0, 14),  # Between ch14 (-6000 Hz) and ch15 (-3000 Hz), closer to ch14
    ])
    def test_off_center_tone_maps_to_nearest_channel(
        self, freq_offset: float, expected_channel: int
    ) -> None:
        """Tones between channel centers should map to the nearest channel."""
        n_channels = 16
        sample_rate = 48_000.0

        ch = PolyphaseChannelizer(n_channels=n_channels, overlap_factor=4)
        frame = _make_tone_frame(
            freq_offset=freq_offset, sample_rate=sample_rate, duration=0.05
        )
        result = ch.channelize(frame)

        energy = np.sum(np.abs(result.channels.astype(np.complex128)) ** 2, axis=1)
        peak_channel = int(np.argmax(energy))

        assert peak_channel == expected_channel, (
            f"Tone at {freq_offset:+.0f} Hz: expected channel {expected_channel}, "
            f"got channel {peak_channel}"
        )

    def test_three_tones_detected_in_correct_channels(self) -> None:
        """Three simultaneous tones should each appear in the correct channel."""
        n_channels = 16
        sample_rate = 48_000.0
        channel_bw = sample_rate / n_channels
        n_samples = int(0.05 * sample_rate)
        t = np.arange(n_samples) / sample_rate

        offsets = [+2000.0, +8000.0, -5000.0]
        signal = sum(np.exp(2j * np.pi * f * t) for f in offsets)
        signal = signal.astype(np.complex64)

        frame = IQFrame(
            receiver_id="RX0", samples=signal,
            sample_rate=sample_rate, center_freq=14.1e6, timestamp=0.0,
        )

        ch = PolyphaseChannelizer(n_channels=n_channels, overlap_factor=4)
        result = ch.channelize(frame)

        energy = np.sum(np.abs(result.channels.astype(np.complex128)) ** 2, axis=1)
        top3 = set(np.argsort(energy)[-3:])

        expected = {
            round(f / channel_bw) % n_channels for f in offsets
        }  # {1, 3, 14}
        assert top3 == expected, f"Top-3 channels {top3} != expected {expected}"


# ---------------------------------------------------------------------------
# FFTChannelizer tests
# ---------------------------------------------------------------------------

class TestFFTChannelizer:
    """Tests for FFTChannelizer."""

    def test_output_shape(self) -> None:
        """FFT channelizer output should have n_channels rows."""
        n_channels = 8
        ch = FFTChannelizer(n_channels=n_channels)
        frame = _make_tone_frame(freq_offset=1000.0)
        result = ch.channelize(frame)

        assert isinstance(result, ChannelizedFrame)
        assert result.channels.shape[0] == n_channels
        # Output should have a positive number of samples per channel.
        assert result.channels.shape[1] > 0

    def test_tone_appears_in_correct_channel(self) -> None:
        """A tone should produce peak energy in the expected FFT channel.

        FFT bin ordering: bin k maps to frequency offset k * (fs/N_fft).
        Channels group consecutive bins, so channel c gets bins
        [c*bins_per_ch, (c+1)*bins_per_ch).
        """
        n_channels = 8
        sample_rate = 48_000.0
        channel_bw = sample_rate / n_channels  # 6000 Hz

        # Tone at +3000 Hz offset → falls in FFT bin corresponding to channel 0
        # (bins 0..bins_per_ch-1 cover 0..channel_bw, tone at +3000 is in middle)
        freq_offset = 3000.0

        ch = FFTChannelizer(n_channels=n_channels)
        frame = _make_tone_frame(freq_offset=freq_offset, sample_rate=sample_rate)
        result = ch.channelize(frame)

        energy_per_channel = np.sum(np.abs(result.channels) ** 2, axis=1)
        peak_channel = int(np.argmax(energy_per_channel))

        # The FFT groups bins 0..bins_per_ch-1 into channel 0, etc.
        # +3000 Hz is bin index = 3000 / (fs/fft_size), belonging to channel 0
        # Just verify the peak channel is consistent
        expected_channel = int(freq_offset / channel_bw) % n_channels

        assert peak_channel == expected_channel, (
            f"Expected tone in channel {expected_channel}, "
            f"but peak energy was in channel {peak_channel}"
        )

    def test_properties(self) -> None:
        """Properties should reflect constructor parameters."""
        ch = FFTChannelizer(n_channels=16, fft_size=128, overlap=32)
        assert ch.n_channels == 16
        assert ch.channel_bandwidth == pytest.approx(1.0 / 16)

    def test_invalid_n_channels(self) -> None:
        """n_channels < 2 should raise ValueError."""
        with pytest.raises(ValueError, match="n_channels"):
            FFTChannelizer(n_channels=0)

    def test_invalid_fft_size(self) -> None:
        """fft_size < n_channels should raise ValueError."""
        with pytest.raises(ValueError, match="fft_size"):
            FFTChannelizer(n_channels=16, fft_size=8)

    def test_invalid_overlap(self) -> None:
        """overlap >= fft_size should raise ValueError."""
        with pytest.raises(ValueError, match="overlap"):
            FFTChannelizer(n_channels=8, fft_size=32, overlap=32)

    def test_output_dtype_is_complex64(self) -> None:
        """Output should be complex64."""
        ch = FFTChannelizer(n_channels=8)
        frame = _make_tone_frame(freq_offset=0.0)
        result = ch.channelize(frame)
        assert result.channels.dtype == np.complex64


# ---------------------------------------------------------------------------
# Cross-implementation tests
# ---------------------------------------------------------------------------

class TestChannelizerComparison:
    """Compare polyphase and FFT channelizer outputs."""

    def test_same_number_of_channels(self) -> None:
        """Both channelizers should produce the same number of channels."""
        n_channels = 8
        poly = PolyphaseChannelizer(n_channels=n_channels)
        fft_ch = FFTChannelizer(n_channels=n_channels)
        frame = _make_tone_frame(freq_offset=2000.0)

        r_poly = poly.channelize(frame)
        r_fft = fft_ch.channelize(frame)

        assert r_poly.channels.shape[0] == r_fft.channels.shape[0] == n_channels

    def test_peak_channel_agreement(self) -> None:
        """Both implementations should identify the same peak-energy channel for a tone."""
        n_channels = 8
        sample_rate = 48_000.0
        freq_offset = 6000.0

        poly = PolyphaseChannelizer(n_channels=n_channels)
        fft_ch = FFTChannelizer(n_channels=n_channels)
        frame = _make_tone_frame(
            freq_offset=freq_offset, sample_rate=sample_rate
        )

        r_poly = poly.channelize(frame)
        r_fft = fft_ch.channelize(frame)

        peak_poly = int(np.argmax(np.sum(np.abs(r_poly.channels) ** 2, axis=1)))
        peak_fft = int(np.argmax(np.sum(np.abs(r_fft.channels) ** 2, axis=1)))

        assert peak_poly == peak_fft


# ---------------------------------------------------------------------------
# Energy / near-perfect reconstruction test
# ---------------------------------------------------------------------------

class TestEnergyConservation:
    """Test that channelization approximately conserves total signal energy."""

    def test_polyphase_energy_conservation(self) -> None:
        """Sum of channel energies should be close to input energy (polyphase)."""
        n_channels = 8
        ch = PolyphaseChannelizer(n_channels=n_channels, overlap_factor=4)
        frame = _make_noise_frame(duration=0.05)

        result = ch.channelize(frame)

        input_energy = float(np.sum(np.abs(frame.samples) ** 2))
        output_energy = float(np.sum(np.abs(result.channels) ** 2))

        # Allow generous tolerance; perfect reconstruction is not guaranteed
        # but total energy should be within the same order of magnitude.
        ratio = output_energy / input_energy
        assert 0.01 < ratio < 100.0, (
            f"Energy ratio {ratio:.4f} outside acceptable range (0.01, 100.0)"
        )

    def test_fft_energy_conservation(self) -> None:
        """Sum of channel energies should be close to input energy (FFT)."""
        n_channels = 8
        ch = FFTChannelizer(n_channels=n_channels)
        frame = _make_noise_frame(duration=0.05)

        result = ch.channelize(frame)

        input_energy = float(np.sum(np.abs(frame.samples) ** 2))
        output_energy = float(np.sum(np.abs(result.channels) ** 2))

        ratio = output_energy / input_energy
        assert 0.1 < ratio < 10.0, (
            f"Energy ratio {ratio:.4f} outside acceptable range (0.1, 10.0)"
        )
