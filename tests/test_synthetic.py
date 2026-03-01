"""Unit tests for the PINGU synthetic data engine.

Tests cover:
  - Signal generators: correct output length, dtype, and spectral properties
  - Noise utilities: SNR accuracy within 1 dB tolerance
  - TDoA scenario: correct differential delays via cross-correlation
"""

from __future__ import annotations

import numpy as np
import pytest
from scipy.signal import correlate

from pingu.types import ReceiverConfig, ModulationType
from pingu.constants import SPEED_OF_LIGHT
from pingu.synthetic.signals import (
    generate_ssb,
    generate_cw,
    generate_am,
    generate_fsk2,
    generate_fsk4,
    generate_bpsk,
    generate_qpsk,
    generate_signal,
    GENERATORS,
)
from pingu.synthetic.noise import add_awgn, generate_noise
from pingu.synthetic.scenarios import TDoAScenario


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

SAMPLE_RATE = 48_000.0
DURATION = 0.1
N_SAMPLES = int(SAMPLE_RATE * DURATION)  # 4800


@pytest.fixture
def rng():
    """Seeded random generator for reproducibility."""
    return np.random.default_rng(12345)


# ---------------------------------------------------------------------------
# Signal generator tests — length and dtype
# ---------------------------------------------------------------------------

class TestSignalGeneratorsBasic:
    """Verify that every generator returns the correct length and dtype."""

    @pytest.mark.parametrize(
        "gen_func",
        [generate_ssb, generate_cw, generate_am, generate_fsk2, generate_fsk4,
         generate_bpsk, generate_qpsk],
        ids=["ssb", "cw", "am", "fsk2", "fsk4", "bpsk", "qpsk"],
    )
    def test_output_length_and_dtype(self, gen_func, rng):
        sig = gen_func(sample_rate=SAMPLE_RATE, duration=DURATION, rng=rng)
        assert sig.dtype == np.complex64, f"Expected complex64, got {sig.dtype}"
        assert len(sig) == N_SAMPLES, f"Expected {N_SAMPLES} samples, got {len(sig)}"

    @pytest.mark.parametrize(
        "gen_func",
        [generate_ssb, generate_cw, generate_am, generate_fsk2, generate_fsk4,
         generate_bpsk, generate_qpsk],
        ids=["ssb", "cw", "am", "fsk2", "fsk4", "bpsk", "qpsk"],
    )
    def test_unit_power(self, gen_func, rng):
        """Signals should be normalised to approximately unit power."""
        sig = gen_func(sample_rate=SAMPLE_RATE, duration=DURATION, rng=rng)
        power = np.mean(np.abs(sig.astype(np.complex128)) ** 2)
        assert 0.5 < power < 2.0, f"Power {power:.4f} is far from unity"

    def test_generate_signal_dispatch(self, rng):
        """generate_signal should dispatch to the correct generator."""
        for mod_type in GENERATORS:
            sig = generate_signal(
                modulation=mod_type,
                sample_rate=SAMPLE_RATE,
                duration=DURATION,
                rng=rng,
            )
            assert sig.dtype == np.complex64
            assert len(sig) == N_SAMPLES

    def test_generate_signal_invalid_modulation(self):
        """generate_signal should raise ValueError for unsupported modulation."""
        with pytest.raises(ValueError, match="No generator"):
            generate_signal(modulation=ModulationType.NOISE)


# ---------------------------------------------------------------------------
# Signal generator tests — spectral properties
# ---------------------------------------------------------------------------

class TestSignalSpectral:
    """Verify that generated signals have expected spectral characteristics."""

    def test_cw_peak_frequency(self, rng):
        """CW tone energy should peak at the specified tone frequency."""
        tone_freq = 1_000.0
        sig = generate_cw(
            sample_rate=SAMPLE_RATE, duration=DURATION,
            tone_freq=tone_freq, rng=rng,
        )
        spectrum = np.abs(np.fft.fft(sig))
        freqs = np.fft.fftfreq(len(sig), d=1.0 / SAMPLE_RATE)
        peak_idx = np.argmax(spectrum)
        peak_freq = freqs[peak_idx]
        # The peak should be within 1 FFT bin of the expected frequency
        freq_resolution = SAMPLE_RATE / len(sig)
        assert abs(peak_freq - tone_freq) <= freq_resolution, (
            f"CW peak at {peak_freq:.1f} Hz, expected ~{tone_freq:.1f} Hz"
        )

    def test_cw_with_center_freq(self, rng):
        """CW with a centre-frequency offset should shift the peak."""
        tone_freq = 600.0
        center_freq = 2_000.0
        expected_freq = tone_freq + center_freq
        sig = generate_cw(
            sample_rate=SAMPLE_RATE, duration=DURATION,
            center_freq=center_freq, tone_freq=tone_freq, rng=rng,
        )
        spectrum = np.abs(np.fft.fft(sig))
        freqs = np.fft.fftfreq(len(sig), d=1.0 / SAMPLE_RATE)
        peak_idx = np.argmax(spectrum)
        peak_freq = freqs[peak_idx]
        freq_resolution = SAMPLE_RATE / len(sig)
        assert abs(peak_freq - expected_freq) <= freq_resolution

    def test_fsk2_has_two_dominant_tones(self, rng):
        """2-FSK should have energy concentrated around two tone frequencies."""
        tone_spacing = 500.0
        sig = generate_fsk2(
            sample_rate=SAMPLE_RATE, duration=0.5,
            tone_spacing=tone_spacing, symbol_rate=50.0, rng=rng,
        )
        spectrum = np.abs(np.fft.fft(sig)) ** 2
        freqs = np.fft.fftfreq(len(sig), d=1.0 / SAMPLE_RATE)

        # Expected tones at +/- tone_spacing/2
        expected_tones = [-tone_spacing / 2.0, tone_spacing / 2.0]
        freq_res = SAMPLE_RATE / len(sig)

        for tone in expected_tones:
            # Find energy near each expected tone (within +/- 50 Hz)
            mask = np.abs(freqs - tone) < 50.0
            tone_energy = np.sum(spectrum[mask])
            total_energy = np.sum(spectrum)
            # Each tone should carry a meaningful fraction of total energy
            assert tone_energy / total_energy > 0.05, (
                f"Too little energy near {tone:.0f} Hz"
            )

    def test_fsk4_has_four_dominant_tones(self, rng):
        """4-FSK should have energy concentrated around four tone frequencies."""
        tone_spacing = 200.0
        sig = generate_fsk4(
            sample_rate=SAMPLE_RATE, duration=0.5,
            tone_spacing=tone_spacing, symbol_rate=50.0, rng=rng,
        )
        spectrum = np.abs(np.fft.fft(sig)) ** 2
        freqs = np.fft.fftfreq(len(sig), d=1.0 / SAMPLE_RATE)

        # Expected tones at -1.5*sp, -0.5*sp, +0.5*sp, +1.5*sp
        expected_tones = [
            -1.5 * tone_spacing,
            -0.5 * tone_spacing,
            0.5 * tone_spacing,
            1.5 * tone_spacing,
        ]
        for tone in expected_tones:
            mask = np.abs(freqs - tone) < 50.0
            tone_energy = np.sum(spectrum[mask])
            total_energy = np.sum(spectrum)
            assert tone_energy / total_energy > 0.02, (
                f"Too little energy near {tone:.0f} Hz"
            )

    def test_am_carrier_present(self, rng):
        """AM signal should have a carrier component at center_freq."""
        center_freq = 1_000.0
        sig = generate_am(
            sample_rate=SAMPLE_RATE, duration=DURATION,
            center_freq=center_freq, mod_freq=300.0, mod_index=0.5, rng=rng,
        )
        spectrum = np.abs(np.fft.fft(sig))
        freqs = np.fft.fftfreq(len(sig), d=1.0 / SAMPLE_RATE)
        peak_idx = np.argmax(spectrum)
        peak_freq = freqs[peak_idx]
        freq_res = SAMPLE_RATE / len(sig)
        assert abs(peak_freq - center_freq) <= freq_res

    def test_cw_keying_pattern(self, rng):
        """CW with keying should have zero-amplitude segments."""
        # Pattern: 0.02s ON, 0.03s OFF, 0.05s ON
        keying = [0.02, 0.03, 0.05]
        sig = generate_cw(
            sample_rate=SAMPLE_RATE, duration=DURATION,
            keying_pattern=keying, rng=rng,
        )
        # The OFF segment (samples 960..2400 approximately) should be near zero
        off_start = int(0.02 * SAMPLE_RATE)
        off_end = int(0.05 * SAMPLE_RATE)
        off_power = np.mean(np.abs(sig[off_start:off_end].astype(np.complex128)) ** 2)
        total_power = np.mean(np.abs(sig.astype(np.complex128)) ** 2)
        # Off-segment power should be much less than total
        assert off_power < 0.01 * total_power, (
            f"Off-segment power {off_power:.6f} should be near zero"
        )

    def test_ssb_sideband_selection(self, rng):
        """USB should have positive-frequency energy; LSB should have negative."""
        usb = generate_ssb(
            sample_rate=SAMPLE_RATE, duration=DURATION, sideband="usb", rng=rng,
        )
        lsb = generate_ssb(
            sample_rate=SAMPLE_RATE, duration=DURATION, sideband="lsb", rng=rng,
        )
        # Compare energy in positive vs negative frequencies
        n = len(usb)
        usb_spec = np.abs(np.fft.fft(usb)) ** 2
        lsb_spec = np.abs(np.fft.fft(lsb)) ** 2

        pos_bins = slice(1, n // 2)
        neg_bins = slice(n // 2 + 1, n)

        usb_pos_energy = np.sum(usb_spec[pos_bins])
        usb_neg_energy = np.sum(usb_spec[neg_bins])
        assert usb_pos_energy > usb_neg_energy, "USB should have more positive-freq energy"

        lsb_pos_energy = np.sum(lsb_spec[pos_bins])
        lsb_neg_energy = np.sum(lsb_spec[neg_bins])
        assert lsb_neg_energy > lsb_pos_energy, "LSB should have more negative-freq energy"


# ---------------------------------------------------------------------------
# Noise tests
# ---------------------------------------------------------------------------

class TestNoise:
    """Verify AWGN noise generation and SNR accuracy."""

    def test_generate_noise_length_and_dtype(self, rng):
        noise = generate_noise(1000, rng=rng)
        assert noise.dtype == np.complex64
        assert len(noise) == 1000

    def test_generate_noise_unit_power(self, rng):
        """Generated noise should have approximately unit power."""
        noise = generate_noise(100_000, rng=rng)
        power = np.mean(np.abs(noise.astype(np.complex128)) ** 2)
        assert abs(power - 1.0) < 0.05, f"Noise power {power:.4f}, expected ~1.0"

    @pytest.mark.parametrize("target_snr_db", [0.0, 10.0, 20.0, -5.0])
    def test_add_awgn_snr_accuracy(self, target_snr_db, rng):
        """add_awgn should produce the target SNR within 1 dB tolerance."""
        # Use a long CW tone for stable power estimation
        sig = generate_cw(sample_rate=SAMPLE_RATE, duration=1.0, rng=rng)
        noisy = add_awgn(sig, target_snr_db, rng=rng)

        # Estimate achieved SNR
        sig_f64 = sig.astype(np.complex128)
        noise_only = noisy.astype(np.complex128) - sig_f64
        sig_power = np.mean(np.abs(sig_f64) ** 2)
        noise_power = np.mean(np.abs(noise_only) ** 2)
        achieved_snr_db = 10.0 * np.log10(sig_power / noise_power)

        assert abs(achieved_snr_db - target_snr_db) < 1.0, (
            f"Target SNR {target_snr_db:.1f} dB, achieved {achieved_snr_db:.1f} dB"
        )

    def test_add_awgn_preserves_length(self, rng):
        sig = generate_cw(sample_rate=SAMPLE_RATE, duration=DURATION, rng=rng)
        noisy = add_awgn(sig, 10.0, rng=rng)
        assert len(noisy) == len(sig)
        assert noisy.dtype == np.complex64


# ---------------------------------------------------------------------------
# TDoA scenario tests
# ---------------------------------------------------------------------------

class TestTDoAScenario:
    """Verify the TDoA scenario generator produces correct delays."""

    @pytest.fixture
    def three_rx(self) -> list[ReceiverConfig]:
        """Three receivers in a simple triangle."""
        return [
            ReceiverConfig(id="RX0", latitude=0.0, longitude=0.0,
                           x=0.0, y=0.0),
            ReceiverConfig(id="RX1", latitude=0.0, longitude=0.0,
                           x=100_000.0, y=0.0),
            ReceiverConfig(id="RX2", latitude=0.0, longitude=0.0,
                           x=50_000.0, y=86_603.0),
        ]

    def test_compute_delays(self, three_rx):
        """Propagation delays should be distance / c."""
        tx = (30_000.0, 20_000.0)
        scenario = TDoAScenario(
            receivers=three_rx,
            tx_position=tx,
        )
        delays = scenario.compute_delays()
        for rx in three_rx:
            dist = np.sqrt((tx[0] - rx.x) ** 2 + (tx[1] - rx.y) ** 2)
            expected_delay = dist / SPEED_OF_LIGHT
            assert abs(delays[rx.id] - expected_delay) < 1e-15

    def test_generate_returns_all_receivers(self, three_rx, rng):
        """generate() should return one IQFrame per receiver."""
        scenario = TDoAScenario(
            receivers=three_rx,
            tx_position=(30_000.0, 20_000.0),
            snr_db=30.0,
        )
        frames = scenario.generate(rng=rng)
        assert set(frames.keys()) == {"RX0", "RX1", "RX2"}
        for frame in frames.values():
            assert frame.samples.dtype == np.complex64
            assert len(frame.samples) == int(SAMPLE_RATE * 0.1)

    def test_iqframe_metadata(self, three_rx, rng):
        """IQFrame fields should match scenario parameters."""
        scenario = TDoAScenario(
            receivers=three_rx,
            tx_position=(30_000.0, 20_000.0),
            sample_rate=SAMPLE_RATE,
            center_freq=14.1e6,
        )
        frames = scenario.generate(rng=rng)
        for rx_id, frame in frames.items():
            assert frame.receiver_id == rx_id
            assert frame.sample_rate == SAMPLE_RATE
            assert frame.center_freq == 14.1e6

    def test_differential_delays_via_cross_correlation(self, rng):
        """Cross-correlating receiver pairs should recover the correct TDOA.

        We use a high SNR broadband signal (BPSK) so the cross-correlation
        peak is sharp and unambiguous. CW is not suitable because a single
        tone produces periodic cross-correlation with ambiguous peaks.
        """
        # Two receivers separated by 150 km on the x-axis
        receivers = [
            ReceiverConfig(id="RX0", latitude=0.0, longitude=0.0,
                           x=0.0, y=0.0),
            ReceiverConfig(id="RX1", latitude=0.0, longitude=0.0,
                           x=150_000.0, y=0.0),
        ]
        tx = (50_000.0, 0.0)  # closer to RX0

        sr = 48_000.0
        dur = 0.5  # longer duration for better xcorr resolution

        scenario = TDoAScenario(
            receivers=receivers,
            tx_position=tx,
            sample_rate=sr,
            snr_db=40.0,  # high SNR for clean cross-correlation
            duration=dur,
            modulation=ModulationType.BPSK,
        )
        frames = scenario.generate(rng=rng)

        s0 = frames["RX0"].samples.astype(np.complex128)
        s1 = frames["RX1"].samples.astype(np.complex128)

        # Cross-correlate
        xcorr = correlate(s0, s1, mode="full")
        lags = np.arange(-(len(s1) - 1), len(s0))
        peak_lag = lags[np.argmax(np.abs(xcorr))]

        # Expected differential delay in samples
        d0 = np.sqrt((tx[0] - 0.0) ** 2 + (tx[1] - 0.0) ** 2)
        d1 = np.sqrt((tx[0] - 150_000.0) ** 2 + (tx[1] - 0.0) ** 2)
        expected_delay_s = (d0 - d1) / SPEED_OF_LIGHT
        expected_delay_samples = expected_delay_s * sr

        # Allow up to 2 samples tolerance (fractional delay + estimation error)
        assert abs(peak_lag - expected_delay_samples) <= 2.0, (
            f"Peak lag {peak_lag} samples, expected ~{expected_delay_samples:.2f} samples"
        )

    def test_differential_delays_three_receivers(self, three_rx, rng):
        """With three receivers the pairwise TDOAs should be consistent."""
        tx = (30_000.0, 20_000.0)
        sr = 48_000.0
        dur = 0.5

        scenario = TDoAScenario(
            receivers=three_rx,
            tx_position=tx,
            sample_rate=sr,
            snr_db=40.0,
            duration=dur,
            modulation=ModulationType.BPSK,
        )
        frames = scenario.generate(rng=rng)

        # Compute expected delays
        dists = {}
        for rx in three_rx:
            dists[rx.id] = np.sqrt((tx[0] - rx.x) ** 2 + (tx[1] - rx.y) ** 2)

        # Check pairwise: RX0-RX1
        s0 = frames["RX0"].samples.astype(np.complex128)
        s1 = frames["RX1"].samples.astype(np.complex128)
        xcorr = correlate(s0, s1, mode="full")
        lags = np.arange(-(len(s1) - 1), len(s0))
        peak_lag_01 = lags[np.argmax(np.abs(xcorr))]

        expected_01 = (dists["RX0"] - dists["RX1"]) / SPEED_OF_LIGHT * sr
        assert abs(peak_lag_01 - expected_01) <= 2.0

        # Check pairwise: RX0-RX2
        s2 = frames["RX2"].samples.astype(np.complex128)
        xcorr = correlate(s0, s2, mode="full")
        lags = np.arange(-(len(s2) - 1), len(s0))
        peak_lag_02 = lags[np.argmax(np.abs(xcorr))]

        expected_02 = (dists["RX0"] - dists["RX2"]) / SPEED_OF_LIGHT * sr
        assert abs(peak_lag_02 - expected_02) <= 2.0
