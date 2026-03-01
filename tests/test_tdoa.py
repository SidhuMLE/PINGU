"""Unit tests for the TDoA estimation sub-package.

Covers:
    - GCC-PHAT / GCC-SCOT / GCC-ML delay recovery with known integer delay
    - Sub-sample interpolation accuracy
    - PairManager combinatorics
    - CRLB monotonicity with SNR
    - Delay recovery at 20 dB SNR within tolerance
"""

from __future__ import annotations

import numpy as np
import pytest

from pingu.tdoa.gcc import estimate_tdoa, gcc_ml, gcc_phat, gcc_scot
from pingu.tdoa.pair_manager import PairManager
from pingu.tdoa.peak_interpolation import parabolic_interpolation, sinc_interpolation
from pingu.tdoa.uncertainty import crlb_tdoa


# -------------------------------------------------------------------- #
# Helpers
# -------------------------------------------------------------------- #

def _make_delayed_pair(
    rng: np.random.Generator,
    fs: float,
    delay_samples: int,
    duration: float = 0.1,
    snr_db: float = 40.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Create a signal pair where *y* is a delayed copy of *x* plus noise.

    The base signal is band-limited Gaussian noise (real-valued) so that
    GCC methods have a well-defined cross-correlation peak.

    Args:
        rng: Numpy random generator.
        fs: Sampling frequency (Hz).
        delay_samples: Integer sample delay to introduce.
        duration: Signal duration in seconds.
        snr_db: Signal-to-noise ratio in dB.

    Returns:
        ``(x, y)`` signal arrays of equal length.
    """
    n_samples = int(fs * duration)
    # Ensure enough room for the delay.
    total = n_samples + abs(delay_samples)
    base = rng.standard_normal(total)

    # Band-limit with a simple low-pass (moving-average) to give a
    # smooth cross-correlation.
    kernel_len = max(int(fs / 4000), 3)
    kernel = np.ones(kernel_len) / kernel_len
    base = np.convolve(base, kernel, mode="same")

    if delay_samples >= 0:
        x = base[:n_samples]
        y = base[delay_samples : delay_samples + n_samples]
    else:
        d = abs(delay_samples)
        y = base[:n_samples]
        x = base[d : d + n_samples]

    # Add noise.
    signal_power = np.mean(x**2)
    snr_linear = 10.0 ** (snr_db / 10.0)
    noise_power = signal_power / snr_linear
    x = x + rng.standard_normal(n_samples) * np.sqrt(noise_power)
    y = y + rng.standard_normal(n_samples) * np.sqrt(noise_power)

    return x.astype(np.float64), y.astype(np.float64)


# -------------------------------------------------------------------- #
# GCC-PHAT: known integer delay
# -------------------------------------------------------------------- #

class TestGccPhat:
    """Tests for the GCC-PHAT method."""

    def test_recover_positive_delay(self, rng, sample_rate):
        """GCC-PHAT should recover a positive integer delay within 1 sample."""
        true_delay = 10
        x, y = _make_delayed_pair(rng, sample_rate, true_delay)
        lags, corr = gcc_phat(x, y, sample_rate)

        peak_idx = np.argmax(corr)
        recovered_delay_sec = lags[peak_idx]
        recovered_samples = recovered_delay_sec * sample_rate

        assert abs(recovered_samples - true_delay) <= 1.0, (
            f"Expected ~{true_delay} samples, got {recovered_samples:.2f}"
        )

    def test_recover_negative_delay(self, rng, sample_rate):
        """GCC-PHAT should recover a negative delay."""
        true_delay = -7
        x, y = _make_delayed_pair(rng, sample_rate, true_delay)
        lags, corr = gcc_phat(x, y, sample_rate)

        peak_idx = np.argmax(corr)
        recovered_samples = lags[peak_idx] * sample_rate

        assert abs(recovered_samples - true_delay) <= 1.0

    def test_zero_delay(self, rng, sample_rate):
        """Two identical signals should give zero delay."""
        n = int(0.1 * sample_rate)
        sig = rng.standard_normal(n)
        lags, corr = gcc_phat(sig, sig, sample_rate)

        peak_idx = np.argmax(corr)
        assert abs(lags[peak_idx]) < 1.0 / sample_rate + 1e-9

    def test_max_delay_clipping(self, rng, sample_rate):
        """With max_delay_samples, lags should be clipped."""
        max_d = 50
        x, y = _make_delayed_pair(rng, sample_rate, 5)
        lags, corr = gcc_phat(x, y, sample_rate, max_delay_samples=max_d)

        max_lag_samples = np.max(np.abs(lags)) * sample_rate
        assert max_lag_samples <= max_d + 0.5  # allow rounding


# -------------------------------------------------------------------- #
# GCC-SCOT and GCC-ML
# -------------------------------------------------------------------- #

class TestGccScot:
    """Tests for the GCC-SCOT method."""

    def test_recover_delay(self, rng, sample_rate):
        true_delay = 12
        x, y = _make_delayed_pair(rng, sample_rate, true_delay)
        lags, corr = gcc_scot(x, y, sample_rate)

        peak_idx = np.argmax(corr)
        recovered = lags[peak_idx] * sample_rate
        assert abs(recovered - true_delay) <= 1.0


class TestGccMl:
    """Tests for the GCC-ML method."""

    def test_recover_delay(self, rng, sample_rate):
        true_delay = 8
        x, y = _make_delayed_pair(rng, sample_rate, true_delay)
        lags, corr = gcc_ml(x, y, sample_rate)

        peak_idx = np.argmax(corr)
        recovered = lags[peak_idx] * sample_rate
        assert abs(recovered - true_delay) <= 1.0


class TestAllMethodsRecoverDelay:
    """All GCC methods should recover the same known delay."""

    @pytest.mark.parametrize("method", ["phat", "scot", "ml"])
    def test_methods_agree(self, rng, sample_rate, method):
        true_delay = 15
        x, y = _make_delayed_pair(rng, sample_rate, true_delay)
        est = estimate_tdoa(x, y, sample_rate, method=method)

        recovered = est.delay * sample_rate
        assert abs(recovered - true_delay) <= 1.5, (
            f"method={method}: expected ~{true_delay}, got {recovered:.2f}"
        )


# -------------------------------------------------------------------- #
# Sub-sample interpolation
# -------------------------------------------------------------------- #

class TestParabolicInterpolation:
    """Tests for parabolic sub-sample peak interpolation."""

    def test_exact_peak(self):
        """If the true peak is at an integer, the refinement should be ~0."""
        corr = np.array([0.5, 1.0, 0.5])
        frac, val = parabolic_interpolation(corr, 1)
        assert abs(frac - 1.0) < 1e-10
        assert abs(val - 1.0) < 1e-10

    def test_fractional_offset(self):
        """A skewed triplet should yield a fractional correction."""
        corr = np.array([0.6, 1.0, 0.8])
        frac, _ = parabolic_interpolation(corr, 1)
        # Peak should shift towards index 2 (higher neighbour on right).
        assert frac > 1.0

    def test_boundary_raises(self):
        corr = np.array([1.0, 0.5, 0.3])
        with pytest.raises(ValueError):
            parabolic_interpolation(corr, 0)

    def test_improves_accuracy(self, rng, sample_rate):
        """Sub-sample interpolation should improve delay accuracy to < 0.1 samples."""
        true_delay = 10
        x, y = _make_delayed_pair(rng, sample_rate, true_delay, snr_db=40.0)
        lags, corr = gcc_phat(x, y, sample_rate)

        peak_idx = int(np.argmax(corr))
        if 0 < peak_idx < len(corr) - 1:
            frac_idx, _ = parabolic_interpolation(corr, peak_idx)
            dt = lags[1] - lags[0]
            refined_sec = lags[0] + frac_idx * dt
            refined_samples = refined_sec * sample_rate
            error = abs(refined_samples - true_delay)
            assert error < 0.1, f"Expected < 0.1, got {error:.4f}"


class TestSincInterpolation:
    """Tests for sinc sub-sample peak interpolation."""

    def test_exact_peak(self):
        corr = np.array([0.0, 0.2, 0.5, 1.0, 0.5, 0.2, 0.0])
        frac, val = sinc_interpolation(corr, 3, n_points=3)
        assert abs(frac - 3.0) < 0.05
        assert val >= 0.99

    def test_returns_tuple(self):
        corr = np.array([0.1, 0.4, 0.9, 1.0, 0.85, 0.3, 0.1])
        result = sinc_interpolation(corr, 3, n_points=3)
        assert len(result) == 2


# -------------------------------------------------------------------- #
# PairManager
# -------------------------------------------------------------------- #

class TestPairManager:
    """Tests for the PairManager class."""

    def test_five_receivers_give_ten_pairs(self):
        pm = PairManager(["RX0", "RX1", "RX2", "RX3", "RX4"])
        pairs = pm.get_pairs()
        assert len(pairs) == 10

    def test_three_receivers_give_three_pairs(self):
        pm = PairManager(["A", "B", "C"])
        pairs = pm.get_pairs()
        assert len(pairs) == 3
        assert ("A", "B") in pairs
        assert ("A", "C") in pairs
        assert ("B", "C") in pairs

    def test_pair_uniqueness(self):
        pm = PairManager(["RX0", "RX1", "RX2", "RX3", "RX4"])
        pairs = pm.get_pairs()
        assert len(pairs) == len(set(pairs))

    def test_n_pairs_property(self):
        pm = PairManager(["RX0", "RX1", "RX2"])
        assert pm.n_pairs == 3

    def test_too_few_receivers_raises(self):
        with pytest.raises(ValueError):
            PairManager(["RX0"])

    def test_estimate_all_tdoas(self, rng, sample_rate):
        """Estimate TDoAs for all pairs of 3 receivers."""
        ids = ["RX0", "RX1", "RX2"]
        pm = PairManager(ids)

        n = int(0.05 * sample_rate)
        signals = {rid: rng.standard_normal(n) for rid in ids}

        results = pm.estimate_all_tdoas(signals, sample_rate, method="phat")
        assert len(results) == 3
        for est in results:
            assert est.receiver_i in ids
            assert est.receiver_j in ids
            assert est.receiver_i != est.receiver_j

    def test_missing_receiver_raises(self, rng, sample_rate):
        pm = PairManager(["RX0", "RX1", "RX2"])
        signals = {"RX0": np.zeros(100), "RX1": np.zeros(100)}
        with pytest.raises(KeyError):
            pm.estimate_all_tdoas(signals, sample_rate)


# -------------------------------------------------------------------- #
# CRLB
# -------------------------------------------------------------------- #

class TestCRLB:
    """Tests for the Cramer-Rao Lower Bound on TDoA variance."""

    def test_positive_value(self):
        var = crlb_tdoa(bandwidth=3000.0, snr_linear=100.0, integration_time=0.1)
        assert var > 0.0

    def test_decreases_with_snr(self):
        """Higher SNR should give a smaller (tighter) CRLB."""
        var_low = crlb_tdoa(bandwidth=3000.0, snr_linear=10.0, integration_time=0.1)
        var_high = crlb_tdoa(bandwidth=3000.0, snr_linear=100.0, integration_time=0.1)
        assert var_high < var_low

    def test_decreases_with_bandwidth(self):
        var_narrow = crlb_tdoa(bandwidth=1000.0, snr_linear=100.0, integration_time=0.1)
        var_wide = crlb_tdoa(bandwidth=5000.0, snr_linear=100.0, integration_time=0.1)
        assert var_wide < var_narrow

    def test_decreases_with_integration_time(self):
        var_short = crlb_tdoa(bandwidth=3000.0, snr_linear=100.0, integration_time=0.01)
        var_long = crlb_tdoa(bandwidth=3000.0, snr_linear=100.0, integration_time=1.0)
        assert var_long < var_short

    def test_tdoa_variance_doubles_single(self):
        """TDoA variance should be 2x the single-delay CRLB."""
        B, snr, T = 3000.0, 50.0, 0.1
        var_tdoa = crlb_tdoa(B, snr, T)
        var_single = 3.0 / (8.0 * np.pi**2 * B**3 * T * snr)
        assert abs(var_tdoa - 2.0 * var_single) < 1e-30

    def test_invalid_inputs(self):
        with pytest.raises(ValueError):
            crlb_tdoa(bandwidth=0.0, snr_linear=10.0, integration_time=0.1)
        with pytest.raises(ValueError):
            crlb_tdoa(bandwidth=3000.0, snr_linear=-1.0, integration_time=0.1)
        with pytest.raises(ValueError):
            crlb_tdoa(bandwidth=3000.0, snr_linear=10.0, integration_time=0.0)


# -------------------------------------------------------------------- #
# Integration: 20 dB SNR delay recovery
# -------------------------------------------------------------------- #

class TestDelayRecoveryAt20dB:
    """At 20 dB SNR, all GCC methods should recover a known delay."""

    @pytest.mark.parametrize("method", ["phat", "scot", "ml"])
    def test_20db_recovery(self, rng, sample_rate, method):
        true_delay = 20  # samples
        x, y = _make_delayed_pair(rng, sample_rate, true_delay, snr_db=20.0)
        est = estimate_tdoa(x, y, sample_rate, method=method)

        recovered = est.delay * sample_rate
        # At 20 dB SNR with 48 kHz / 0.1 s, 1-sample tolerance is reasonable.
        assert abs(recovered - true_delay) <= 1.5, (
            f"method={method}: expected ~{true_delay}, got {recovered:.2f}"
        )
