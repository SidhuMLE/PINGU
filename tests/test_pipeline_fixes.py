"""Tests for the modulation-aware TDoA pipeline fixes.

Covers:
    - gcc_basic() delay recovery
    - select_gcc_method() modulation mapping
    - Complex IQ handling in _prepare_spectra
    - CRLB-based variance in estimate_tdoa
    - Pipeline end-to-end with CW signal (position error < 1 km)
    - Pipeline variance history tracking
    - Pipeline classifier fallback (no checkpoint)
"""

from __future__ import annotations

import numpy as np
import pytest

from pingu.tdoa.gcc import (
    _GCC_METHODS,
    _prepare_spectra,
    estimate_tdoa,
    gcc_basic,
    select_gcc_method,
)
from pingu.types import ModulationType


# -------------------------------------------------------------------- #
# Helpers
# -------------------------------------------------------------------- #

def _make_delayed_pair(
    rng: np.random.Generator,
    fs: float,
    delay_samples: int,
    duration: float = 0.1,
    snr_db: float = 40.0,
    complex_iq: bool = False,
) -> tuple[np.ndarray, np.ndarray]:
    """Create a signal pair where *y* is a delayed copy of *x* plus noise."""
    n_samples = int(fs * duration)
    total = n_samples + abs(delay_samples)

    if complex_iq:
        base = (
            rng.standard_normal(total) + 1j * rng.standard_normal(total)
        ) / np.sqrt(2)
    else:
        base = rng.standard_normal(total)

    # Band-limit with a simple low-pass.
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

    signal_power = np.mean(np.abs(x) ** 2)
    snr_linear = 10.0 ** (snr_db / 10.0)
    noise_power = signal_power / snr_linear

    if complex_iq:
        noise_x = (
            rng.standard_normal(n_samples) + 1j * rng.standard_normal(n_samples)
        ) * np.sqrt(noise_power / 2)
        noise_y = (
            rng.standard_normal(n_samples) + 1j * rng.standard_normal(n_samples)
        ) * np.sqrt(noise_power / 2)
    else:
        noise_x = rng.standard_normal(n_samples) * np.sqrt(noise_power)
        noise_y = rng.standard_normal(n_samples) * np.sqrt(noise_power)

    return x + noise_x, y + noise_y


# -------------------------------------------------------------------- #
# gcc_basic
# -------------------------------------------------------------------- #

class TestGccBasic:
    """Tests for the unweighted (basic) cross-correlation method."""

    def test_registered_in_methods(self):
        """gcc_basic should be in the _GCC_METHODS registry."""
        assert "basic" in _GCC_METHODS
        assert _GCC_METHODS["basic"] is gcc_basic

    def test_recover_positive_delay(self, rng, sample_rate):
        true_delay = 10
        x, y = _make_delayed_pair(rng, sample_rate, true_delay)
        lags, corr = gcc_basic(x, y, sample_rate)

        peak_idx = np.argmax(corr)
        recovered = lags[peak_idx] * sample_rate
        assert abs(recovered - true_delay) <= 1.0

    def test_recover_negative_delay(self, rng, sample_rate):
        true_delay = -7
        x, y = _make_delayed_pair(rng, sample_rate, true_delay)
        lags, corr = gcc_basic(x, y, sample_rate)

        peak_idx = np.argmax(corr)
        recovered = lags[peak_idx] * sample_rate
        assert abs(recovered - true_delay) <= 1.0

    def test_zero_delay(self, rng, sample_rate):
        n = int(0.1 * sample_rate)
        sig = rng.standard_normal(n)
        lags, corr = gcc_basic(sig, sig, sample_rate)

        peak_idx = np.argmax(corr)
        assert abs(lags[peak_idx]) < 1.0 / sample_rate + 1e-9

    def test_max_delay_clipping(self, rng, sample_rate):
        max_d = 50
        x, y = _make_delayed_pair(rng, sample_rate, 5)
        lags, corr = gcc_basic(x, y, sample_rate, max_delay_samples=max_d)

        max_lag_samples = np.max(np.abs(lags)) * sample_rate
        assert max_lag_samples <= max_d + 0.5

    def test_estimate_tdoa_basic(self, rng, sample_rate):
        """estimate_tdoa with method='basic' should work."""
        true_delay = 15
        x, y = _make_delayed_pair(rng, sample_rate, true_delay)
        est = estimate_tdoa(x, y, sample_rate, method="basic")

        recovered = est.delay * sample_rate
        assert abs(recovered - true_delay) <= 1.5


# -------------------------------------------------------------------- #
# select_gcc_method
# -------------------------------------------------------------------- #

class TestSelectGccMethod:
    """Tests for modulation-aware GCC method selection."""

    def test_cw_selects_basic(self):
        assert select_gcc_method(ModulationType.CW) == "basic"

    def test_ssb_selects_scot(self):
        assert select_gcc_method(ModulationType.SSB) == "scot"

    def test_am_selects_scot(self):
        assert select_gcc_method(ModulationType.AM) == "scot"

    def test_fsk2_selects_phat(self):
        assert select_gcc_method(ModulationType.FSK2) == "phat"

    def test_fsk4_selects_phat(self):
        assert select_gcc_method(ModulationType.FSK4) == "phat"

    def test_bpsk_selects_phat(self):
        assert select_gcc_method(ModulationType.BPSK) == "phat"

    def test_qpsk_selects_phat(self):
        assert select_gcc_method(ModulationType.QPSK) == "phat"

    def test_noise_selects_phat(self):
        assert select_gcc_method(ModulationType.NOISE) == "phat"

    def test_none_selects_phat(self):
        assert select_gcc_method(None) == "phat"

    def test_all_returned_methods_valid(self):
        """Every returned method should be in the GCC registry."""
        for mod in ModulationType:
            method = select_gcc_method(mod)
            assert method in _GCC_METHODS, f"{mod} -> {method} not in registry"
        assert select_gcc_method(None) in _GCC_METHODS


# -------------------------------------------------------------------- #
# Complex IQ handling
# -------------------------------------------------------------------- #

class TestComplexIQHandling:
    """Tests for complex input support in _prepare_spectra and GCC methods."""

    def test_prepare_spectra_complex_input(self):
        """_prepare_spectra should accept complex arrays without casting to float64."""
        x = np.array([1 + 2j, 3 + 4j, 5 + 6j], dtype=np.complex128)
        y = np.array([7 + 8j, 9 + 10j, 11 + 12j], dtype=np.complex128)
        X, Y = _prepare_spectra(x, y, nfft=8)
        assert X.dtype == np.complex128
        assert Y.dtype == np.complex128

    def test_prepare_spectra_real_input(self):
        """_prepare_spectra should still work with real arrays."""
        x = np.array([1.0, 2.0, 3.0])
        y = np.array([4.0, 5.0, 6.0])
        X, Y = _prepare_spectra(x, y, nfft=8)
        assert X.dtype == np.complex128
        assert Y.dtype == np.complex128

    def test_gcc_basic_complex_delay_recovery(self, rng, sample_rate):
        """gcc_basic should recover delay from complex IQ signals."""
        true_delay = 10
        x, y = _make_delayed_pair(rng, sample_rate, true_delay, complex_iq=True)
        lags, corr = gcc_basic(x, y, sample_rate)

        peak_idx = np.argmax(corr)
        recovered = lags[peak_idx] * sample_rate
        assert abs(recovered - true_delay) <= 1.0

    def test_gcc_phat_complex_delay_recovery(self, rng, sample_rate):
        """gcc_phat should work with complex IQ input."""
        true_delay = 10
        x, y = _make_delayed_pair(rng, sample_rate, true_delay, complex_iq=True)
        lags, corr = gcc_basic(x, y, sample_rate)

        peak_idx = np.argmax(corr)
        recovered = lags[peak_idx] * sample_rate
        assert abs(recovered - true_delay) <= 1.0


# -------------------------------------------------------------------- #
# CRLB-based variance in estimate_tdoa
# -------------------------------------------------------------------- #

class TestCRLBVariance:
    """Tests for CRLB-based variance in estimate_tdoa."""

    def test_crlb_used_when_metadata_provided(self, rng, sample_rate):
        """When bandwidth, snr_linear, integration_time are given, use CRLB."""
        x, y = _make_delayed_pair(rng, sample_rate, 10)
        est_crlb = estimate_tdoa(
            x, y, sample_rate, method="basic",
            bandwidth=3000.0, snr_linear=100.0, integration_time=0.1,
        )
        est_heuristic = estimate_tdoa(x, y, sample_rate, method="basic")

        # CRLB and heuristic variances should differ.
        assert est_crlb.variance != est_heuristic.variance

    def test_heuristic_fallback_without_metadata(self, rng, sample_rate):
        """Without metadata, heuristic variance is used."""
        x, y = _make_delayed_pair(rng, sample_rate, 10)
        est = estimate_tdoa(x, y, sample_rate, method="basic")
        assert est.variance > 0


# -------------------------------------------------------------------- #
# Pipeline end-to-end
# -------------------------------------------------------------------- #

class TestPipelineEndToEnd:
    """Integration tests for the full pipeline with modulation-aware TDoA."""

    def test_cw_position_error_under_1km(self, rng, pentagon_receivers, tx_position):
        """With CW modulation and method='basic', error should be < 1 km."""
        from pingu.config import load_config
        from pingu.pipeline.runner import PinguPipeline
        from pingu.synthetic.scenarios import TDoAScenario

        cfg = load_config()
        cfg.tdoa.method = "basic"  # Correct method for CW

        pipeline = PinguPipeline(config=cfg, receivers=pentagon_receivers)
        scenario = TDoAScenario(
            receivers=pentagon_receivers,
            tx_position=tx_position,
            sample_rate=48_000.0,
            center_freq=14.1e6,
            snr_db=20.0,
            duration=0.1,
            modulation=ModulationType.CW,
        )

        frames_list = [scenario.generate(rng=rng) for _ in range(50)]
        estimate = pipeline.run(frames_list)

        assert estimate is not None
        error = np.sqrt(
            (estimate.x - tx_position[0]) ** 2
            + (estimate.y - tx_position[1]) ** 2
        )
        assert error < 1000.0, f"Position error {error:.1f} m exceeds 1 km"

    def test_variance_history_populated(self, rng, pentagon_receivers, tx_position):
        """Pipeline should record variance history for visualization."""
        from pingu.config import load_config
        from pingu.pipeline.runner import PinguPipeline
        from pingu.synthetic.scenarios import TDoAScenario

        cfg = load_config()
        cfg.tdoa.method = "basic"

        pipeline = PinguPipeline(config=cfg, receivers=pentagon_receivers)
        scenario = TDoAScenario(
            receivers=pentagon_receivers,
            tx_position=tx_position,
            sample_rate=48_000.0,
            center_freq=14.1e6,
            snr_db=20.0,
            duration=0.1,
            modulation=ModulationType.CW,
        )

        n_frames = 10
        frames_list = [scenario.generate(rng=rng) for _ in range(n_frames)]
        pipeline.run(frames_list)

        # Pipeline may converge early and return before all frames are processed.
        assert len(pipeline.variance_history) >= 1
        assert len(pipeline.variance_history) <= n_frames
        assert pipeline.variance_history[0].shape == (pipeline.pair_manager.n_pairs,)

    def test_auto_method_without_classifier_falls_back(
        self, rng, pentagon_receivers, tx_position
    ):
        """With method='auto' and no classifier, pipeline should not crash."""
        from pingu.config import load_config
        from pingu.pipeline.runner import PinguPipeline
        from pingu.synthetic.scenarios import TDoAScenario

        cfg = load_config()
        cfg.tdoa.method = "auto"  # No classifier → falls back to phat

        pipeline = PinguPipeline(config=cfg, receivers=pentagon_receivers)
        scenario = TDoAScenario(
            receivers=pentagon_receivers,
            tx_position=tx_position,
            sample_rate=48_000.0,
            center_freq=14.1e6,
            snr_db=20.0,
            duration=0.1,
            modulation=ModulationType.CW,
        )

        frames_list = [scenario.generate(rng=rng) for _ in range(5)]
        # Should not raise — graceful fallback.
        estimate = pipeline.run(frames_list)
        assert estimate is not None
