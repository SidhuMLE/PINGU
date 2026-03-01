"""Unit tests for the Bayesian integration module (Kalman filter + convergence)."""

from __future__ import annotations

import numpy as np
import pytest

from pingu.integrator.convergence import ConvergenceMonitor
from pingu.integrator.kalman import TDoAKalmanFilter
from pingu.types import IntegratedTDoA


# =====================================================================
# Fixtures
# =====================================================================

@pytest.fixture
def rng():
    """Seeded random generator for reproducibility."""
    return np.random.default_rng(12345)


@pytest.fixture
def n_pairs() -> int:
    """Number of TDoA pairs for 5 receivers: C(5,2) = 10."""
    return 10


@pytest.fixture
def kf(n_pairs: int) -> TDoAKalmanFilter:
    """Fresh Kalman filter with default parameters."""
    return TDoAKalmanFilter(n_pairs=n_pairs)


# =====================================================================
# Kalman filter — basic functionality
# =====================================================================

class TestTDoAKalmanFilterBasic:
    """Basic constructor and state management tests."""

    def test_initial_state_is_zero(self, kf: TDoAKalmanFilter, n_pairs: int):
        """State vector should be initialised to zeros."""
        state = kf.get_state()
        np.testing.assert_array_equal(state.delays, np.zeros(n_pairs))

    def test_initial_covariance_is_diagonal(self, kf: TDoAKalmanFilter, n_pairs: int):
        """Initial covariance should be initial_variance * I."""
        state = kf.get_state()
        expected = np.eye(n_pairs) * 1e-6
        np.testing.assert_array_almost_equal(state.covariance, expected)

    def test_get_state_returns_integrated_tdoa(self, kf: TDoAKalmanFilter):
        """get_state() should return an IntegratedTDoA dataclass."""
        state = kf.get_state()
        assert isinstance(state, IntegratedTDoA)

    def test_n_updates_initially_zero(self, kf: TDoAKalmanFilter):
        """Update counter starts at zero."""
        assert kf.n_updates == 0
        assert kf.get_state().n_updates == 0

    def test_pair_labels_generated(self, kf: TDoAKalmanFilter, n_pairs: int):
        """Pair labels should be auto-generated if not supplied."""
        state = kf.get_state()
        assert len(state.pair_labels) == n_pairs

    def test_custom_pair_labels(self):
        """User-supplied pair labels should be preserved."""
        labels = [("RX0", "RX1"), ("RX0", "RX2"), ("RX1", "RX2")]
        kf = TDoAKalmanFilter(n_pairs=3, pair_labels=labels)
        assert kf.get_state().pair_labels == labels

    def test_invalid_pair_labels_length(self):
        """Mismatched pair_labels length should raise ValueError."""
        with pytest.raises(ValueError, match="pair_labels"):
            TDoAKalmanFilter(n_pairs=3, pair_labels=[("A", "B")])

    def test_invalid_n_pairs(self):
        """n_pairs < 1 should raise ValueError."""
        with pytest.raises(ValueError, match="n_pairs"):
            TDoAKalmanFilter(n_pairs=0)


# =====================================================================
# Kalman filter — convergence to constant measurements
# =====================================================================

class TestKalmanConvergence:
    """Verify that repeated identical measurements cause the filter to
    converge to the true value."""

    def test_constant_measurements_converge(self, n_pairs: int):
        """With constant (noiseless) measurements the posterior mean should
        converge to the measurement value."""
        true_tdoa = np.linspace(-1e-4, 1e-4, n_pairs)
        noise_var = np.full(n_pairs, 1e-10)

        kf = TDoAKalmanFilter(n_pairs=n_pairs)
        for _ in range(50):
            kf.predict()
            kf.update(true_tdoa, noise_var)

        state = kf.get_state()
        np.testing.assert_allclose(state.delays, true_tdoa, atol=1e-10)

    def test_variance_decreases_with_updates(self, n_pairs: int):
        """Diagonal variance should monotonically decrease when measurements
        are consistent."""
        true_tdoa = np.ones(n_pairs) * 5e-5
        noise_var = np.full(n_pairs, 1e-8)

        kf = TDoAKalmanFilter(n_pairs=n_pairs, initial_variance=1e-6)
        prev_var = np.diag(kf.get_state().covariance).copy()

        for _ in range(20):
            kf.predict()
            kf.update(true_tdoa, noise_var)
            cur_var = np.diag(kf.get_state().covariance)
            assert np.all(cur_var < prev_var), (
                "Variance should strictly decrease with consistent measurements"
            )
            prev_var = cur_var.copy()

    def test_variance_shrinks_as_one_over_n(self, n_pairs: int):
        """After N identical-noise updates the posterior variance should
        approximately follow sigma^2 / N  (steady-state Kalman)."""
        measurement_var = 1e-8
        noise_var = np.full(n_pairs, measurement_var)
        true_tdoa = np.zeros(n_pairs)

        # Use a very large initial variance so the prior is uninformative.
        kf = TDoAKalmanFilter(n_pairs=n_pairs, initial_variance=1.0, process_noise=0.0)

        N = 200
        for _ in range(N):
            kf.predict()
            kf.update(true_tdoa, noise_var)

        posterior_var = np.diag(kf.get_state().covariance)
        expected_var = measurement_var / N

        # Allow 10% tolerance because the first few updates are transient.
        np.testing.assert_allclose(posterior_var, expected_var, rtol=0.1)


# =====================================================================
# Kalman filter — noisy measurements
# =====================================================================

class TestKalmanNoisy:
    """Verify filtering quality with stochastic measurements."""

    def test_filtered_closer_to_truth_than_single_measurement(
        self, n_pairs: int, rng: np.random.Generator
    ):
        """The Kalman-filtered estimate should be closer to the true TDoA
        than any individual noisy measurement."""
        true_tdoa = rng.uniform(-1e-4, 1e-4, size=n_pairs)
        measurement_std = 1e-6
        noise_var = np.full(n_pairs, measurement_std**2)

        kf = TDoAKalmanFilter(n_pairs=n_pairs, initial_variance=1e-4, process_noise=0.0)

        single_measurement_errors: list[float] = []
        for _ in range(100):
            noisy = true_tdoa + rng.normal(0, measurement_std, size=n_pairs)
            single_measurement_errors.append(float(np.linalg.norm(noisy - true_tdoa)))
            kf.predict()
            kf.update(noisy, noise_var)

        filtered_error = float(np.linalg.norm(kf.get_state().delays - true_tdoa))
        mean_single_error = float(np.mean(single_measurement_errors))

        assert filtered_error < mean_single_error, (
            f"Filtered error ({filtered_error:.3e}) should be less than "
            f"mean single-measurement error ({mean_single_error:.3e})"
        )

    def test_n_updates_counter(self, kf: TDoAKalmanFilter, n_pairs: int):
        """n_updates should increment with each update call."""
        noise_var = np.ones(n_pairs) * 1e-8
        for k in range(1, 11):
            kf.predict()
            kf.update(np.zeros(n_pairs), noise_var)
            assert kf.n_updates == k


# =====================================================================
# Kalman filter — 10x variance reduction after 100 updates (spec)
# =====================================================================

class TestKalmanSpecRequirements:
    """Tests derived from the project specification."""

    def test_10x_variance_reduction_after_100_updates(self, n_pairs: int):
        """Per spec: 10x variance reduction after 100 updates with constant
        measurement noise."""
        measurement_var = 1e-8
        noise_var = np.full(n_pairs, measurement_var)
        true_tdoa = np.zeros(n_pairs)

        initial_variance = 1e-6
        kf = TDoAKalmanFilter(
            n_pairs=n_pairs,
            initial_variance=initial_variance,
            process_noise=0.0,
        )

        initial_var = np.mean(np.diag(kf.get_state().covariance))

        for _ in range(100):
            kf.predict()
            kf.update(true_tdoa, noise_var)

        final_var = np.mean(np.diag(kf.get_state().covariance))
        ratio = initial_var / final_var
        assert ratio >= 10.0, (
            f"Expected >= 10x variance reduction after 100 updates, got {ratio:.1f}x"
        )


# =====================================================================
# Kalman filter — reset
# =====================================================================

class TestKalmanReset:
    """Verify that reset() restores the filter to its initial state."""

    def test_reset_restores_initial_state(self, n_pairs: int):
        """After updates + reset, the state should match a freshly constructed
        filter."""
        kf = TDoAKalmanFilter(n_pairs=n_pairs, initial_variance=1e-6)

        # Perform some updates.
        noise_var = np.ones(n_pairs) * 1e-8
        for _ in range(10):
            kf.predict()
            kf.update(np.ones(n_pairs) * 1e-5, noise_var)

        # Reset.
        kf.reset()

        state = kf.get_state()
        np.testing.assert_array_equal(state.delays, np.zeros(n_pairs))
        np.testing.assert_array_almost_equal(state.covariance, np.eye(n_pairs) * 1e-6)
        assert state.n_updates == 0

    def test_reset_allows_reuse(self, n_pairs: int):
        """After reset the filter should work identically to a new instance."""
        kf = TDoAKalmanFilter(n_pairs=n_pairs, initial_variance=1e-6, process_noise=0.0)

        # Run once.
        noise_var = np.ones(n_pairs) * 1e-8
        tdoa_val = np.ones(n_pairs) * 3e-5
        for _ in range(20):
            kf.predict()
            kf.update(tdoa_val, noise_var)
        first_state = kf.get_state()

        # Reset and run again with the same inputs.
        kf.reset()
        for _ in range(20):
            kf.predict()
            kf.update(tdoa_val, noise_var)
        second_state = kf.get_state()

        np.testing.assert_allclose(first_state.delays, second_state.delays)
        np.testing.assert_allclose(first_state.covariance, second_state.covariance)


# =====================================================================
# Kalman filter — input validation
# =====================================================================

class TestKalmanValidation:
    """Input validation edge cases."""

    def test_wrong_measurement_length(self, kf: TDoAKalmanFilter):
        """Measurement vector with wrong length should raise ValueError."""
        with pytest.raises(ValueError, match="measurements"):
            kf.update(np.zeros(3), np.ones(10) * 1e-8)

    def test_wrong_noise_length(self, kf: TDoAKalmanFilter):
        """Noise vector with wrong length should raise ValueError."""
        with pytest.raises(ValueError, match="measurement_noise"):
            kf.update(np.zeros(10), np.ones(3) * 1e-8)


# =====================================================================
# Convergence monitor
# =====================================================================

class TestConvergenceMonitor:
    """Tests for the ConvergenceMonitor class."""

    def test_not_converged_before_min_updates(self, n_pairs: int):
        """Should not declare convergence before min_updates snapshots."""
        monitor = ConvergenceMonitor(n_pairs=n_pairs, min_updates=5)
        cov = np.eye(n_pairs) * 1e-20  # extremely small variance
        for _ in range(4):
            monitor.update(cov)
        assert not monitor.is_converged()

    def test_converged_by_rate(self, n_pairs: int):
        """With steadily shrinking variance and no CRLB, convergence should
        be declared once the rate of change drops below threshold."""
        monitor = ConvergenceMonitor(n_pairs=n_pairs, rate_threshold=0.01, min_updates=3)

        # Simulate rapidly decreasing variance, then plateau.
        for v in [1e-4, 1e-5, 1e-6, 1e-7, 1e-7 * 0.999]:
            monitor.update(np.eye(n_pairs) * v)

        assert monitor.is_converged()

    def test_converged_by_crlb(self, n_pairs: int):
        """When CRLB is provided, convergence requires variances < factor * CRLB."""
        crlb = np.eye(n_pairs) * 1e-8
        monitor = ConvergenceMonitor(n_pairs=n_pairs, min_updates=3)

        # Variance well below 0.1 * CRLB.
        small_cov = np.eye(n_pairs) * 1e-10
        for _ in range(5):
            monitor.update(small_cov, crlb=crlb)

        assert monitor.is_converged(factor=0.1)

    def test_not_converged_above_crlb(self, n_pairs: int):
        """Variance above factor * CRLB should not be converged."""
        crlb = np.eye(n_pairs) * 1e-8
        monitor = ConvergenceMonitor(n_pairs=n_pairs, min_updates=3)

        large_cov = np.eye(n_pairs) * 1e-6  # 100x above CRLB
        for _ in range(5):
            monitor.update(large_cov, crlb=crlb)

        assert not monitor.is_converged(factor=0.1)

    def test_variance_history_shape(self, n_pairs: int):
        """Variance history should have shape (n_updates, n_pairs)."""
        monitor = ConvergenceMonitor(n_pairs=n_pairs)
        for _ in range(7):
            monitor.update(np.eye(n_pairs) * 1e-6)

        history = monitor.get_variance_history()
        assert history.shape == (7, n_pairs)

    def test_empty_variance_history(self, n_pairs: int):
        """Before any updates, variance history should be empty."""
        monitor = ConvergenceMonitor(n_pairs=n_pairs)
        history = monitor.get_variance_history()
        assert history.shape == (0, n_pairs)

    def test_improvement_ratio(self, n_pairs: int):
        """Improvement ratio should reflect initial / current variance."""
        monitor = ConvergenceMonitor(n_pairs=n_pairs)
        monitor.update(np.eye(n_pairs) * 1e-4)
        monitor.update(np.eye(n_pairs) * 1e-5)

        ratio = monitor.get_improvement_ratio()
        np.testing.assert_almost_equal(ratio, 10.0)

    def test_improvement_ratio_before_updates(self, n_pairs: int):
        """Improvement ratio should be 1.0 before any updates."""
        monitor = ConvergenceMonitor(n_pairs=n_pairs)
        assert monitor.get_improvement_ratio() == 1.0

    def test_convergence_monitor_with_kalman(self, n_pairs: int):
        """End-to-end: run Kalman filter and monitor convergence together."""
        kf = TDoAKalmanFilter(n_pairs=n_pairs, initial_variance=1e-6, process_noise=0.0)
        monitor = ConvergenceMonitor(n_pairs=n_pairs, rate_threshold=0.01, min_updates=5)

        true_tdoa = np.zeros(n_pairs)
        noise_var = np.full(n_pairs, 1e-8)

        for _ in range(200):
            kf.predict()
            kf.update(true_tdoa, noise_var)
            state = kf.get_state()
            monitor.update(state.covariance)

        assert monitor.is_converged(), "Filter should have converged after 200 updates"
        assert monitor.get_improvement_ratio() > 10.0, (
            "Improvement ratio should exceed 10x after 200 updates"
        )

    def test_invalid_n_pairs(self):
        """n_pairs < 1 should raise ValueError."""
        with pytest.raises(ValueError, match="n_pairs"):
            ConvergenceMonitor(n_pairs=0)


# =====================================================================
# Kalman filter — predict step
# =====================================================================

class TestKalmanPredict:
    """Tests for the predict-only step."""

    def test_predict_increases_covariance(self, kf: TDoAKalmanFilter, n_pairs: int):
        """Predict step should increase the covariance by Q."""
        before = kf.get_state().covariance.copy()
        kf.predict()
        after = kf.get_state().covariance

        expected = before + np.eye(n_pairs) * 1e-12  # default process_noise
        np.testing.assert_allclose(after, expected)

    def test_predict_does_not_change_state(self, kf: TDoAKalmanFilter):
        """Predict step should not alter the state vector (identity dynamics)."""
        before = kf.get_state().delays.copy()
        kf.predict()
        after = kf.get_state().delays
        np.testing.assert_array_equal(before, after)
