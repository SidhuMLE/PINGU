"""Unit tests for the position estimation (locator) module."""

from __future__ import annotations

import numpy as np
import pytest
from scipy.stats import chi2 as chi2_dist

from pingu.constants import SPEED_OF_LIGHT
from pingu.types import ReceiverConfig, PositionEstimate
from pingu.locator.geometry import ReceiverGeometry
from pingu.locator.cost_functions import tdoa_residuals, tdoa_jacobian
from pingu.locator.solvers import TDoASolver
from pingu.locator.posterior import confidence_ellipse, confidence_radius, position_uncertainty


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def triangle_receivers() -> list[ReceiverConfig]:
    """Three receivers in an equilateral triangle, ~100 km baseline."""
    R = 100_000.0  # 100 km
    receivers = []
    for i in range(3):
        angle = 2 * np.pi * i / 3
        receivers.append(
            ReceiverConfig(
                id=f"RX{i}",
                latitude=0.0,
                longitude=0.0,
                x=R * np.cos(angle),
                y=R * np.sin(angle),
            )
        )
    return receivers


@pytest.fixture
def geometry(triangle_receivers) -> ReceiverGeometry:
    return ReceiverGeometry(triangle_receivers)


@pytest.fixture
def known_tx_pos() -> np.ndarray:
    """A known transmitter position for testing."""
    return np.array([30_000.0, -20_000.0], dtype=np.float64)


# ---------------------------------------------------------------------------
# Geometry tests
# ---------------------------------------------------------------------------

class TestReceiverGeometry:
    """Tests for ReceiverGeometry."""

    def test_positions_shape(self, geometry: ReceiverGeometry):
        """Positions array has correct shape."""
        pos = geometry.get_positions()
        assert pos.shape == (3, 3)
        assert pos.dtype == np.float64

    def test_pair_indices_count(self, geometry: ReceiverGeometry):
        """Number of pairs is C(n,2)."""
        pairs = geometry.get_pair_indices()
        assert len(pairs) == 3  # C(3,2) = 3

    def test_pair_indices_ordered(self, geometry: ReceiverGeometry):
        """All pair indices satisfy i < j."""
        for i, j in geometry.get_pair_indices():
            assert i < j

    def test_baseline_symmetric(self, geometry: ReceiverGeometry):
        """Baseline distance is symmetric."""
        d01 = geometry.get_baseline(0, 1)
        d10 = geometry.get_baseline(1, 0)
        assert d01 == pytest.approx(d10, rel=1e-12)

    def test_baseline_positive(self, geometry: ReceiverGeometry):
        """All baselines are positive."""
        for i, j in geometry.get_pair_indices():
            assert geometry.get_baseline(i, j) > 0.0

    def test_equilateral_baselines_equal(self, geometry: ReceiverGeometry):
        """Equilateral triangle has equal baselines."""
        baselines = [geometry.get_baseline(i, j) for i, j in geometry.get_pair_indices()]
        assert baselines[0] == pytest.approx(baselines[1], rel=1e-10)
        assert baselines[0] == pytest.approx(baselines[2], rel=1e-10)

    def test_compute_tdoa_known_position(self, geometry: ReceiverGeometry, known_tx_pos):
        """TDoA computation matches manual calculation for a known position."""
        pos = geometry.get_positions()
        pair = (0, 1)
        d0 = np.linalg.norm(known_tx_pos - pos[0, :2])
        d1 = np.linalg.norm(known_tx_pos - pos[1, :2])
        expected_tdoa = (d0 - d1) / SPEED_OF_LIGHT
        computed_tdoa = geometry.compute_tdoa(known_tx_pos, pair)
        assert computed_tdoa == pytest.approx(expected_tdoa, rel=1e-12)

    def test_compute_all_tdoas_shape(self, geometry: ReceiverGeometry, known_tx_pos):
        """compute_all_tdoas returns correct number of values."""
        tdoas = geometry.compute_all_tdoas(known_tx_pos)
        assert tdoas.shape == (3,)

    def test_tdoa_at_centroid_near_zero(self, geometry: ReceiverGeometry):
        """TDoAs at the receiver centroid should be small for symmetric layout."""
        centroid = geometry.get_positions()[:, :2].mean(axis=0)
        tdoas = geometry.compute_all_tdoas(centroid)
        # For an equilateral triangle, centroid is equidistant from all receivers
        assert np.allclose(tdoas, 0.0, atol=1e-15)

    def test_minimum_receivers(self):
        """Should raise ValueError with fewer than 2 receivers."""
        with pytest.raises(ValueError, match="At least 2"):
            ReceiverGeometry([ReceiverConfig(id="RX0", latitude=0, longitude=0)])


# ---------------------------------------------------------------------------
# Cost function tests
# ---------------------------------------------------------------------------

class TestCostFunctions:
    """Tests for tdoa_residuals and tdoa_jacobian."""

    def test_residuals_zero_at_true_position(self, geometry: ReceiverGeometry, known_tx_pos):
        """Residuals should be zero when tx_pos produces the measured TDoAs exactly."""
        pairs = geometry.get_pair_indices()
        true_tdoas = geometry.compute_all_tdoas(known_tx_pos)
        weights = np.ones(len(pairs), dtype=np.float64)
        positions = geometry.get_positions()

        residuals = tdoa_residuals(known_tx_pos, positions, true_tdoas, pairs, weights)
        assert np.allclose(residuals, 0.0, atol=1e-15)

    def test_residuals_nonzero_at_wrong_position(self, geometry: ReceiverGeometry, known_tx_pos):
        """Residuals should be nonzero at a wrong position."""
        pairs = geometry.get_pair_indices()
        true_tdoas = geometry.compute_all_tdoas(known_tx_pos)
        weights = np.ones(len(pairs), dtype=np.float64)
        positions = geometry.get_positions()

        wrong_pos = known_tx_pos + np.array([10_000.0, 10_000.0])
        residuals = tdoa_residuals(wrong_pos, positions, true_tdoas, pairs, weights)
        assert not np.allclose(residuals, 0.0)

    def test_residuals_weighted(self, geometry: ReceiverGeometry, known_tx_pos):
        """Doubling weights should double residuals (away from zero)."""
        pairs = geometry.get_pair_indices()
        true_tdoas = geometry.compute_all_tdoas(known_tx_pos)
        positions = geometry.get_positions()
        wrong_pos = known_tx_pos + np.array([5_000.0, 5_000.0])

        w1 = np.ones(len(pairs))
        w2 = 2.0 * np.ones(len(pairs))
        r1 = tdoa_residuals(wrong_pos, positions, true_tdoas, pairs, w1)
        r2 = tdoa_residuals(wrong_pos, positions, true_tdoas, pairs, w2)
        np.testing.assert_allclose(r2, 2.0 * r1, rtol=1e-12)

    def test_jacobian_against_finite_differences(self, geometry: ReceiverGeometry, known_tx_pos):
        """Analytical Jacobian should match numerical finite-difference approximation."""
        pairs = geometry.get_pair_indices()
        true_tdoas = geometry.compute_all_tdoas(known_tx_pos)
        weights = np.ones(len(pairs), dtype=np.float64)
        positions = geometry.get_positions()

        # Evaluate at a point away from true position for non-degenerate gradients
        test_pos = known_tx_pos + np.array([5_000.0, -3_000.0])

        # Analytical Jacobian
        jac_analytical = tdoa_jacobian(test_pos, positions, pairs, weights)

        # Numerical Jacobian via central finite differences
        eps = 1.0  # 1 meter perturbation
        ndim = len(test_pos)
        n_pairs = len(pairs)
        jac_numerical = np.empty((n_pairs, ndim), dtype=np.float64)

        for d in range(ndim):
            p_plus = test_pos.copy()
            p_minus = test_pos.copy()
            p_plus[d] += eps
            p_minus[d] -= eps
            r_plus = tdoa_residuals(p_plus, positions, true_tdoas, pairs, weights)
            r_minus = tdoa_residuals(p_minus, positions, true_tdoas, pairs, weights)
            jac_numerical[:, d] = (r_plus - r_minus) / (2.0 * eps)

        np.testing.assert_allclose(jac_analytical, jac_numerical, rtol=1e-6, atol=1e-12)

    def test_jacobian_shape(self, geometry: ReceiverGeometry, known_tx_pos):
        """Jacobian has correct shape (n_pairs, ndim)."""
        pairs = geometry.get_pair_indices()
        weights = np.ones(len(pairs))
        positions = geometry.get_positions()
        jac = tdoa_jacobian(known_tx_pos, positions, pairs, weights)
        assert jac.shape == (3, 2)


# ---------------------------------------------------------------------------
# Solver tests
# ---------------------------------------------------------------------------

class TestTDoASolver:
    """Tests for TDoASolver."""

    def test_solver_exact_recovery(self, geometry: ReceiverGeometry, known_tx_pos):
        """Solver should exactly recover position from noiseless TDoAs."""
        true_tdoas = geometry.compute_all_tdoas(known_tx_pos)
        weights = np.ones(len(true_tdoas), dtype=np.float64)

        solver = TDoASolver(geometry)
        result = solver.solve(true_tdoas, weights, initial_guess=known_tx_pos + 1000.0)

        recovered = np.array([result.x, result.y])
        np.testing.assert_allclose(recovered, known_tx_pos, atol=1.0)  # within 1 m

    def test_solver_exact_recovery_no_initial_guess(
        self, geometry: ReceiverGeometry, known_tx_pos
    ):
        """Solver should recover position even without an initial guess."""
        true_tdoas = geometry.compute_all_tdoas(known_tx_pos)
        weights = np.ones(len(true_tdoas), dtype=np.float64)

        solver = TDoASolver(geometry, config={"extent_km": 200, "n_grid_points": 30})
        result = solver.solve(true_tdoas, weights)

        recovered = np.array([result.x, result.y])
        # With grid search initialization, should still converge well
        np.testing.assert_allclose(recovered, known_tx_pos, atol=10.0)  # within 10 m

    def test_solver_residual_near_zero_for_exact(
        self, geometry: ReceiverGeometry, known_tx_pos
    ):
        """Residual should be near zero for exact TDoAs."""
        true_tdoas = geometry.compute_all_tdoas(known_tx_pos)
        weights = np.ones(len(true_tdoas), dtype=np.float64)

        solver = TDoASolver(geometry)
        result = solver.solve(true_tdoas, weights, initial_guess=known_tx_pos)

        assert result.residual < 1e-20

    def test_solver_noisy_tdoas_within_10km(self, geometry: ReceiverGeometry, known_tx_pos):
        """Solver recovers position within 10 km for ~100 km baseline at 20 dB SNR.

        At 20 dB SNR with 48 kHz bandwidth, the TDoA noise standard deviation
        is approximately 1/(BW * sqrt(SNR_linear)) ~ 2 microseconds. We add
        Gaussian noise at this level and verify the solution is within 10 km.
        """
        rng = np.random.default_rng(12345)
        true_tdoas = geometry.compute_all_tdoas(known_tx_pos)

        # SNR = 20 dB -> linear factor = 100
        # TDoA std ~ 1/(BW * sqrt(SNR)) = 1/(48000 * 10) ~ 2.08 us
        snr_linear = 100.0
        bw = 48_000.0
        tdoa_std = 1.0 / (bw * np.sqrt(snr_linear))
        noise = rng.normal(0.0, tdoa_std, size=len(true_tdoas))
        noisy_tdoas = true_tdoas + noise

        # Weights = 1/sigma (inverse of standard deviation)
        weights = np.full(len(true_tdoas), 1.0 / tdoa_std, dtype=np.float64)

        solver = TDoASolver(geometry, config={"extent_km": 200, "n_grid_points": 30})
        result = solver.solve(noisy_tdoas, weights)

        recovered = np.array([result.x, result.y])
        error_m = np.linalg.norm(recovered - known_tx_pos)
        assert error_m < 10_000.0, f"Position error {error_m/1e3:.1f} km exceeds 10 km limit"

    def test_solver_covariance_shape(self, geometry: ReceiverGeometry, known_tx_pos):
        """Covariance in the result should be 2x2."""
        true_tdoas = geometry.compute_all_tdoas(known_tx_pos)
        weights = np.ones(len(true_tdoas), dtype=np.float64)

        solver = TDoASolver(geometry)
        result = solver.solve(true_tdoas, weights, initial_guess=known_tx_pos)

        assert result.covariance.shape == (2, 2)

    def test_solver_covariance_positive_definite(
        self, geometry: ReceiverGeometry, known_tx_pos
    ):
        """Covariance should be symmetric positive definite."""
        true_tdoas = geometry.compute_all_tdoas(known_tx_pos)
        weights = np.ones(len(true_tdoas), dtype=np.float64)

        solver = TDoASolver(geometry)
        result = solver.solve(true_tdoas, weights, initial_guess=known_tx_pos)

        cov = result.covariance
        # Symmetric
        np.testing.assert_allclose(cov, cov.T, atol=1e-10)
        # Positive eigenvalues
        eigvals = np.linalg.eigvalsh(cov)
        assert np.all(eigvals > 0), f"Non-positive eigenvalues: {eigvals}"

    def test_solver_with_pentagon_receivers(self, pentagon_receivers, known_tx_pos):
        """Solver works with 5 receivers (10 pairs)."""
        geometry = ReceiverGeometry(pentagon_receivers)
        true_tdoas = geometry.compute_all_tdoas(known_tx_pos)
        weights = np.ones(len(true_tdoas), dtype=np.float64)

        solver = TDoASolver(geometry)
        result = solver.solve(true_tdoas, weights, initial_guess=known_tx_pos + 500.0)

        recovered = np.array([result.x, result.y])
        np.testing.assert_allclose(recovered, known_tx_pos, atol=1.0)


# ---------------------------------------------------------------------------
# Grid search tests
# ---------------------------------------------------------------------------

class TestGridSearch:
    """Tests for grid search initialization."""

    def test_grid_search_near_true_position(self, geometry: ReceiverGeometry, known_tx_pos):
        """Grid search should find a point reasonably close to the true position."""
        true_tdoas = geometry.compute_all_tdoas(known_tx_pos)
        weights = np.ones(len(true_tdoas), dtype=np.float64)

        solver = TDoASolver(geometry, config={"extent_km": 200, "n_grid_points": 50})
        init_pos = solver._grid_search_init(true_tdoas, weights)

        error_m = np.linalg.norm(init_pos - known_tx_pos)
        # Grid spacing is ~8 km at 50 points over 400 km, so should be within ~15 km
        assert error_m < 20_000.0, (
            f"Grid search init error {error_m/1e3:.1f} km exceeds 20 km limit"
        )

    def test_grid_search_returns_2d(self, geometry: ReceiverGeometry, known_tx_pos):
        """Grid search should return a 2-D position."""
        true_tdoas = geometry.compute_all_tdoas(known_tx_pos)
        weights = np.ones(len(true_tdoas))

        solver = TDoASolver(geometry)
        init_pos = solver._grid_search_init(true_tdoas, weights, n_points=10, extent_km=100)
        assert init_pos.shape == (2,)


# ---------------------------------------------------------------------------
# Posterior / confidence ellipse tests
# ---------------------------------------------------------------------------

class TestPosterior:
    """Tests for confidence ellipse and uncertainty functions."""

    def test_confidence_ellipse_identity_covariance(self):
        """Identity covariance gives equal semi-axes."""
        cov = np.eye(2)
        semi_major, semi_minor, angle = confidence_ellipse(cov, confidence=0.95)
        # Both eigenvalues are 1, so semi-axes should be equal
        assert semi_major == pytest.approx(semi_minor, rel=1e-10)
        # chi2(0.95, 2) ~ 5.991, so semi-axis ~ sqrt(5.991) ~ 2.448
        expected = np.sqrt(chi2_dist.ppf(0.95, df=2))
        assert semi_major == pytest.approx(expected, rel=1e-6)

    def test_confidence_ellipse_diagonal_covariance(self):
        """Diagonal covariance gives axes aligned with coordinate axes."""
        cov = np.diag([4.0, 1.0])
        semi_major, semi_minor, angle = confidence_ellipse(cov, confidence=0.95)
        chi2_val = chi2_dist.ppf(0.95, df=2)
        assert semi_major == pytest.approx(np.sqrt(chi2_val * 4.0), rel=1e-6)
        assert semi_minor == pytest.approx(np.sqrt(chi2_val * 1.0), rel=1e-6)
        # Major axis should be along x (angle ~ 0 degrees)
        assert abs(angle) < 1.0 or abs(abs(angle) - 180.0) < 1.0

    def test_confidence_ellipse_positive_axes(self):
        """Semi-axes should always be non-negative."""
        cov = np.array([[2.0, 0.5], [0.5, 1.0]])
        semi_major, semi_minor, _ = confidence_ellipse(cov, confidence=0.95)
        assert semi_major >= 0.0
        assert semi_minor >= 0.0
        assert semi_major >= semi_minor

    def test_confidence_radius_larger_than_semi_minor(self):
        """Circular radius should be >= semi-minor axis."""
        cov = np.array([[4.0, 1.0], [1.0, 2.0]])
        semi_major, semi_minor, _ = confidence_ellipse(cov, confidence=0.95)
        radius = confidence_radius(cov, confidence=0.95)
        assert radius >= semi_minor - 1e-10
        assert radius == pytest.approx(semi_major, rel=1e-10)

    def test_confidence_radius_scales_with_confidence(self):
        """Higher confidence level should give a larger radius."""
        cov = np.array([[3.0, 0.5], [0.5, 1.0]])
        r_90 = confidence_radius(cov, confidence=0.90)
        r_95 = confidence_radius(cov, confidence=0.95)
        r_99 = confidence_radius(cov, confidence=0.99)
        assert r_90 < r_95 < r_99

    def test_position_uncertainty_dict_keys(self):
        """position_uncertainty returns a dict with expected keys."""
        est = PositionEstimate(
            x=0.0,
            y=0.0,
            covariance=np.diag([100.0, 50.0]),
        )
        result = position_uncertainty(est, confidence=0.95)
        assert "semi_major" in result
        assert "semi_minor" in result
        assert "angle_degrees" in result
        assert "radius" in result
        assert "cep" in result

    def test_cep_less_than_95_radius(self):
        """CEP (50% radius) should be smaller than 95% radius."""
        est = PositionEstimate(
            x=0.0,
            y=0.0,
            covariance=np.array([[4.0, 1.0], [1.0, 2.0]]),
        )
        result = position_uncertainty(est, confidence=0.95)
        assert result["cep"] < result["radius"]

    def test_confidence_ellipse_rejects_wrong_shape(self):
        """Should raise ValueError for non-2x2 input."""
        with pytest.raises(ValueError, match="2x2"):
            confidence_ellipse(np.eye(3))

    def test_confidence_radius_rejects_wrong_shape(self):
        """Should raise ValueError for non-2x2 input."""
        with pytest.raises(ValueError, match="2x2"):
            confidence_radius(np.eye(3))
