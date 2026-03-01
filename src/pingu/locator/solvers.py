"""Levenberg-Marquardt TDoA solver with grid-search initialization."""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray
from scipy.optimize import least_squares

from pingu.types import PositionEstimate
from pingu.locator.geometry import ReceiverGeometry
from pingu.locator.cost_functions import tdoa_residuals, tdoa_jacobian


class TDoASolver:
    """Non-linear least-squares solver for TDoA-based geolocation.

    Uses scipy.optimize.least_squares with the Levenberg-Marquardt algorithm
    to minimize weighted TDoA residuals. When no initial guess is provided,
    a coarse grid search around the receiver centroid is used for initialization.
    """

    def __init__(
        self,
        geometry: ReceiverGeometry,
        config: dict | None = None,
    ) -> None:
        """Initialize the solver.

        Args:
            geometry: ReceiverGeometry instance describing station layout.
            config: Optional dict of solver parameters. Supported keys:
                - n_grid_points (int): Points per axis in grid search. Default 50.
                - extent_km (float): Half-width of search grid in km. Default 500.
                - ftol (float): Function tolerance for least_squares. Default 1e-12.
                - xtol (float): Variable tolerance for least_squares. Default 1e-12.
                - gtol (float): Gradient tolerance for least_squares. Default 1e-12.
                - max_nfev (int): Max function evaluations. Default 1000.
        """
        self._geometry = geometry
        self._config = config or {}
        self._pair_indices = geometry.get_pair_indices()
        self._positions = geometry.get_positions()

    def solve(
        self,
        tdoas: NDArray[np.float64],
        weights: NDArray[np.float64],
        initial_guess: NDArray[np.float64] | None = None,
    ) -> PositionEstimate:
        """Estimate the transmitter position from TDoA measurements.

        Args:
            tdoas: Measured TDoA values, shape (n_pairs,).
            weights: Per-pair weights, shape (n_pairs,). Typically 1/sigma_k.
            initial_guess: Starting position, shape (2,) or (3,). If None, a
                grid search is performed to find a reasonable starting point.

        Returns:
            PositionEstimate with solved position, covariance, and residual.
        """
        tdoas = np.asarray(tdoas, dtype=np.float64)
        weights = np.asarray(weights, dtype=np.float64)

        if initial_guess is None:
            x0 = self._grid_search_init(tdoas, weights)
        else:
            x0 = np.asarray(initial_guess, dtype=np.float64)

        ndim = len(x0)

        # Partial-bind the receiver positions and pair indices for scipy interface
        def residual_fn(p: NDArray) -> NDArray:
            return tdoa_residuals(p, self._positions, tdoas, self._pair_indices, weights)

        def jacobian_fn(p: NDArray) -> NDArray:
            return tdoa_jacobian(p, self._positions, self._pair_indices, weights)

        result = least_squares(
            residual_fn,
            x0,
            jac=jacobian_fn,
            method="lm",
            ftol=self._config.get("ftol", 1e-12),
            xtol=self._config.get("xtol", 1e-12),
            gtol=self._config.get("gtol", 1e-12),
            max_nfev=self._config.get("max_nfev", 1000),
        )

        # Extract solution
        pos = result.x
        residual = float(np.sum(result.fun ** 2))

        # Compute covariance: cov = inv(J^T W J)
        # Since weights are already baked into the Jacobian and residuals,
        # the Jacobian at the solution already includes the weighting.
        # cov = inv(J^T J) gives the covariance of the weighted problem.
        jac_at_sol = jacobian_fn(pos)
        jtj = jac_at_sol.T @ jac_at_sol
        try:
            covariance = np.linalg.inv(jtj)
        except np.linalg.LinAlgError:
            # Fallback: use pseudo-inverse for rank-deficient cases
            covariance = np.linalg.pinv(jtj)

        # Build 2x2 covariance for the PositionEstimate (x, y components)
        cov_2x2 = covariance[:2, :2].copy()

        # Compute 95% confidence radius from covariance
        eigenvalues = np.linalg.eigvalsh(cov_2x2)
        # chi2 value for 95% confidence with 2 DOF is ~5.991
        from scipy.stats import chi2 as chi2_dist

        chi2_val = chi2_dist.ppf(0.95, df=2)
        conf_radius = float(np.sqrt(chi2_val * np.max(eigenvalues)))

        return PositionEstimate(
            x=float(pos[0]),
            y=float(pos[1]),
            z=float(pos[2]) if ndim > 2 else 0.0,
            covariance=cov_2x2,
            residual=residual,
            confidence_radius_95=conf_radius,
        )

    def _grid_search_init(
        self,
        tdoas: NDArray[np.float64],
        weights: NDArray[np.float64],
        n_points: int = 50,
        extent_km: float = 500.0,
    ) -> NDArray[np.float64]:
        """Find initial guess via brute-force grid search.

        Evaluates the cost function on a regular 2-D grid centered on the
        receiver centroid and returns the grid point with the lowest cost.

        Args:
            tdoas: Measured TDoA values, shape (n_pairs,).
            weights: Per-pair weights, shape (n_pairs,).
            n_points: Number of grid points per axis.
            extent_km: Half-width of the search area in kilometers.

        Returns:
            Best grid point as shape (2,) array.
        """
        n_points = self._config.get("n_grid_points", n_points)
        extent_km = self._config.get("extent_km", extent_km)

        # Centroid of receivers in 2-D
        centroid = self._positions[:, :2].mean(axis=0)
        extent_m = extent_km * 1e3

        xs = np.linspace(centroid[0] - extent_m, centroid[0] + extent_m, n_points)
        ys = np.linspace(centroid[1] - extent_m, centroid[1] + extent_m, n_points)

        best_cost = np.inf
        best_pos = centroid.copy()

        for xi in xs:
            for yi in ys:
                candidate = np.array([xi, yi], dtype=np.float64)
                res = tdoa_residuals(
                    candidate, self._positions, tdoas, self._pair_indices, weights
                )
                cost = float(np.sum(res ** 2))
                if cost < best_cost:
                    best_cost = cost
                    best_pos = candidate

        return best_pos
