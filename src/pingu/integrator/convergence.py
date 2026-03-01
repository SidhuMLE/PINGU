"""Convergence monitoring for the Bayesian TDoA integration loop.

Tracks the evolution of the posterior covariance over successive Kalman filter
updates and determines when the filter has effectively converged (i.e., further
observations yield negligible improvement).
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray


class ConvergenceMonitor:
    """Monitor posterior variance convergence during Kalman integration.

    After each filter update the caller should pass the current covariance
    matrix (and optionally the CRLB) to :meth:`update`.  The monitor records
    the diagonal variances and can test for convergence.

    Parameters
    ----------
    n_pairs : int
        Number of TDoA pairs being tracked.
    rate_threshold : float, optional
        Fractional change in mean variance below which the filter is considered
        converged when no CRLB is available.  Default is ``0.01`` (1 %).
    min_updates : int, optional
        Minimum number of updates before convergence can be declared.
        Default is ``5``.
    """

    def __init__(
        self,
        n_pairs: int,
        rate_threshold: float = 0.01,
        min_updates: int = 5,
    ) -> None:
        if n_pairs < 1:
            raise ValueError(f"n_pairs must be >= 1, got {n_pairs}")

        self._n_pairs = n_pairs
        self._rate_threshold = rate_threshold
        self._min_updates = min_updates

        # History of diagonal variances: list of 1-D arrays, shape (n_pairs,).
        self._variance_history: list[NDArray[np.float64]] = []

        # Optional CRLB diagonal, updated each call.
        self._crlb_diag: NDArray[np.float64] | None = None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    @property
    def n_updates(self) -> int:
        """Number of covariance snapshots recorded so far."""
        return len(self._variance_history)

    def update(
        self,
        covariance: NDArray[np.float64],
        crlb: NDArray[np.float64] | None = None,
    ) -> None:
        """Record the current posterior covariance.

        Parameters
        ----------
        covariance : ndarray, shape (n_pairs, n_pairs)
            Full posterior covariance matrix from the Kalman filter.
        crlb : ndarray, shape (n_pairs, n_pairs) or (n_pairs,), optional
            Cramer-Rao lower bound.  May be a full matrix (diagonal is used)
            or a 1-D vector of per-pair lower bounds.
        """
        cov = np.asarray(covariance, dtype=np.float64)
        variances = np.diag(cov) if cov.ndim == 2 else cov.ravel().copy()
        self._variance_history.append(variances)

        if crlb is not None:
            crlb_arr = np.asarray(crlb, dtype=np.float64)
            self._crlb_diag = np.diag(crlb_arr) if crlb_arr.ndim == 2 else crlb_arr.ravel().copy()

    def is_converged(self, factor: float = 0.1) -> bool:
        """Check whether the filter has converged.

        Convergence criteria (evaluated in order):

        1. If a CRLB is available, convergence is declared when *all*
           posterior variances are below ``factor * CRLB_diag``.
        2. Otherwise, convergence is declared when the relative change in
           mean variance between the last two updates is below
           ``rate_threshold``.

        In both cases, at least ``min_updates`` snapshots must have been
        recorded.

        Parameters
        ----------
        factor : float, optional
            Multiplier on the CRLB diagonal.  Default is ``0.1`` (variances
            must be within 10 % of the theoretical lower bound).

        Returns
        -------
        bool
        """
        if len(self._variance_history) < self._min_updates:
            return False

        current = self._variance_history[-1]

        # Criterion 1: CRLB-based.
        if self._crlb_diag is not None:
            return bool(np.all(current < factor * self._crlb_diag))

        # Criterion 2: rate-of-change based.
        previous = self._variance_history[-2]
        mean_prev = np.mean(previous)
        if mean_prev == 0.0:
            return True
        relative_change = np.abs(np.mean(current) - mean_prev) / mean_prev
        return bool(relative_change < self._rate_threshold)

    def get_variance_history(self) -> NDArray[np.float64]:
        """Return the recorded variance history.

        Returns
        -------
        ndarray, shape (n_updates, n_pairs)
            Each row is the diagonal of the covariance at that update step.
        """
        if not self._variance_history:
            return np.empty((0, self._n_pairs), dtype=np.float64)
        return np.array(self._variance_history, dtype=np.float64)

    def get_improvement_ratio(self) -> float:
        """Ratio of initial to current mean variance.

        Returns
        -------
        float
            ``initial_mean_variance / current_mean_variance``.  A value of 10
            means the variance has been reduced by a factor of 10.  Returns 1.0
            if fewer than 2 updates have been recorded.
        """
        if len(self._variance_history) < 2:
            return 1.0

        initial_mean = float(np.mean(self._variance_history[0]))
        current_mean = float(np.mean(self._variance_history[-1]))

        if current_mean == 0.0:
            return float("inf")
        return initial_mean / current_mean
