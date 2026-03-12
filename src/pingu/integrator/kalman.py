"""Kalman filter for stationary-target TDoA integration.

Implements a linear Kalman filter where the state vector holds the TDoA values
for each receiver pair.  Since the target is stationary the TDoAs do not change,
so the state transition model is the identity matrix.  Each ``update`` step
fuses a new vector of TDoA observations, progressively shrinking the posterior
covariance toward the Cramer-Rao lower bound.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from pingu.types import IntegratedTDoA


class TDoAKalmanFilter:
    """Linear Kalman filter for integrating TDoA measurements over time.

    Parameters
    ----------
    n_pairs : int
        Number of receiver pairs tracked in the state vector.  For 5 receivers
        there are C(5,2) = 10 unique pairs.
    process_noise : float, optional
        Scalar variance added per prediction step along the diagonal of Q.
        Should be very small for a stationary target.  Default is ``1e-12``.
    initial_variance : float, optional
        Initial diagonal variance of the state covariance P.  Default is
        ``1e-6`` (seconds squared).
    pair_labels : list[tuple[str, str]] | None, optional
        Human-readable labels for each receiver pair.  If *None*, generic
        labels ``("i", "j")`` are generated.
    """

    def __init__(
        self,
        n_pairs: int,
        process_noise: float = 1e-12,
        initial_variance: float = 1e-6,
        pair_labels: list[tuple[str, str]] | None = None,
    ) -> None:
        if n_pairs < 1:
            raise ValueError(f"n_pairs must be >= 1, got {n_pairs}")

        self._n_pairs = n_pairs
        self._process_noise = process_noise
        self._initial_variance = initial_variance

        # Pair labels for IntegratedTDoA output.
        if pair_labels is not None:
            if len(pair_labels) != n_pairs:
                raise ValueError(
                    f"pair_labels length ({len(pair_labels)}) must match n_pairs ({n_pairs})"
                )
            self._pair_labels = list(pair_labels)
        else:
            self._pair_labels = [(str(i), str(j)) for i, j in _default_pair_indices(n_pairs)]

        # State vector: n_pairs TDoA values, initialised to zero.
        self._x: NDArray[np.float64] = np.zeros(n_pairs, dtype=np.float64)

        # State covariance: diagonal with initial_variance.
        self._P: NDArray[np.float64] = np.eye(n_pairs, dtype=np.float64) * initial_variance

        # Process noise covariance (constant).
        self._Q: NDArray[np.float64] = np.eye(n_pairs, dtype=np.float64) * process_noise

        # Count of measurement updates applied.
        self._n_updates: int = 0

        # Timestamp of the most recent update.
        self._timestamp: float = 0.0

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    @property
    def n_pairs(self) -> int:
        """Number of TDoA pairs in the state vector."""
        return self._n_pairs

    @property
    def n_updates(self) -> int:
        """Number of measurement updates applied so far."""
        return self._n_updates

    def predict(self) -> None:
        """Propagate the state and covariance one step forward.

        Because the target is stationary the state transition matrix is the
        identity, so the predicted state equals the current state and only the
        covariance grows by the process noise Q.
        """
        # x_pred = F @ x = I @ x = x  (no change)
        # P_pred = F @ P @ F.T + Q = P + Q
        self._P = self._P + self._Q

    def update(
        self,
        measurements: NDArray[np.float64],
        measurement_noise: NDArray[np.float64],
        timestamp: float = 0.0,
    ) -> NDArray[np.float64]:
        """Incorporate a new set of TDoA observations.

        Parameters
        ----------
        measurements : ndarray, shape (n_pairs,)
            Observed TDoA delays for each receiver pair (seconds).
        measurement_noise : ndarray, shape (n_pairs,)
            Variance of each TDoA measurement (seconds squared).  These form
            the diagonal of the measurement noise covariance R.
        timestamp : float, optional
            UNIX epoch time associated with this measurement set.

        Returns
        -------
        ndarray, shape (n_pairs,)
            Innovation vector (z - x_prior).
        """
        z = np.asarray(measurements, dtype=np.float64).ravel()
        r = np.asarray(measurement_noise, dtype=np.float64).ravel()

        if z.shape[0] != self._n_pairs:
            raise ValueError(
                f"measurements length ({z.shape[0]}) must equal n_pairs ({self._n_pairs})"
            )
        if r.shape[0] != self._n_pairs:
            raise ValueError(
                f"measurement_noise length ({r.shape[0]}) must equal n_pairs ({self._n_pairs})"
            )

        # Observation model: H = I (we directly observe the state).
        # Innovation: y = z - H @ x_pred = z - x
        y = z - self._x

        # Measurement noise covariance (diagonal).
        R = np.diag(r)

        # Innovation covariance: S = H @ P @ H.T + R = P + R
        S = self._P + R

        # Kalman gain: K = P @ H.T @ inv(S) = P @ inv(S)
        K = np.linalg.solve(S.T, self._P.T).T  # equivalent to P @ inv(S), numerically stable

        # State update.
        self._x = self._x + K @ y

        # Covariance update using the Joseph stabilised form:
        #   P = (I - K @ H) @ P @ (I - K @ H).T + K @ R @ K.T
        I_KH = np.eye(self._n_pairs) - K  # since H = I
        self._P = I_KH @ self._P @ I_KH.T + K @ R @ K.T

        # Enforce symmetry to prevent numerical drift.
        self._P = 0.5 * (self._P + self._P.T)

        self._n_updates += 1
        self._timestamp = timestamp

        return y.copy()

    def get_state(self) -> IntegratedTDoA:
        """Return the current filtered TDoA estimates as an ``IntegratedTDoA``.

        Returns
        -------
        IntegratedTDoA
            Contains the posterior mean delays, full covariance matrix, pair
            labels, update count, and timestamp.
        """
        return IntegratedTDoA(
            delays=self._x.copy(),
            covariance=self._P.copy(),
            pair_labels=list(self._pair_labels),
            n_updates=self._n_updates,
            timestamp=self._timestamp,
        )

    def reset(self) -> None:
        """Re-initialise the filter to its prior state."""
        self._x[:] = 0.0
        self._P = np.eye(self._n_pairs, dtype=np.float64) * self._initial_variance
        self._n_updates = 0
        self._timestamp = 0.0


# ------------------------------------------------------------------
# Helper
# ------------------------------------------------------------------

def _default_pair_indices(n_pairs: int) -> list[tuple[int, int]]:
    """Generate sequential pair indices (i, j) for *n_pairs* pairs.

    The pairs correspond to the upper triangle of a receiver matrix.  We
    reconstruct the number of receivers from n_pairs = C(N, 2) and enumerate
    accordingly.  If n_pairs does not correspond to a valid C(N, 2) we fall
    back to simple ``(0, k)`` indices.
    """
    # Solve N*(N-1)/2 = n_pairs for N.
    N = int(0.5 + 0.5 * (1 + np.sqrt(1 + 8 * n_pairs)))
    if N * (N - 1) // 2 == n_pairs:
        pairs: list[tuple[int, int]] = []
        for i in range(N):
            for j in range(i + 1, N):
                pairs.append((i, j))
        return pairs
    # Fallback: flat numbering.
    return [(0, k) for k in range(n_pairs)]
