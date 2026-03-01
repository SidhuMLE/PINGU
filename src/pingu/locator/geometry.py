"""Receiver geometry management for TDoA geolocation."""

from __future__ import annotations

from itertools import combinations

import numpy as np
from numpy.typing import NDArray

from pingu.constants import SPEED_OF_LIGHT
from pingu.types import ReceiverConfig


class ReceiverGeometry:
    """Manages receiver positions and computes geometric relationships.

    Stores receiver station positions in Cartesian coordinates and provides
    methods for computing baselines, pair indices, and theoretical TDoA values
    for arbitrary transmitter positions.
    """

    def __init__(self, receivers: list[ReceiverConfig]) -> None:
        """Initialize geometry from a list of receiver configurations.

        Args:
            receivers: List of ReceiverConfig objects with Cartesian x, y, z fields.
        """
        if len(receivers) < 2:
            raise ValueError("At least 2 receivers are required for TDoA geolocation.")
        self._receivers = receivers
        # Build (n_receivers, 3) position array from Cartesian coordinates.
        self._positions: NDArray[np.float64] = np.array(
            [[r.x, r.y, r.z] for r in receivers], dtype=np.float64
        )

    @property
    def n_receivers(self) -> int:
        """Number of receiver stations."""
        return len(self._receivers)

    def get_positions(self) -> NDArray[np.float64]:
        """Return receiver positions as an array.

        Returns:
            Array of shape (n_receivers, 3) with columns (x, y, z).
            If all z values are zero the full 3-column array is still returned;
            callers may slice to (n_receivers, 2) for 2-D problems.
        """
        return self._positions.copy()

    def get_pair_indices(self) -> list[tuple[int, int]]:
        """Return all C(n, 2) receiver pair indices.

        Returns:
            List of (i, j) tuples with i < j, covering every unique pair.
        """
        return list(combinations(range(self.n_receivers), 2))

    def get_baseline(self, i: int, j: int) -> float:
        """Euclidean distance between receivers i and j.

        Args:
            i: Index of the first receiver.
            j: Index of the second receiver.

        Returns:
            Baseline distance in meters.
        """
        diff = self._positions[i] - self._positions[j]
        return float(np.linalg.norm(diff))

    def compute_tdoa(self, tx_pos: NDArray[np.float64], pair: tuple[int, int]) -> float:
        """Compute the true TDoA for a transmitter position and a receiver pair.

        TDoA is defined as tau_ij = (||tx - r_i|| - ||tx - r_j||) / c, where
        a positive value means the signal arrives at receiver i *after* receiver j.

        Args:
            tx_pos: Transmitter position, shape (2,) or (3,).
            pair: Tuple (i, j) of receiver indices.

        Returns:
            Time difference of arrival in seconds.
        """
        tx = np.asarray(tx_pos, dtype=np.float64)
        i, j = pair
        ri = self._positions[i, : len(tx)]
        rj = self._positions[j, : len(tx)]
        dist_i = float(np.linalg.norm(tx - ri))
        dist_j = float(np.linalg.norm(tx - rj))
        return (dist_i - dist_j) / SPEED_OF_LIGHT

    def compute_all_tdoas(self, tx_pos: NDArray[np.float64]) -> NDArray[np.float64]:
        """Compute TDoAs for all receiver pairs.

        Args:
            tx_pos: Transmitter position, shape (2,) or (3,).

        Returns:
            Array of shape (n_pairs,) with TDoA values in seconds, ordered
            consistently with get_pair_indices().
        """
        pairs = self.get_pair_indices()
        return np.array([self.compute_tdoa(tx_pos, p) for p in pairs], dtype=np.float64)
