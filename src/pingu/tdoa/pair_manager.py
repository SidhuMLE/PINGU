"""Receiver-pair management for TDoA estimation.

Given a list of *N* receivers, this module generates the C(N, 2) unique pairs
and orchestrates TDoA estimation across all of them.
"""

from __future__ import annotations

from itertools import combinations
from typing import Literal

import numpy as np
from numpy.typing import NDArray

from pingu.tdoa.gcc import estimate_tdoa
from pingu.types import TDoAEstimate


class PairManager:
    """Manage all unique receiver-pair combinations.

    Args:
        receiver_ids: Sequence of receiver identifiers (e.g.
            ``["RX0", "RX1", "RX2", "RX3", "RX4"]``).
    """

    def __init__(self, receiver_ids: list[str]) -> None:
        if len(receiver_ids) < 2:
            raise ValueError("Need at least 2 receivers to form pairs.")
        self._ids = list(receiver_ids)
        self._pairs: list[tuple[str, str]] = list(combinations(self._ids, 2))

    # ------------------------------------------------------------------ #
    # Properties
    # ------------------------------------------------------------------ #

    @property
    def receiver_ids(self) -> list[str]:
        """Return the list of receiver identifiers."""
        return list(self._ids)

    @property
    def n_pairs(self) -> int:
        """Number of unique pairs (C(N, 2))."""
        return len(self._pairs)

    # ------------------------------------------------------------------ #
    # Public methods
    # ------------------------------------------------------------------ #

    def get_pairs(self) -> list[tuple[str, str]]:
        """Return all unique receiver pairs.

        Returns:
            List of ``(receiver_i, receiver_j)`` tuples with ``i < j`` in
            the original ordering.
        """
        return list(self._pairs)

    def estimate_all_tdoas(
        self,
        signals: dict[str, NDArray],
        fs: float,
        method: Literal["phat", "scot", "ml"] = "phat",
        center_freq: float = 0.0,
        timestamp: float = 0.0,
        **kwargs,
    ) -> list[TDoAEstimate]:
        """Estimate TDoA for every receiver pair.

        Args:
            signals: Mapping from receiver ID to its signal array.
                Every receiver listed in :pyattr:`receiver_ids` must have
                an entry.
            fs: Sampling frequency (Hz), assumed identical for all receivers.
            method: GCC weighting method (``"phat"``, ``"scot"``, ``"ml"``).
            center_freq: Centre frequency of the signal (Hz).
            timestamp: Epoch timestamp of the measurement.
            **kwargs: Additional keyword arguments forwarded to
                :func:`estimate_tdoa`.

        Returns:
            A list of :class:`TDoAEstimate` objects, one per pair.

        Raises:
            KeyError: If a required receiver ID is not present in *signals*.
        """
        missing = set(self._ids) - set(signals)
        if missing:
            raise KeyError(f"Missing signal data for receiver(s): {missing}")

        estimates: list[TDoAEstimate] = []
        for rx_i, rx_j in self._pairs:
            est = estimate_tdoa(
                x=signals[rx_i],
                y=signals[rx_j],
                fs=fs,
                method=method,
                receiver_i=rx_i,
                receiver_j=rx_j,
                center_freq=center_freq,
                timestamp=timestamp,
                **kwargs,
            )
            estimates.append(est)

        return estimates
