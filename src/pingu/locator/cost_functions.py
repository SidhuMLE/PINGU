"""Weighted NLLS cost functions for TDoA-based geolocation."""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from pingu.constants import SPEED_OF_LIGHT


def tdoa_residuals(
    tx_pos: NDArray[np.float64],
    receiver_positions: NDArray[np.float64],
    tdoa_measurements: NDArray[np.float64],
    pair_indices: list[tuple[int, int]],
    weights: NDArray[np.float64],
) -> NDArray[np.float64]:
    """Compute weighted residual vector for TDoA least-squares.

    Each residual is:
        r_k = w_k * (tau_hat_k - (1/c) * (||p - r_i|| - ||p - r_j||))

    where tau_hat_k is the measured TDoA for pair (i, j), p is the candidate
    transmitter position, and w_k is the weight for that pair.

    Args:
        tx_pos: Candidate transmitter position, shape (2,) or (3,).
        receiver_positions: Receiver positions, shape (n_receivers, D) where D >= len(tx_pos).
        tdoa_measurements: Measured TDoA values, shape (n_pairs,).
        pair_indices: List of (i, j) index tuples for each measurement.
        weights: Per-pair weights, shape (n_pairs,).

    Returns:
        Weighted residual vector, shape (n_pairs,).
    """
    tx = np.asarray(tx_pos, dtype=np.float64)
    ndim = len(tx)
    residuals = np.empty(len(pair_indices), dtype=np.float64)

    for k, (i, j) in enumerate(pair_indices):
        ri = receiver_positions[i, :ndim]
        rj = receiver_positions[j, :ndim]
        dist_i = np.linalg.norm(tx - ri)
        dist_j = np.linalg.norm(tx - rj)
        predicted_tdoa = (dist_i - dist_j) / SPEED_OF_LIGHT
        residuals[k] = weights[k] * (tdoa_measurements[k] - predicted_tdoa)

    return residuals


def tdoa_jacobian(
    tx_pos: NDArray[np.float64],
    receiver_positions: NDArray[np.float64],
    pair_indices: list[tuple[int, int]],
    weights: NDArray[np.float64],
) -> NDArray[np.float64]:
    """Analytical Jacobian of the weighted TDoA residuals w.r.t. tx_pos.

    For pair k = (i, j), the Jacobian row is:
        J_k = -(w_k / c) * ((p - r_i) / ||p - r_i|| - (p - r_j) / ||p - r_j||)

    The negative sign arises because the residual is (measured - predicted), so
    d(residual)/d(p) = -d(predicted)/d(p).

    Args:
        tx_pos: Candidate transmitter position, shape (D,).
        receiver_positions: Receiver positions, shape (n_receivers, D') where D' >= D.
        pair_indices: List of (i, j) index tuples.
        weights: Per-pair weights, shape (n_pairs,).

    Returns:
        Jacobian matrix, shape (n_pairs, D).
    """
    tx = np.asarray(tx_pos, dtype=np.float64)
    ndim = len(tx)
    n_pairs = len(pair_indices)
    jac = np.empty((n_pairs, ndim), dtype=np.float64)

    for k, (i, j) in enumerate(pair_indices):
        ri = receiver_positions[i, :ndim]
        rj = receiver_positions[j, :ndim]
        diff_i = tx - ri
        diff_j = tx - rj
        dist_i = np.linalg.norm(diff_i)
        dist_j = np.linalg.norm(diff_j)

        # Unit vectors from receiver toward transmitter
        uhat_i = diff_i / dist_i if dist_i > 0.0 else np.zeros(ndim)
        uhat_j = diff_j / dist_j if dist_j > 0.0 else np.zeros(ndim)

        # d(predicted_tdoa)/d(p) = (1/c)(uhat_i - uhat_j)
        # d(residual)/d(p) = -w * d(predicted)/d(p)
        jac[k, :] = -(weights[k] / SPEED_OF_LIGHT) * (uhat_i - uhat_j)

    return jac
