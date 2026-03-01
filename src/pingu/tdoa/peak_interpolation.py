"""Sub-sample peak interpolation for cross-correlation refinement.

After the coarse (integer-sample) peak of a GCC cross-correlation has been
identified, these routines refine the peak location to fractional-sample
precision.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray


def parabolic_interpolation(
    corr: NDArray[np.floating],
    peak_idx: int,
) -> tuple[float, float]:
    """Fit a parabola to three points around the peak and return the vertex.

    Given three consecutive correlation values ``corr[peak_idx-1]``,
    ``corr[peak_idx]``, ``corr[peak_idx+1]``, this fits a second-order
    polynomial and solves for the vertex, yielding sub-sample precision.

    Args:
        corr: 1-D array of cross-correlation values.
        peak_idx: Integer index of the coarse peak (must satisfy
            ``0 < peak_idx < len(corr) - 1``).

    Returns:
        ``(fractional_index, peak_value)`` where *fractional_index* is the
        refined peak position (relative to the start of *corr*) and
        *peak_value* is the interpolated correlation amplitude.

    Raises:
        ValueError: If *peak_idx* is on the boundary (no neighbour available).
    """
    if peak_idx <= 0 or peak_idx >= len(corr) - 1:
        raise ValueError(
            f"peak_idx={peak_idx} must be in range [1, {len(corr) - 2}] "
            "so that both neighbours are available."
        )

    alpha = float(corr[peak_idx - 1])
    beta = float(corr[peak_idx])
    gamma = float(corr[peak_idx + 1])

    # Vertex of the parabola through (-1, alpha), (0, beta), (1, gamma):
    #   delta = 0.5 * (alpha - gamma) / (alpha - 2*beta + gamma)
    denom = alpha - 2.0 * beta + gamma
    if abs(denom) < 1e-30:
        # Flat region -- no refinement possible.
        return float(peak_idx), beta

    delta = 0.5 * (alpha - gamma) / denom
    refined_idx = peak_idx + delta
    refined_val = beta - 0.25 * (alpha - gamma) * delta

    return refined_idx, refined_val


def sinc_interpolation(
    corr: NDArray[np.floating],
    peak_idx: int,
    n_points: int = 5,
) -> tuple[float, float]:
    """Sinc interpolation around the peak for higher-accuracy refinement.

    Uses an initial parabolic estimate to find the fractional offset, then
    evaluates the Whittaker-Shannon (sinc) interpolant over a neighbourhood
    of *n_points* samples on each side.  The interpolant is maximised on a
    fine grid to yield the refined peak location.

    Args:
        corr: 1-D array of cross-correlation values.
        peak_idx: Integer index of the coarse peak (must have at least
            *n_points* neighbours on each side).
        n_points: Number of samples on each side of the peak to include
            in the sinc kernel.  Defaults to 5.

    Returns:
        ``(fractional_index, peak_value)`` with higher accuracy than
        parabolic interpolation for band-limited correlation functions.
    """
    half = n_points
    lo = max(peak_idx - half, 0)
    hi = min(peak_idx + half + 1, len(corr))
    local = np.asarray(corr[lo:hi], dtype=np.float64)
    local_indices = np.arange(lo, hi, dtype=np.float64)

    # Fine grid: search +/-1 sample around the coarse peak at 0.01-sample
    # resolution.
    fine_grid = np.linspace(peak_idx - 1.0, peak_idx + 1.0, 201)

    # Evaluate sinc interpolant on fine grid.
    interp_vals = np.zeros_like(fine_grid)
    for k, idx_f in enumerate(fine_grid):
        sinc_kernel = np.sinc(idx_f - local_indices)
        interp_vals[k] = np.dot(local, sinc_kernel)

    best = int(np.argmax(interp_vals))
    return float(fine_grid[best]), float(interp_vals[best])
