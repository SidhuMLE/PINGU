"""Cross-correlation visualization for TDoA analysis.

Plots the generalized cross-correlation output with an optional marker at the
detected peak lag, useful for inspecting TDoA estimation quality.
"""

from __future__ import annotations

import matplotlib
matplotlib.use("Agg")  # Headless backend; must be set before importing pyplot

import matplotlib.figure
import matplotlib.pyplot as plt
import numpy as np
from numpy.typing import NDArray


def plot_cross_correlation(
    lags: NDArray[np.floating],
    corr: NDArray[np.floating],
    peak_lag: float | None = None,
    title: str = "",
    ax: plt.Axes | None = None,
) -> matplotlib.figure.Figure:
    """Plot the cross-correlation function with an optional peak marker.

    Args:
        lags: Lag values (e.g. in seconds or samples), shape ``(N,)``.
        corr: Correlation values corresponding to each lag, shape ``(N,)``.
        peak_lag: If provided, a vertical line and marker are drawn at this
            lag value to highlight the detected peak.
        title: Optional title for the plot.
        ax: Optional matplotlib Axes to draw on.  If ``None``, a new figure
            and axes are created.

    Returns:
        The matplotlib Figure containing the plot.
    """
    created_figure = ax is None
    if created_figure:
        fig, ax = plt.subplots(figsize=(10, 4))
    else:
        fig = ax.get_figure()

    ax.plot(lags, corr, linewidth=0.8, label="GCC")

    if peak_lag is not None:
        # Interpolate the correlation value at peak_lag for the marker.
        peak_corr = float(np.interp(peak_lag, lags, corr))
        ax.axvline(peak_lag, color="red", linestyle="--", linewidth=0.8, alpha=0.7)
        ax.plot(peak_lag, peak_corr, "ro", markersize=8, label=f"Peak @ {peak_lag:.6g}")
        ax.legend(loc="upper right")

    ax.set_xlabel("Lag (s)")
    ax.set_ylabel("Correlation")
    if title:
        ax.set_title(title)

    if created_figure:
        fig.tight_layout()

    return fig
