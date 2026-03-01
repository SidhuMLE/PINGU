"""Convergence visualization for the Bayesian TDoA integration loop.

Plots the evolution of posterior variance across Kalman filter updates for
each TDoA pair, optionally overlaying the Cramer-Rao Lower Bound as
horizontal dashed reference lines.
"""

from __future__ import annotations

import matplotlib
matplotlib.use("Agg")  # Headless backend; must be set before importing pyplot

import matplotlib.figure
import matplotlib.pyplot as plt
import numpy as np
from numpy.typing import NDArray


def plot_convergence(
    variance_history: NDArray[np.floating],
    crlb: NDArray[np.floating] | None = None,
    title: str = "",
    ax: plt.Axes | None = None,
) -> matplotlib.figure.Figure:
    """Plot variance convergence over successive Kalman filter updates.

    Each column of *variance_history* corresponds to a TDoA pair and each
    row to an update step.  The CRLB (if provided) is shown as horizontal
    dashed lines for reference.

    Args:
        variance_history: Array of shape ``(n_updates, n_pairs)`` containing
            the diagonal posterior variance at each update step.
        crlb: Optional array of shape ``(n_pairs,)`` or ``(n_pairs, n_pairs)``
            giving the Cramer-Rao lower bound for each pair.  If a 2-D matrix
            is supplied, only the diagonal is used.
        title: Optional title for the plot.
        ax: Optional matplotlib Axes to draw on.  If ``None``, a new figure
            and axes are created.

    Returns:
        The matplotlib Figure containing the convergence plot.
    """
    variance_history = np.asarray(variance_history, dtype=np.float64)
    if variance_history.ndim == 1:
        variance_history = variance_history[:, np.newaxis]

    n_updates, n_pairs = variance_history.shape

    created_figure = ax is None
    if created_figure:
        fig, ax = plt.subplots(figsize=(10, 5))
    else:
        fig = ax.get_figure()

    updates = np.arange(1, n_updates + 1)

    for p in range(n_pairs):
        ax.semilogy(updates, variance_history[:, p], linewidth=1.0, label=f"Pair {p}")

    # Overlay CRLB if provided.
    if crlb is not None:
        crlb_arr = np.asarray(crlb, dtype=np.float64)
        if crlb_arr.ndim == 2:
            crlb_diag = np.diag(crlb_arr)
        else:
            crlb_diag = crlb_arr.ravel()

        for p in range(min(n_pairs, len(crlb_diag))):
            ax.axhline(
                crlb_diag[p],
                color=f"C{p}",
                linestyle="--",
                linewidth=0.8,
                alpha=0.7,
            )
        # Add a single legend entry for CRLB lines.
        ax.plot([], [], "k--", linewidth=0.8, label="CRLB")

    ax.set_xlabel("Update Number")
    ax.set_ylabel("Variance (s^2)")
    if title:
        ax.set_title(title)
    else:
        ax.set_title("TDoA Variance Convergence")
    ax.legend(loc="upper right", fontsize=8, ncol=2)
    ax.grid(True, alpha=0.3, which="both")

    if created_figure:
        fig.tight_layout()

    return fig
