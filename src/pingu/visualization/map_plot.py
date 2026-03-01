"""Position map visualization for TDoA geolocation results.

Plots receiver positions, the estimated transmitter position, the true
transmitter position (if known), and a confidence ellipse around the
estimate.
"""

from __future__ import annotations

import matplotlib
matplotlib.use("Agg")  # Headless backend; must be set before importing pyplot

import matplotlib.figure
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np

from pingu.locator.posterior import confidence_ellipse
from pingu.types import PositionEstimate, ReceiverConfig


def plot_position_map(
    receivers: list[ReceiverConfig],
    estimate: PositionEstimate | None = None,
    true_pos: tuple[float, float] | None = None,
    confidence: float = 0.95,
    ax: plt.Axes | None = None,
) -> matplotlib.figure.Figure:
    """Plot a 2-D map of receivers, estimated position, and confidence ellipse.

    Receiver positions are shown as blue triangles, the estimated transmitter
    position as a gold star, and the true transmitter position (if provided) as
    a red X.  A confidence ellipse is drawn around the estimate using the
    covariance from the :class:`PositionEstimate`.

    All axis coordinates are displayed in kilometres.

    Args:
        receivers: List of receiver configurations (uses ``x`` and ``y`` fields).
        estimate: Estimated transmitter position with covariance.  If ``None``,
            no estimate or confidence ellipse is drawn.
        true_pos: True transmitter ``(x, y)`` in metres.  If ``None``, no true
            position marker is drawn.
        confidence: Confidence level for the ellipse (default 0.95).
        ax: Optional matplotlib Axes to draw on.  If ``None``, a new figure
            and axes are created.

    Returns:
        The matplotlib Figure containing the map.
    """
    created_figure = ax is None
    if created_figure:
        fig, ax = plt.subplots(figsize=(8, 8))
    else:
        fig = ax.get_figure()

    # Conversion factor: metres -> kilometres.
    M_TO_KM = 1e-3

    # Plot receivers as blue triangles.
    rx_x = [r.x * M_TO_KM for r in receivers]
    rx_y = [r.y * M_TO_KM for r in receivers]
    ax.plot(rx_x, rx_y, "b^", markersize=10, label="Receivers")
    for r in receivers:
        ax.annotate(
            r.id,
            (r.x * M_TO_KM, r.y * M_TO_KM),
            textcoords="offset points",
            xytext=(5, 5),
            fontsize=8,
        )

    # Plot true position as a red X.
    if true_pos is not None:
        ax.plot(
            true_pos[0] * M_TO_KM,
            true_pos[1] * M_TO_KM,
            "rx",
            markersize=12,
            markeredgewidth=2,
            label="True Position",
        )

    # Plot estimate as a gold star and draw confidence ellipse.
    if estimate is not None:
        ax.plot(
            estimate.x * M_TO_KM,
            estimate.y * M_TO_KM,
            "*",
            color="gold",
            markersize=15,
            markeredgecolor="black",
            markeredgewidth=0.5,
            label="Estimate",
        )

        # Compute confidence ellipse parameters (in metres).
        semi_major, semi_minor, angle_deg = confidence_ellipse(
            estimate.covariance, confidence=confidence
        )

        # Draw the ellipse (convert semi-axes to km for display).
        ellipse = mpatches.Ellipse(
            xy=(estimate.x * M_TO_KM, estimate.y * M_TO_KM),
            width=2 * semi_major * M_TO_KM,
            height=2 * semi_minor * M_TO_KM,
            angle=angle_deg,
            edgecolor="darkorange",
            facecolor="orange",
            alpha=0.25,
            linewidth=1.5,
            label=f"{confidence * 100:.0f}% Confidence",
        )
        ax.add_patch(ellipse)

    ax.set_xlabel("x (km)")
    ax.set_ylabel("y (km)")
    ax.set_title("TDoA Position Map")
    ax.legend(loc="best")
    ax.set_aspect("equal", adjustable="datalim")
    ax.grid(True, alpha=0.3)

    if created_figure:
        fig.tight_layout()

    return fig
