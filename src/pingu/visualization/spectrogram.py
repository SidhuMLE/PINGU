"""Spectrogram visualization for IQ signal analysis.

Provides a convenience function to plot time-frequency spectrograms using
matplotlib, suitable for inspecting channelized or raw IQ data.
"""

from __future__ import annotations

import matplotlib
matplotlib.use("Agg")  # Headless backend; must be set before importing pyplot

import matplotlib.figure
import matplotlib.pyplot as plt
import numpy as np
from numpy.typing import NDArray


def plot_spectrogram(
    samples: NDArray[np.complexfloating] | NDArray[np.floating],
    sample_rate: float,
    title: str = "",
    ax: plt.Axes | None = None,
) -> matplotlib.figure.Figure:
    """Plot the spectrogram of IQ or real-valued samples.

    Uses ``plt.specgram`` internally, which computes a short-time Fourier
    transform and displays the magnitude in dB.

    Args:
        samples: 1-D array of IQ (complex) or real samples.
        sample_rate: Sampling frequency in Hz.
        title: Optional title for the plot.
        ax: Optional matplotlib Axes to draw on.  If ``None``, a new figure
            and axes are created.

    Returns:
        The matplotlib Figure containing the spectrogram.
    """
    # Use real part of complex signals for spectrogram display.
    data = np.real(samples) if np.iscomplexobj(samples) else np.asarray(samples)

    created_figure = ax is None
    if created_figure:
        fig, ax = plt.subplots(figsize=(10, 4))
    else:
        fig = ax.get_figure()

    ax.specgram(data, Fs=sample_rate, NFFT=256, noverlap=128, cmap="viridis")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Frequency (Hz)")
    if title:
        ax.set_title(title)

    if created_figure:
        fig.tight_layout()

    return fig
