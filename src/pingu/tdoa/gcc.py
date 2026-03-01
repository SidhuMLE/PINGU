"""Generalized Cross-Correlation methods for TDoA estimation.

Implements GCC-PHAT, GCC-SCOT, and GCC-ML weighting functions for
time-delay estimation between pairs of receivers.
"""

from __future__ import annotations

from typing import Literal

import numpy as np
from numpy.typing import NDArray
from scipy.fft import fft, ifft

from pingu.tdoa.peak_interpolation import parabolic_interpolation
from pingu.types import TDoAEstimate


# Regularization floor to avoid division by zero in spectral weighting.
_EPS = 1e-12


def _next_pow2(n: int) -> int:
    """Return the smallest power of 2 >= *n*."""
    return 1 << (n - 1).bit_length()


def _prepare_spectra(
    x: NDArray[np.floating | np.complexfloating],
    y: NDArray[np.floating | np.complexfloating],
    nfft: int,
) -> tuple[NDArray[np.complex128], NDArray[np.complex128]]:
    """Compute zero-padded FFTs of the two input signals.

    Args:
        x: First signal array.
        y: Second signal array.
        nfft: FFT length (should be >= len(x) + len(y) - 1 for linear
              cross-correlation).

    Returns:
        Tuple of (X, Y) complex spectra, each of length *nfft*.
    """
    X = fft(np.asarray(x, dtype=np.float64), n=nfft)
    Y = fft(np.asarray(y, dtype=np.float64), n=nfft)
    return X, Y


def _trim_correlation(
    corr: NDArray[np.float64],
    nfft: int,
    max_delay_samples: int | None,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Extract the valid lag region from a full circular correlation.

    The IFFT produces a circular correlation of length *nfft*.  This helper
    rearranges it into a centred, two-sided correlation and optionally
    truncates to ``[-max_delay_samples, +max_delay_samples]``.

    Args:
        corr: Full circular cross-correlation (length *nfft*).
        nfft: FFT length used.
        max_delay_samples: If given, keep only lags in
            ``[-max_delay_samples, max_delay_samples]``.

    Returns:
        (lags, correlation) where *lags* is an integer-sample lag array and
        *correlation* contains the corresponding GCC values.
    """
    # Shift so that lag 0 is in the centre.
    corr_shifted = np.fft.fftshift(corr)
    lags_full = np.arange(nfft) - nfft // 2

    if max_delay_samples is not None:
        mask = np.abs(lags_full) <= max_delay_samples
        lags_full = lags_full[mask]
        corr_shifted = corr_shifted[mask]

    return lags_full.astype(np.float64), corr_shifted


# --------------------------------------------------------------------------- #
# Public GCC variants
# --------------------------------------------------------------------------- #

def gcc_phat(
    x: NDArray,
    y: NDArray,
    fs: float,
    max_delay_samples: int | None = None,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """GCC-PHAT: cross-spectrum weighted by ``1 / |G_xy|``.

    The phase transform whitens the cross-spectrum magnitude so that only
    phase information is used for delay estimation, yielding a sharp
    correlation peak.

    Args:
        x: Signal from receiver *i*.
        y: Signal from receiver *j*.
        fs: Sampling frequency (Hz).
        max_delay_samples: Optional maximum lag to retain.

    Returns:
        ``(lags_seconds, correlation)`` as float64 arrays.
    """
    n = len(x) + len(y) - 1
    nfft = _next_pow2(n)
    X, Y = _prepare_spectra(x, y, nfft)

    G_xy = X * np.conj(Y)
    weight = np.abs(G_xy) + _EPS
    corr = np.real(ifft(G_xy / weight, n=nfft)).astype(np.float64)

    lags, cc = _trim_correlation(corr, nfft, max_delay_samples)
    return lags / fs, cc


def gcc_scot(
    x: NDArray,
    y: NDArray,
    fs: float,
    max_delay_samples: int | None = None,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """GCC-SCOT: Smoothed Coherence Transform.

    Weighted by ``1 / sqrt(G_xx * G_yy)`` which normalises by the
    geometric mean of the auto-spectra.

    Args:
        x: Signal from receiver *i*.
        y: Signal from receiver *j*.
        fs: Sampling frequency (Hz).
        max_delay_samples: Optional maximum lag to retain.

    Returns:
        ``(lags_seconds, correlation)`` as float64 arrays.
    """
    n = len(x) + len(y) - 1
    nfft = _next_pow2(n)
    X, Y = _prepare_spectra(x, y, nfft)

    G_xy = X * np.conj(Y)
    G_xx = np.abs(X) ** 2
    G_yy = np.abs(Y) ** 2
    weight = np.sqrt(G_xx * G_yy) + _EPS
    corr = np.real(ifft(G_xy / weight, n=nfft)).astype(np.float64)

    lags, cc = _trim_correlation(corr, nfft, max_delay_samples)
    return lags / fs, cc


def gcc_ml(
    x: NDArray,
    y: NDArray,
    fs: float,
    max_delay_samples: int | None = None,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """GCC-ML: Maximum Likelihood weighting.

    Weighted by ``|G_xy|^2 / (G_xx * G_yy * (1 - |gamma_xy|^2))``,
    where ``gamma_xy`` is the magnitude-squared coherence.  This gives the
    ML estimate under a Gaussian signal-in-noise model.

    A small regularisation constant prevents division by zero in spectral
    nulls and in frequency bins with perfect coherence.

    Args:
        x: Signal from receiver *i*.
        y: Signal from receiver *j*.
        fs: Sampling frequency (Hz).
        max_delay_samples: Optional maximum lag to retain.

    Returns:
        ``(lags_seconds, correlation)`` as float64 arrays.
    """
    n = len(x) + len(y) - 1
    nfft = _next_pow2(n)
    X, Y = _prepare_spectra(x, y, nfft)

    G_xy = X * np.conj(Y)
    G_xx = np.abs(X) ** 2
    G_yy = np.abs(Y) ** 2

    # Magnitude-squared coherence per frequency bin.
    denom_coh = G_xx * G_yy + _EPS
    gamma_sq = np.abs(G_xy) ** 2 / denom_coh
    # Clamp to avoid 1 - gamma_sq becoming zero or negative.
    gamma_sq = np.clip(gamma_sq, 0.0, 1.0 - _EPS)

    weight = denom_coh * (1.0 - gamma_sq) + _EPS
    W = np.abs(G_xy) ** 2 / weight
    corr = np.real(ifft(W * np.exp(1j * np.angle(G_xy)), n=nfft)).astype(np.float64)

    lags, cc = _trim_correlation(corr, nfft, max_delay_samples)
    return lags / fs, cc


# --------------------------------------------------------------------------- #
# Dispatcher / convenience
# --------------------------------------------------------------------------- #

_GCC_METHODS = {
    "phat": gcc_phat,
    "scot": gcc_scot,
    "ml": gcc_ml,
}


def estimate_tdoa(
    x: NDArray,
    y: NDArray,
    fs: float,
    method: Literal["phat", "scot", "ml"] = "phat",
    receiver_i: str = "RX0",
    receiver_j: str = "RX1",
    center_freq: float = 0.0,
    timestamp: float = 0.0,
    **kwargs,
) -> TDoAEstimate:
    """Estimate the TDoA between two signals using GCC + peak interpolation.

    This is the top-level convenience function that:
    1. Computes the generalised cross-correlation using the chosen *method*.
    2. Locates the correlation peak.
    3. Refines it via parabolic sub-sample interpolation.
    4. Returns a fully populated :class:`TDoAEstimate`.

    Args:
        x: Signal from receiver *i*.
        y: Signal from receiver *j*.
        fs: Sampling frequency (Hz).
        method: GCC weighting --- ``"phat"``, ``"scot"``, or ``"ml"``.
        receiver_i: Identifier for the first receiver.
        receiver_j: Identifier for the second receiver.
        center_freq: Centre frequency of the signal (Hz).
        timestamp: Epoch timestamp of the measurement.
        **kwargs: Forwarded to the underlying GCC function (e.g.
            ``max_delay_samples``).

    Returns:
        A :class:`TDoAEstimate` with delay, variance estimate, and
        correlation peak value.
    """
    if method not in _GCC_METHODS:
        raise ValueError(f"Unknown GCC method '{method}'. Choose from {list(_GCC_METHODS)}")

    gcc_func = _GCC_METHODS[method]
    lags_sec, corr = gcc_func(x, y, fs, **kwargs)

    # Find coarse peak.
    peak_idx = int(np.argmax(corr))

    # Sub-sample refinement via parabolic interpolation (needs neighbours).
    if 0 < peak_idx < len(corr) - 1:
        frac_idx, peak_val = parabolic_interpolation(corr, peak_idx)
    else:
        frac_idx = float(peak_idx)
        peak_val = float(corr[peak_idx])

    # Convert fractional index back to seconds.
    dt = lags_sec[1] - lags_sec[0] if len(lags_sec) > 1 else 1.0 / fs
    delay_sec = lags_sec[0] + frac_idx * dt

    # Rough variance estimate: inversely proportional to squared peak value.
    # This is a heuristic; for a proper bound see uncertainty.crlb_tdoa.
    variance = (1.0 / (fs * max(abs(peak_val), _EPS))) ** 2

    return TDoAEstimate(
        receiver_i=receiver_i,
        receiver_j=receiver_j,
        delay=delay_sec,
        variance=variance,
        correlation_peak=peak_val,
        center_freq=center_freq,
        timestamp=timestamp,
    )
