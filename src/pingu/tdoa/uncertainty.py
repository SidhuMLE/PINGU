"""Cramer-Rao Lower Bound for TDoA estimation uncertainty.

Provides a theoretical minimum-variance bound on time-delay (and
time-difference-of-arrival) estimates, useful for evaluating estimator
performance and for initialising Bayesian covariance matrices.
"""

from __future__ import annotations

import numpy as np


def crlb_tdoa(
    bandwidth: float,
    snr_linear: float,
    integration_time: float,
    center_freq: float = 0.0,
) -> float:
    """Cramer-Rao Lower Bound on the variance of a TDoA estimate.

    For a single delay estimate of a signal with flat power spectral
    density over bandwidth *B*, the CRLB on delay variance is:

        Var(tau) >= 3 / (8 * pi^2 * B^3 * T * SNR)

    where ``B`` is the one-sided bandwidth, ``T`` the coherent integration
    time, and ``SNR`` the linear (not dB) signal-to-noise ratio.

    The effective RMS bandwidth for a flat spectrum of width *B* is
    ``B_eff = B / sqrt(3)`` (the standard deviation of a uniform
    distribution on ``[-B/2, B/2]``).  Substituting into the general form
    ``Var(tau) >= 1 / (8 pi^2 SNR T B_eff^2)`` reproduces the formula
    above.

    A TDoA is the *difference* of two independent delay estimates, so the
    variance doubles:

        Var(delta_tau) >= 2 * Var(tau)

    Args:
        bandwidth: Signal bandwidth *B* in Hz (one-sided).
        snr_linear: Signal-to-noise ratio in **linear** scale
            (not decibels).  Must be > 0.
        integration_time: Coherent integration time *T* in seconds.
        center_freq: Centre frequency of the signal in Hz.  Currently
            unused but accepted for interface compatibility (may be used
            in future refinements involving carrier-phase information).

    Returns:
        Minimum achievable variance of the TDoA estimate, in seconds^2.

    Raises:
        ValueError: If any physical parameter is non-positive.
    """
    if bandwidth <= 0:
        raise ValueError(f"bandwidth must be positive, got {bandwidth}")
    if snr_linear <= 0:
        raise ValueError(f"snr_linear must be positive, got {snr_linear}")
    if integration_time <= 0:
        raise ValueError(f"integration_time must be positive, got {integration_time}")

    # Single-delay CRLB for flat-spectrum signal.
    var_single = 3.0 / (8.0 * np.pi**2 * bandwidth**3 * integration_time * snr_linear)

    # TDoA = difference of two independent delays => variance doubles.
    return 2.0 * var_single
