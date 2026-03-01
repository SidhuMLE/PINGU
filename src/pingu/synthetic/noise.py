"""AWGN noise generation and injection utilities.

All functions operate on complex baseband IQ signals stored as numpy
complex64 arrays.  SNR is defined as signal power over noise power in
linear scale (i.e. ``SNR_dB = 10 * log10(P_signal / P_noise)``).
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray


def generate_noise(
    n_samples: int,
    rng: np.random.Generator | None = None,
) -> NDArray[np.complex64]:
    """Generate unit-power complex white Gaussian noise.

    Each component (real, imaginary) is drawn from ``N(0, 1/2)`` so that
    the total expected power ``E[|z|^2] = 1``.

    Args:
        n_samples: Number of complex samples to generate.
        rng: Optional numpy random generator for reproducibility.

    Returns:
        Complex64 ndarray of shape ``(n_samples,)`` with unit average power.
    """
    if rng is None:
        rng = np.random.default_rng()

    noise = (
        rng.standard_normal(n_samples) + 1j * rng.standard_normal(n_samples)
    ).astype(np.complex128) / np.sqrt(2.0)

    return noise.astype(np.complex64)


def add_awgn(
    signal: NDArray[np.complex64],
    snr_db: float,
    rng: np.random.Generator | None = None,
) -> NDArray[np.complex64]:
    """Add white Gaussian noise to a signal at a specified SNR.

    The noise power is chosen so that the *output* SNR matches *snr_db*:

        ``P_noise = P_signal / 10^(snr_db / 10)``

    Args:
        signal: Input complex IQ signal.
        snr_db: Desired signal-to-noise ratio in dB.
        rng: Optional numpy random generator for reproducibility.

    Returns:
        Noisy complex64 signal with the same shape as *signal*.
    """
    if rng is None:
        rng = np.random.default_rng()

    # Compute signal power (mean |x|^2) in float64 for precision
    sig_f64 = signal.astype(np.complex128)
    signal_power = np.mean(np.abs(sig_f64) ** 2)

    # Target noise power
    snr_linear = 10.0 ** (snr_db / 10.0)
    noise_power = signal_power / snr_linear

    # Generate unit-power noise and scale to desired power
    noise = generate_noise(len(signal), rng=rng).astype(np.complex128)
    noise *= np.sqrt(noise_power)

    noisy = sig_f64 + noise
    return noisy.astype(np.complex64)
