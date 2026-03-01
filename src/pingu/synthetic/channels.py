"""Watterson HF channel model -- Phase 5 placeholder.

This module will eventually implement the Watterson ionospheric fading
channel model for realistic HF propagation simulation, including
multipath delay spreads, Doppler shifts, and time-varying Rayleigh
fading taps.

Currently provides a pass-through stub so that the pipeline interface
is established and downstream modules can reference it.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray


def apply_watterson(
    signal: NDArray[np.complexfloating],
    delay_spread: float = 0.0,
    doppler_spread: float = 0.0,
    sample_rate: float = 48_000.0,
    rng: np.random.Generator | None = None,
) -> NDArray[np.complexfloating]:
    """Apply the Watterson HF channel model to a signal.

    .. note::
        This is a Phase 5 placeholder.  The signal is returned unmodified.

    Args:
        signal: Input complex IQ signal.
        delay_spread: Multipath delay spread in seconds (unused).
        doppler_spread: Doppler spread in Hz (unused).
        sample_rate: Sample rate in Hz (unused).
        rng: Optional random generator for reproducibility (unused).

    Returns:
        The input signal, unmodified (pass-through).
    """
    return signal
