"""TDoA scenario generator for multi-receiver IQ simulation.

Given a set of receiver positions and a transmitter location, this module
produces per-receiver IQ frames with realistic propagation delays and
independent AWGN noise.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from pingu.constants import SPEED_OF_LIGHT
from pingu.types import IQFrame, ReceiverConfig, ModulationType
from pingu.synthetic.signals import generate_signal
from pingu.synthetic.noise import add_awgn


def _fractional_delay(
    signal: NDArray[np.complex64],
    delay_samples: float,
) -> NDArray[np.complex64]:
    """Apply a fractional-sample delay via FFT phase-shift method.

    The signal is transformed to the frequency domain, a linear phase
    ramp corresponding to the delay is applied, and the result is
    inverse-transformed back to the time domain.

    Args:
        signal: Input complex IQ signal.
        delay_samples: Delay in fractional samples (positive = later arrival).

    Returns:
        Delayed complex64 signal with the same length as *signal*.
    """
    n = len(signal)
    # Frequency bins: [0, 1, ..., N/2-1, -N/2, ..., -1] / N  (normalised)
    freqs = np.fft.fftfreq(n)
    # Phase shift: exp(-j * 2pi * f * delay)
    phase_shift = np.exp(-2j * np.pi * freqs * delay_samples)

    sig_fft = np.fft.fft(signal.astype(np.complex128))
    delayed_fft = sig_fft * phase_shift
    delayed = np.fft.ifft(delayed_fft)

    return delayed.astype(np.complex64)


def _distance_2d(
    tx: tuple[float, float],
    rx: ReceiverConfig,
) -> float:
    """Compute Euclidean distance in the 2-D Cartesian plane.

    Uses the receiver's ``x`` and ``y`` fields.

    Args:
        tx: Transmitter position ``(x, y)`` in metres.
        rx: Receiver configuration (uses ``rx.x``, ``rx.y``).

    Returns:
        Distance in metres.
    """
    dx = tx[0] - rx.x
    dy = tx[1] - rx.y
    return float(np.sqrt(dx * dx + dy * dy))


@dataclass
class TDoAScenario:
    """Multi-receiver TDoA scenario generator.

    Given receiver positions and a transmitter location, this class produces
    IQ frames for each receiver with the correct propagation delays and
    independent additive noise.

    Attributes:
        receivers: List of receiver configurations.
        tx_position: Transmitter ``(x, y)`` in metres (Cartesian).
        sample_rate: Common sample rate in Hz for all receivers.
        center_freq: Nominal centre frequency in Hz.
        snr_db: Per-receiver SNR in dB (same for all receivers).
        duration: Signal duration in seconds.
        modulation: Modulation type for the transmitted signal.
    """

    receivers: list[ReceiverConfig]
    tx_position: tuple[float, float]
    sample_rate: float = 48_000.0
    center_freq: float = 14.1e6
    snr_db: float = 20.0
    duration: float = 0.1
    modulation: ModulationType = ModulationType.CW

    def compute_delays(self) -> dict[str, float]:
        """Compute one-way propagation delays for each receiver.

        Returns:
            Dict mapping ``receiver_id`` to delay in seconds.
        """
        delays: dict[str, float] = {}
        for rx in self.receivers:
            dist = _distance_2d(self.tx_position, rx)
            delays[rx.id] = dist / SPEED_OF_LIGHT
        return delays

    def generate(
        self,
        rng: np.random.Generator | None = None,
        **signal_kwargs,
    ) -> dict[str, IQFrame]:
        """Generate multi-receiver IQ frames with propagation delays and noise.

        A single transmit signal is synthesised, then independently delayed
        and corrupted with AWGN for each receiver.

        Args:
            rng: Optional numpy random generator for reproducibility.
            **signal_kwargs: Extra keyword arguments forwarded to the signal
                generator (e.g. ``tone_freq``, ``symbol_rate``).

        Returns:
            Dict mapping ``receiver_id`` to an :class:`IQFrame` containing
            the delayed, noisy received signal.
        """
        if rng is None:
            rng = np.random.default_rng()

        # Generate the common transmitted signal
        tx_signal = generate_signal(
            modulation=self.modulation,
            sample_rate=self.sample_rate,
            duration=self.duration,
            center_freq=0.0,  # delays are applied in baseband
            rng=rng,
            **signal_kwargs,
        )

        delays = self.compute_delays()
        frames: dict[str, IQFrame] = {}

        for rx in self.receivers:
            delay_s = delays[rx.id]
            delay_samples = delay_s * self.sample_rate

            # Apply fractional-sample delay
            delayed_signal = _fractional_delay(tx_signal, delay_samples)

            # Add independent AWGN at the specified SNR
            noisy_signal = add_awgn(delayed_signal, self.snr_db, rng=rng)

            frames[rx.id] = IQFrame(
                receiver_id=rx.id,
                samples=noisy_signal,
                sample_rate=self.sample_rate,
                center_freq=self.center_freq,
                timestamp=0.0,
            )

        return frames
