"""Synthetic signal generators for HF modulation types.

Each generator produces a complex baseband IQ signal (numpy complex64 ndarray)
suitable for training the AMC classifier or for TDoA scenario simulation.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray
from scipy.signal import hilbert, firwin, lfilter

from pingu.types import ModulationType


# Reference sample rate used to define base signal parameters.
_BASE_RATE = 48_000.0


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _time_vector(sample_rate: float, duration: float) -> NDArray[np.float64]:
    """Return a time vector [0, duration) at the given sample rate."""
    n_samples = int(sample_rate * duration)
    return np.arange(n_samples, dtype=np.float64) / sample_rate


def _scale_rate(base_value: float, sample_rate: float) -> float:
    """Scale a parameter proportionally from the 48 kHz base rate.

    Args:
        base_value: Parameter value at 48 kHz sample rate.
        sample_rate: Actual sample rate in Hz.

    Returns:
        Scaled parameter value.
    """
    return base_value * (sample_rate / _BASE_RATE)


def _root_raised_cosine(
    n_taps: int,
    samples_per_symbol: int,
    rolloff: float,
) -> NDArray[np.float64]:
    """Design a root-raised-cosine (RRC) FIR filter.

    Args:
        n_taps: Number of filter taps (must be odd for symmetry).
        samples_per_symbol: Oversampling factor (samples per symbol period).
        rolloff: Roll-off factor in [0, 1].

    Returns:
        Normalised RRC impulse response of length *n_taps*.
    """
    T = float(samples_per_symbol)
    beta = rolloff
    # Ensure odd length for symmetry
    if n_taps % 2 == 0:
        n_taps += 1

    center = n_taps // 2
    h = np.zeros(n_taps, dtype=np.float64)

    for i in range(n_taps):
        t = (i - center) / T  # normalised time in symbol periods

        if t == 0.0:
            h[i] = 1.0 - beta + 4.0 * beta / np.pi
        elif beta != 0.0 and abs(t) == 1.0 / (4.0 * beta):
            h[i] = (beta / np.sqrt(2.0)) * (
                (1.0 + 2.0 / np.pi) * np.sin(np.pi / (4.0 * beta))
                + (1.0 - 2.0 / np.pi) * np.cos(np.pi / (4.0 * beta))
            )
        else:
            num = np.sin(np.pi * t * (1.0 - beta)) + 4.0 * beta * t * np.cos(
                np.pi * t * (1.0 + beta)
            )
            den = np.pi * t * (1.0 - (4.0 * beta * t) ** 2)
            if den == 0.0:
                h[i] = 0.0
            else:
                h[i] = num / den

    # Normalise so that the filter has unit energy
    h /= np.sqrt(np.sum(h ** 2))
    return h


def _freq_shift(
    signal: NDArray[np.complex64],
    shift_hz: float,
    sample_rate: float,
) -> NDArray[np.complex64]:
    """Apply a frequency shift to a complex signal.

    Args:
        signal: Input IQ signal.
        shift_hz: Frequency shift in Hz (positive = up).
        sample_rate: Sample rate in Hz.

    Returns:
        Frequency-shifted complex64 signal.
    """
    n = len(signal)
    t = np.arange(n, dtype=np.float64) / sample_rate
    shifter = np.exp(2j * np.pi * shift_hz * t)
    return (signal * shifter).astype(np.complex64)


# ---------------------------------------------------------------------------
# Signal generators
# ---------------------------------------------------------------------------

def generate_ssb(
    sample_rate: float = 2_000_000.0,
    duration: float = 0.1,
    center_freq: float = 0.0,
    sideband: str = "usb",
    audio_bw: float | None = None,
    rng: np.random.Generator | None = None,
) -> NDArray[np.complex64]:
    """Generate a single-sideband (SSB) voice-like signal.

    A band-limited noise signal (simulating speech) is created, the analytic
    signal is formed via the Hilbert transform, and then shifted to produce
    USB (upper sideband) or LSB (lower sideband).

    Args:
        sample_rate: Sample rate in Hz.
        duration: Signal duration in seconds.
        center_freq: Relative centre frequency offset in Hz.
        sideband: ``"usb"`` or ``"lsb"``.
        audio_bw: Audio bandwidth in Hz. If None, auto-scaled from sample rate.
        rng: Optional numpy random generator for reproducibility.

    Returns:
        Complex64 IQ array of length ``int(sample_rate * duration)``.
    """
    if rng is None:
        rng = np.random.default_rng()

    if audio_bw is None:
        audio_bw = _scale_rate(3_000.0, sample_rate)

    n_samples = int(sample_rate * duration)

    # Synthesise band-limited audio noise (300 Hz -- audio_bw)
    audio = rng.standard_normal(n_samples).astype(np.float64)
    # Band-pass filter the audio to speech-like bandwidth
    low_base = _scale_rate(300.0, sample_rate)
    low = low_base / (sample_rate / 2.0)
    high = min(audio_bw, sample_rate / 2.0 - 1.0) / (sample_rate / 2.0)
    if high <= low:
        high = low + 0.01
    taps = firwin(101, [low, high], pass_zero=False)
    audio = lfilter(taps, 1.0, audio)

    # Analytic signal via Hilbert transform (single-sideband suppressed carrier)
    analytic = hilbert(audio).astype(np.complex128)

    # For LSB, conjugate to flip the spectrum
    if sideband.lower() == "lsb":
        analytic = np.conj(analytic)

    # Normalise to unit power
    power = np.mean(np.abs(analytic) ** 2)
    if power > 0:
        analytic /= np.sqrt(power)

    # Apply centre-frequency offset
    sig = analytic.astype(np.complex64)
    if center_freq != 0.0:
        sig = _freq_shift(sig, center_freq, sample_rate)

    return sig


def generate_cw(
    sample_rate: float = 2_000_000.0,
    duration: float = 0.1,
    center_freq: float = 0.0,
    tone_freq: float | None = None,
    keying_pattern: list[float] | None = None,
    rng: np.random.Generator | None = None,
) -> NDArray[np.complex64]:
    """Generate a continuous-wave (CW) signal.

    Produces a complex sinusoid at *tone_freq* Hz (relative to baseband),
    optionally gated by a keying pattern for Morse-like on/off behaviour.

    Args:
        sample_rate: Sample rate in Hz.
        duration: Signal duration in seconds.
        center_freq: Additional relative frequency offset in Hz.
        tone_freq: Tone frequency in Hz relative to baseband.
            If None, defaults to 600 Hz (kept low to avoid TDoA ambiguity).
        keying_pattern: List of durations in seconds ``[on, off, on, off, ...]``.
            If ``None`` the tone is continuous.
        rng: Optional numpy random generator (unused, accepted for API uniformity).

    Returns:
        Complex64 IQ array.
    """
    if tone_freq is None:
        # CW tone must stay low to avoid periodic ambiguity in TDoA
        # cross-correlation.  600 Hz gives unambiguous delays up to ~830 us
        # (>200 km baseline).
        tone_freq = 600.0

    t = _time_vector(sample_rate, duration)
    n_samples = len(t)
    total_freq = tone_freq + center_freq

    sig = np.exp(2j * np.pi * total_freq * t)

    # Apply on/off keying envelope if provided
    if keying_pattern is not None:
        envelope = np.zeros(n_samples, dtype=np.float64)
        idx = 0
        on = True
        for dur in keying_pattern:
            n = int(dur * sample_rate)
            end = min(idx + n, n_samples)
            if on:
                envelope[idx:end] = 1.0
            idx = end
            on = not on
            if idx >= n_samples:
                break
        sig = sig * envelope

    # Normalise to unit power (over nonzero samples)
    power = np.mean(np.abs(sig) ** 2)
    if power > 0:
        sig /= np.sqrt(power)

    return sig.astype(np.complex64)


def generate_am(
    sample_rate: float = 2_000_000.0,
    duration: float = 0.1,
    center_freq: float = 0.0,
    mod_freq: float | None = None,
    mod_index: float = 0.8,
    rng: np.random.Generator | None = None,
) -> NDArray[np.complex64]:
    """Generate an amplitude-modulated (AM) signal.

    The signal is ``(1 + m * audio(t)) * exp(j 2 pi fc t)`` where *m* is the
    modulation index and *audio* is a simple tone at *mod_freq*.

    Args:
        sample_rate: Sample rate in Hz.
        duration: Signal duration in seconds.
        center_freq: Carrier frequency offset in Hz.
        mod_freq: Modulating audio frequency in Hz.
            If None, auto-scaled from sample rate.
        mod_index: Modulation depth in [0, 1].
        rng: Optional numpy random generator (unused, accepted for API uniformity).

    Returns:
        Complex64 IQ array.
    """
    if mod_freq is None:
        # AM with a pure-tone modulator produces only 3 spectral lines
        # (carrier ± mod_freq), so it has periodic cross-correlation
        # ambiguity at 1/mod_freq intervals.  Keep low to avoid this.
        mod_freq = 1_000.0

    t = _time_vector(sample_rate, duration)

    # Modulating tone
    audio = np.cos(2.0 * np.pi * mod_freq * t)

    # AM envelope: carrier + modulated audio
    envelope = 1.0 + mod_index * audio

    # Carrier at centre frequency
    carrier = np.exp(2j * np.pi * center_freq * t)

    sig = envelope * carrier

    # Normalise to unit power
    power = np.mean(np.abs(sig) ** 2)
    if power > 0:
        sig /= np.sqrt(power)

    return sig.astype(np.complex64)


def generate_fsk(
    sample_rate: float = 2_000_000.0,
    duration: float = 0.1,
    center_freq: float = 0.0,
    n_tones: int = 2,
    symbol_rate: float | None = None,
    tone_spacing: float | None = None,
    rng: np.random.Generator | None = None,
) -> NDArray[np.complex64]:
    """Generate a frequency-shift keying (FSK) signal with 2 or 4 tones.

    Random symbols are generated and mapped to tone frequencies symmetrically
    placed around *center_freq* with the given *tone_spacing*.

    Args:
        sample_rate: Sample rate in Hz.
        duration: Signal duration in seconds.
        center_freq: Centre frequency offset in Hz.
        n_tones: Number of FSK tones (2 or 4).
        symbol_rate: Symbol rate in symbols/s (baud).
            If None, auto-scaled from sample rate.
        tone_spacing: Frequency separation between adjacent tones in Hz.
            If None, auto-scaled from sample rate.
        rng: Optional numpy random generator for reproducibility.

    Returns:
        Complex64 IQ array.
    """
    if rng is None:
        rng = np.random.default_rng()
    if n_tones not in (2, 4):
        raise ValueError(f"n_tones must be 2 or 4, got {n_tones}")

    if symbol_rate is None:
        symbol_rate = _scale_rate(100.0, sample_rate)
    if tone_spacing is None:
        tone_spacing = _scale_rate(200.0, sample_rate)

    n_samples = int(sample_rate * duration)
    samples_per_symbol = int(sample_rate / symbol_rate)
    n_symbols = int(np.ceil(n_samples / samples_per_symbol))

    # Tone frequencies relative to centre
    # For 2-FSK: [-spacing/2, +spacing/2]
    # For 4-FSK: [-1.5*spacing, -0.5*spacing, +0.5*spacing, +1.5*spacing]
    tone_offsets = (np.arange(n_tones) - (n_tones - 1) / 2.0) * tone_spacing

    # Random symbol sequence
    symbols = rng.integers(0, n_tones, size=n_symbols)

    # Build continuous-phase FSK signal
    phase = np.zeros(n_samples, dtype=np.float64)
    t_sample = 1.0 / sample_rate

    current_phase = 0.0
    idx = 0
    for sym in symbols:
        freq = center_freq + tone_offsets[sym]
        end = min(idx + samples_per_symbol, n_samples)
        n = end - idx
        phase[idx:end] = current_phase + 2.0 * np.pi * freq * np.arange(n) * t_sample
        current_phase = phase[end - 1] + 2.0 * np.pi * freq * t_sample
        idx = end
        if idx >= n_samples:
            break

    sig = np.exp(1j * phase)

    # Normalise to unit power
    power = np.mean(np.abs(sig) ** 2)
    if power > 0:
        sig /= np.sqrt(power)

    return sig.astype(np.complex64)


def generate_fsk2(
    sample_rate: float = 2_000_000.0,
    duration: float = 0.1,
    center_freq: float = 0.0,
    symbol_rate: float | None = None,
    tone_spacing: float | None = None,
    rng: np.random.Generator | None = None,
) -> NDArray[np.complex64]:
    """Convenience wrapper for 2-FSK generation.  See :func:`generate_fsk`."""
    return generate_fsk(
        sample_rate=sample_rate,
        duration=duration,
        center_freq=center_freq,
        n_tones=2,
        symbol_rate=symbol_rate,
        tone_spacing=tone_spacing,
        rng=rng,
    )


def generate_fsk4(
    sample_rate: float = 2_000_000.0,
    duration: float = 0.1,
    center_freq: float = 0.0,
    symbol_rate: float | None = None,
    tone_spacing: float | None = None,
    rng: np.random.Generator | None = None,
) -> NDArray[np.complex64]:
    """Convenience wrapper for 4-FSK generation.  See :func:`generate_fsk`."""
    return generate_fsk(
        sample_rate=sample_rate,
        duration=duration,
        center_freq=center_freq,
        n_tones=4,
        symbol_rate=symbol_rate,
        tone_spacing=tone_spacing,
        rng=rng,
    )


def generate_bpsk(
    sample_rate: float = 2_000_000.0,
    duration: float = 0.1,
    center_freq: float = 0.0,
    symbol_rate: float | None = None,
    rolloff: float = 0.35,
    rng: np.random.Generator | None = None,
) -> NDArray[np.complex64]:
    """Generate a BPSK signal with root-raised-cosine pulse shaping.

    Random binary symbols {-1, +1} are upsampled and filtered through an
    RRC filter, then frequency-shifted to *center_freq*.

    Args:
        sample_rate: Sample rate in Hz.
        duration: Signal duration in seconds.
        center_freq: Frequency offset in Hz.
        symbol_rate: Symbol rate in baud.
            If None, auto-scaled from sample rate.
        rolloff: RRC roll-off factor.
        rng: Optional numpy random generator for reproducibility.

    Returns:
        Complex64 IQ array.
    """
    if rng is None:
        rng = np.random.default_rng()

    if symbol_rate is None:
        symbol_rate = _scale_rate(300.0, sample_rate)

    n_samples = int(sample_rate * duration)
    sps = int(sample_rate / symbol_rate)  # samples per symbol
    n_symbols = int(np.ceil(n_samples / sps)) + 10  # pad for filter transient

    # Random BPSK symbols
    bits = rng.integers(0, 2, size=n_symbols)
    symbols = 2.0 * bits - 1.0  # map {0,1} -> {-1, +1}

    # Upsample
    upsampled = np.zeros(n_symbols * sps, dtype=np.float64)
    upsampled[::sps] = symbols

    # RRC pulse-shape filter
    n_taps = 11 * sps + 1  # span 11 symbols
    rrc = _root_raised_cosine(n_taps, sps, rolloff)
    shaped = np.convolve(upsampled, rrc, mode="same")

    # Trim to desired length
    shaped = shaped[:n_samples]

    # Convert to complex baseband
    sig = shaped.astype(np.complex128)

    # Normalise to unit power
    power = np.mean(np.abs(sig) ** 2)
    if power > 0:
        sig /= np.sqrt(power)

    # Apply centre-frequency offset
    sig = sig.astype(np.complex64)
    if center_freq != 0.0:
        sig = _freq_shift(sig, center_freq, sample_rate)

    return sig


def generate_qpsk(
    sample_rate: float = 2_000_000.0,
    duration: float = 0.1,
    center_freq: float = 0.0,
    symbol_rate: float | None = None,
    rolloff: float = 0.35,
    rng: np.random.Generator | None = None,
) -> NDArray[np.complex64]:
    """Generate a QPSK signal with root-raised-cosine pulse shaping.

    Random 2-bit symbols are mapped to one of four constellation points
    at phase offsets {pi/4, 3pi/4, 5pi/4, 7pi/4}, then pulse-shaped.

    Args:
        sample_rate: Sample rate in Hz.
        duration: Signal duration in seconds.
        center_freq: Frequency offset in Hz.
        symbol_rate: Symbol rate in baud.
            If None, auto-scaled from sample rate.
        rolloff: RRC roll-off factor.
        rng: Optional numpy random generator for reproducibility.

    Returns:
        Complex64 IQ array.
    """
    if rng is None:
        rng = np.random.default_rng()

    if symbol_rate is None:
        symbol_rate = _scale_rate(300.0, sample_rate)

    n_samples = int(sample_rate * duration)
    sps = int(sample_rate / symbol_rate)
    n_symbols = int(np.ceil(n_samples / sps)) + 10

    # QPSK constellation: exp(j*(pi/4 + k*pi/2)), k = 0,1,2,3
    constellation = np.exp(1j * (np.pi / 4.0 + np.arange(4) * np.pi / 2.0))
    sym_indices = rng.integers(0, 4, size=n_symbols)
    symbols = constellation[sym_indices]

    # Upsample I and Q independently
    upsampled_i = np.zeros(n_symbols * sps, dtype=np.float64)
    upsampled_q = np.zeros(n_symbols * sps, dtype=np.float64)
    upsampled_i[::sps] = np.real(symbols)
    upsampled_q[::sps] = np.imag(symbols)

    # RRC pulse-shape filter
    n_taps = 11 * sps + 1
    rrc = _root_raised_cosine(n_taps, sps, rolloff)
    shaped_i = np.convolve(upsampled_i, rrc, mode="same")
    shaped_q = np.convolve(upsampled_q, rrc, mode="same")

    sig = (shaped_i + 1j * shaped_q)[:n_samples]

    # Normalise to unit power
    power = np.mean(np.abs(sig) ** 2)
    if power > 0:
        sig /= np.sqrt(power)

    sig = sig.astype(np.complex64)
    if center_freq != 0.0:
        sig = _freq_shift(sig, center_freq, sample_rate)

    return sig


def generate_noise_signal(
    sample_rate: float = 2_000_000.0,
    duration: float = 0.1,
    center_freq: float = 0.0,
    rng: np.random.Generator | None = None,
) -> NDArray[np.complex64]:
    """Generate a complex Gaussian noise signal.

    Args:
        sample_rate: Sample rate in Hz.
        duration: Signal duration in seconds.
        center_freq: Frequency offset in Hz (applied as frequency shift).
        rng: Optional numpy random generator for reproducibility.

    Returns:
        Complex64 IQ array with unit power.
    """
    if rng is None:
        rng = np.random.default_rng()

    n_samples = int(sample_rate * duration)
    sig = (
        rng.standard_normal(n_samples) + 1j * rng.standard_normal(n_samples)
    ).astype(np.complex64) / np.sqrt(2.0)

    if center_freq != 0.0:
        sig = _freq_shift(sig, center_freq, sample_rate)

    return sig.astype(np.complex64)


# ---------------------------------------------------------------------------
# Lookup table: ModulationType -> generator function
# ---------------------------------------------------------------------------

GENERATORS: dict = {
    ModulationType.SSB: generate_ssb,
    ModulationType.CW: generate_cw,
    ModulationType.AM: generate_am,
    ModulationType.FSK2: generate_fsk2,
    ModulationType.FSK4: generate_fsk4,
    ModulationType.BPSK: generate_bpsk,
    ModulationType.QPSK: generate_qpsk,
    ModulationType.NOISE: generate_noise_signal,
}


def generate_signal(
    modulation: ModulationType,
    sample_rate: float = 2_000_000.0,
    duration: float = 0.1,
    center_freq: float = 0.0,
    rng: np.random.Generator | None = None,
    **kwargs,
) -> NDArray[np.complex64]:
    """Dispatch to the appropriate signal generator by modulation type.

    Args:
        modulation: The desired modulation type.
        sample_rate: Sample rate in Hz.
        duration: Duration in seconds.
        center_freq: Relative centre frequency in Hz.
        rng: Optional numpy random generator.
        **kwargs: Additional keyword arguments forwarded to the generator.

    Returns:
        Complex64 IQ array.

    Raises:
        ValueError: If *modulation* has no registered generator.
    """
    gen = GENERATORS.get(modulation)
    if gen is None:
        raise ValueError(f"No generator registered for modulation {modulation!r}")
    return gen(
        sample_rate=sample_rate,
        duration=duration,
        center_freq=center_freq,
        rng=rng,
        **kwargs,
    )
