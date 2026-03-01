"""PyTorch Dataset for on-the-fly synthetic AMC data generation.

Generates IQ samples for each supported modulation type with additive white
Gaussian noise at a random SNR drawn from a configurable range. Provides
inline fallback signal generators so the dataset works standalone even if the
``pingu.synthetic`` subpackage is not yet implemented.
"""

from __future__ import annotations

import numpy as np
import torch
from torch.utils.data import Dataset

from pingu.types import ModulationType

# ---------------------------------------------------------------------------
# Attempt to import canonical synthetic generators; fall back to local stubs.
# ---------------------------------------------------------------------------
_USE_SYNTHETIC_PACKAGE = False

try:
    from pingu.synthetic.signals import (  # type: ignore[import-untyped]
        generate_am,
        generate_bpsk,
        generate_cw,
        generate_fsk2,
        generate_fsk4,
        generate_qpsk,
        generate_ssb,
    )
    from pingu.synthetic.noise import add_awgn  # type: ignore[import-untyped]

    _USE_SYNTHETIC_PACKAGE = True
except ImportError:
    pass


# ---------------------------------------------------------------------------
# Inline fallback generators (simple, deterministic approximations).
# ---------------------------------------------------------------------------

def _fallback_generate_cw(
    n_samples: int,
    sample_rate: float = 48_000.0,
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    """Generate a continuous-wave (pure tone) signal."""
    rng = rng or np.random.default_rng()
    freq = rng.uniform(200, sample_rate / 4)
    t = np.arange(n_samples) / sample_rate
    return np.exp(2j * np.pi * freq * t).astype(np.complex64)


def _fallback_generate_am(
    n_samples: int,
    sample_rate: float = 48_000.0,
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    """Generate an AM signal (carrier + sinusoidal modulation)."""
    rng = rng or np.random.default_rng()
    f_carrier = rng.uniform(1000, sample_rate / 4)
    f_mod = rng.uniform(50, 500)
    mod_index = rng.uniform(0.3, 0.9)
    t = np.arange(n_samples) / sample_rate
    envelope = 1.0 + mod_index * np.sin(2 * np.pi * f_mod * t)
    carrier = np.exp(2j * np.pi * f_carrier * t)
    return (envelope * carrier).astype(np.complex64)


def _fallback_generate_ssb(
    n_samples: int,
    sample_rate: float = 48_000.0,
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    """Generate a crude SSB-like signal (analytic signal of band-limited noise)."""
    rng = rng or np.random.default_rng()
    # Band-limited baseband noise as the message
    msg = rng.standard_normal(n_samples)
    # Simple low-pass via moving average
    kernel_len = max(3, n_samples // 100)
    kernel = np.ones(kernel_len) / kernel_len
    msg = np.convolve(msg, kernel, mode="same")
    # Hilbert-style analytic signal (FFT method)
    spectrum = np.fft.fft(msg)
    spectrum[n_samples // 2 + 1:] = 0.0
    spectrum[1:n_samples // 2] *= 2.0
    analytic = np.fft.ifft(spectrum)
    return analytic.astype(np.complex64)


def _fallback_generate_bpsk(
    n_samples: int,
    sample_rate: float = 48_000.0,
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    """Generate a BPSK signal."""
    rng = rng or np.random.default_rng()
    symbol_rate = rng.uniform(100, 2000)
    samples_per_symbol = max(1, int(sample_rate / symbol_rate))
    n_symbols = n_samples // samples_per_symbol + 1
    symbols = rng.choice([-1.0, 1.0], size=n_symbols)
    signal = np.repeat(symbols, samples_per_symbol)[:n_samples]
    return signal.astype(np.complex64)


def _fallback_generate_qpsk(
    n_samples: int,
    sample_rate: float = 48_000.0,
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    """Generate a QPSK signal."""
    rng = rng or np.random.default_rng()
    symbol_rate = rng.uniform(100, 2000)
    samples_per_symbol = max(1, int(sample_rate / symbol_rate))
    n_symbols = n_samples // samples_per_symbol + 1
    phases = rng.choice([np.pi / 4, 3 * np.pi / 4, 5 * np.pi / 4, 7 * np.pi / 4], size=n_symbols)
    symbols = np.exp(1j * phases)
    signal = np.repeat(symbols, samples_per_symbol)[:n_samples]
    return signal.astype(np.complex64)


def _fallback_generate_fsk2(
    n_samples: int,
    sample_rate: float = 48_000.0,
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    """Generate a 2-FSK signal."""
    rng = rng or np.random.default_rng()
    symbol_rate = rng.uniform(100, 1000)
    freq_dev = rng.uniform(200, 1000)
    samples_per_symbol = max(1, int(sample_rate / symbol_rate))
    n_symbols = n_samples // samples_per_symbol + 1
    bits = rng.choice([0, 1], size=n_symbols)
    freqs = np.where(bits, freq_dev, -freq_dev)
    inst_freq = np.repeat(freqs, samples_per_symbol)[:n_samples]
    phase = np.cumsum(2 * np.pi * inst_freq / sample_rate)
    return np.exp(1j * phase).astype(np.complex64)


def _fallback_generate_fsk4(
    n_samples: int,
    sample_rate: float = 48_000.0,
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    """Generate a 4-FSK signal."""
    rng = rng or np.random.default_rng()
    symbol_rate = rng.uniform(100, 1000)
    freq_dev = rng.uniform(200, 800)
    samples_per_symbol = max(1, int(sample_rate / symbol_rate))
    n_symbols = n_samples // samples_per_symbol + 1
    symbols = rng.choice([0, 1, 2, 3], size=n_symbols)
    freq_map = np.array([-3, -1, 1, 3]) * freq_dev
    inst_freq = np.repeat(freq_map[symbols], samples_per_symbol)[:n_samples]
    phase = np.cumsum(2 * np.pi * inst_freq / sample_rate)
    return np.exp(1j * phase).astype(np.complex64)


def _fallback_generate_noise(
    n_samples: int,
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    """Generate complex Gaussian noise (no signal)."""
    rng = rng or np.random.default_rng()
    return (
        rng.standard_normal(n_samples) + 1j * rng.standard_normal(n_samples)
    ).astype(np.complex64) / np.sqrt(2)


def _fallback_add_awgn(
    signal: np.ndarray,
    snr_db: float,
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    """Add AWGN to a complex signal at the specified SNR (dB)."""
    rng = rng or np.random.default_rng()
    sig_power = float(np.mean(np.abs(signal) ** 2))
    if sig_power == 0:
        sig_power = 1.0
    noise_power = sig_power / (10.0 ** (snr_db / 10.0))
    noise = np.sqrt(noise_power / 2) * (
        rng.standard_normal(len(signal)) + 1j * rng.standard_normal(len(signal))
    )
    return (signal + noise).astype(np.complex64)


# ---------------------------------------------------------------------------
# Dispatch table mapping ModulationType -> generator callable
# ---------------------------------------------------------------------------

_FALLBACK_GENERATORS: dict[ModulationType, callable] = {
    ModulationType.CW: _fallback_generate_cw,
    ModulationType.AM: _fallback_generate_am,
    ModulationType.SSB: _fallback_generate_ssb,
    ModulationType.BPSK: _fallback_generate_bpsk,
    ModulationType.QPSK: _fallback_generate_qpsk,
    ModulationType.FSK2: _fallback_generate_fsk2,
    ModulationType.FSK4: _fallback_generate_fsk4,
}


class AMCDataset(Dataset):
    """On-the-fly synthetic dataset for automatic modulation classification.

    Each call to ``__getitem__`` randomly selects a modulation type, generates
    the corresponding IQ signal, adds AWGN at a random SNR, and returns a
    ``(2, input_length)`` tensor (I/Q channels) with its integer class label.

    Parameters
    ----------
    samples_per_class : int
        Number of samples to provide per modulation class.
    snr_range_db : tuple[float, float]
        ``(min_snr, max_snr)`` in dB for the random SNR draw.
    input_length : int
        Number of time-domain samples per IQ segment.
    modulations : list[str] | None
        Ordered list of modulation names (must match ``ModulationType`` values).
        Defaults to all eight supported types.
    sample_rate : float
        Nominal sample rate for the signal generators.
    seed : int | None
        Base random seed (offset by index for reproducibility).
    """

    def __init__(
        self,
        samples_per_class: int = 10_000,
        snr_range_db: tuple[float, float] = (-10.0, 30.0),
        input_length: int = 1024,
        modulations: list[str] | None = None,
        sample_rate: float = 48_000.0,
        seed: int | None = None,
    ) -> None:
        super().__init__()
        self.samples_per_class = samples_per_class
        self.snr_range_db = tuple(snr_range_db)
        self.input_length = input_length
        self.sample_rate = sample_rate
        self.seed = seed

        if modulations is None:
            modulations = [m.value for m in ModulationType]
        self.modulations = [ModulationType(m) for m in modulations]
        self.num_classes = len(self.modulations)

    def __len__(self) -> int:
        return self.samples_per_class * self.num_classes

    def __getitem__(self, index: int) -> tuple[torch.Tensor, int]:
        """Generate a single (IQ tensor, label) pair.

        Parameters
        ----------
        index : int
            Dataset index.

        Returns
        -------
        tuple[Tensor, int]
            ``(iq, label)`` where ``iq`` has shape ``(2, input_length)`` and
            ``label`` is the integer class index.
        """
        # Determine class from index
        class_idx = index % self.num_classes
        mod_type = self.modulations[class_idx]

        # Per-sample RNG for reproducibility
        rng = np.random.default_rng(self.seed + index if self.seed is not None else None)

        # Random SNR for this sample
        snr_db = float(rng.uniform(self.snr_range_db[0], self.snr_range_db[1]))

        # Generate clean signal
        iq = self._generate_signal(mod_type, rng)

        # Add noise (for NOISE class the "signal" is already pure noise)
        if mod_type != ModulationType.NOISE:
            iq = self._add_noise(iq, snr_db, rng)

        # Normalise to unit power
        power = np.mean(np.abs(iq) ** 2)
        if power > 0:
            iq = iq / np.sqrt(power)

        # Split into 2-channel real tensor (I, Q)
        iq_tensor = torch.zeros(2, self.input_length, dtype=torch.float32)
        iq_tensor[0] = torch.from_numpy(iq.real.astype(np.float32))
        iq_tensor[1] = torch.from_numpy(iq.imag.astype(np.float32))

        return iq_tensor, class_idx

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _generate_signal(
        self,
        mod_type: ModulationType,
        rng: np.random.Generator,
    ) -> np.ndarray:
        """Generate a clean IQ signal for the given modulation type."""
        if mod_type == ModulationType.NOISE:
            if _USE_SYNTHETIC_PACKAGE:
                # Pure noise -- no clean signal component
                return _fallback_generate_noise(self.input_length, rng)
            return _fallback_generate_noise(self.input_length, rng)

        if _USE_SYNTHETIC_PACKAGE:
            # Use canonical generators from pingu.synthetic.signals
            generator_map = {
                ModulationType.CW: generate_cw,
                ModulationType.AM: generate_am,
                ModulationType.SSB: generate_ssb,
                ModulationType.BPSK: generate_bpsk,
                ModulationType.QPSK: generate_qpsk,
                ModulationType.FSK2: generate_fsk2,
                ModulationType.FSK4: generate_fsk4,
            }
            gen_fn = generator_map[mod_type]
            try:
                return gen_fn(
                    n_samples=self.input_length,
                    sample_rate=self.sample_rate,
                    rng=rng,
                )
            except TypeError:
                # Fallback if the signature doesn't match expectations
                pass

        # Use inline fallback generators
        gen_fn = _FALLBACK_GENERATORS[mod_type]
        return gen_fn(self.input_length, self.sample_rate, rng)

    def _add_noise(
        self,
        signal: np.ndarray,
        snr_db: float,
        rng: np.random.Generator,
    ) -> np.ndarray:
        """Add AWGN to a complex signal at the given SNR."""
        if _USE_SYNTHETIC_PACKAGE:
            try:
                return add_awgn(signal, snr_db, rng=rng)
            except TypeError:
                pass
        return _fallback_add_awgn(signal, snr_db, rng)
