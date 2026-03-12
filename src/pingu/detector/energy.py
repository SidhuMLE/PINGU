"""Neyman-Pearson energy detector with CFAR-style adaptive threshold.

Implements a cell-averaging CFAR (Constant False Alarm Rate) energy detector
for narrowband signal detection in channelized IQ data. The threshold is
derived from a specified probability of false alarm (Pfa) using the
chi-squared distribution.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray
from scipy import stats

from pingu.types import ChannelizedFrame, Detection


class EnergyDetector:
    """CFAR energy detector for channelized IQ frames.

    Uses reference cells surrounding each test cell (with guard cells to avoid
    signal self-masking) to estimate the local noise floor. A detection is
    declared when the test-cell energy exceeds the adaptive threshold derived
    from the desired probability of false alarm.

    Parameters
    ----------
    pfa : float
        Probability of false alarm (e.g. 1e-6).
    block_size : int
        Number of time-domain samples per detection block.
    guard_cells : int
        Number of guard cells on each side of the test cell.
    reference_cells : int
        Number of reference cells on each side of the test cell.
    """

    def __init__(
        self,
        pfa: float = 1e-6,
        block_size: int = 1024,
        guard_cells: int = 4,
        reference_cells: int = 16,
    ) -> None:
        if not (0.0 < pfa < 1.0):
            raise ValueError(f"pfa must be in (0, 1), got {pfa}")
        if guard_cells < 0:
            raise ValueError(f"guard_cells must be >= 0, got {guard_cells}")
        if reference_cells < 1:
            raise ValueError(f"reference_cells must be >= 1, got {reference_cells}")

        self.pfa = pfa
        self.block_size = block_size
        self.guard_cells = guard_cells
        self.reference_cells = reference_cells

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def detect(self, frame: ChannelizedFrame) -> list[Detection]:
        """Scan all channels in a channelized frame and return detections.

        For each channel the time-domain samples are segmented into
        non-overlapping blocks of ``block_size``. Each block is treated as a
        test cell evaluated against an adaptive noise-floor estimate built
        from its surrounding reference cells (skipping guard cells).

        Parameters
        ----------
        frame : ChannelizedFrame
            Channelized output with shape ``(n_channels, n_samples)``.

        Returns
        -------
        list[Detection]
            Detections found across all channels.
        """
        n_channels, n_samples = frame.channels.shape
        detections: list[Detection] = []

        # Compute per-block energy for every channel
        n_blocks = n_samples // self.block_size
        if n_blocks == 0:
            # Not enough samples for even one block; fall back to whole-channel
            n_blocks = 1
            effective_block_size = n_samples
        else:
            effective_block_size = self.block_size

        # Energy matrix: (n_channels, n_blocks)
        energy = np.empty((n_channels, n_blocks), dtype=np.float64)
        for ch in range(n_channels):
            for blk in range(n_blocks):
                start = blk * effective_block_size
                end = start + effective_block_size
                segment = frame.channels[ch, start:end]
                energy[ch, blk] = float(np.sum(np.abs(segment) ** 2))

        # Determine CFAR mode: time-domain (many blocks per channel) or
        # frequency-domain (scan across channels when blocks are scarce).
        min_cfar_cells = 2 * (self.guard_cells + self.reference_cells) + 1
        use_freq_cfar = n_blocks < min_cfar_cells and n_channels >= min_cfar_cells

        if use_freq_cfar:
            # Frequency-domain CFAR: compare each channel's total energy
            # against its neighboring channels.
            channel_energies = energy.sum(axis=1)  # shape (n_channels,)
            threshold_factor = self._compute_threshold(
                self.pfa, effective_block_size * n_blocks
            )

            for ch in range(n_channels):
                ref_indices = self._freq_cfar_refs(ch, n_channels)
                if not ref_indices:
                    continue
                noise_estimate = float(np.mean(channel_energies[ref_indices]))
                dof = 2 * effective_block_size * n_blocks
                sigma2_hat = noise_estimate / dof
                adaptive_threshold = threshold_factor * sigma2_hat

                if channel_energies[ch] > adaptive_threshold:
                    snr_linear = (
                        channel_energies[ch] / noise_estimate
                        if noise_estimate > 0
                        else float("inf")
                    )
                    snr_db = float(10.0 * np.log10(max(snr_linear, 1e-30)))
                    detections.append(
                        Detection(
                            receiver_id=frame.receiver_id,
                            channel_index=ch,
                            center_freq=float(frame.channel_freqs[ch]),
                            bandwidth=frame.channel_bw,
                            snr_estimate=snr_db,
                            timestamp=frame.timestamp,
                        )
                    )
        else:
            # Time-domain CFAR: scan over blocks within each channel.
            threshold_factor = self._compute_threshold(
                self.pfa, effective_block_size
            )
            for ch in range(n_channels):
                channel_energy = energy[ch]
                detected = self._cfar_scan(channel_energy, threshold_factor, effective_block_size)

                if detected:
                    noise_floor = self._estimate_noise_floor(channel_energy)
                    peak_energy = float(np.max(channel_energy))
                    snr_linear = (
                        peak_energy / noise_floor
                        if noise_floor > 0
                        else float("inf")
                    )
                    snr_db = float(10.0 * np.log10(max(snr_linear, 1e-30)))

                    detections.append(
                        Detection(
                            receiver_id=frame.receiver_id,
                            channel_index=ch,
                            center_freq=float(frame.channel_freqs[ch]),
                            bandwidth=frame.channel_bw,
                            snr_estimate=snr_db,
                            timestamp=frame.timestamp,
                        )
                    )

        return detections

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _compute_threshold(pfa: float, n_samples: int) -> float:
        """Compute the detection threshold from the desired Pfa.

        Under H0 (noise only), the energy of *n_samples* complex Gaussian
        samples follows a chi-squared distribution with ``2 * n_samples``
        degrees of freedom (real and imaginary parts each contribute one
        degree of freedom per sample). The threshold is the inverse CDF
        (percent-point function) at ``1 - pfa``.

        Parameters
        ----------
        pfa : float
            Probability of false alarm.
        n_samples : int
            Number of complex samples in the test cell.

        Returns
        -------
        float
            Threshold value (energy units, normalized by noise variance).
        """
        dof = 2 * n_samples
        return float(stats.chi2.ppf(1.0 - pfa, dof))

    def _cfar_scan(
        self,
        energy_cells: NDArray[np.float64],
        threshold_factor: float,
        effective_block_size: int | None = None,
    ) -> bool:
        """Run a 1-D CFAR scan over energy cells for a single channel.

        Returns True if *any* cell exceeds the adaptive threshold.

        Parameters
        ----------
        energy_cells : ndarray
            Per-block energy values, shape ``(n_blocks,)``.
        threshold_factor : float
            Chi-squared threshold factor from :meth:`_compute_threshold`.

        Returns
        -------
        bool
            True if a detection is declared in this channel.
        """
        if effective_block_size is None:
            effective_block_size = self.block_size
        n_cells = len(energy_cells)
        half_window = self.guard_cells + self.reference_cells

        for i in range(n_cells):
            # Gather reference-cell indices (excluding guard cells)
            ref_indices: list[int] = []
            for j in range(i - half_window, i - self.guard_cells):
                if 0 <= j < n_cells:
                    ref_indices.append(j)
            for j in range(i + self.guard_cells + 1, i + half_window + 1):
                if 0 <= j < n_cells:
                    ref_indices.append(j)

            if len(ref_indices) == 0:
                # Edge case: not enough reference cells, use all others
                ref_indices = [j for j in range(n_cells) if j != i]

            if len(ref_indices) == 0:
                continue

            # Adaptive threshold: scale the chi-squared threshold by the
            # average reference-cell energy (noise-floor estimate) divided
            # by the expected mean under H0.
            noise_estimate = float(np.mean(energy_cells[ref_indices]))
            # Under H0 the expected energy equals dof * sigma^2.  The
            # threshold_factor already encodes the chi-squared critical
            # value for unit variance, so we normalise accordingly.
            dof = 2 * effective_block_size
            sigma2_hat = noise_estimate / dof
            adaptive_threshold = threshold_factor * sigma2_hat

            if energy_cells[i] > adaptive_threshold:
                return True

        return False

    def _freq_cfar_refs(self, ch: int, n_channels: int) -> list[int]:
        """Return reference-cell indices for frequency-domain CFAR.

        Wraps around channel boundaries so that edge channels still get a
        full set of reference cells.
        """
        refs: list[int] = []
        half_window = self.guard_cells + self.reference_cells
        for offset in range(-half_window, half_window + 1):
            if offset == 0 or abs(offset) <= self.guard_cells:
                continue
            refs.append((ch + offset) % n_channels)
        return refs

    def _estimate_noise_floor(self, energy_cells: NDArray[np.float64]) -> float:
        """Estimate the noise floor from the energy cells.

        Uses the median energy (robust to signal outliers) as the noise-floor
        estimate.

        Parameters
        ----------
        energy_cells : ndarray
            Per-block energy values.

        Returns
        -------
        float
            Estimated noise-floor energy.
        """
        return float(np.median(energy_cells))
