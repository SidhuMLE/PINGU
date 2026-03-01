"""FFT / overlap-save channelizer.

A simpler reference channelizer that segments the input, applies an FFT to
each segment, extracts the bins belonging to each channel, and applies an
IFFT to recover per-channel time-domain samples.  Overlapping segments are
used to reduce spectral leakage artefacts.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from pingu.channelizer.base import BaseChannelizer
from pingu.types import ChannelizedFrame, IQFrame


class FFTChannelizer(BaseChannelizer):
    """FFT-based overlap-save channelizer.

    Parameters
    ----------
    n_channels : int
        Number of output channels.  Must be >= 2.
    fft_size : int, optional
        FFT length used for each segment.  Must be >= ``n_channels``.
        Defaults to ``n_channels * 4`` for a reasonable frequency resolution.
    overlap : int, optional
        Number of overlapping samples between consecutive segments.
        Defaults to ``fft_size // 2``.
    """

    def __init__(
        self,
        n_channels: int,
        fft_size: int | None = None,
        overlap: int | None = None,
    ) -> None:
        if n_channels < 2:
            raise ValueError(f"n_channels must be >= 2, got {n_channels}")

        self._n_channels = n_channels
        self._fft_size = fft_size if fft_size is not None else n_channels * 4

        if self._fft_size < n_channels:
            raise ValueError(
                f"fft_size ({self._fft_size}) must be >= n_channels ({n_channels})"
            )

        self._overlap = overlap if overlap is not None else self._fft_size // 2

        if self._overlap >= self._fft_size:
            raise ValueError(
                f"overlap ({self._overlap}) must be < fft_size ({self._fft_size})"
            )

    # ------------------------------------------------------------------
    # BaseChannelizer interface
    # ------------------------------------------------------------------

    @property
    def n_channels(self) -> int:  # noqa: D401
        """Number of output channels."""
        return self._n_channels

    @property
    def channel_bandwidth(self) -> float:
        """Fractional bandwidth of each channel (multiply by fs to get Hz)."""
        return 1.0 / self._n_channels

    def channelize(self, frame: IQFrame) -> ChannelizedFrame:
        """Channelize a wideband IQ frame using the FFT overlap-save method.

        For each overlapping segment of the input:
        1. Apply a Hann window and compute the FFT.
        2. Partition the FFT bins evenly into ``n_channels`` groups.
        3. Shift each group to baseband and IFFT to produce per-channel
           time-domain samples.

        The resulting per-channel sample count equals the number of segments
        times the number of bins per channel.

        Args:
            frame: Wideband IQ frame.

        Returns:
            ChannelizedFrame with ``channels`` of shape
            ``(n_channels, n_samples_per_channel)``.
        """
        M = self._n_channels
        N = self._fft_size
        hop = N - self._overlap
        samples = frame.samples.astype(np.complex64)

        # Pad the input so we can form at least one full segment.
        n_input = len(samples)
        if n_input < N:
            samples = np.concatenate([samples, np.zeros(N - n_input, dtype=np.complex64)])
            n_input = N

        # Number of segments.
        n_segments = max(1, (n_input - self._overlap) // hop)

        # Number of FFT bins assigned to each channel.
        bins_per_channel = N // M

        # Window for spectral leakage control.
        window = np.hanning(N).astype(np.float32)

        # Output: each segment contributes bins_per_channel time-domain
        # samples per channel via IFFT.
        out_samples_per_channel = n_segments * bins_per_channel
        channels: NDArray[np.complex64] = np.empty(
            (M, out_samples_per_channel), dtype=np.complex64
        )

        for seg_idx in range(n_segments):
            start = seg_idx * hop
            segment = samples[start : start + N] * window
            spectrum = np.fft.fft(segment)

            for ch in range(M):
                # Extract the bins for this channel.
                bin_start = ch * bins_per_channel
                bin_end = bin_start + bins_per_channel
                channel_spectrum = spectrum[bin_start:bin_end]

                # IFFT to get time-domain samples for this channel/segment.
                channel_td = np.fft.ifft(channel_spectrum).astype(np.complex64)

                out_start = seg_idx * bins_per_channel
                out_end = out_start + bins_per_channel
                channels[ch, out_start:out_end] = channel_td

        # Compute channel center frequencies.
        channel_bw = frame.sample_rate / M
        channel_freqs = (
            frame.center_freq
            - frame.sample_rate / 2
            + np.arange(M) * channel_bw
            + channel_bw / 2
        )

        return ChannelizedFrame(
            receiver_id=frame.receiver_id,
            channels=channels,
            channel_freqs=channel_freqs,
            channel_bw=channel_bw,
            sample_rate=channel_bw,  # decimated rate = fs / M
            timestamp=frame.timestamp,
        )
