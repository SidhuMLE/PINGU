"""Polyphase filterbank channelizer.

Implements an M-path polyphase analysis filterbank.  A prototype lowpass
filter is designed with ``scipy.signal.firwin``, decomposed into M polyphase
branches, and each input block is filtered through the branches before an
M-point FFT recovers the individual channel signals.

When the prototype filter length is an integer multiple of *M* and channels
overlap correctly (controlled by ``overlap_factor``), the filterbank achieves
near-perfect reconstruction.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray
from scipy.signal import firwin

from pingu.channelizer.base import BaseChannelizer
from pingu.types import ChannelizedFrame, IQFrame


class PolyphaseChannelizer(BaseChannelizer):
    """Polyphase filterbank channelizer.

    Parameters
    ----------
    n_channels : int
        Number of output channels (*M*).  Must be >= 2.
    overlap_factor : int, optional
        Oversampling / overlap factor that controls the prototype filter
        length.  The filter will have ``n_channels * overlap_factor`` taps.
        Higher values improve stopband attenuation at the cost of latency.
        Default is 4.
    window : str, optional
        Window type passed to ``scipy.signal.firwin`` for prototype filter
        design.  Default is ``"hamming"``.
    """

    def __init__(
        self,
        n_channels: int,
        overlap_factor: int = 4,
        window: str = "hamming",
    ) -> None:
        if n_channels < 2:
            raise ValueError(f"n_channels must be >= 2, got {n_channels}")
        if overlap_factor < 1:
            raise ValueError(f"overlap_factor must be >= 1, got {overlap_factor}")

        self._n_channels = n_channels
        self._overlap_factor = overlap_factor
        self._window = window

        # Design the prototype lowpass filter.
        # Cutoff at 1/M of Nyquist (i.e. channel bandwidth = fs / M).
        n_taps = n_channels * overlap_factor
        # Cutoff normalized to Nyquist (scipy convention): 1/M of Nyquist.
        self._prototype: NDArray[np.float64] = firwin(
            n_taps, cutoff=1.0 / n_channels, window=window
        )

        # Decompose into polyphase branches: shape (n_channels, overlap_factor)
        # Branch k contains taps h[k], h[k+M], h[k+2M], ...
        self._polyphase_branches: NDArray[np.float64] = self._prototype.reshape(
            overlap_factor, n_channels
        ).T

    # ------------------------------------------------------------------
    # BaseChannelizer interface
    # ------------------------------------------------------------------

    @property
    def n_channels(self) -> int:  # noqa: D401
        """Number of output channels."""
        return self._n_channels

    @property
    def channel_bandwidth(self) -> float:
        """Bandwidth of each output channel in Hz.

        The actual bandwidth depends on the input sample rate, so this
        returns the *fractional* bandwidth (1 / n_channels) which should
        be multiplied by the input sample rate to get Hz.
        """
        # The concrete value is computed per-call in channelize(); expose
        # the fractional value here for introspection.
        return 1.0 / self._n_channels

    def channelize(self, frame: IQFrame) -> ChannelizedFrame:
        """Channelize a wideband IQ frame using the polyphase filterbank.

        The input is segmented into blocks of length *M* (``n_channels``),
        each block is multiplied element-wise by the polyphase branches,
        and an M-point FFT yields one output sample per channel.

        Args:
            frame: Wideband IQ frame.

        Returns:
            ChannelizedFrame with ``channels`` of shape
            ``(n_channels, n_output_samples)``.
        """
        M = self._n_channels
        P = self._overlap_factor
        samples = frame.samples.astype(np.complex64)

        # Pad to an integer multiple of M so we can reshape cleanly.
        n_input = len(samples)
        pad_len = (-n_input) % M
        if pad_len:
            samples = np.concatenate([samples, np.zeros(pad_len, dtype=np.complex64)])

        n_blocks = len(samples) // M

        # We need P blocks of history for the polyphase filter.  Pre-pad
        # with zeros so the first output sample sees a full filter span.
        padded = np.concatenate([np.zeros((P - 1) * M, dtype=np.complex64), samples])

        n_output = n_blocks  # one output sample per original block

        # Output buffer: (n_channels, n_output)
        out = np.empty((M, n_output), dtype=np.complex64)

        for b in range(n_output):
            # Gather P consecutive M-length blocks ending at the current block.
            # block index in *padded* array: b + (P-1) is the "current" block,
            # so we need blocks from b to b + P - 1 inclusive.
            block_start = b * M
            block_end = block_start + P * M
            segment = padded[block_start:block_end]  # length P*M

            # Reshape to (P, M) and reverse so that the most recent block
            # is at index 0 (convolution ordering).
            segment_2d = segment.reshape(P, M)[::-1]

            # Multiply each branch (column k) by its filter coefficients
            # and sum across the P taps.
            filtered = np.sum(segment_2d * self._polyphase_branches.T, axis=0)  # length M

            # M-point FFT to recover channel outputs.
            out[:, b] = np.fft.fft(filtered)

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
            channels=out,
            channel_freqs=channel_freqs,
            channel_bw=channel_bw,
            sample_rate=channel_bw,  # decimated rate = fs / M
            timestamp=frame.timestamp,
        )
