"""Abstract base class for channelizer implementations."""

from __future__ import annotations

from abc import ABC, abstractmethod

from pingu.types import ChannelizedFrame, IQFrame


class BaseChannelizer(ABC):
    """Base class defining the channelizer interface.

    A channelizer splits a wideband IQ stream into multiple narrowband
    channels, each centered on an evenly-spaced sub-band.
    """

    @abstractmethod
    def channelize(self, frame: IQFrame) -> ChannelizedFrame:
        """Split a wideband IQ frame into narrowband channels.

        Args:
            frame: Wideband IQ frame from a single receiver.

        Returns:
            ChannelizedFrame with shape (n_channels, n_samples_per_channel).
        """

    @property
    @abstractmethod
    def n_channels(self) -> int:
        """Number of output channels."""

    @property
    @abstractmethod
    def channel_bandwidth(self) -> float:
        """Bandwidth of each output channel in Hz."""
