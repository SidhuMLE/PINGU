"""End-to-end PINGU pipeline orchestrator.

The :class:`PinguPipeline` class wires together every stage of the HF TDoA
geolocation chain -- channelization, detection, TDoA estimation, Bayesian
integration, convergence monitoring, and position solving -- into a single
coherent processing loop driven by per-frame IQ data.
"""

from __future__ import annotations

import logging

import numpy as np
from numpy.typing import NDArray
from omegaconf import DictConfig
from scipy.signal import firwin, lfilter

from pingu.channelizer.polyphase import PolyphaseChannelizer
from pingu.channelizer.fft import FFTChannelizer
from pingu.detector.energy import EnergyDetector
from pingu.integrator.convergence import ConvergenceMonitor
from pingu.integrator.kalman import TDoAKalmanFilter
from pingu.locator.geometry import ReceiverGeometry
from pingu.locator.solvers import TDoASolver
from pingu.tdoa.gcc import select_gcc_method
from pingu.tdoa.pair_manager import PairManager
from pingu.types import (
    ChannelizedFrame,
    Detection,
    FrameTrace,
    IQFrame,
    PositionEstimate,
    ReceiverConfig,
)

logger = logging.getLogger(__name__)


def _bandpass_filter(
    signal: NDArray[np.complex64],
    center_freq: float,
    bandwidth: float,
    fs: float,
) -> NDArray[np.complex64]:
    """Bandpass filter a wideband IQ signal around a channel's frequency band.

    Shifts the signal so the target channel is at DC, applies a lowpass FIR
    filter of width *bandwidth*, then shifts back.  This works correctly for
    both positive and negative baseband offsets.

    Args:
        signal: Wideband complex IQ signal.
        center_freq: Baseband offset of the target channel (Hz), i.e.
            the channel's absolute RF frequency minus the IQ center frequency.
        bandwidth: Bandwidth of the channel (Hz).
        fs: Sample rate of the signal (Hz).

    Returns:
        Bandpass-filtered complex64 signal at full sample rate.
    """
    n = len(signal)
    nyquist = fs / 2.0

    # Frequency-shift to bring the channel to DC
    t = np.arange(n, dtype=np.float64) / fs
    shifted = signal * np.exp(-2j * np.pi * center_freq * t)

    # Lowpass filter: keep ±bandwidth/2 around DC (with 20% margin)
    cutoff = bandwidth * 0.6 / nyquist
    cutoff = min(cutoff, 0.999)
    if cutoff <= 0.001:
        return signal

    n_taps = 101
    taps = firwin(n_taps, cutoff)
    filtered = lfilter(taps, 1.0, shifted)

    # Shift back to original frequency
    filtered = filtered * np.exp(2j * np.pi * center_freq * t)

    return filtered.astype(np.complex64)


def _compute_max_delay_from_geometry(
    receivers: list[ReceiverConfig],
    fs: float,
    margin: float = 1.5,
) -> int:
    """Compute max delay samples from receiver baseline geometry.

    Args:
        receivers: List of receiver configurations.
        fs: Sample rate in Hz.
        margin: Safety margin factor (default 1.5 = 50% extra).

    Returns:
        Maximum delay in samples.
    """
    from pingu.constants import SPEED_OF_LIGHT

    max_dist = 0.0
    for i in range(len(receivers)):
        for j in range(i + 1, len(receivers)):
            dx = receivers[i].x - receivers[j].x
            dy = receivers[i].y - receivers[j].y
            dist = np.sqrt(dx * dx + dy * dy)
            max_dist = max(max_dist, dist)

    max_delay_s = max_dist / SPEED_OF_LIGHT
    return int(np.ceil(max_delay_s * fs * margin))


class PinguPipeline:
    """End-to-end TDoA geolocation pipeline.

    Orchestrates channelization, signal detection, TDoA estimation, Kalman
    integration, convergence monitoring, and non-linear position solving.

    Parameters
    ----------
    config : DictConfig
        Full PINGU configuration (as loaded by :func:`pingu.config.load_config`).
    receivers : list[ReceiverConfig] | None
        Explicit list of receiver configurations.  If ``None``, a default
        pentagon geometry is constructed from the config.
    """

    def __init__(
        self,
        config: DictConfig,
        receivers: list[ReceiverConfig] | None = None,
    ) -> None:
        self._cfg = config

        # --- Receivers ---------------------------------------------------
        if receivers is not None:
            self._receivers = list(receivers)
        else:
            self._receivers = self._build_default_receivers(config)

        self._receiver_ids = [r.id for r in self._receivers]

        # --- Channelizer -------------------------------------------------
        chan_cfg = config.channelizer
        method = chan_cfg.get("method", "polyphase")
        n_channels = int(chan_cfg.get("n_channels", 64))
        if method == "polyphase":
            overlap_factor = int(chan_cfg.get("overlap_factor", 4))
            window = chan_cfg.get("window", "hann")
            self.channelizer = PolyphaseChannelizer(
                n_channels=n_channels,
                overlap_factor=overlap_factor,
                window=window,
            )
        else:
            self.channelizer = FFTChannelizer(n_channels=n_channels)

        # --- Detector ----------------------------------------------------
        det_cfg = config.detector
        self.detector = EnergyDetector(
            pfa=float(det_cfg.get("pfa", 1e-6)),
            block_size=int(det_cfg.get("block_size", 32768)),
            guard_cells=int(det_cfg.get("guard_cells", 4)),
            reference_cells=int(det_cfg.get("reference_cells", 16)),
        )

        # --- Classifier (optional) ----------------------------------------
        self._classifier = None
        cls_cfg = config.get("classifier", {})
        checkpoint = cls_cfg.get("checkpoint", None)
        if checkpoint is not None:
            from pingu.classifier.inference import AMCInference
            self._classifier = AMCInference(
                checkpoint_path=checkpoint,
                input_length=int(cls_cfg.get("input_length", 1024)),
                class_names=list(cls_cfg.get("classes", [])) or None,
            )
            logger.info("AMCCNN classifier loaded from %s", checkpoint)
        self._confidence_threshold: float = float(
            cls_cfg.get("confidence_threshold", 0.5)
        )

        # --- TDoA pair management ----------------------------------------
        self.pair_manager = PairManager(self._receiver_ids)
        self._tdoa_method: str = config.tdoa.get("method", "phat")
        self._fft_size: int = int(config.tdoa.get("fft_size", 131072))

        # --- max_delay_samples from config or geometry ---
        cfg_max_delay = config.tdoa.get("max_delay_samples", None)
        if cfg_max_delay is not None:
            self._max_delay_samples: int = int(cfg_max_delay)
        else:
            fs = float(config.receivers.get("sample_rate", 2_000_000.0))
            self._max_delay_samples = _compute_max_delay_from_geometry(
                self._receivers, fs
            )

        # --- Kalman filter -----------------------------------------------
        n_pairs = self.pair_manager.n_pairs
        int_cfg = config.integrator
        self.kalman = TDoAKalmanFilter(
            n_pairs=n_pairs,
            process_noise=float(int_cfg.get("process_noise", 1e-12)),
            initial_variance=float(int_cfg.get("initial_variance", 1e-6)),
            pair_labels=self.pair_manager.get_pairs(),
        )

        # --- Convergence monitor -----------------------------------------
        self.convergence_monitor = ConvergenceMonitor(n_pairs=n_pairs)
        self._convergence_factor: float = float(
            int_cfg.get("convergence_factor", 0.1)
        )

        # Diagnostic history for visualization.
        self.variance_history: list[np.ndarray] = []

        # Per-frame traces.
        self.traces: list[FrameTrace] = []
        self._frame_counter: int = 0

        # --- Locator / solver -------------------------------------------
        geometry = ReceiverGeometry(self._receivers)
        loc_cfg = config.locator
        solver_config = {
            "n_grid_points": int(loc_cfg.grid_search.get("n_points", 50)),
            "extent_km": float(loc_cfg.grid_search.get("extent_km", 500)),
            "ftol": float(loc_cfg.get("tolerance", 1e-8)),
            "xtol": float(loc_cfg.get("tolerance", 1e-8)),
            "gtol": float(loc_cfg.get("tolerance", 1e-8)),
            "max_nfev": int(loc_cfg.get("max_iterations", 100)) * 10,
        }
        self.solver = TDoASolver(geometry=geometry, config=solver_config)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def process_frame(
        self, frames: dict[str, IQFrame]
    ) -> PositionEstimate | None:
        """Process a single time-frame of IQ data from all receivers.

        Steps:
            1. Channelize each receiver's wideband IQ.
            2. Detect signals in each channelized output.
            3. Bandpass filter wideband IQ to the detected channel, then
               estimate TDoAs at full sample rate.
            4. Update the Kalman filter.
            5. If converged, solve for the transmitter position.

        Args:
            frames: Mapping from receiver ID to its :class:`IQFrame`.

        Returns:
            A :class:`PositionEstimate` if the filter has converged and a
            position was successfully solved, otherwise ``None``.
        """
        self._frame_counter += 1
        timestamp = next(iter(frames.values())).timestamp
        trace = FrameTrace(
            frame_index=self._frame_counter,
            timestamp=timestamp,
        )

        # 1. Channelize
        channelized: dict[str, ChannelizedFrame] = {}
        for rx_id, frame in frames.items():
            channelized[rx_id] = self.channelizer.channelize(frame)

        # Trace: channels with energy
        if channelized:
            first_ch = next(iter(channelized.values()))
            ch_energy = np.sum(np.abs(first_ch.channels) ** 2, axis=1)
            trace.n_channels_with_energy = int(np.sum(ch_energy > 0))

        # 2. Detect signals per receiver
        detections: dict[str, list[Detection]] = {}
        for rx_id, ch_frame in channelized.items():
            det = self.detector.detect(ch_frame)
            if det:
                detections[rx_id] = det

        # Trace: detections
        all_dets = [d for dets in detections.values() for d in dets]
        trace.n_detections = len(all_dets)
        trace.detected_channels = sorted(set(d.channel_index for d in all_dets))
        trace.detection_snrs_db = [d.snr_estimate for d in all_dets]

        # 2b. Classify detected signals (if classifier is available).
        detected_modulation = None
        if self._classifier is not None and detections:
            for rx_id, det_list in detections.items():
                if rx_id in frames:
                    iq = frames[rx_id].samples
                    for det in det_list:
                        try:
                            mod, conf = self._classifier.classify(iq)
                            if conf >= self._confidence_threshold:
                                det.modulation = mod
                                det.confidence = conf
                                detected_modulation = mod
                                trace.classified_modulation = mod.value if hasattr(mod, 'value') else str(mod)
                                trace.classification_confidence = conf
                        except Exception:
                            logger.debug(
                                "Classification failed for %s", rx_id,
                                exc_info=True,
                            )

        # 2c. Select GCC method.
        if self._tdoa_method == "auto":
            tdoa_method = select_gcc_method(detected_modulation)
            logger.debug("Auto-selected GCC method: %s (mod=%s)", tdoa_method, detected_modulation)
        else:
            tdoa_method = self._tdoa_method
        trace.gcc_method_used = tdoa_method

        # 3. Select IQ for TDoA estimation.
        #    Bandpass filter wideband IQ to detected channel band, keeping
        #    full sample rate for maximum time-delay resolution.
        sample_rate = next(iter(frames.values())).sample_rate
        signals_for_tdoa: dict[str, np.ndarray] = {}

        common_channel = self._find_common_channel(detections, channelized)
        if common_channel is not None and channelized:
            # Get channel parameters from any channelized frame
            ref_ch_frame = next(iter(channelized.values()))
            ch_bandwidth = ref_ch_frame.channel_bw
            n_ch = ref_ch_frame.channels.shape[0]

            # Compute baseband offset from channel index.
            # The polyphase channelizer's FFT output: channel k corresponds
            # to baseband frequency k * bw for k < N/2, and (k - N) * bw
            # for k >= N/2 (standard FFT ordering).
            if common_channel < n_ch // 2:
                ch_offset = common_channel * ch_bandwidth
            else:
                ch_offset = (common_channel - n_ch) * ch_bandwidth

            trace.selected_channel = common_channel
            trace.selected_channel_freq = float(ref_ch_frame.channel_freqs[common_channel])

            for rx_id, frame in frames.items():
                signals_for_tdoa[rx_id] = _bandpass_filter(
                    frame.samples, ch_offset, ch_bandwidth, sample_rate
                )

            # After bandpass filtering, always use basic cross-correlation.
            # The filter already removes out-of-band noise; PHAT/SCOT would
            # whiten the mostly-empty spectrum and amplify stopband leakage.
            tdoa_method = "basic"
            trace.gcc_method_used = tdoa_method
        else:
            # Fallback: use full wideband IQ
            for rx_id, frame in frames.items():
                signals_for_tdoa[rx_id] = frame.samples

        # 4. Estimate TDoAs for all pairs.
        tdoa_estimates = self.pair_manager.estimate_all_tdoas(
            signals=signals_for_tdoa,
            fs=sample_rate,
            method=tdoa_method,
            max_delay_samples=self._max_delay_samples,
        )

        # Trace: TDoA results
        trace.tdoa_delays_s = [est.delay for est in tdoa_estimates]
        trace.tdoa_variances = [est.variance for est in tdoa_estimates]
        trace.tdoa_peak_values = [est.correlation_peak for est in tdoa_estimates]

        # Build measurement vectors for the Kalman filter.
        measurements = np.array(
            [est.delay for est in tdoa_estimates], dtype=np.float64
        )
        variances = np.array(
            [est.variance for est in tdoa_estimates], dtype=np.float64
        )

        # 5. Kalman predict + update.
        self.kalman.predict()
        innovation = self.kalman.update(measurements, variances, timestamp=timestamp)

        # Trace: Kalman state
        trace.kalman_innovation = innovation
        state = self.kalman.get_state()
        trace.kalman_covariance_diag = np.diag(state.covariance).copy()

        # Update convergence monitor and record variance history.
        self.convergence_monitor.update(state.covariance)
        self.variance_history.append(np.diag(state.covariance).copy())

        # 6. Check convergence and solve.
        result = None
        if self.convergence_monitor.is_converged(factor=self._convergence_factor):
            integrated = self.kalman.get_state()
            # Weights for the solver: 1 / sigma (std dev).
            diag_var = np.diag(integrated.covariance)
            weights = 1.0 / np.sqrt(np.maximum(diag_var, 1e-30))

            try:
                position = self.solver.solve(
                    tdoas=integrated.delays,
                    weights=weights,
                )
                logger.info(
                    "Position solved: (%.1f, %.1f) m after %d updates",
                    position.x,
                    position.y,
                    self.kalman.n_updates,
                )
                trace.position_estimate = position
                trace.solver_converged = True
                result = position
            except Exception:
                logger.warning(
                    "Solver failed after %d updates; continuing integration.",
                    self.kalman.n_updates,
                    exc_info=True,
                )

        self.traces.append(trace)
        return result

    def run(
        self, scenario_frames: list[dict[str, IQFrame]]
    ) -> PositionEstimate | None:
        """Process multiple frames until convergence or exhaustion.

        Iterates through the provided frame sequence, calling
        :meth:`process_frame` for each.  Returns the first successful
        position estimate, or ``None`` if convergence is never reached.

        Args:
            scenario_frames: Ordered list of per-receiver IQ frame dicts,
                one entry per time step.

        Returns:
            The first converged :class:`PositionEstimate`, or ``None``.
        """
        for i, frames in enumerate(scenario_frames):
            logger.debug("Processing frame %d / %d", i + 1, len(scenario_frames))
            result = self.process_frame(frames)
            if result is not None:
                return result

        # If we exhaust all frames without convergence, attempt a solve
        # with the current best state anyway.
        logger.info(
            "All %d frames processed without convergence; attempting final solve.",
            len(scenario_frames),
        )
        integrated = self.kalman.get_state()
        diag_var = np.diag(integrated.covariance)
        weights = 1.0 / np.sqrt(np.maximum(diag_var, 1e-30))
        try:
            return self.solver.solve(tdoas=integrated.delays, weights=weights)
        except Exception:
            logger.error("Final solve failed.", exc_info=True)
            return None

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _build_default_receivers(config: DictConfig) -> list[ReceiverConfig]:
        """Build a default pentagon receiver layout from config."""
        n = int(config.receivers.get("count", 5))
        R = 100_000.0  # 100 km radius
        sample_rate = float(config.receivers.get("sample_rate", 2_000_000.0))
        receivers = []
        for i in range(n):
            angle = np.pi / 2 + 2 * np.pi * i / n
            receivers.append(
                ReceiverConfig(
                    id=f"RX{i}",
                    latitude=40.0 + 0.3 * np.sin(angle),
                    longitude=-75.0 + 0.4 * np.cos(angle),
                    x=R * np.cos(angle),
                    y=R * np.sin(angle),
                    sample_rate=sample_rate,
                )
            )
        return receivers

    @staticmethod
    def _find_common_channel(
        detections: dict[str, list[Detection]],
        channelized: dict[str, ChannelizedFrame],
    ) -> int | None:
        """Find the best channel detected across multiple receivers.

        First filters to channels detected in >= 2 receivers, then picks the
        one with the highest average SNR.  This avoids selecting leakage
        channels that appear in many receivers but carry little signal energy.

        Returns the channel index, or ``None`` if no common channel is found.
        """
        # Track detection count and cumulative SNR per channel.
        channel_counts: dict[int, int] = {}
        channel_snr_sum: dict[int, float] = {}
        for rx_id, det_list in detections.items():
            for det in det_list:
                ch = det.channel_index
                channel_counts[ch] = channel_counts.get(ch, 0) + 1
                channel_snr_sum[ch] = channel_snr_sum.get(ch, 0.0) + det.snr_estimate

        if not channel_counts:
            return None

        # Keep only channels detected in >= 2 receivers.
        candidates = {ch for ch, cnt in channel_counts.items() if cnt >= 2}
        if not candidates:
            return None

        # Among candidates, pick the one with highest average SNR.
        best_channel = max(
            candidates,
            key=lambda ch: channel_snr_sum[ch] / channel_counts[ch],
        )
        return best_channel
