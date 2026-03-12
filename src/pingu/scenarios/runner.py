"""Scenario execution engine.

Provides :class:`ScenarioRunner` for batch-executing multiple TDoA pipeline
scenarios with different parameters, and :func:`build_receivers` for
constructing receiver geometries.
"""

from __future__ import annotations

import time
from dataclasses import dataclass

import numpy as np
from omegaconf import DictConfig, OmegaConf

from pingu.constants import SPEED_OF_LIGHT
from pingu.pipeline.runner import PinguPipeline
from pingu.scenarios.spec import ScenarioSpec
from pingu.synthetic.scenarios import TDoAScenario
from pingu.tdoa.gcc import select_gcc_method
from pingu.types import FrameTrace, PositionEstimate, ReceiverConfig


def build_receivers(
    n: int = 5,
    radius: float = 100_000.0,
    sample_rate: float = 2_000_000.0,
) -> list[ReceiverConfig]:
    """Create a regular polygon of receivers.

    Args:
        n: Number of receiver stations.
        radius: Radius of the polygon in metres.
        sample_rate: Sample rate in Hz for all receivers.

    Returns:
        List of ReceiverConfig with Cartesian x, y positions.
    """
    receivers = []
    for i in range(n):
        angle = np.pi / 2 + 2 * np.pi * i / n
        receivers.append(
            ReceiverConfig(
                id=f"RX{i}",
                latitude=40.0 + 0.3 * np.sin(angle),
                longitude=-75.0 + 0.4 * np.cos(angle),
                x=radius * np.cos(angle),
                y=radius * np.sin(angle),
                sample_rate=sample_rate,
            )
        )
    return receivers


@dataclass
class ScenarioResult:
    """Result of a single scenario execution.

    Parameters
    ----------
    spec : ScenarioSpec
        The scenario that was executed.
    estimate : PositionEstimate | None
        Estimated transmitter position, or ``None`` if the solver failed.
    position_error_m : float | None
        Euclidean 2D position error in metres, or ``None``.
    converged : bool
        Whether the Kalman filter converged before exhausting all frames.
    n_kalman_updates : int
        Number of Kalman filter updates performed.
    elapsed_seconds : float
        Wall-clock execution time in seconds.
    variance_history : list[np.ndarray]
        Kalman diagonal variance per update (for convergence plots).
    """

    spec: ScenarioSpec
    estimate: PositionEstimate | None
    position_error_m: float | None
    converged: bool
    n_kalman_updates: int
    elapsed_seconds: float
    variance_history: list[np.ndarray]
    traces: list[FrameTrace] = None  # type: ignore[assignment]


class ScenarioRunner:
    """Execute a batch of scenario specifications against the pipeline.

    Parameters
    ----------
    config : DictConfig
        Base pipeline configuration (cloned per scenario).
    receivers : list[ReceiverConfig] | None
        Explicit receiver list. If ``None``, built from config.
    n_receivers : int
        Number of receivers when building default geometry.
    receiver_radius : float
        Radius in metres for the default polygon geometry.
    verbose : bool
        Print per-scenario progress to stdout.
    """

    def __init__(
        self,
        config: DictConfig,
        receivers: list[ReceiverConfig] | None = None,
        n_receivers: int = 5,
        receiver_radius: float = 100_000.0,
        verbose: bool = True,
    ) -> None:
        self._base_config = config
        self._verbose = verbose

        if receivers is not None:
            self._receivers = list(receivers)
        else:
            n = int(config.receivers.get("count", n_receivers))
            sample_rate = float(config.receivers.get("sample_rate", 2_000_000.0))
            self._receivers = build_receivers(
                n=n, radius=receiver_radius, sample_rate=sample_rate
            )

    def run_all(self, specs: list[ScenarioSpec]) -> list[ScenarioResult]:
        """Execute all scenarios sequentially.

        Args:
            specs: List of scenario specifications to execute.

        Returns:
            List of :class:`ScenarioResult` in the same order as *specs*.
        """
        results: list[ScenarioResult] = []
        total = len(specs)

        for i, spec in enumerate(specs):
            if self._verbose:
                print(
                    f"[{i + 1}/{total}] Running {spec.name} "
                    f"(mod={spec.modulation.value}, SNR={spec.snr_db} dB) ...",
                    flush=True,
                )
            result = self._run_one(spec)
            results.append(result)

            if self._verbose:
                if result.position_error_m is not None:
                    print(
                        f"         -> error: {result.position_error_m:.1f} m "
                        f"({result.position_error_m / 1000:.2f} km), "
                        f"time: {result.elapsed_seconds:.2f}s"
                    )
                else:
                    print(
                        f"         -> FAILED (no estimate), "
                        f"time: {result.elapsed_seconds:.2f}s"
                    )

        return results

    def _run_one(self, spec: ScenarioSpec) -> ScenarioResult:
        """Execute a single scenario."""
        t0 = time.monotonic()

        # Clone config and apply per-scenario overrides.
        cfg = OmegaConf.create(OmegaConf.to_container(self._base_config, resolve=True))
        if spec.config_overrides:
            override_cfg = OmegaConf.create(spec.config_overrides)
            cfg = OmegaConf.merge(cfg, override_cfg)

        # Select optimal GCC method for this modulation.
        if cfg.tdoa.get("method", "auto") == "auto":
            cfg.tdoa.method = select_gcc_method(spec.modulation)

        # Create fresh pipeline (Kalman state must not leak between scenarios).
        pipeline = PinguPipeline(config=cfg, receivers=self._receivers)

        # Create scenario and generate frames.
        scenario = TDoAScenario(
            receivers=self._receivers,
            tx_position=spec.tx_position,
            sample_rate=spec.sample_rate,
            center_freq=spec.center_freq,
            snr_db=spec.snr_db,
            duration=spec.duration,
            modulation=spec.modulation,
        )

        rng = np.random.default_rng(spec.seed)
        frames_list = [scenario.generate(rng=rng) for _ in range(spec.n_frames)]

        # Run pipeline.
        estimate = pipeline.run(frames_list)

        elapsed = time.monotonic() - t0

        # Compute position error.
        position_error = None
        if estimate is not None:
            position_error = float(
                np.sqrt(
                    (estimate.x - spec.tx_position[0]) ** 2
                    + (estimate.y - spec.tx_position[1]) ** 2
                )
            )

        converged = pipeline.kalman.n_updates < spec.n_frames

        # Compute true TDoAs for trace comparison.
        true_delays = self._compute_true_tdoas(
            spec.tx_position,
            self._receivers,
            pipeline.pair_manager.get_pairs(),
        )
        for trace in pipeline.traces:
            trace.tdoa_true_delays_s = true_delays

        return ScenarioResult(
            spec=spec,
            estimate=estimate,
            position_error_m=position_error,
            converged=converged,
            n_kalman_updates=pipeline.kalman.n_updates,
            elapsed_seconds=elapsed,
            variance_history=pipeline.variance_history,
            traces=pipeline.traces,
        )

    @staticmethod
    def _compute_true_tdoas(
        tx_position: tuple[float, float],
        receivers: list[ReceiverConfig],
        pairs: list[tuple[str, str]],
    ) -> list[float]:
        """Compute ground-truth TDoA delays for all receiver pairs."""
        rx_map = {r.id: r for r in receivers}
        true_delays = []
        for rx_i, rx_j in pairs:
            ri, rj = rx_map[rx_i], rx_map[rx_j]
            di = np.sqrt((tx_position[0] - ri.x) ** 2 + (tx_position[1] - ri.y) ** 2)
            dj = np.sqrt((tx_position[0] - rj.x) ** 2 + (tx_position[1] - rj.y) ** 2)
            true_delays.append((di - dj) / SPEED_OF_LIGHT)
        return true_delays
