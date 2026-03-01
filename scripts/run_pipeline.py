#!/usr/bin/env python3
"""Run the PINGU TDoA geolocation pipeline on a synthetic scenario.

Loads configuration, creates a synthetic multi-receiver scenario, runs the
end-to-end pipeline, and prints the estimated transmitter position alongside
the true position and the resulting error.

Usage
-----
    python scripts/run_pipeline.py
    python scripts/run_pipeline.py --config configs/default.yaml --snr 20 --n-frames 10
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

# Ensure the project root is on sys.path when running as a script.
_project_root = Path(__file__).resolve().parents[1]
if str(_project_root / "src") not in sys.path:
    sys.path.insert(0, str(_project_root / "src"))

from pingu.config import load_config
from pingu.pipeline.runner import PinguPipeline
from pingu.synthetic.scenarios import TDoAScenario
from pingu.types import ModulationType, ReceiverConfig


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Run the PINGU TDoA pipeline on a synthetic scenario"
    )
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Path to YAML config file (default: configs/default.yaml)",
    )
    parser.add_argument(
        "--snr",
        type=float,
        default=20.0,
        help="SNR in dB for the synthetic scenario (default: 20.0)",
    )
    parser.add_argument(
        "--n-frames",
        type=int,
        default=10,
        help="Number of frames to process (default: 10)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility (default: 42)",
    )
    return parser.parse_args()


def build_receivers(n: int = 5, radius: float = 100_000.0) -> list[ReceiverConfig]:
    """Create a regular polygon of receivers.

    Args:
        n: Number of receiver stations.
        radius: Radius of the polygon in metres.

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
            )
        )
    return receivers


def main() -> None:
    """Entry point for the pipeline demo."""
    args = parse_args()
    rng = np.random.default_rng(args.seed)

    # Load configuration.
    cfg = load_config(args.config)

    # Build receivers and pipeline.
    receivers = build_receivers(n=int(cfg.receivers.get("count", 5)))
    pipeline = PinguPipeline(config=cfg, receivers=receivers)

    # True transmitter position (inside the receiver polygon).
    tx_position = (30_000.0, -20_000.0)

    # Create the synthetic scenario.
    scenario = TDoAScenario(
        receivers=receivers,
        tx_position=tx_position,
        sample_rate=float(cfg.receivers.get("sample_rate", 48_000.0)),
        center_freq=14.1e6,
        snr_db=args.snr,
        duration=0.1,
        modulation=ModulationType.CW,
    )

    # Generate multiple frames (independent noise realisations).
    scenario_frames = []
    for _ in range(args.n_frames):
        frames = scenario.generate(rng=rng)
        scenario_frames.append(frames)

    # Run the pipeline.
    print(f"Running pipeline with {args.n_frames} frames at SNR = {args.snr} dB ...")
    estimate = pipeline.run(scenario_frames)

    if estimate is not None:
        error = np.sqrt(
            (estimate.x - tx_position[0]) ** 2
            + (estimate.y - tx_position[1]) ** 2
        )
        print(f"\n--- Results ---")
        print(f"True position:      ({tx_position[0]:.1f}, {tx_position[1]:.1f}) m")
        print(f"Estimated position: ({estimate.x:.1f}, {estimate.y:.1f}) m")
        print(f"Position error:     {error:.1f} m ({error / 1000:.2f} km)")
        print(f"95% conf. radius:   {estimate.confidence_radius_95:.1f} m")
        print(f"Residual:           {estimate.residual:.4e}")
        print(f"Kalman updates:     {pipeline.kalman.n_updates}")
    else:
        print("Pipeline did not produce a position estimate.")


if __name__ == "__main__":
    main()
