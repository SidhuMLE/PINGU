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
from pingu.scenarios.runner import build_receivers
from pingu.synthetic.scenarios import TDoAScenario
from pingu.tdoa.gcc import select_gcc_method
from pingu.types import ModulationType


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
    parser.add_argument(
        "--plot",
        action="store_true",
        help="Generate diagnostic plots (position map + convergence)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="output",
        help="Directory to save plots (default: output/)",
    )
    return parser.parse_args()


def main() -> None:
    """Entry point for the pipeline demo."""
    args = parse_args()
    rng = np.random.default_rng(args.seed)

    # Load configuration.
    cfg = load_config(args.config)

    # Build receivers and pipeline.
    receivers = build_receivers(n=int(cfg.receivers.get("count", 5)))

    # For synthetic scenarios with known modulation, override auto method
    # selection to use the correct GCC method directly.
    modulation = ModulationType.CW
    if cfg.tdoa.get("method", "auto") == "auto":
        cfg.tdoa.method = select_gcc_method(modulation)

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

    # --- Visualization ---------------------------------------------------
    if args.plot:
        from pingu.visualization import plot_position_map, plot_convergence

        out_dir = Path(args.output_dir)
        out_dir.mkdir(parents=True, exist_ok=True)

        # Position map
        fig_map = plot_position_map(
            receivers=receivers,
            estimate=estimate,
            true_pos=tx_position,
        )
        map_path = out_dir / "position_map.png"
        fig_map.savefig(map_path, dpi=150)
        print(f"\nPosition map saved to {map_path}")

        # Convergence plot
        if pipeline.variance_history:
            var_hist = np.array(pipeline.variance_history)
            fig_conv = plot_convergence(var_hist, title="TDoA Variance Convergence")
            conv_path = out_dir / "convergence.png"
            fig_conv.savefig(conv_path, dpi=150)
            print(f"Convergence plot saved to {conv_path}")


if __name__ == "__main__":
    main()
