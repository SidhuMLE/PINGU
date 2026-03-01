#!/usr/bin/env python3
"""Generate synthetic TDoA datasets for testing and validation.

Uses the :class:`TDoAScenario` to create multi-receiver IQ data with known
transmitter positions and saves the frames to disk as compressed NumPy
archives.

Usage
-----
    python scripts/generate_synthetic.py
    python scripts/generate_synthetic.py --n-scenarios 5 --snr 15 --output data/synthetic/
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

from pingu.synthetic.scenarios import TDoAScenario
from pingu.types import ModulationType, ReceiverConfig


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Generate synthetic TDoA datasets for testing"
    )
    parser.add_argument(
        "--n-scenarios",
        type=int,
        default=3,
        help="Number of scenario snapshots to generate (default: 3)",
    )
    parser.add_argument(
        "--snr",
        type=float,
        default=20.0,
        help="SNR in dB (default: 20.0)",
    )
    parser.add_argument(
        "--duration",
        type=float,
        default=0.1,
        help="Signal duration in seconds (default: 0.1)",
    )
    parser.add_argument(
        "--sample-rate",
        type=float,
        default=48_000.0,
        help="Sample rate in Hz (default: 48000)",
    )
    parser.add_argument(
        "--n-receivers",
        type=int,
        default=5,
        help="Number of receivers (default: 5)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output directory (default: data/synthetic/)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed (default: 42)",
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
                sample_rate=48_000.0,
            )
        )
    return receivers


def main() -> None:
    """Entry point for synthetic data generation."""
    args = parse_args()
    rng = np.random.default_rng(args.seed)

    output_dir = Path(args.output) if args.output else _project_root / "data" / "synthetic"
    output_dir.mkdir(parents=True, exist_ok=True)

    receivers = build_receivers(n=args.n_receivers)

    # Available modulations for variety.
    modulations = [
        ModulationType.CW,
        ModulationType.SSB,
        ModulationType.AM,
        ModulationType.BPSK,
        ModulationType.QPSK,
        ModulationType.FSK2,
        ModulationType.FSK4,
    ]

    print(f"Generating {args.n_scenarios} synthetic scenarios ...")
    print(f"  Receivers:   {args.n_receivers}")
    print(f"  SNR:         {args.snr} dB")
    print(f"  Duration:    {args.duration} s")
    print(f"  Sample rate: {args.sample_rate} Hz")
    print(f"  Output:      {output_dir}")

    for scenario_idx in range(args.n_scenarios):
        # Random transmitter position within the receiver polygon.
        tx_x = rng.uniform(-80_000, 80_000)
        tx_y = rng.uniform(-80_000, 80_000)
        tx_position = (float(tx_x), float(tx_y))

        # Cycle through modulations.
        mod = modulations[scenario_idx % len(modulations)]

        scenario = TDoAScenario(
            receivers=receivers,
            tx_position=tx_position,
            sample_rate=args.sample_rate,
            center_freq=14.1e6,
            snr_db=args.snr,
            duration=args.duration,
            modulation=mod,
        )

        frames = scenario.generate(rng=rng)

        # Pack into a dictionary for saving.
        save_dict = {
            "tx_x": tx_position[0],
            "tx_y": tx_position[1],
            "snr_db": args.snr,
            "modulation": mod.value,
            "sample_rate": args.sample_rate,
            "n_receivers": args.n_receivers,
        }
        for rx_id, frame in frames.items():
            save_dict[f"{rx_id}_samples"] = frame.samples
            save_dict[f"{rx_id}_x"] = next(
                r.x for r in receivers if r.id == rx_id
            )
            save_dict[f"{rx_id}_y"] = next(
                r.y for r in receivers if r.id == rx_id
            )

        out_path = output_dir / f"scenario_{scenario_idx:04d}.npz"
        np.savez_compressed(str(out_path), **save_dict)
        print(
            f"  [{scenario_idx + 1}/{args.n_scenarios}] {mod.value:>5s} "
            f"TX=({tx_x:+.0f}, {tx_y:+.0f}) m -> {out_path.name}"
        )

    print("Done.")


if __name__ == "__main__":
    main()
