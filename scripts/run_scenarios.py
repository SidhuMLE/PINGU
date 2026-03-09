#!/usr/bin/env python3
"""Run multiple PINGU TDoA scenarios and compare results.

Supports two modes:

1. **Quick sweep** via CLI flags::

    python scripts/run_scenarios.py --modulations cw bpsk --snr 20 --n-frames 10
    python scripts/run_scenarios.py --modulations all --snr-start -10 --snr-stop 30 --snr-step 10

2. **YAML manifest** for complex or pre-defined scenarios::

    python scripts/run_scenarios.py --manifest configs/scenarios/example_sweep.yaml
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Ensure the project root is on sys.path when running as a script.
_project_root = Path(__file__).resolve().parents[1]
if str(_project_root / "src") not in sys.path:
    sys.path.insert(0, str(_project_root / "src"))

from pingu.config import load_config
from pingu.scenarios import (
    ScenarioReport,
    ScenarioRunner,
    load_scenario_manifest,
    specs_from_cli_args,
)


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Run multiple PINGU TDoA scenarios and compare results",
    )

    # Manifest mode.
    parser.add_argument(
        "--manifest",
        type=str,
        default=None,
        help="Path to YAML scenario manifest (overrides sweep flags)",
    )

    # Sweep mode: modulations.
    parser.add_argument(
        "--modulations",
        nargs="+",
        default=["cw"],
        help=(
            "Modulation types to test: cw ssb am bpsk qpsk fsk2 fsk4, "
            "or 'all' for all types (default: cw)"
        ),
    )

    # Sweep mode: SNR.
    parser.add_argument(
        "--snr",
        type=float,
        default=None,
        help="Single SNR value in dB (default: 20.0)",
    )
    parser.add_argument(
        "--snr-start",
        type=float,
        default=None,
        help="Start of SNR sweep range in dB",
    )
    parser.add_argument(
        "--snr-stop",
        type=float,
        default=None,
        help="End of SNR sweep range in dB (inclusive)",
    )
    parser.add_argument(
        "--snr-step",
        type=float,
        default=5.0,
        help="Step size for SNR sweep (default: 5.0)",
    )

    # Sweep mode: positions.
    parser.add_argument(
        "--position",
        nargs=2,
        type=float,
        default=None,
        metavar=("X", "Y"),
        help="Single TX position in metres (default: 30000 -20000)",
    )
    parser.add_argument(
        "--positions",
        nargs="+",
        default=None,
        help="Multiple TX positions as 'x,y' strings (e.g. 30000,-20000 50000,10000)",
    )

    # Common parameters.
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Path to YAML config file (default: configs/default.yaml)",
    )
    parser.add_argument(
        "--n-frames",
        type=int,
        default=10,
        help="Number of frames per scenario (default: 10)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Base random seed (default: 42)",
    )
    parser.add_argument(
        "--plot",
        action="store_true",
        help="Generate comparison plots",
    )
    parser.add_argument(
        "--csv",
        action="store_true",
        help="Export results to CSV",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="output/scenarios",
        help="Directory to save plots and CSV (default: output/scenarios/)",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Suppress per-scenario progress output",
    )

    return parser.parse_args()


def main() -> None:
    """Entry point."""
    args = parse_args()
    cfg = load_config(args.config)

    # Build scenario specs.
    if args.manifest:
        specs = load_scenario_manifest(args.manifest)
    else:
        specs = specs_from_cli_args(args)

    if not specs:
        print("No scenarios to run.")
        return

    print(f"Running {len(specs)} scenario(s) ...\n")

    # Execute.
    runner = ScenarioRunner(config=cfg, verbose=not args.quiet)
    results = runner.run_all(specs)

    # Report.
    report = ScenarioReport(results)
    report.summary_table()
    report.print_stats()

    # Optional CSV.
    if args.csv:
        out_dir = Path(args.output_dir)
        csv_path = out_dir / "results.csv"
        report.to_csv(csv_path)
        print(f"CSV saved to {csv_path}")

    # Optional plots.
    if args.plot:
        out_dir = Path(args.output_dir)
        out_dir.mkdir(parents=True, exist_ok=True)

        fig_snr = report.plot_error_vs_snr(save_path=out_dir / "error_vs_snr.png")
        print(f"Error vs SNR plot saved to {out_dir / 'error_vs_snr.png'}")

        fig_conv = report.plot_convergence_comparison(
            save_path=out_dir / "convergence_comparison.png"
        )
        print(f"Convergence plot saved to {out_dir / 'convergence_comparison.png'}")


if __name__ == "__main__":
    main()
