"""Scenario results reporting — tables, stats, CSV, and comparison plots."""

from __future__ import annotations

import csv
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

from pingu.scenarios.runner import ScenarioResult


class ScenarioReport:
    """Format and display scenario comparison results.

    Parameters
    ----------
    results : list[ScenarioResult]
        Results from :meth:`ScenarioRunner.run_all`.
    """

    def __init__(self, results: list[ScenarioResult]) -> None:
        self._results = results

    # ------------------------------------------------------------------
    # Summary table
    # ------------------------------------------------------------------

    def summary_table(self, file=None) -> None:
        """Print a formatted ASCII table of results."""
        if file is None:
            file = sys.stdout

        n_success = sum(1 for r in self._results if r.estimate is not None)
        total = len(self._results)

        print(
            f"\n{'=' * 78}",
            file=file,
        )
        print(
            f" Scenario Results ({total} scenarios, {n_success} succeeded)",
            file=file,
        )
        print(f"{'=' * 78}", file=file)

        # Header
        header = (
            f"{'Scenario':<22} | {'Mod':<5} | {'SNR':>5} | "
            f"{'Error (m)':>10} | {'Error (km)':>10} | "
            f"{'Conv':>4} | {'Updates':>7} | {'Time':>6}"
        )
        print(header, file=file)
        print("-" * len(header), file=file)

        for r in self._results:
            err_m = f"{r.position_error_m:10.1f}" if r.position_error_m is not None else "     FAILED"
            err_km = (
                f"{r.position_error_m / 1000:10.2f}"
                if r.position_error_m is not None
                else "          -"
            )
            conv = "Yes" if r.converged else "No"
            print(
                f"{r.spec.name:<22} | "
                f"{r.spec.modulation.value:<5} | "
                f"{r.spec.snr_db:5.1f} | "
                f"{err_m} | "
                f"{err_km} | "
                f"{conv:>4} | "
                f"{r.n_kalman_updates:>7} | "
                f"{r.elapsed_seconds:5.2f}s",
                file=file,
            )

        print(f"{'=' * 78}\n", file=file)

    # ------------------------------------------------------------------
    # Aggregate stats
    # ------------------------------------------------------------------

    def stats_by_modulation(self) -> dict[str, dict]:
        """Aggregate statistics grouped by modulation type."""
        return self._group_stats(key_fn=lambda r: r.spec.modulation.value)

    def stats_by_snr(self) -> dict[float, dict]:
        """Aggregate statistics grouped by SNR level."""
        return self._group_stats(key_fn=lambda r: r.spec.snr_db)

    def print_stats(self, file=None) -> None:
        """Print aggregate stats by modulation and by SNR."""
        if file is None:
            file = sys.stdout

        # By modulation
        mod_stats = self.stats_by_modulation()
        if mod_stats:
            print("--- Stats by Modulation ---", file=file)
            header = f"{'Mod':<8} | {'Success':>8} | {'Mean Err (m)':>12} | {'Median Err (m)':>14} | {'Std Err (m)':>12}"
            print(header, file=file)
            print("-" * len(header), file=file)
            for mod, s in sorted(mod_stats.items()):
                print(
                    f"{mod:<8} | "
                    f"{s['success_rate']:>7.0f}% | "
                    f"{s['mean_error']:>12.1f} | "
                    f"{s['median_error']:>14.1f} | "
                    f"{s['std_error']:>12.1f}",
                    file=file,
                )
            print(file=file)

        # By SNR
        snr_stats = self.stats_by_snr()
        if snr_stats:
            print("--- Stats by SNR ---", file=file)
            header = f"{'SNR (dB)':>8} | {'Success':>8} | {'Mean Err (m)':>12} | {'Median Err (m)':>14} | {'Std Err (m)':>12}"
            print(header, file=file)
            print("-" * len(header), file=file)
            for snr, s in sorted(snr_stats.items()):
                print(
                    f"{snr:>8.1f} | "
                    f"{s['success_rate']:>7.0f}% | "
                    f"{s['mean_error']:>12.1f} | "
                    f"{s['median_error']:>14.1f} | "
                    f"{s['std_error']:>12.1f}",
                    file=file,
                )
            print(file=file)

    def _group_stats(self, key_fn) -> dict:
        """Compute grouped statistics using an arbitrary key function."""
        groups: dict = defaultdict(list)
        for r in self._results:
            groups[key_fn(r)].append(r)

        stats = {}
        for key, group in groups.items():
            errors = [
                r.position_error_m for r in group if r.position_error_m is not None
            ]
            n_success = len(errors)
            n_total = len(group)
            if errors:
                stats[key] = {
                    "success_rate": 100.0 * n_success / n_total,
                    "mean_error": float(np.mean(errors)),
                    "median_error": float(np.median(errors)),
                    "std_error": float(np.std(errors)) if len(errors) > 1 else 0.0,
                    "min_error": float(np.min(errors)),
                    "max_error": float(np.max(errors)),
                    "n_total": n_total,
                    "n_success": n_success,
                }
            else:
                stats[key] = {
                    "success_rate": 0.0,
                    "mean_error": float("inf"),
                    "median_error": float("inf"),
                    "std_error": 0.0,
                    "min_error": float("inf"),
                    "max_error": float("inf"),
                    "n_total": n_total,
                    "n_success": 0,
                }
        return stats

    # ------------------------------------------------------------------
    # CSV export
    # ------------------------------------------------------------------

    def to_csv(self, path: str | Path) -> None:
        """Export results as CSV."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        with open(path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([
                "scenario", "modulation", "snr_db", "tx_x", "tx_y",
                "est_x", "est_y", "error_m", "error_km",
                "converged", "n_updates", "elapsed_s",
            ])
            for r in self._results:
                est_x = r.estimate.x if r.estimate else ""
                est_y = r.estimate.y if r.estimate else ""
                err_m = r.position_error_m if r.position_error_m is not None else ""
                err_km = (
                    r.position_error_m / 1000
                    if r.position_error_m is not None
                    else ""
                )
                writer.writerow([
                    r.spec.name,
                    r.spec.modulation.value,
                    r.spec.snr_db,
                    r.spec.tx_position[0],
                    r.spec.tx_position[1],
                    est_x,
                    est_y,
                    err_m,
                    err_km,
                    r.converged,
                    r.n_kalman_updates,
                    f"{r.elapsed_seconds:.3f}",
                ])

    # ------------------------------------------------------------------
    # Frame traces
    # ------------------------------------------------------------------

    def print_frame_traces(
        self,
        scenario_index: int = 0,
        max_frames: int = 20,
        file=None,
    ) -> None:
        """Print per-frame diagnostic traces for a scenario.

        Args:
            scenario_index: Index of the scenario in the results list.
            max_frames: Maximum number of frames to display.
            file: Output file object (default stdout).
        """
        if file is None:
            file = sys.stdout

        if scenario_index >= len(self._results):
            print(f"Scenario index {scenario_index} out of range.", file=file)
            return

        result = self._results[scenario_index]
        traces = getattr(result, "traces", None)
        if not traces:
            print("No trace data available for this scenario.", file=file)
            return

        print(
            f"\n--- Frame Traces: {result.spec.name} ---",
            file=file,
        )
        header = (
            f"{'Frame':>5} | {'Dets':>4} | {'Ch':>6} | "
            f"{'GCC':>5} | {'Mean TDoA Err':>14} | "
            f"{'Kalman Var':>10} | {'Position':>20} | {'Solver':>6}"
        )
        print(header, file=file)
        print("-" * len(header), file=file)

        for trace in traces[:max_frames]:
            # Detection info
            ch_str = str(trace.detected_channels[:3]) if trace.detected_channels else "-"
            if len(ch_str) > 6:
                ch_str = ch_str[:6]

            # TDoA error
            if trace.tdoa_true_delays_s and trace.tdoa_delays_s:
                errors = [
                    abs(e - t)
                    for e, t in zip(trace.tdoa_delays_s, trace.tdoa_true_delays_s)
                ]
                mean_err_us = np.mean(errors) * 1e6
                err_str = f"{mean_err_us:12.2f} us"
            else:
                err_str = "             -"

            # Kalman variance
            if trace.kalman_covariance_diag is not None:
                mean_var = float(np.mean(trace.kalman_covariance_diag))
                var_str = f"{mean_var:10.2e}"
            else:
                var_str = "         -"

            # Position
            if trace.position_estimate is not None:
                pos = trace.position_estimate
                pos_str = f"({pos.x:.0f}, {pos.y:.0f})"
            else:
                pos_str = "-"

            # Solver
            solver_str = "OK" if trace.solver_converged else "-"

            print(
                f"{trace.frame_index:>5} | "
                f"{trace.n_detections:>4} | "
                f"{ch_str:>6} | "
                f"{trace.gcc_method_used:>5} | "
                f"{err_str} | "
                f"{var_str} | "
                f"{pos_str:>20} | "
                f"{solver_str:>6}",
                file=file,
            )

        print(file=file)

    # ------------------------------------------------------------------
    # Comparison plots
    # ------------------------------------------------------------------

    def plot_error_vs_snr(self, save_path: str | Path | None = None):
        """Plot position error vs SNR, one line per modulation type.

        Returns the matplotlib Figure.
        """
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        # Group results by (modulation, snr).
        mod_snr_data: dict[str, dict[float, list[float]]] = defaultdict(
            lambda: defaultdict(list)
        )
        for r in self._results:
            if r.position_error_m is not None:
                mod_snr_data[r.spec.modulation.value][r.spec.snr_db].append(
                    r.position_error_m
                )

        fig, ax = plt.subplots(figsize=(10, 6))
        for mod_name in sorted(mod_snr_data.keys()):
            snr_errors = mod_snr_data[mod_name]
            snrs = sorted(snr_errors.keys())
            mean_errors = [np.mean(snr_errors[s]) for s in snrs]
            ax.semilogy(snrs, mean_errors, "o-", label=mod_name.upper(), markersize=6)

        ax.set_xlabel("SNR (dB)")
        ax.set_ylabel("Position Error (m)")
        ax.set_title("Position Error vs SNR by Modulation")
        ax.legend()
        ax.grid(True, alpha=0.3)
        fig.tight_layout()

        if save_path:
            fig.savefig(save_path, dpi=150)

        return fig

    def plot_convergence_comparison(
        self,
        max_scenarios: int = 8,
        save_path: str | Path | None = None,
    ):
        """Plot convergence curves for multiple scenarios.

        Returns the matplotlib Figure.
        """
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        # Pick scenarios with variance history.
        candidates = [r for r in self._results if r.variance_history]
        candidates = candidates[:max_scenarios]

        if not candidates:
            fig, ax = plt.subplots()
            ax.text(0.5, 0.5, "No convergence data", ha="center", va="center")
            return fig

        fig, ax = plt.subplots(figsize=(10, 6))
        for r in candidates:
            var_hist = np.array(r.variance_history)
            mean_var = var_hist.mean(axis=1)
            ax.semilogy(
                range(1, len(mean_var) + 1),
                mean_var,
                "o-",
                label=r.spec.name,
                markersize=4,
            )

        ax.set_xlabel("Update")
        ax.set_ylabel("Mean TDoA Variance")
        ax.set_title("Kalman Filter Convergence")
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)
        fig.tight_layout()

        if save_path:
            fig.savefig(save_path, dpi=150)

        return fig
