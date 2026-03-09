"""Tests for the multi-scenario runner."""

from __future__ import annotations

import csv
import tempfile
from pathlib import Path

import numpy as np
import pytest
from omegaconf import OmegaConf

from pingu.config import load_config
from pingu.scenarios.report import ScenarioReport
from pingu.scenarios.runner import ScenarioResult, ScenarioRunner, build_receivers
from pingu.scenarios.spec import (
    ScenarioSpec,
    expand_sweep,
    load_scenario_manifest,
)
from pingu.types import ModulationType, PositionEstimate


# ---------------------------------------------------------------------------
# build_receivers
# ---------------------------------------------------------------------------

class TestBuildReceivers:
    """Tests for the build_receivers utility."""

    def test_default_pentagon(self) -> None:
        """Default call should produce 5 receivers."""
        rxs = build_receivers()
        assert len(rxs) == 5

    def test_custom_count(self) -> None:
        rxs = build_receivers(n=3)
        assert len(rxs) == 3

    def test_receiver_ids_unique(self) -> None:
        rxs = build_receivers(n=7)
        ids = [r.id for r in rxs]
        assert len(set(ids)) == 7

    def test_radius_affects_positions(self) -> None:
        small = build_receivers(n=3, radius=1000.0)
        large = build_receivers(n=3, radius=200_000.0)
        small_dist = np.sqrt(small[0].x ** 2 + small[0].y ** 2)
        large_dist = np.sqrt(large[0].x ** 2 + large[0].y ** 2)
        assert large_dist > small_dist * 10


# ---------------------------------------------------------------------------
# ScenarioSpec / expand_sweep
# ---------------------------------------------------------------------------

class TestExpandSweep:
    """Tests for sweep expansion."""

    def test_single_modulation_single_snr(self) -> None:
        specs = expand_sweep({"modulations": ["cw"], "snr_db": 20.0})
        assert len(specs) == 1
        assert specs[0].modulation == ModulationType.CW
        assert specs[0].snr_db == 20.0

    def test_cartesian_product(self) -> None:
        """2 modulations x 3 SNRs x 1 position = 6 specs."""
        specs = expand_sweep({
            "modulations": ["cw", "bpsk"],
            "snr_range": [0, 20, 10],  # 0, 10, 20
            "positions": [[30000, -20000]],
        })
        assert len(specs) == 6

    def test_all_modulations(self) -> None:
        """'all' should expand to all modulations except NOISE."""
        specs = expand_sweep({"modulations": "all", "snr_db": 20.0})
        mods = {s.modulation for s in specs}
        assert ModulationType.NOISE not in mods
        assert ModulationType.CW in mods
        assert ModulationType.BPSK in mods
        assert len(mods) == 7  # SSB, CW, AM, FSK2, FSK4, BPSK, QPSK

    def test_multiple_positions(self) -> None:
        specs = expand_sweep({
            "modulations": ["cw"],
            "snr_db": 20.0,
            "positions": [[30000, -20000], [50000, 10000]],
        })
        assert len(specs) == 2
        assert specs[0].tx_position != specs[1].tx_position

    def test_unique_seeds(self) -> None:
        """Each spec should get a unique seed."""
        specs = expand_sweep({
            "modulations": ["cw", "bpsk"],
            "snr_range": [0, 10, 10],
        })
        seeds = [s.seed for s in specs]
        assert len(set(seeds)) == len(seeds)

    def test_auto_name_generation(self) -> None:
        specs = expand_sweep({"modulations": ["cw"], "snr_db": 20.0})
        assert "CW" in specs[0].name
        assert "SNR20" in specs[0].name

    def test_snr_range_inclusive(self) -> None:
        """snr_range stop should be inclusive."""
        specs = expand_sweep({
            "modulations": ["cw"],
            "snr_range": [0, 30, 10],
        })
        snrs = [s.snr_db for s in specs]
        assert 0.0 in snrs
        assert 30.0 in snrs
        assert len(snrs) == 4  # 0, 10, 20, 30


# ---------------------------------------------------------------------------
# YAML manifest loading
# ---------------------------------------------------------------------------

class TestLoadManifest:
    """Tests for YAML scenario manifest loading."""

    def test_sweep_format(self, tmp_path: Path) -> None:
        manifest = tmp_path / "sweep.yaml"
        manifest.write_text(
            "sweep:\n"
            "  modulations: [cw, bpsk]\n"
            "  snr_range: [10, 20, 10]\n"
            "  positions:\n"
            "    - [30000, -20000]\n"
            "  n_frames: 5\n"
        )
        specs = load_scenario_manifest(manifest)
        assert len(specs) == 4  # 2 mods x 2 SNRs
        assert all(s.n_frames == 5 for s in specs)

    def test_explicit_format(self, tmp_path: Path) -> None:
        manifest = tmp_path / "explicit.yaml"
        manifest.write_text(
            "scenarios:\n"
            "  - name: test_cw\n"
            "    modulation: cw\n"
            "    snr_db: 25.0\n"
            "    tx_position: [40000, -10000]\n"
            "  - name: test_bpsk\n"
            "    modulation: bpsk\n"
            "    snr_db: 15.0\n"
        )
        specs = load_scenario_manifest(manifest)
        assert len(specs) == 2
        assert specs[0].name == "test_cw"
        assert specs[0].snr_db == 25.0
        assert specs[1].modulation == ModulationType.BPSK

    def test_invalid_manifest_raises(self, tmp_path: Path) -> None:
        manifest = tmp_path / "bad.yaml"
        manifest.write_text("something_wrong: true\n")
        with pytest.raises(ValueError, match="sweep.*scenarios"):
            load_scenario_manifest(manifest)


# ---------------------------------------------------------------------------
# ScenarioRunner
# ---------------------------------------------------------------------------

class TestScenarioRunner:
    """Tests for the scenario execution engine."""

    def test_single_cw_scenario(self) -> None:
        """Running a single CW scenario should produce a result with low error."""
        cfg = load_config()
        runner = ScenarioRunner(config=cfg, verbose=False)

        spec = ScenarioSpec(
            name="test_cw",
            modulation=ModulationType.CW,
            snr_db=20.0,
            tx_position=(30_000.0, -20_000.0),
            n_frames=20,
            seed=42,
        )
        results = runner.run_all([spec])

        assert len(results) == 1
        r = results[0]
        assert r.estimate is not None
        assert r.position_error_m is not None
        assert r.position_error_m < 1000.0  # < 1 km

    def test_multiple_scenarios(self) -> None:
        """Running multiple scenarios should return one result per spec."""
        cfg = load_config()
        runner = ScenarioRunner(config=cfg, verbose=False)

        specs = expand_sweep({
            "modulations": ["cw", "bpsk"],
            "snr_db": 20.0,
            "n_frames": 5,
        })
        results = runner.run_all(specs)
        assert len(results) == len(specs)

    def test_result_has_variance_history(self) -> None:
        cfg = load_config()
        runner = ScenarioRunner(config=cfg, verbose=False)

        spec = ScenarioSpec(
            name="test_vh",
            modulation=ModulationType.CW,
            snr_db=20.0,
            tx_position=(30_000.0, -20_000.0),
            n_frames=10,
            seed=42,
        )
        results = runner.run_all([spec])
        assert len(results[0].variance_history) >= 1


# ---------------------------------------------------------------------------
# ScenarioReport
# ---------------------------------------------------------------------------

def _make_mock_results() -> list[ScenarioResult]:
    """Create mock results for report testing."""
    results = []
    for mod, err in [
        (ModulationType.CW, 50.0),
        (ModulationType.CW, 100.0),
        (ModulationType.BPSK, 5000.0),
        (ModulationType.BPSK, None),
    ]:
        spec = ScenarioSpec(
            name=f"{mod.value}_test",
            modulation=mod,
            snr_db=20.0,
            tx_position=(30_000.0, -20_000.0),
        )
        estimate = None
        if err is not None:
            estimate = PositionEstimate(
                x=30_000.0 + err, y=-20_000.0,
                latitude=0.0, longitude=0.0,
                covariance=np.eye(2),
                residual=0.0,
                confidence_radius_95=10.0,
            )
        results.append(ScenarioResult(
            spec=spec,
            estimate=estimate,
            position_error_m=err,
            converged=err is not None and err < 1000,
            n_kalman_updates=5,
            elapsed_seconds=0.5,
            variance_history=[],
        ))
    return results


class TestScenarioReport:
    """Tests for report generation."""

    def test_summary_table_runs(self) -> None:
        """summary_table should print without error."""
        report = ScenarioReport(_make_mock_results())
        import io
        buf = io.StringIO()
        report.summary_table(file=buf)
        output = buf.getvalue()
        assert "Scenario Results" in output
        assert "cw" in output

    def test_stats_by_modulation(self) -> None:
        report = ScenarioReport(_make_mock_results())
        stats = report.stats_by_modulation()
        assert "cw" in stats
        assert "bpsk" in stats
        assert stats["cw"]["mean_error"] == pytest.approx(75.0)
        assert stats["bpsk"]["success_rate"] == 50.0

    def test_stats_by_snr(self) -> None:
        report = ScenarioReport(_make_mock_results())
        stats = report.stats_by_snr()
        assert 20.0 in stats

    def test_csv_export(self, tmp_path: Path) -> None:
        report = ScenarioReport(_make_mock_results())
        csv_path = tmp_path / "results.csv"
        report.to_csv(csv_path)
        assert csv_path.exists()

        with open(csv_path) as f:
            reader = csv.reader(f)
            rows = list(reader)
        assert len(rows) == 5  # header + 4 results
        assert rows[0][0] == "scenario"

    def test_print_stats_runs(self) -> None:
        report = ScenarioReport(_make_mock_results())
        import io
        buf = io.StringIO()
        report.print_stats(file=buf)
        output = buf.getvalue()
        assert "Stats by Modulation" in output
        assert "Stats by SNR" in output
