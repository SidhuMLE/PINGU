"""Scenario specification and parameter sweep expansion.

Defines the :class:`ScenarioSpec` dataclass for fully-resolved scenario
parameters, and utilities for expanding sweep definitions and loading
YAML scenario manifests.
"""

from __future__ import annotations

import itertools
from dataclasses import dataclass, field
from pathlib import Path

from omegaconf import OmegaConf

from pingu.types import ModulationType


@dataclass
class ScenarioSpec:
    """Fully-resolved specification for a single scenario run.

    Parameters
    ----------
    name : str
        Human-readable identifier (e.g. ``"CW_SNR20"``).
    modulation : ModulationType
        Modulation type for the transmitted signal.
    snr_db : float
        Per-receiver SNR in dB.
    tx_position : tuple[float, float]
        Transmitter position ``(x, y)`` in metres (Cartesian).
    sample_rate : float
        Common sample rate in Hz.
    center_freq : float
        Nominal centre frequency in Hz.
    duration : float
        Signal duration in seconds.
    n_frames : int
        Number of independent frames to generate and process.
    seed : int
        Random seed for reproducibility.
    config_overrides : dict | None
        Optional OmegaConf-style config overrides for this scenario.
    """

    name: str
    modulation: ModulationType
    snr_db: float
    tx_position: tuple[float, float]
    sample_rate: float = 48_000.0
    center_freq: float = 14.1e6
    duration: float = 0.1
    n_frames: int = 10
    seed: int = 42
    config_overrides: dict | None = None


def _parse_modulation(name: str) -> ModulationType:
    """Parse a modulation name string into a ModulationType enum."""
    return ModulationType(name.lower())


def _auto_name(mod: ModulationType, snr: float, pos: tuple[float, float]) -> str:
    """Generate a descriptive scenario name from parameters."""
    pos_str = f"{int(pos[0])}_{int(pos[1])}"
    snr_str = f"SNR{int(snr)}" if snr == int(snr) else f"SNR{snr}"
    return f"{mod.value.upper()}_{snr_str}_pos{pos_str}"


def expand_sweep(sweep_def: dict) -> list[ScenarioSpec]:
    """Expand a sweep definition into individual ScenarioSpec instances.

    The sweep definition supports:

    - ``modulations``: list of modulation name strings, or ``"all"``
    - ``snr_range``: ``[start, stop, step]`` or explicit list of values
    - ``positions``: list of ``[x, y]`` pairs
    - ``n_frames``, ``seed``, ``sample_rate``, ``duration``, ``center_freq``
    - ``config_overrides``: dict applied to all generated specs

    Returns the Cartesian product of all parameter axes.
    """
    # Parse modulations.
    mod_raw = sweep_def.get("modulations", ["cw"])
    if mod_raw == "all" or mod_raw == ["all"]:
        modulations = [m for m in ModulationType if m != ModulationType.NOISE]
    else:
        modulations = [_parse_modulation(m) for m in mod_raw]

    # Parse SNR values.
    snr_raw = sweep_def.get("snr_range", None)
    snr_single = sweep_def.get("snr_db", None)
    if snr_raw is not None:
        if len(snr_raw) == 3:
            start, stop, step = snr_raw
            snr_values = []
            v = start
            while v <= stop + 1e-9:
                snr_values.append(float(v))
                v += step
        else:
            snr_values = [float(s) for s in snr_raw]
    elif snr_single is not None:
        snr_values = [float(snr_single)]
    else:
        snr_values = [20.0]

    # Parse positions.
    pos_raw = sweep_def.get("positions", [[30_000.0, -20_000.0]])
    positions = [tuple(p) for p in pos_raw]

    # Shared parameters.
    n_frames = int(sweep_def.get("n_frames", 10))
    base_seed = int(sweep_def.get("seed", 42))
    sample_rate = float(sweep_def.get("sample_rate", 48_000.0))
    center_freq = float(sweep_def.get("center_freq", 14.1e6))
    duration = float(sweep_def.get("duration", 0.1))
    config_overrides = sweep_def.get("config_overrides", None)

    specs: list[ScenarioSpec] = []
    for i, (mod, snr, pos) in enumerate(
        itertools.product(modulations, snr_values, positions)
    ):
        specs.append(
            ScenarioSpec(
                name=_auto_name(mod, snr, pos),
                modulation=mod,
                snr_db=snr,
                tx_position=pos,
                sample_rate=sample_rate,
                center_freq=center_freq,
                duration=duration,
                n_frames=n_frames,
                seed=base_seed + i,
                config_overrides=config_overrides,
            )
        )
    return specs


def load_scenario_manifest(path: str | Path) -> list[ScenarioSpec]:
    """Load a YAML scenario manifest file.

    Supports two formats:

    1. **Sweep format**: a top-level ``sweep`` key with parameter ranges,
       expanded via :func:`expand_sweep`.
    2. **Explicit format**: a top-level ``scenarios`` key with a list of
       individual scenario definitions.
    """
    raw = OmegaConf.load(path)
    data = OmegaConf.to_container(raw, resolve=True)

    if "sweep" in data:
        return expand_sweep(data["sweep"])

    if "scenarios" in data:
        specs = []
        for i, s in enumerate(data["scenarios"]):
            mod = _parse_modulation(s["modulation"])
            pos = tuple(s.get("tx_position", [30_000.0, -20_000.0]))
            name = s.get("name", _auto_name(mod, s["snr_db"], pos))
            specs.append(
                ScenarioSpec(
                    name=name,
                    modulation=mod,
                    snr_db=float(s["snr_db"]),
                    tx_position=pos,
                    sample_rate=float(s.get("sample_rate", 48_000.0)),
                    center_freq=float(s.get("center_freq", 14.1e6)),
                    duration=float(s.get("duration", 0.1)),
                    n_frames=int(s.get("n_frames", 10)),
                    seed=int(s.get("seed", 42 + i)),
                    config_overrides=s.get("config_overrides", None),
                )
            )
        return specs

    raise ValueError(
        f"Manifest must contain a 'sweep' or 'scenarios' key, got: {list(data.keys())}"
    )


def specs_from_cli_args(args) -> list[ScenarioSpec]:
    """Build a list of ScenarioSpec from parsed CLI arguments.

    Constructs a sweep definition dict from the CLI namespace and delegates
    to :func:`expand_sweep`.
    """
    # Parse modulations.
    if args.modulations == ["all"] or args.modulations == "all":
        mod_list = "all"
    else:
        mod_list = args.modulations

    # Parse SNR.
    sweep_def: dict = {"modulations": mod_list}

    if args.snr is not None:
        sweep_def["snr_db"] = args.snr
    elif args.snr_start is not None and args.snr_stop is not None:
        sweep_def["snr_range"] = [args.snr_start, args.snr_stop, args.snr_step]
    else:
        sweep_def["snr_db"] = 20.0

    # Parse positions.
    if args.positions:
        positions = []
        for p in args.positions:
            parts = p.split(",")
            positions.append([float(parts[0]), float(parts[1])])
        sweep_def["positions"] = positions
    elif args.position:
        sweep_def["positions"] = [args.position]

    sweep_def["n_frames"] = args.n_frames
    sweep_def["seed"] = args.seed

    return expand_sweep(sweep_def)
