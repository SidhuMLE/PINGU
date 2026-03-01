"""YAML configuration loader using OmegaConf."""

from __future__ import annotations

from pathlib import Path

from omegaconf import DictConfig, OmegaConf


_DEFAULT_CONFIG_PATH = Path(__file__).resolve().parents[2] / "configs" / "default.yaml"


def load_config(
    path: str | Path | None = None,
    overrides: dict | None = None,
) -> DictConfig:
    """Load configuration from YAML file with optional overrides.

    Args:
        path: Path to YAML config file. Defaults to configs/default.yaml.
        overrides: Dict of dot-notation overrides (e.g. {"tdoa.method": "scot"}).

    Returns:
        Merged OmegaConf DictConfig.
    """
    path = Path(path) if path is not None else _DEFAULT_CONFIG_PATH

    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")

    base = OmegaConf.load(path)

    if overrides:
        override_cfg = OmegaConf.create(overrides)
        base = OmegaConf.merge(base, override_cfg)

    return base


def load_and_merge(base_path: str | Path, *overlay_paths: str | Path) -> DictConfig:
    """Load a base config and merge one or more overlay configs on top."""
    cfg = OmegaConf.load(base_path)
    for overlay in overlay_paths:
        overlay_cfg = OmegaConf.load(overlay)
        cfg = OmegaConf.merge(cfg, overlay_cfg)
    return cfg
