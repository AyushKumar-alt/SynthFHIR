"""Configuration loader — reads settings.yaml into a typed Config dataclass."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

import yaml

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class Config:
    """Immutable runtime configuration."""

    raw_data_dir: Path
    processed_dir: Path
    ready_dir: Path
    reports_dir: Path
    figures_dir: Path
    logs_dir: Path
    log_file: Path
    log_level: str
    file_extension: str
    top_n: int
    top_n_loinc: int
    min_loinc_frequency: int


def load_config(config_path: str | Path) -> Config:
    """Load and parse settings.yaml into a Config object.

    All relative paths in the YAML are resolved against the project root
    (the parent directory of the config/ folder).

    Args:
        config_path: Path to settings.yaml.

    Returns:
        Populated, immutable Config instance.

    Raises:
        FileNotFoundError: If the config file does not exist.
    """
    config_path = Path(config_path).resolve()
    if not config_path.exists():
        raise FileNotFoundError(f"Config not found: {config_path}")

    with open(config_path, encoding="utf-8") as fh:
        raw = yaml.safe_load(fh)

    project_root = config_path.parent.parent

    def resolve(p: str) -> Path:
        path = Path(p)
        return path if path.is_absolute() else (project_root / path).resolve()

    paths = raw["paths"]
    prep = raw.get("preprocessing", {})

    return Config(
        raw_data_dir=resolve(paths["raw_data_dir"]),
        processed_dir=resolve(paths["processed_dir"]),
        ready_dir=resolve(paths.get("ready_dir", "data/ready")),
        reports_dir=resolve(paths["reports_dir"]),
        figures_dir=resolve(paths["figures_dir"]),
        logs_dir=resolve(paths["logs_dir"]),
        log_file=resolve(paths["log_file"]),
        log_level=raw["logging"]["level"],
        file_extension=raw["extraction"]["file_extension"],
        top_n=int(raw["profiling"]["top_n"]),
        top_n_loinc=int(prep.get("top_n_loinc", 30)),
        min_loinc_frequency=int(prep.get("min_loinc_frequency", 100)),
    )


def ensure_directories(config: Config) -> None:
    """Create all output directories if they do not already exist."""
    for directory in (
        config.processed_dir,
        config.ready_dir,
        config.reports_dir,
        config.figures_dir,
        config.logs_dir,
    ):
        directory.mkdir(parents=True, exist_ok=True)
