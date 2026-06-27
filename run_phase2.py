"""Entry point for Phase 2: FHIR Dataset Explorer & Parsing Engine.

Usage:
    python run_phase2.py
    python run_phase2.py --config config/settings.yaml
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

from src.config_loader import load_config, ensure_directories
from src.pipeline import run


def setup_logging(log_file: Path, level: str) -> None:
    """Configure root logger to write to both a log file and stdout."""
    log_file.parent.mkdir(parents=True, exist_ok=True)

    numeric_level = getattr(logging, level.upper(), logging.INFO)
    formatter = logging.Formatter(
        fmt="%(asctime)s | %(levelname)-8s | %(name)s — %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    root = logging.getLogger()
    root.setLevel(numeric_level)

    for handler in [
        logging.FileHandler(log_file, encoding="utf-8"),
        logging.StreamHandler(sys.stdout),
    ]:
        handler.setFormatter(formatter)
        root.addHandler(handler)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Synthetic Health — Phase 2: FHIR Dataset Explorer & Parsing Engine"
    )
    parser.add_argument(
        "--config",
        default="config/settings.yaml",
        help="Path to settings.yaml (default: config/settings.yaml)",
    )
    args = parser.parse_args()

    # Config must be loaded before logging (log path comes from it)
    config = load_config(args.config)
    setup_logging(config.log_file, config.log_level)
    ensure_directories(config)

    logger = logging.getLogger(__name__)
    logger.info("Config:       %s", Path(args.config).resolve())
    logger.info("Raw data dir: %s", config.raw_data_dir)
    logger.info("Output dir:   %s", config.processed_dir.parent.parent)

    run(config)


if __name__ == "__main__":
    main()
