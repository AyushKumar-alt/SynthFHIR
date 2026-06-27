"""Entry point for Phase 3: Data Preparation & Feature Engineering.

Usage:
    python run_phase3.py
    python run_phase3.py --config config/settings.yaml
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

from src.config_loader import load_config, ensure_directories
from src.preprocessor import run


def setup_logging(log_file: Path, level: str) -> None:
    """Configure root logger to write to both log file and stdout."""
    log_file.parent.mkdir(parents=True, exist_ok=True)
    numeric_level = getattr(logging, level.upper(), logging.INFO)
    formatter = logging.Formatter(
        fmt="%(asctime)s | %(levelname)-8s | %(name)s — %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    root = logging.getLogger()
    root.setLevel(numeric_level)
    for handler in [
        logging.FileHandler(log_file, encoding="utf-8", mode="a"),
        logging.StreamHandler(sys.stdout),
    ]:
        handler.setFormatter(formatter)
        root.addHandler(handler)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Synthetic Health — Phase 3: Data Preparation & Feature Engineering"
    )
    parser.add_argument(
        "--config",
        default="config/settings.yaml",
        help="Path to settings.yaml (default: config/settings.yaml)",
    )
    args = parser.parse_args()

    config = load_config(args.config)
    setup_logging(config.log_file, config.log_level)
    ensure_directories(config)

    logger = logging.getLogger(__name__)
    logger.info("Config:       %s", Path(args.config).resolve())
    logger.info("Processed dir: %s", config.processed_dir)
    logger.info("Ready dir:     %s", config.ready_dir)

    run(config)


if __name__ == "__main__":
    main()
