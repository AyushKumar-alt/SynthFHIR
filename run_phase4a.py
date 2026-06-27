"""Phase 4A — Synthesis Engine Setup & Smoke Test.

Runs a fast 5-epoch CTGAN training pass on the patients table, generates
20 synthetic rows, and validates them against 8 sanity checks.

If all checks PASS → safe to proceed to Phase 4B (full training).
If any check FAILS → fix the reported issue before proceeding.

Usage:
    python run_phase4a.py
    python run_phase4a.py --config config/settings.yaml
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path


def _require_sdv() -> None:
    """Hard-fail with an installation hint if SDV is not installed."""
    try:
        import sdv  # noqa: F401
    except ImportError:
        print(
            "\n[ERROR] SDV is not installed.\n"
            "Install it with:  pip install sdv\n"
        )
        sys.exit(1)


def setup_logging(log_file: Path, level: str) -> None:
    log_file.parent.mkdir(parents=True, exist_ok=True)
    numeric_level = getattr(logging, level.upper(), logging.INFO)
    formatter = logging.Formatter(
        fmt="%(asctime)s | %(levelname)-8s | %(message)s",
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
    _require_sdv()

    parser = argparse.ArgumentParser(
        description="Synthetic Health — Phase 4A: Smoke Test"
    )
    parser.add_argument(
        "--config",
        default="config/settings.yaml",
        help="Path to settings.yaml (default: config/settings.yaml)",
    )
    args = parser.parse_args()

    # Import after SDV check so error messages are clean
    from src.config_loader import load_config, ensure_directories
    from src.synthesis.config import load_synthesis_config
    from src.synthesis.pipeline import SynthesisPipeline

    config       = load_config(args.config)
    synth_config = load_synthesis_config(args.config)
    setup_logging(config.log_file, config.log_level)
    ensure_directories(config)

    # Ensure synthesis output dirs exist
    synth_config.synthetic_dir.mkdir(parents=True, exist_ok=True)
    synth_config.model_dir.mkdir(parents=True, exist_ok=True)

    logger = logging.getLogger(__name__)
    logger.info("Config:          %s", Path(args.config).resolve())
    logger.info("Patient model:   %s", synth_config.patient_model)
    logger.info("Smoke epochs:    %d", synth_config.smoke_test_epochs)
    logger.info("Smoke rows:      %d", synth_config.smoke_test_n_rows)
    logger.info("Ready dir:       %s", config.ready_dir)

    pipeline = SynthesisPipeline(
        synth_config=synth_config,
        ready_dir=config.ready_dir,
        reports_dir=config.reports_dir,
    )

    passed = pipeline.smoke_test()

    if not passed:
        logger.error(
            "Smoke test FAILED. Fix the issues above before running Phase 4B."
        )
        sys.exit(1)

    logger.info(
        "Smoke test PASSED. Ready for Phase 4B: python run_phase4b.py"
    )


if __name__ == "__main__":
    main()
