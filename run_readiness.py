"""Entry point for the Dataset Readiness Report + SDV Pre-flight.

Runs three stages in sequence:
  1. Readiness checks    — row counts, missing values, FK integrity, chronology
  2. Distribution profile — descriptive stats for all numerical features
  3. SDV compatibility   — cardinality, finite values, metadata load

Outputs
-------
outputs/reports/readiness_report.md
outputs/reports/data_profile_before_training.md

Usage:
    python run_readiness.py
    python run_readiness.py --config config/settings.yaml
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

import pandas as pd

from src.config_loader import load_config, ensure_directories
from src.readiness import build_report, print_report, save_report
from src.distribution_profiler import (
    build_distribution_profile,
    print_distribution_summary,
    save_distribution_report,
)
from src.sdv_compat import (
    run_compat_checks,
    print_compat_summary,
    save_compat_report,
)


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


def _load_ready_tables(ready_dir: Path) -> dict[str, pd.DataFrame]:
    names = ["patients", "encounters", "observations", "conditions", "medications"]
    tables = {}
    for name in names:
        path = ready_dir / f"{name}_ready.csv"
        if not path.exists():
            raise FileNotFoundError(
                f"Ready table missing: {path}\n"
                "Run Phase 3 first: python run_phase3.py"
            )
        tables[name] = pd.read_csv(path, low_memory=False)
    return tables


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Synthetic Health — Dataset Readiness + SDV Pre-flight"
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

    # ── Stage 1: Data readiness ───────────────────────────────────────────
    logger.info("=== Stage 1 / 3: Data Readiness ===")
    readiness = build_report(config)
    print_report(readiness)
    report_path = config.reports_dir / "readiness_report.md"
    save_report(readiness, config)

    # ── Stage 2: Distribution profile ────────────────────────────────────
    logger.info("=== Stage 2 / 3: Distribution Profile ===")
    tables = _load_ready_tables(config.ready_dir)
    profile = build_distribution_profile(tables, label="pre-synthesis")
    print_distribution_summary(profile)
    dist_path = config.reports_dir / "data_profile_before_training.md"
    save_distribution_report(profile, dist_path)

    # ── Stage 3: SDV compatibility ────────────────────────────────────────
    logger.info("=== Stage 3 / 3: SDV Compatibility ===")
    meta_path = config.ready_dir / "metadata.json"
    with open(meta_path, encoding="utf-8") as fh:
        metadata = json.load(fh)

    compat_results = run_compat_checks(tables, metadata, meta_path)
    print_compat_summary(compat_results)
    save_compat_report(compat_results, report_path)

    # ── Final verdict ─────────────────────────────────────────────────────
    compat_fail = any(r.status == "FAIL" for r in compat_results)
    if not readiness.ready_for_sdv or compat_fail:
        logger.error("Pre-flight FAILED — fix issues above before Phase 4.")
        sys.exit(1)

    logger.info("Pre-flight COMPLETE — pipeline is ready for Phase 4 (SDV training).")


if __name__ == "__main__":
    main()
