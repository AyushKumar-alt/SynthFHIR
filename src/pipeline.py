"""Pipeline orchestrator for Phase 2 — FHIR Dataset Explorer & Parsing Engine.

Execution order:
  Step 1 — Explorer:    scan all files, count resources, write resource_summary.*
  Step 2 — Extraction: single-pass through all bundles, write 5 CSVs with checkpoints
  Step 3 — Validation: referential integrity + temporal checks, write reports
  Step 4 — Profiling:  descriptive statistics, write dataset_profile.*
  Step 5 — Visualise:  generate 10 charts to outputs/figures/
"""

from __future__ import annotations

import json
import logging
import time
from pathlib import Path

import pandas as pd
from tqdm import tqdm

from .config_loader import Config
from .explorer import ExplorerResult, scan_dataset, save_resource_summary
from .parsers import RESOURCE_EXTRACTORS, RESOURCE_CSV_NAMES
from . import validator, profiler, visualizer

logger = logging.getLogger(__name__)


# ── Step 2: single-pass extraction ───────────────────────────────────────────

def _extract_resources(config: Config) -> None:
    """Read every FHIR bundle once and extract all 5 resource types simultaneously.

    Writing each CSV immediately after the full pass provides checkpoints:
    if any later step fails, the extracted CSVs are already on disk.
    """
    raw_dir = config.raw_data_dir
    files = sorted(raw_dir.glob(f"*{config.file_extension}"))

    # Accumulate rows in plain Python lists — far faster than repeated
    # DataFrame concatenation for 500K+ rows.
    rows: dict[str, list[dict]] = {rt: [] for rt in RESOURCE_EXTRACTORS}
    parse_errors = 0

    logger.info("Extracting resources from %d files (single pass)...", len(files))

    for fpath in tqdm(files, desc="Parsing bundles", unit="file"):
        try:
            with open(fpath, encoding="utf-8", errors="replace") as fh:
                bundle = json.load(fh)
        except json.JSONDecodeError as exc:
            parse_errors += 1
            logger.warning("Skipping malformed file: %s — %s", fpath.name, exc)
            continue

        for entry in bundle.get("entry", []):
            resource = entry.get("resource", {})
            rt = resource.get("resourceType", "")
            extractor = RESOURCE_EXTRACTORS.get(rt)
            if extractor is not None:
                row = extractor(resource)
                if row is not None:
                    rows[rt].append(row)

    if parse_errors:
        logger.warning("%d malformed files were skipped during extraction", parse_errors)

    # Write one CSV per resource type immediately (checkpoint behaviour)
    logger.info("Writing extracted CSVs...")
    for rt, csv_name in RESOURCE_CSV_NAMES.items():
        df = pd.DataFrame(rows[rt])
        out_path = config.processed_dir / csv_name
        df.to_csv(out_path, index=False)
        logger.info("  ✓ %-22s saved — %8d rows, %2d columns",
                    csv_name, len(df), len(df.columns))

    logger.info("Extraction complete.")


# ── Public entry point ────────────────────────────────────────────────────────

def run(config: Config) -> None:
    """Execute the full Phase 2 pipeline end-to-end."""
    wall_start = time.time()

    sep = "=" * 56
    logger.info(sep)
    logger.info("  SYNTHETIC HEALTH — Phase 2: FHIR Parsing Engine")
    logger.info(sep)

    # ── Step 1 ────────────────────────────────────────────────────────────
    t0 = time.time()
    logger.info("[Step 1/5] Dataset Explorer")
    result: ExplorerResult = scan_dataset(config)
    save_resource_summary(result, config)
    logger.info("  Done in %.1f s", time.time() - t0)

    # ── Step 2 ────────────────────────────────────────────────────────────
    t0 = time.time()
    logger.info("[Step 2/5] Resource Extraction")
    _extract_resources(config)
    logger.info("  Done in %.1f s", time.time() - t0)

    # ── Step 3 ────────────────────────────────────────────────────────────
    t0 = time.time()
    logger.info("[Step 3/5] Validation")
    validator.run(config)
    logger.info("  Done in %.1f s", time.time() - t0)

    # ── Step 4 ────────────────────────────────────────────────────────────
    t0 = time.time()
    logger.info("[Step 4/5] Profiling")
    profiler.run(config)
    logger.info("  Done in %.1f s", time.time() - t0)

    # ── Step 5 ────────────────────────────────────────────────────────────
    t0 = time.time()
    logger.info("[Step 5/5] Visualizations")
    visualizer.run(config)
    logger.info("  Done in %.1f s", time.time() - t0)

    total = time.time() - wall_start
    logger.info(sep)
    logger.info("  Phase 2 complete in %.1f seconds (%.1f min)", total, total / 60)
    logger.info(sep)
