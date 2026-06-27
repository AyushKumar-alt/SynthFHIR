"""Dataset Explorer — scans all FHIR JSON files, validates, and counts resources."""

from __future__ import annotations

import json
import logging
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd
from tqdm import tqdm

from .config_loader import Config

logger = logging.getLogger(__name__)


@dataclass
class ExplorerResult:
    """Summary of the dataset scan."""

    total_files: int = 0
    valid_files: int = 0
    malformed_files: int = 0
    malformed_file_list: list[str] = field(default_factory=list)
    resource_counts: Counter = field(default_factory=Counter)
    bundles_per_resource: Counter = field(default_factory=Counter)


def scan_dataset(config: Config) -> ExplorerResult:
    """Iterate every FHIR JSON file, validate it, and count resource types.

    Malformed files are skipped safely and logged — they do not crash the scan.

    Args:
        config: Runtime configuration.

    Returns:
        ExplorerResult with totals and per-resource counts.
    """
    raw_dir = config.raw_data_dir
    files = sorted(raw_dir.glob(f"*{config.file_extension}"))

    if not files:
        raise FileNotFoundError(
            f"No {config.file_extension} files found in: {raw_dir}"
        )

    result = ExplorerResult(total_files=len(files))
    logger.info("Dataset scan started — %d files found in %s", len(files), raw_dir)

    for fpath in tqdm(files, desc="Scanning bundles", unit="file"):
        try:
            with open(fpath, encoding="utf-8", errors="replace") as fh:
                bundle = json.load(fh)

            types_in_bundle: set[str] = set()
            for entry in bundle.get("entry", []):
                rt = entry.get("resource", {}).get("resourceType")
                if rt:
                    result.resource_counts[rt] += 1
                    types_in_bundle.add(rt)

            for rt in types_in_bundle:
                result.bundles_per_resource[rt] += 1

            result.valid_files += 1

        except (json.JSONDecodeError, UnicodeDecodeError, KeyError) as exc:
            result.malformed_files += 1
            result.malformed_file_list.append(fpath.name)
            logger.warning("Malformed file skipped: %s — %s", fpath.name, exc)

    logger.info(
        "Scan complete — valid: %d | malformed: %d | resource types: %d",
        result.valid_files,
        result.malformed_files,
        len(result.resource_counts),
    )
    return result


def save_resource_summary(result: ExplorerResult, config: Config) -> None:
    """Write resource_summary.csv and resource_summary.json to reports dir.

    Args:
        result: Output of scan_dataset().
        config: Runtime configuration.
    """
    records = [
        {
            "resource_type": rt,
            "total_records": count,
            "bundles_present": result.bundles_per_resource.get(rt, 0),
        }
        for rt, count in result.resource_counts.most_common()
    ]

    df = pd.DataFrame(records)
    csv_path = config.reports_dir / "resource_summary.csv"
    df.to_csv(csv_path, index=False)
    logger.info("Saved: %s", csv_path)

    summary = {
        "scan_summary": {
            "total_files": result.total_files,
            "valid_files": result.valid_files,
            "malformed_files": result.malformed_files,
            "malformed_file_list": result.malformed_file_list,
        },
        "resource_counts": {
            rt: {"total_records": cnt, "bundles_present": result.bundles_per_resource.get(rt, 0)}
            for rt, cnt in result.resource_counts.most_common()
        },
    }

    json_path = config.reports_dir / "resource_summary.json"
    with open(json_path, "w", encoding="utf-8") as fh:
        json.dump(summary, fh, indent=2)
    logger.info("Saved: %s", json_path)

    # Human-readable log summary
    logger.info("=" * 50)
    logger.info("  FILES:  total=%d  valid=%d  malformed=%d",
                result.total_files, result.valid_files, result.malformed_files)
    logger.info("  RESOURCE TYPES FOUND: %d", len(result.resource_counts))
    for rt, cnt in result.resource_counts.most_common(10):
        logger.info("    %-35s %8d", rt, cnt)
    if len(result.resource_counts) > 10:
        logger.info("    ... and %d more types", len(result.resource_counts) - 10)
    logger.info("=" * 50)
