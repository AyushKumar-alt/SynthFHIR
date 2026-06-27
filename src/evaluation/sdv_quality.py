"""Module F — SDV quality report wrapper.

Calls sdv.evaluation.single_table.evaluate_quality() when available.
Skips gracefully (with INFO log) when SDV is not installed or the API
has changed — never raises, never blocks the rest of Phase 5.

SDV quality API (SDV >= 1.9.0)
--------------------------------
from sdv.evaluation.single_table import evaluate_quality
report = evaluate_quality(real_data, synthetic_data, metadata, verbose=False)
score  = report.get_score()          # float 0–1
props  = report.get_properties()     # DataFrame: Property | Score
"""

from __future__ import annotations

import json
import logging
import warnings
from pathlib import Path

from .loader import TablePair

logger = logging.getLogger(__name__)


def run_sdv_quality(
    pairs:         dict[str, TablePair],
    metadata_path: Path,
) -> dict[str, dict]:
    """Try evaluate_quality() for each table. Returns graceful results dict."""
    try:
        from sdv.evaluation.single_table import evaluate_quality
    except ImportError:
        logger.info(
            "[SDV QUALITY] sdv.evaluation.single_table not available — skipping all tables."
        )
        return {t: {"status": "sdv_unavailable"} for t in pairs}

    raw_meta = _load_metadata(metadata_path)
    results: dict[str, dict] = {}

    for table_name, (original, synthetic) in pairs.items():
        if original is None or synthetic is None:
            results[table_name] = {"status": "missing_data"}
            continue
        results[table_name] = _evaluate_table(
            table_name, original, synthetic, raw_meta, evaluate_quality
        )

    return results


# ── Per-table ─────────────────────────────────────────────────────────────────

def _evaluate_table(
    table_name:      str,
    original,
    synthetic,
    raw_meta:        dict,
    evaluate_quality,
) -> dict:
    try:
        from sdv.metadata import SingleTableMetadata

        table_dict = raw_meta.get("tables", {}).get(table_name, {})
        if not table_dict:
            logger.warning("[SDV QUALITY] %s — table not found in metadata.json", table_name)
            return {"status": "metadata_missing"}

        meta_dict = {**table_dict, "METADATA_SPEC_VERSION": "V1"}
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            meta = SingleTableMetadata.load_from_dict(meta_dict)

        report = evaluate_quality(original, synthetic, meta, verbose=False)
        score  = float(report.get_score())
        props  = report.get_properties()

        props_records: list[dict] = (
            props.round(4).to_dict("records")
            if hasattr(props, "to_dict")
            else [{"raw": str(props)}]
        )

        logger.info(
            "[SDV QUALITY] %-14s  score=%.4f  properties=%s",
            table_name,
            score,
            {r.get("Property", "?"): round(r.get("Score", 0), 3) for r in props_records},
        )

        return {
            "status":        "ok",
            "overall_score": round(score, 4),
            "properties":    props_records,
        }

    except Exception as exc:
        logger.warning("[SDV QUALITY] %s — evaluation failed: %s", table_name, exc)
        return {"status": "error", "error": str(exc)}


# ── Metadata loader ───────────────────────────────────────────────────────────

def _load_metadata(path: Path) -> dict:
    if not path.exists():
        logger.warning("[SDV QUALITY] metadata.json not found at %s", path)
        return {}
    try:
        with open(path, encoding="utf-8") as fh:
            return json.load(fh)
    except Exception as exc:
        logger.warning("[SDV QUALITY] Failed to read metadata.json: %s", exc)
        return {}
