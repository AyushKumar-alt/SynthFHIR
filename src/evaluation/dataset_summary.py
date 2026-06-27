"""Module A — Dataset summary: row counts, columns, missing values, dtypes."""

from __future__ import annotations

import logging

import pandas as pd

from .config import EvaluationConfig
from .loader import TablePair

logger = logging.getLogger(__name__)


def summarize_table(
    table_name: str,
    original:   pd.DataFrame | None,
    synthetic:  pd.DataFrame | None,
) -> dict:
    """Build a summary dict for one (original, synthetic) pair."""
    result: dict = {"table": table_name}

    if original is not None:
        null_cells = int(original.isna().sum().sum())
        result.update(
            original_rows        = len(original),
            original_cols        = len(original.columns),
            original_missing_pct = _null_pct(original),
            original_dtypes      = original.dtypes.astype(str).to_dict(),
        )
    else:
        result.update(
            original_rows        = None,
            original_cols        = None,
            original_missing_pct = None,
            original_dtypes      = {},
        )

    if synthetic is not None:
        result.update(
            synthetic_rows        = len(synthetic),
            synthetic_cols        = len(synthetic.columns),
            synthetic_missing_pct = _null_pct(synthetic),
            synthetic_dtypes      = synthetic.dtypes.astype(str).to_dict(),
        )
    else:
        result.update(
            synthetic_rows        = None,
            synthetic_cols        = None,
            synthetic_missing_pct = None,
            synthetic_dtypes      = {},
        )

    if original is not None and synthetic is not None:
        orig_cols  = set(original.columns)
        synth_cols = set(synthetic.columns)
        shared     = sorted(orig_cols & synth_cols)
        result["shared_columns"]               = shared
        result["columns_only_in_original"]     = sorted(orig_cols  - synth_cols)
        result["columns_only_in_synthetic"]    = sorted(synth_cols - orig_cols)
        result["column_count_match"]           = len(original.columns) == len(synthetic.columns)
        result["dtype_mismatches"]             = {
            col: {
                "original":  str(original[col].dtype),
                "synthetic": str(synthetic[col].dtype),
            }
            for col in shared
            if str(original[col].dtype) != str(synthetic[col].dtype)
        }
    else:
        result["shared_columns"]            = []
        result["columns_only_in_original"]  = []
        result["columns_only_in_synthetic"] = []
        result["column_count_match"]        = None
        result["dtype_mismatches"]          = {}

    return result


def run_dataset_summary(
    pairs: dict[str, TablePair],
) -> list[dict]:
    """Run dataset summary for all tables.

    Returns a list of per-table summary dicts, one per table.
    """
    summaries = []
    for table_name, (original, synthetic) in pairs.items():
        s = summarize_table(table_name, original, synthetic)
        _log_summary(s)
        summaries.append(s)
    return summaries


# ── Helpers ───────────────────────────────────────────────────────────────────

def _null_pct(df: pd.DataFrame) -> float:
    total = df.size
    if total == 0:
        return 0.0
    return round(100.0 * df.isna().sum().sum() / total, 2)


def _log_summary(s: dict) -> None:
    table = s["table"]
    orig  = s.get("original_rows")
    synth = s.get("synthetic_rows")
    miss  = s.get("synthetic_missing_pct")
    n_mis = len(s.get("dtype_mismatches", {}))

    logger.info(
        "[SUMMARY] %-14s  original=%-7s  synthetic=%-7s  synth_missing=%-6s  dtype_mismatches=%d",
        table,
        f"{orig:,}" if orig is not None else "N/A",
        f"{synth:,}" if synth is not None else "N/A",
        f"{miss}%" if miss is not None else "N/A",
        n_mis,
    )
    if n_mis:
        for col, info in s["dtype_mismatches"].items():
            logger.warning(
                "  [DTYPE MISMATCH] %s.%s: original=%s  synthetic=%s",
                table, col, info["original"], info["synthetic"],
            )
