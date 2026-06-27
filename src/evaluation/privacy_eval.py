"""Module E — Privacy checks: duplicate rows, exact overlap, uniqueness."""

from __future__ import annotations

import logging

import pandas as pd

from .loader import TablePair

logger = logging.getLogger(__name__)

# Columns that are always unique by construction (UUIDs, sequence counters).
# Excluding them lets us detect exact-match content rows without being
# confused by intentionally distinct identifiers.
_ID_COLUMNS: frozenset[str] = frozenset({
    "patient_id", "encounter_id", "observation_id",
    "condition_id", "medication_id", "sequence_index",
})


def run_privacy_evaluation(
    pairs:  dict[str, TablePair],
    pk_map: dict[str, str],
) -> dict[str, dict]:
    """Run privacy checks for every table."""
    all_results: dict[str, dict] = {}
    for table_name, (original, synthetic) in pairs.items():
        if original is None or synthetic is None:
            logger.warning("[PRIVACY] %s — missing data, skipped", table_name)
            all_results[table_name] = {"table": table_name, "status": "missing_data"}
            continue
        pk = pk_map.get(table_name)
        result = _evaluate_table(original, synthetic, table_name, pk)
        _log_result(result)
        all_results[table_name] = result
    return all_results


# ── Per-table logic ───────────────────────────────────────────────────────────

def _evaluate_table(
    original:   pd.DataFrame,
    synthetic:  pd.DataFrame,
    table_name: str,
    pk_col:     str | None,
) -> dict:
    n_synth = len(synthetic)
    result  = {"table": table_name, "status": "ok", "n_synthetic_rows": n_synth}

    # ── 1. Duplicates within synthetic ───────────────────────────────────
    n_dup = int(synthetic.duplicated().sum())
    result["synthetic_duplicate_rows"] = n_dup
    result["synthetic_duplicate_pct"]  = _pct(n_dup, n_synth)
    result["synthetic_unique_rows"]    = n_synth - n_dup
    result["synthetic_unique_pct"]     = _pct(n_synth - n_dup, n_synth)

    # ── 2. Exact row overlap with original (excluding ID columns) ─────────
    drop_cols = _build_drop_cols(original, synthetic, pk_col)
    shared    = [c for c in original.columns if c in synthetic.columns and c not in drop_cols]

    if shared:
        orig_tuples  = set(_to_tuples(original[shared]))
        synth_tuples = set(_to_tuples(synthetic[shared]))
        n_overlap    = len(orig_tuples & synth_tuples)
        result["exact_row_overlap"]     = n_overlap
        result["exact_overlap_pct"]     = _pct(n_overlap, len(synth_tuples))
        result["columns_compared"]      = shared
    else:
        result["exact_row_overlap"]     = None
        result["exact_overlap_pct"]     = None
        result["columns_compared"]      = []

    # ── 3. Privacy score ──────────────────────────────────────────────────
    # High uniqueness + low exact overlap → good privacy.
    uniqueness     = result["synthetic_unique_pct"] / 100.0
    overlap_frac   = (result.get("exact_overlap_pct") or 0.0) / 100.0
    result["privacy_score"] = round(uniqueness * (1.0 - overlap_frac), 4)

    return result


# ── Helpers ───────────────────────────────────────────────────────────────────

def _build_drop_cols(
    original:  pd.DataFrame,
    synthetic: pd.DataFrame,
    pk_col:    str | None,
) -> set[str]:
    """Build set of columns to exclude from the exact-overlap comparison."""
    drop = set(_ID_COLUMNS)
    if pk_col:
        drop.add(pk_col)
    return drop


def _to_tuples(df: pd.DataFrame) -> list[tuple]:
    return list(map(tuple, df.fillna("__NULL__").astype(str).values))


def _pct(num: int, denom: int) -> float:
    return round(100.0 * num / denom, 2) if denom > 0 else 0.0


def _log_result(result: dict) -> None:
    table = result["table"]
    if result.get("status") != "ok":
        logger.info("[PRIVACY] %s — %s", table, result["status"])
        return
    logger.info(
        "[PRIVACY] %-14s  unique=%.1f%%  duplicates=%d  "
        "exact_overlap=%s  privacy_score=%.4f",
        table,
        result["synthetic_unique_pct"],
        result["synthetic_duplicate_rows"],
        f"{result.get('exact_overlap_pct', 'N/A')}%" if result.get("exact_overlap_pct") is not None else "N/A",
        result["privacy_score"],
    )
