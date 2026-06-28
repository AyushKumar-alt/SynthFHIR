"""Module H — k-Anonymity evaluation of synthetic data.

For each table, automatically identifies quasi-identifier (QI) columns
by keyword matching and computes k-anonymity statistics over the
synthetic dataset.

k-anonymity definition
----------------------
A dataset satisfies k-anonymity if every combination of QI values
appears in at least k records.  Low k values indicate higher risk of
re-identification.

Quasi-identifiers detected (column-name keyword matching)
---------------------------------------------------------
age, gender, sex, race, ethnicity, city, state, county, zip,
postal, birth, dob, nationality, marital, occupation
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import pandas as pd

from .loader import TablePair

logger = logging.getLogger(__name__)

_QI_KEYWORDS: tuple[str, ...] = (
    "age", "gender", "sex", "race", "ethnicity",
    "city", "state", "county", "zip", "postal",
    "birth", "dob", "nationality", "marital", "occupation",
)

# Columns with more unique values than this are treated as IDs/free-text.
_MAX_QI_CARDINALITY = 200

# Need at least this many synthetic rows to compute meaningful k-anon.
_MIN_ROWS = 20


def run_k_anonymity(
    pairs:      dict[str, TablePair],
    output_dir: Path,
) -> dict[str, dict]:
    """Compute k-anonymity for the synthetic dataset in every table.

    Parameters
    ----------
    pairs      : table-name → (original_df, synthetic_df)
    output_dir : outputs/evaluation/ — where CSV/JSON are written

    Returns
    -------
    dict keyed by table name::

        {
            "status":            "ok" | "missing_data" | "insufficient_rows" | "no_qi_found",
            "quasi_identifiers": [str, ...],
            "n_records":         int,
            "n_groups":          int,
            "min_k":             int,
            "mean_k":            float,
            "max_k":             int,
            "pct_k_ge_5":        float,   # % records in groups of size ≥ 5
            "pct_k_ge_10":       float,
        }
    """
    all_results: dict[str, dict] = {}

    for table_name, (original, synthetic) in pairs.items():
        if synthetic is None:
            logger.warning("[K-ANON] %s — missing synthetic data, skipped", table_name)
            all_results[table_name] = {"status": "missing_data"}
            continue

        result = _evaluate_table(synthetic, table_name)
        _log_result(table_name, result)
        all_results[table_name] = result

    _save_csv(output_dir, all_results)
    _save_json(output_dir, all_results)
    return all_results


# ── Per-table logic ───────────────────────────────────────────────────────────

def _evaluate_table(df: pd.DataFrame, table_name: str) -> dict:
    if len(df) < _MIN_ROWS:
        return {"status": "insufficient_rows", "n_records": int(len(df))}

    qi_cols = _detect_qi(df)
    if not qi_cols:
        return {
            "status":            "no_qi_found",
            "n_records":         int(len(df)),
            "quasi_identifiers": [],
        }

    df_qi = df[qi_cols].fillna("__NULL__").astype(str)

    try:
        group_sizes = df_qi.groupby(qi_cols).size()
    except Exception as exc:
        logger.warning("[K-ANON] %s — groupby failed: %s", table_name, exc)
        return {"status": "error", "error": str(exc)}

    n_records = int(len(df))
    n_groups  = int(len(group_sizes))
    min_k     = int(group_sizes.min())
    max_k     = int(group_sizes.max())
    mean_k    = round(float(group_sizes.mean()), 2)

    # Per-record k values
    record_k = df_qi.groupby(qi_cols)[qi_cols[0]].transform("count")
    pct_k5   = round(float((record_k >= 5).sum()  / n_records * 100), 2)
    pct_k10  = round(float((record_k >= 10).sum() / n_records * 100), 2)

    return {
        "status":            "ok",
        "quasi_identifiers": qi_cols,
        "n_records":         n_records,
        "n_groups":          n_groups,
        "min_k":             min_k,
        "mean_k":            mean_k,
        "max_k":             max_k,
        "pct_k_ge_5":        pct_k5,
        "pct_k_ge_10":       pct_k10,
    }


# ── QI detection ──────────────────────────────────────────────────────────────

def _detect_qi(df: pd.DataFrame) -> list[str]:
    """Return columns whose names match QI keywords and have bounded cardinality."""
    qi = []
    for col in df.columns:
        if not any(kw in col.lower() for kw in _QI_KEYWORDS):
            continue
        if df[col].nunique(dropna=True) > _MAX_QI_CARDINALITY:
            continue
        qi.append(col)
    return qi


# ── File I/O ──────────────────────────────────────────────────────────────────

def _save_csv(output_dir: Path, results: dict) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for table, r in results.items():
        rows.append({
            "table":             table,
            "status":            r.get("status"),
            "quasi_identifiers": "|".join(r.get("quasi_identifiers", [])),
            "n_records":         r.get("n_records"),
            "n_groups":          r.get("n_groups"),
            "min_k":             r.get("min_k"),
            "mean_k":            r.get("mean_k"),
            "max_k":             r.get("max_k"),
            "pct_k_ge_5":        r.get("pct_k_ge_5"),
            "pct_k_ge_10":       r.get("pct_k_ge_10"),
        })
    path = output_dir / "kanonymity.csv"
    pd.DataFrame(rows).to_csv(path, index=False)
    logger.info("Saved: %s", path)


def _save_json(output_dir: Path, results: dict) -> None:
    path = output_dir / "kanonymity.json"
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(results, fh, indent=2, default=str)
    logger.info("Saved: %s", path)


def _log_result(table: str, result: dict) -> None:
    status = result.get("status")
    if status != "ok":
        logger.info("[K-ANON] %-14s  %s", table, status)
        return
    logger.info(
        "[K-ANON] %-14s  QI=%s  min_k=%d  mean_k=%.1f  "
        "pct_k≥5=%.1f%%  pct_k≥10=%.1f%%",
        table,
        result["quasi_identifiers"],
        result["min_k"],
        result["mean_k"],
        result["pct_k_ge_5"],
        result["pct_k_ge_10"],
    )
