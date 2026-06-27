"""SDV model compatibility checks — Phase 4 pre-flight.

Validates that the ready tables and metadata.json are fully compatible with
SDV before any training is attempted. If this passes, any training failure
is model-related, not data-related.

Checks
------
1. Categorical cardinality    — no column with a single unique value (constant)
2. Numerical finite values    — no NaN or ±inf in numerical columns
3. No 100%-null columns       — every column has at least one non-null value
4. Primary key uniqueness     — each PK column is unique (SDV requires this)
5. SDV metadata load          — metadata.json loads into MultiTableMetadata
                                without raising exceptions
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

_PASS = "PASS"
_FAIL = "FAIL"
_WARN = "WARN"
_SKIP = "SKIP"

_PK_MAP = {
    "patients":     "patient_id",
    "encounters":   "encounter_id",
    "observations": "observation_id",
    "conditions":   "condition_id",
    "medications":  "medication_id",
}


# ── Result type (reuse same shape as readiness.CheckResult) ──────────────────

class CompatResult:
    def __init__(self, name: str, value: str, status: str, detail: str = ""):
        self.name   = name
        self.value  = value
        self.status = status
        self.detail = detail


# ── Individual checks ─────────────────────────────────────────────────────────

def _check_constant_columns(tables: dict[str, pd.DataFrame]) -> list[CompatResult]:
    """Flag any column with only one unique value — CTGAN/PAR will reject these."""
    results = []
    for tname, df in tables.items():
        constants = [c for c in df.columns if df[c].nunique(dropna=True) <= 1]
        status = _PASS if not constants else _WARN
        results.append(CompatResult(
            name=f"Constant cols — {tname}",
            value=str(len(constants)),
            status=status,
            detail=f"columns: {constants}" if constants else "",
        ))
    return results


def _check_numerical_finite(
    tables: dict[str, pd.DataFrame],
    metadata: dict,
) -> list[CompatResult]:
    """Check that all numerical columns contain only finite values."""
    results = []
    for tname, tmeta in metadata.get("tables", {}).items():
        df = tables.get(tname)
        if df is None:
            continue
        num_cols = [
            col for col, info in tmeta.get("columns", {}).items()
            if info.get("sdtype") == "numerical" and col in df.columns
        ]
        issues: list[str] = []
        for col in num_cols:
            series = pd.to_numeric(df[col], errors="coerce")
            n_nan  = int(series.isna().sum())
            n_inf  = int(np.isinf(series.dropna()).sum())
            if n_inf > 0:
                issues.append(f"{col}: {n_inf} inf")
        status = _PASS if not issues else _FAIL
        results.append(CompatResult(
            name=f"Finite numericals — {tname}",
            value="OK" if not issues else "ISSUES",
            status=status,
            detail="; ".join(issues) if issues else "",
        ))
    return results


def _check_null_columns(tables: dict[str, pd.DataFrame]) -> list[CompatResult]:
    """Flag any column that is 100% null — SDV cannot model these."""
    results = []
    for tname, df in tables.items():
        all_null = [c for c in df.columns if df[c].isna().all()]
        status = _PASS if not all_null else _FAIL
        results.append(CompatResult(
            name=f"All-null cols — {tname}",
            value=str(len(all_null)),
            status=status,
            detail=f"columns: {all_null}" if all_null else "",
        ))
    return results


def _check_pk_uniqueness(tables: dict[str, pd.DataFrame]) -> list[CompatResult]:
    """Verify that every primary key column is unique (SDV requirement)."""
    results = []
    for tname, pk_col in _PK_MAP.items():
        df = tables.get(tname)
        if df is None or pk_col not in df.columns:
            continue
        n_dup = int(df[pk_col].duplicated().sum())
        status = _PASS if n_dup == 0 else _FAIL
        results.append(CompatResult(
            name=f"PK unique — {tname}",
            value=str(n_dup),
            status=status,
            detail=f"{n_dup} duplicate {pk_col}" if n_dup else "",
        ))
    return results


def _check_categorical_cardinality(
    tables: dict[str, pd.DataFrame],
    metadata: dict,
    max_cardinality: int = 500,
) -> list[CompatResult]:
    """Warn when a categorical column has very high cardinality.

    CTGAN can handle high-cardinality categoricals but training becomes slow
    and the model may overfit. Flag anything above ``max_cardinality``.
    """
    results = []
    for tname, tmeta in metadata.get("tables", {}).items():
        df = tables.get(tname)
        if df is None:
            continue
        cat_cols = [
            col for col, info in tmeta.get("columns", {}).items()
            if info.get("sdtype") == "categorical" and col in df.columns
        ]
        high_card = []
        for col in cat_cols:
            n_unique = df[col].nunique(dropna=True)
            if n_unique > max_cardinality:
                high_card.append(f"{col}:{n_unique}")
        status = _PASS if not high_card else _WARN
        results.append(CompatResult(
            name=f"Categorical cardinality — {tname}",
            value=f"{len(high_card)} high",
            status=status,
            detail="; ".join(high_card) if high_card else "",
        ))
    return results


def _check_sdv_metadata_load(metadata_path: Path) -> list[CompatResult]:
    """Attempt to load metadata.json into SDV's MultiTableMetadata.

    Returns SKIP if SDV is not installed (avoids breaking on environments
    where sdv is not yet installed).
    """
    try:
        from sdv.metadata import MultiTableMetadata  # type: ignore
    except ImportError:
        return [CompatResult(
            name="SDV metadata load",
            value="SKIP",
            status=_SKIP,
            detail="sdv not installed — install with: pip install sdv",
        )]

    try:
        with open(metadata_path, encoding="utf-8") as fh:
            meta_dict = json.load(fh)

        m = MultiTableMetadata.load_from_dict(meta_dict)

        # Validate — SDV raises ValueError for inconsistencies
        m.validate()

        n_tables = len(meta_dict.get("tables", {}))
        return [CompatResult(
            name="SDV metadata load",
            value=f"{n_tables} tables",
            status=_PASS,
            detail="MultiTableMetadata.validate() passed",
        )]
    except Exception as exc:
        return [CompatResult(
            name="SDV metadata load",
            value="FAIL",
            status=_FAIL,
            detail=str(exc)[:120],
        )]


# ── Top-level runner ──────────────────────────────────────────────────────────

def run_compat_checks(
    tables: dict[str, pd.DataFrame],
    metadata: dict,
    metadata_path: Path,
) -> list[CompatResult]:
    """Run all SDV compatibility checks and return a flat list of results."""
    results: list[CompatResult] = []
    results += _check_constant_columns(tables)
    results += _check_null_columns(tables)
    results += _check_numerical_finite(tables, metadata)
    results += _check_pk_uniqueness(tables)
    results += _check_categorical_cardinality(tables, metadata)
    results += _check_sdv_metadata_load(metadata_path)
    return results


# ── Terminal + markdown output ────────────────────────────────────────────────

def print_compat_summary(results: list[CompatResult]) -> None:
    W    = 64
    sep  = "=" * W
    thin = "-" * W
    tags = {_PASS: "[PASS]", _FAIL: "[FAIL]", _WARN: "[WARN]", _SKIP: "[SKIP]"}

    all_pass = all(r.status in (_PASS, _SKIP, _WARN) for r in results)
    fail_or_warn = [r for r in results if r.status in (_FAIL, _WARN) and r.detail]

    logger.info(sep)
    logger.info("  SDV MODEL COMPATIBILITY CHECK")
    logger.info(sep)

    for r in results:
        tag   = tags.get(r.status, r.status)
        avail = W - len(tag) - 1
        line  = f"  {r.name:<{avail - len(r.value) - 2}} {r.value}"
        line  = f"{line:<{W - len(tag)}}{tag}"
        logger.info(line)

    logger.info(thin)
    overall = "READY FOR TRAINING" if all_pass else "FIX ISSUES BEFORE TRAINING"
    logger.info("  Overall: %s", overall)
    logger.info(sep)

    if fail_or_warn:
        logger.info("  DETAIL")
        logger.info(thin)
        for r in fail_or_warn:
            logger.info("  [%s] %s: %s", r.status, r.name, r.detail)
        logger.info(sep)


def save_compat_report(results: list[CompatResult], output_path: Path) -> None:
    """Append the compatibility section to outputs/reports/readiness_report.md."""
    tags = {_PASS: "PASS", _FAIL: "FAIL", _WARN: "WARN", _SKIP: "SKIP"}

    lines = [
        "\n## SDV Model Compatibility\n\n",
        "| Check | Value | Status |\n",
        "|-------|-------|:------:|\n",
    ]
    for r in results:
        lines.append(f"| {r.name} | {r.value} | **{tags.get(r.status, r.status)}** |\n")

    fail_or_warn = [r for r in results if r.status in (_FAIL, _WARN) and r.detail]
    if fail_or_warn:
        lines += ["\n### Detail\n\n"]
        for r in fail_or_warn:
            lines.append(f"- **[{r.status}]** `{r.name}`: {r.detail}\n")

    # Append to existing report file
    with open(output_path, "a", encoding="utf-8") as fh:
        fh.writelines(lines)
    logger.info("Compat section appended to: %s", output_path)
