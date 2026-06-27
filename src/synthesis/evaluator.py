"""Smoke-test evaluator and quick sanity report for synthetic data.

Smoke test (Phase 4A)
---------------------
8 quick checks run immediately after generating a small synthetic sample.
If any FAIL, full training is blocked. Designed to catch metadata/schema
issues before wasting hours on a full training run.

Quick sanity report (Phase 4B / post-synthesis)
------------------------------------------------
A judge-friendly table summarising key statistics from the synthetic output:
patient count, avg age, avg encounters per patient, avg conditions, null %.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

_PASS = "PASS"
_FAIL = "FAIL"
_WARN = "WARN"


# ── Result type ───────────────────────────────────────────────────────────────

@dataclass
class SmokeCheckResult:
    name:   str
    value:  str
    status: str
    detail: str = ""


# ── Individual smoke checks ───────────────────────────────────────────────────

def _check_row_count(df: pd.DataFrame, expected: int) -> SmokeCheckResult:
    n = len(df)
    ok = n == expected
    return SmokeCheckResult(
        name="Row count",
        value=str(n),
        status=_PASS if ok else _FAIL,
        detail=f"expected {expected}" if not ok else "",
    )


def _check_column_match(
    synthetic: pd.DataFrame, real: pd.DataFrame
) -> SmokeCheckResult:
    missing = set(real.columns) - set(synthetic.columns)
    extra   = set(synthetic.columns) - set(real.columns)
    ok = not missing and not extra
    detail = ""
    if missing:
        detail += f"missing: {sorted(missing)}"
    if extra:
        detail += f"  extra: {sorted(extra)}"
    return SmokeCheckResult(
        name="Column match",
        value="OK" if ok else "MISMATCH",
        status=_PASS if ok else _FAIL,
        detail=detail.strip(),
    )


def _check_no_allnull(df: pd.DataFrame) -> SmokeCheckResult:
    all_null = [c for c in df.columns if df[c].isna().all()]
    return SmokeCheckResult(
        name="All-null columns",
        value=str(len(all_null)),
        status=_PASS if not all_null else _FAIL,
        detail=f"columns: {all_null}" if all_null else "",
    )


def _check_missing_pct(df: pd.DataFrame, threshold: float = 20.0) -> SmokeCheckResult:
    pct = 100 * df.isna().sum().sum() / max(df.size, 1)
    status = _PASS if pct <= threshold else _WARN
    return SmokeCheckResult(
        name="Missing values",
        value=f"{pct:.1f}%",
        status=status,
        detail=f"threshold {threshold}%",
    )


def _check_no_inf(df: pd.DataFrame) -> SmokeCheckResult:
    num_cols = df.select_dtypes(include="number").columns
    n_inf = sum(int(np.isinf(df[c].dropna()).sum()) for c in num_cols)
    return SmokeCheckResult(
        name="Infinite values",
        value=str(n_inf),
        status=_PASS if n_inf == 0 else _FAIL,
        detail=f"{n_inf} inf values in numerical columns" if n_inf else "",
    )


def _check_age_range(df: pd.DataFrame) -> SmokeCheckResult:
    if "age" not in df.columns:
        return SmokeCheckResult(name="Age range", value="N/A", status=_WARN,
                                detail="'age' column not found")
    age = pd.to_numeric(df["age"], errors="coerce").dropna()
    invalid = int(((age < 0) | (age > 130)).sum())
    return SmokeCheckResult(
        name="Age in [0, 130]",
        value=str(invalid),
        status=_PASS if invalid == 0 else _WARN,
        detail=f"{invalid} rows with age out of range" if invalid else "",
    )


def _check_unique_ids(df: pd.DataFrame, id_col: str) -> SmokeCheckResult:
    if id_col not in df.columns:
        return SmokeCheckResult(name=f"Unique {id_col}", value="N/A", status=_WARN,
                                detail=f"column '{id_col}' not found")
    n_dup = int(df[id_col].duplicated().sum())
    return SmokeCheckResult(
        name=f"Unique {id_col}",
        value=str(n_dup),
        status=_PASS if n_dup == 0 else _FAIL,
        detail=f"{n_dup} duplicate IDs" if n_dup else "",
    )


def _check_no_training_leakage(
    synthetic: pd.DataFrame, real: pd.DataFrame, id_col: str
) -> SmokeCheckResult:
    """Verify no synthetic IDs match real training IDs."""
    if id_col not in synthetic.columns or id_col not in real.columns:
        return SmokeCheckResult(name="No ID leakage", value="N/A", status=_WARN)
    real_ids    = set(real[id_col].astype(str))
    synth_ids   = set(synthetic[id_col].astype(str))
    leaked = len(real_ids & synth_ids)
    return SmokeCheckResult(
        name="No ID leakage",
        value=str(leaked),
        status=_PASS if leaked == 0 else _WARN,
        detail=f"{leaked} synthetic IDs match real training IDs" if leaked else "",
    )


# ── Top-level smoke test ──────────────────────────────────────────────────────

def run_smoke_test(
    synthetic: pd.DataFrame,
    real: pd.DataFrame,
    table_name: str,
    pk_col: str,
    expected_rows: int,
) -> list[SmokeCheckResult]:
    """Run 8 smoke checks on a synthetic sample.

    Returns a list of ``SmokeCheckResult`` objects. If any status is FAIL,
    the pipeline should abort before full training.
    """
    return [
        _check_row_count(synthetic, expected_rows),
        _check_column_match(synthetic, real),
        _check_no_allnull(synthetic),
        _check_missing_pct(synthetic),
        _check_no_inf(synthetic),
        _check_age_range(synthetic),
        _check_unique_ids(synthetic, pk_col),
        _check_no_training_leakage(synthetic, real, pk_col),
    ]


# ── Terminal + markdown output ────────────────────────────────────────────────

def print_smoke_results(
    results: list[SmokeCheckResult],
    table_name: str,
    synthetic: pd.DataFrame,
) -> None:
    W    = 60
    sep  = "=" * W
    thin = "-" * W
    tags = {_PASS: "[PASS]", _FAIL: "[FAIL]", _WARN: "[WARN]"}

    n_fail = sum(1 for r in results if r.status == _FAIL)
    n_warn = sum(1 for r in results if r.status == _WARN)
    verdict = "PASS" if n_fail == 0 else "FAIL"

    logger.info(sep)
    logger.info("  SMOKE TEST RESULTS — %s", table_name.upper())
    logger.info(sep)

    for r in results:
        tag   = tags.get(r.status, r.status)
        avail = W - len(tag) - 1
        line  = f"  {r.name:<{avail - len(r.value) - 2}} {r.value}"
        line  = f"{line:<{W - len(tag)}}{tag}"
        logger.info(line)

    logger.info(thin)
    logger.info("  Result: %s  (%d fail, %d warn)", verdict, n_fail, n_warn)
    logger.info(sep)

    # Print first 5 rows of synthetic sample
    logger.info("  SYNTHETIC SAMPLE (first 5 rows):")
    logger.info(thin)
    for i, row in synthetic.head(5).iterrows():
        age_val = row.get("age", "n/a")
        gender_val = row.get("gender", "n/a")
        logger.info("  [%d] age=%-6s gender=%s", i, age_val, gender_val)
    logger.info(sep)


def save_smoke_report(
    results: list[SmokeCheckResult],
    table_name: str,
    synthetic: pd.DataFrame,
    output_path: Path,
) -> None:
    """Save smoke test results as markdown."""
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    n_fail = sum(1 for r in results if r.status == _FAIL)
    verdict = "PASS" if n_fail == 0 else "FAIL"

    lines = [
        f"# Smoke Test Report — {table_name}\n\n",
        f"**Generated:** {now}  \n",
        f"**Verdict:** **{verdict}**  \n",
        f"**Rows generated:** {len(synthetic):,}\n\n",

        "## Checks\n\n",
        "| Check | Value | Status |\n",
        "|-------|-------|:------:|\n",
    ]
    tags = {_PASS: "PASS", _FAIL: "FAIL", _WARN: "WARN"}
    for r in results:
        lines.append(f"| {r.name} | {r.value} | **{tags.get(r.status, r.status)}** |\n")

    detail_rows = [r for r in results if r.detail and r.status != _PASS]
    if detail_rows:
        lines += ["\n## Detail\n\n"]
        for r in detail_rows:
            lines.append(f"- **[{r.status}]** {r.name}: {r.detail}\n")

    # Sample rows
    lines += [
        "\n## Synthetic Sample (first 10 rows)\n\n",
        synthetic.head(10).to_markdown(index=False), "\n",
    ]

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("".join(lines), encoding="utf-8")
    logger.info("Saved: %s", output_path)


# ── Quick sanity report (Phase 4B post-synthesis) ─────────────────────────────

def quick_sanity_report(
    synthetic_tables: dict[str, pd.DataFrame],
    output_path: Path,
) -> None:
    """Generate the post-synthesis sanity table for judges / demo.

    Mirrors the user's requested format:
    | Metric               | Value |
    | Synthetic Patients   | 1000  |
    | Avg Age              | 48.2  |
    | ...                  | ...   |
    """
    patients = synthetic_tables.get("patients", pd.DataFrame())
    encounters = synthetic_tables.get("encounters", pd.DataFrame())
    conditions = synthetic_tables.get("conditions", pd.DataFrame())
    medications = synthetic_tables.get("medications", pd.DataFrame())

    n_patients   = len(patients)
    avg_age      = round(float(pd.to_numeric(patients.get("age", pd.Series()), errors="coerce").mean()), 1) if n_patients else 0.0
    avg_enc      = round(len(encounters) / max(n_patients, 1), 1)
    avg_cond     = round(len(conditions) / max(n_patients, 1), 1)
    avg_med      = round(len(medications) / max(n_patients, 1), 1)
    total_cells  = sum(df.size for df in synthetic_tables.values())
    total_null   = sum(int(df.isna().sum().sum()) for df in synthetic_tables.values())
    null_pct     = round(100 * total_null / max(total_cells, 1), 1)

    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    lines = [
        "# Synthetic Data — Quick Sanity Report\n\n",
        f"**Generated:** {now}\n\n",
        "| Metric | Value |\n",
        "|--------|------:|\n",
        f"| Synthetic Patients | {n_patients:,} |\n",
        f"| Avg Age (years) | {avg_age} |\n",
        f"| Avg Encounters / Patient | {avg_enc} |\n",
        f"| Avg Conditions / Patient | {avg_cond} |\n",
        f"| Avg Medications / Patient | {avg_med} |\n",
        f"| Null Values (overall) | {null_pct}% |\n",
        f"| Total Synthetic Rows | {sum(len(df) for df in synthetic_tables.values()):,} |\n",
    ]

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("".join(lines), encoding="utf-8")

    # Also log to console
    logger.info("=" * 50)
    logger.info("  SYNTHETIC DATA SANITY REPORT")
    logger.info("-" * 50)
    logger.info("  Synthetic Patients      : %d", n_patients)
    logger.info("  Avg Age                 : %.1f yrs", avg_age)
    logger.info("  Avg Encounters/Patient  : %.1f", avg_enc)
    logger.info("  Avg Conditions/Patient  : %.1f", avg_cond)
    logger.info("  Null Values             : %.1f%%", null_pct)
    logger.info("=" * 50)
    logger.info("Saved: %s", output_path)
