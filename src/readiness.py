"""Dataset Readiness Report — Phase 4 pre-flight checks.

Loads the five ``*_ready.csv`` tables from ``data/ready/``, runs six
validation categories, and produces:
  - a formatted terminal summary
  - ``outputs/reports/readiness_report.md``

Check categories
----------------
1. Row counts        — confirm every table has rows
2. Missing values    — overall % and per-table
3. Duplicate PKs     — no duplicate primary keys
4. FK integrity      — no orphan records
5. Metadata          — metadata.json is valid and well-formed
6. Chronology        — temporal columns are internally consistent
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

import pandas as pd

from .config_loader import Config

logger = logging.getLogger(__name__)

# ── Data structures ───────────────────────────────────────────────────────────

_PASS = "PASS"
_FAIL = "FAIL"
_WARN = "WARN"

_PK_MAP = {
    "patients":     "patient_id",
    "encounters":   "encounter_id",
    "observations": "observation_id",
    "conditions":   "condition_id",
    "medications":  "medication_id",
}

_FK_CHECKS = [
    # (child_table, fk_col, parent_table, parent_pk)
    ("encounters",   "patient_id",   "patients",   "patient_id"),
    ("observations", "patient_id",   "patients",   "patient_id"),
    ("observations", "encounter_id", "encounters", "encounter_id"),
    ("conditions",   "patient_id",   "patients",   "patient_id"),
    ("conditions",   "encounter_id", "encounters", "encounter_id"),
    ("medications",  "patient_id",   "patients",   "patient_id"),
    ("medications",  "encounter_id", "encounters", "encounter_id"),
]

_EXPECTED_RELATIONSHIPS = 4
_EXPECTED_TABLES = {"patients", "encounters", "observations", "conditions", "medications"}


@dataclass
class CheckResult:
    name: str
    value: str
    status: str       # PASS | FAIL | WARN
    detail: str = ""  # optional detail line shown in the report


@dataclass
class ReadinessReport:
    timestamp: str
    row_counts: list[CheckResult]
    missing: list[CheckResult]
    duplicates: list[CheckResult]
    fk_integrity: list[CheckResult]
    metadata_checks: list[CheckResult]
    chronology: list[CheckResult]

    @property
    def all_checks(self) -> list[CheckResult]:
        return (
            self.row_counts
            + self.missing
            + self.duplicates
            + self.fk_integrity
            + self.metadata_checks
            + self.chronology
        )

    @property
    def ready_for_sdv(self) -> bool:
        return all(c.status != _FAIL for c in self.all_checks)

    @property
    def summary_status(self) -> str:
        return _PASS if self.ready_for_sdv else _FAIL


# ── Loaders ───────────────────────────────────────────────────────────────────

def _load_ready(config: Config) -> dict[str, pd.DataFrame]:
    tables: dict[str, pd.DataFrame] = {}
    for name in _PK_MAP:
        path = config.ready_dir / f"{name}_ready.csv"
        if not path.exists():
            raise FileNotFoundError(
                f"Ready table not found: {path}\n"
                "Run Phase 3 first: python run_phase3.py"
            )
        tables[name] = pd.read_csv(path, low_memory=False)
    return tables


# ── Check functions ───────────────────────────────────────────────────────────

def _check_row_counts(tables: dict[str, pd.DataFrame]) -> list[CheckResult]:
    results = []
    for tname, df in tables.items():
        n = len(df)
        status = _PASS if n > 0 else _FAIL
        results.append(CheckResult(
            name=f"Rows — {tname}",
            value=f"{n:,}",
            status=status,
            detail="" if n > 0 else "table is empty",
        ))
    return results


def _check_missing(tables: dict[str, pd.DataFrame]) -> list[CheckResult]:
    results = []
    total_cells = sum(df.size for df in tables.values())
    total_missing = sum(int(df.isna().sum().sum()) for df in tables.values())
    pct = 100 * total_missing / max(total_cells, 1)

    status = _PASS if pct < 5 else (_WARN if pct < 20 else _FAIL)
    results.append(CheckResult(
        name="Missing Values (overall)",
        value=f"{pct:.2f}%",
        status=status,
        detail=f"{total_missing:,} of {total_cells:,} cells",
    ))

    for tname, df in tables.items():
        miss = int(df.isna().sum().sum())
        cells = df.size
        t_pct = 100 * miss / max(cells, 1)
        t_status = _PASS if t_pct < 5 else (_WARN if t_pct < 20 else _FAIL)
        top_cols = df.isna().sum().nlargest(3)
        top_cols = top_cols[top_cols > 0]
        detail = ", ".join(f"{c}:{n}" for c, n in top_cols.items()) if len(top_cols) else ""
        results.append(CheckResult(
            name=f"Missing — {tname}",
            value=f"{t_pct:.2f}%",
            status=t_status,
            detail=detail,
        ))

    return results


def _check_duplicates(tables: dict[str, pd.DataFrame]) -> list[CheckResult]:
    results = []
    for tname, pk in _PK_MAP.items():
        df = tables[tname]
        if pk not in df.columns:
            results.append(CheckResult(name=f"Duplicates — {tname}", value="N/A",
                                       status=_WARN, detail=f"{pk} column missing"))
            continue
        n_dup = int(df[pk].duplicated().sum())
        status = _PASS if n_dup == 0 else _FAIL
        results.append(CheckResult(
            name=f"Duplicate PKs — {tname}",
            value=str(n_dup),
            status=status,
            detail=f"{n_dup} duplicate {pk} values" if n_dup else "",
        ))
    return results


def _check_fk(tables: dict[str, pd.DataFrame]) -> list[CheckResult]:
    results = []
    for child_tbl, fk_col, parent_tbl, parent_pk in _FK_CHECKS:
        child_df  = tables[child_tbl]
        parent_df = tables[parent_tbl]
        if fk_col not in child_df.columns:
            results.append(CheckResult(
                name=f"FK {child_tbl}.{fk_col}",
                value="N/A", status=_WARN, detail="column missing",
            ))
            continue
        valid = set(parent_df[parent_pk].dropna().astype(str))
        orphans = int((~child_df[fk_col].astype(str).isin(valid)).sum())
        status = _PASS if orphans == 0 else _FAIL
        results.append(CheckResult(
            name=f"FK {child_tbl}.{fk_col}",
            value=str(orphans),
            status=status,
            detail=f"{orphans:,} orphan records" if orphans else "",
        ))
    return results


def _check_metadata(config: Config) -> list[CheckResult]:
    results = []
    meta_path = config.ready_dir / "metadata.json"

    # File exists
    if not meta_path.exists():
        results.append(CheckResult(
            name="metadata.json exists",
            value="NO", status=_FAIL, detail=str(meta_path),
        ))
        return results
    results.append(CheckResult(name="metadata.json exists", value="YES", status=_PASS))

    with open(meta_path, encoding="utf-8") as fh:
        meta = json.load(fh)

    # Spec version
    spec = meta.get("METADATA_SPEC_VERSION", "?")
    results.append(CheckResult(
        name="Metadata spec version",
        value=spec,
        status=_PASS if spec == "V1" else _FAIL,
        detail="" if spec == "V1" else "expected V1",
    ))

    # Table coverage
    tables_present = set(meta.get("tables", {}).keys())
    missing_tables = _EXPECTED_TABLES - tables_present
    results.append(CheckResult(
        name="Metadata tables",
        value=f"{len(tables_present)}/5",
        status=_PASS if not missing_tables else _FAIL,
        detail=f"missing: {missing_tables}" if missing_tables else "",
    ))

    # Relationships count
    n_rel = len(meta.get("relationships", []))
    results.append(CheckResult(
        name="Metadata relationships",
        value=str(n_rel),
        status=_PASS if n_rel >= _EXPECTED_RELATIONSHIPS else _WARN,
        detail=f"expected {_EXPECTED_RELATIONSHIPS}" if n_rel < _EXPECTED_RELATIONSHIPS else "",
    ))

    # Each table has a primary_key
    no_pk = [t for t, tmeta in meta.get("tables", {}).items() if "primary_key" not in tmeta]
    results.append(CheckResult(
        name="Metadata primary keys",
        value="OK" if not no_pk else f"{len(no_pk)} missing",
        status=_PASS if not no_pk else _FAIL,
        detail=f"tables without pk: {no_pk}" if no_pk else "",
    ))

    return results


def _check_chronology(tables: dict[str, pd.DataFrame]) -> list[CheckResult]:
    results = []

    # days_since_birth must be >= 0 in all tables that have it
    for tname in ("encounters", "observations", "conditions", "medications"):
        df = tables[tname]
        col = "days_since_birth" if tname != "conditions" else "onset_days_since_birth"
        if col not in df.columns:
            continue
        neg = int((df[col].dropna() < 0).sum())
        status = _PASS if neg == 0 else _FAIL
        results.append(CheckResult(
            name=f"Chronology days>=0 — {tname}",
            value=str(neg),
            status=status,
            detail=f"{neg:,} negative values in {col}" if neg else "",
        ))

    # Conditions: onset <= abatement where both exist
    cond = tables["conditions"]
    if "onset_days_since_birth" in cond.columns and "abatement_days_since_birth" in cond.columns:
        both = cond.dropna(subset=["onset_days_since_birth", "abatement_days_since_birth"])
        inverted = int((both["onset_days_since_birth"] > both["abatement_days_since_birth"]).sum())
        results.append(CheckResult(
            name="Chronology onset<=abatement",
            value=str(inverted),
            status=_PASS if inverted == 0 else _WARN,
            detail=f"{inverted:,} conditions onset > abatement" if inverted else "",
        ))

    # Encounter duration >= 0
    enc = tables["encounters"]
    if "encounter_duration_hours" in enc.columns:
        neg_dur = int((enc["encounter_duration_hours"].dropna() < 0).sum())
        results.append(CheckResult(
            name="Encounter duration >= 0",
            value=str(neg_dur),
            status=_PASS if neg_dur == 0 else _WARN,
            detail=f"{neg_dur:,} negative durations" if neg_dur else "",
        ))

    # Age sanity: 0 <= age <= 130
    pat = tables["patients"]
    if "age" in pat.columns:
        age = pat["age"].dropna()
        invalid = int(((age < 0) | (age > 130)).sum())
        results.append(CheckResult(
            name="Patient age in [0, 130]",
            value=str(invalid),
            status=_PASS if invalid == 0 else _WARN,
            detail=f"{invalid:,} patients with age out of range" if invalid else "",
        ))

    # Sequence index: must start at 0 or 1 and be positive integers
    for tname in ("encounters", "observations", "conditions", "medications"):
        df = tables[tname]
        if "sequence_index" not in df.columns:
            continue
        neg_seq = int((df["sequence_index"].dropna() < 0).sum())
        results.append(CheckResult(
            name=f"Sequence index >= 0 — {tname}",
            value=str(neg_seq),
            status=_PASS if neg_seq == 0 else _FAIL,
            detail=f"{neg_seq:,} negative sequence values" if neg_seq else "",
        ))

    return results


# ── Build report ──────────────────────────────────────────────────────────────

def build_report(config: Config) -> ReadinessReport:
    logger.info("Loading ready tables...")
    tables = _load_ready(config)

    logger.info("Running readiness checks...")
    return ReadinessReport(
        timestamp=datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        row_counts=_check_row_counts(tables),
        missing=_check_missing(tables),
        duplicates=_check_duplicates(tables),
        fk_integrity=_check_fk(tables),
        metadata_checks=_check_metadata(config),
        chronology=_check_chronology(tables),
    )


# ── Terminal output ───────────────────────────────────────────────────────────

def print_report(report: ReadinessReport) -> None:
    """Print a clean, judge-friendly terminal summary."""
    W = 56
    sep  = "=" * W
    thin = "-" * W

    def _status_tag(status: str) -> str:
        return {"PASS": "[PASS]", "FAIL": "[FAIL]", "WARN": "[WARN]"}.get(status, status)

    def _row(label: str, value: str, status: str) -> None:
        tag   = _status_tag(status)
        avail = W - len(tag) - 2
        label_w = max(avail - len(value) - 2, 12)
        line  = f"  {label:<{label_w}} {value}"
        line  = f"{line:<{W - len(tag) - 1}}{tag}"
        logger.info(line)

    logger.info(sep)
    logger.info("  DATASET READINESS REPORT")
    logger.info(f"  Generated: {report.timestamp}")
    logger.info(sep)

    # Row counts block
    logger.info("  ROW COUNTS")
    logger.info(thin)
    for c in report.row_counts:
        tname = c.name.split(" — ", 1)[-1]
        _row(f"  {tname}", c.value, c.status)

    # Summary checks block
    logger.info(thin)
    logger.info("  VALIDATION SUMMARY")
    logger.info(thin)

    # Overall missing
    overall_miss = next(c for c in report.missing if "overall" in c.name)
    _row("  Missing Values", overall_miss.value, overall_miss.status)

    # Duplicates: total across tables
    total_dup = sum(int(c.value) for c in report.duplicates if c.value.isdigit())
    dup_status = _PASS if total_dup == 0 else _FAIL
    _row("  Duplicate Rows", str(total_dup), dup_status)

    # FK: total broken
    total_broken = sum(int(c.value) for c in report.fk_integrity if c.value.isdigit())
    fk_status = _PASS if total_broken == 0 else _FAIL
    _row("  Broken FK", str(total_broken), fk_status)

    # Metadata overall
    meta_fail = any(c.status == _FAIL for c in report.metadata_checks)
    meta_warn = any(c.status == _WARN for c in report.metadata_checks)
    meta_status = _FAIL if meta_fail else (_WARN if meta_warn else _PASS)
    _row("  Metadata Validation", _status_tag(meta_status).strip("[]"), meta_status)

    # Chronology overall
    chron_fail = any(c.status == _FAIL for c in report.chronology)
    chron_warn = any(c.status == _WARN for c in report.chronology)
    chron_status = _FAIL if chron_fail else (_WARN if chron_warn else _PASS)
    _row("  Chronology Validation", _status_tag(chron_status).strip("[]"), chron_status)

    logger.info(thin)
    final_tag = _status_tag(report.summary_status)
    line = f"  Ready for SDV"
    line = f"  {'Ready for SDV':<{W - len(final_tag) - len('YES') - 3}} {'YES' if report.ready_for_sdv else 'NO'} {final_tag}"
    logger.info(line)
    logger.info(sep)

    # Detail block for failures / warnings
    failures = [c for c in report.all_checks if c.status in (_FAIL, _WARN) and c.detail]
    if failures:
        logger.info("  DETAIL (failures / warnings)")
        logger.info(thin)
        for c in failures:
            logger.info("  [%s] %s: %s", c.status, c.name, c.detail)
        logger.info(sep)


# ── Markdown report ───────────────────────────────────────────────────────────

def save_report(report: ReadinessReport, config: Config) -> None:
    """Write outputs/reports/readiness_report.md."""

    def _icon(status: str) -> str:
        return {"PASS": "OK", "FAIL": "FAIL", "WARN": "WARN"}.get(status, status)

    lines = [
        "# Dataset Readiness Report\n\n",
        f"**Generated:** {report.timestamp}\n\n",

        "## Row Counts\n\n",
        "| Table | Rows | Status |\n",
        "|-------|-----:|:------:|\n",
    ]
    for c in report.row_counts:
        tname = c.name.split(" — ", 1)[-1]
        lines.append(f"| {tname} | {c.value} | {_icon(c.status)} |\n")

    lines += [
        "\n## Validation Summary\n\n",
        "| Check | Value | Status |\n",
        "|-------|-------|:------:|\n",
    ]

    overall_miss = next(c for c in report.missing if "overall" in c.name)
    total_dup    = sum(int(c.value) for c in report.duplicates if c.value.isdigit())
    dup_status   = _PASS if total_dup == 0 else _FAIL
    total_broken = sum(int(c.value) for c in report.fk_integrity if c.value.isdigit())
    fk_status    = _PASS if total_broken == 0 else _FAIL

    meta_fail   = any(c.status == _FAIL for c in report.metadata_checks)
    meta_status = _FAIL if meta_fail else (
        _WARN if any(c.status == _WARN for c in report.metadata_checks) else _PASS)

    chron_fail   = any(c.status == _FAIL for c in report.chronology)
    chron_status = _FAIL if chron_fail else (
        _WARN if any(c.status == _WARN for c in report.chronology) else _PASS)

    summary_rows = [
        ("Missing Values", overall_miss.value, overall_miss.status),
        ("Duplicate Rows", str(total_dup),    dup_status),
        ("Broken FK",      str(total_broken),  fk_status),
        ("Metadata Validation", _icon(meta_status), meta_status),
        ("Chronology Validation", _icon(chron_status), chron_status),
        ("Ready for SDV",  "YES" if report.ready_for_sdv else "NO", report.summary_status),
    ]
    for name, val, status in summary_rows:
        lines.append(f"| {name} | {val} | **{_icon(status)}** |\n")

    # Missing per-table detail
    lines += [
        "\n## Missing Values Per Table\n\n",
        "| Table | Missing % | Top Missing Columns |\n",
        "|-------|:---------:|---------------------|\n",
    ]
    for c in report.missing:
        if "overall" in c.name:
            continue
        tname = c.name.split(" — ", 1)[-1]
        lines.append(f"| {tname} | {c.value} | {c.detail or 'None'} |\n")

    # FK detail
    lines += [
        "\n## Foreign Key Integrity\n\n",
        "| Check | Orphan Records | Status |\n",
        "|-------|:--------------:|:------:|\n",
    ]
    for c in report.fk_integrity:
        fk_name = c.name.replace("FK ", "")
        lines.append(f"| {fk_name} | {c.value} | {_icon(c.status)} |\n")

    # Metadata detail
    lines += [
        "\n## Metadata Validation\n\n",
        "| Check | Value | Status |\n",
        "|-------|-------|:------:|\n",
    ]
    for c in report.metadata_checks:
        lines.append(f"| {c.name} | {c.value} | {_icon(c.status)} |\n")

    # Chronology detail
    lines += [
        "\n## Chronology Validation\n\n",
        "| Check | Violations | Status |\n",
        "|-------|:----------:|:------:|\n",
    ]
    for c in report.chronology:
        lines.append(f"| {c.name} | {c.value} | {_icon(c.status)} |\n")

    path = config.reports_dir / "readiness_report.md"
    path.write_text("".join(lines), encoding="utf-8")
    logger.info("Saved: %s", path)
