"""Relationship and temporal validation for extracted FHIR CSV tables.

Reads from data/processed/ after extraction is complete.
Writes relationship_report.md and timeline_report.md to outputs/reports/.
"""

from __future__ import annotations

import logging
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd

from .config_loader import Config

logger = logging.getLogger(__name__)

# ── Helpers ───────────────────────────────────────────────────────────────────

def _load_tables(config: Config) -> dict[str, pd.DataFrame]:
    """Load all five extracted CSVs into a dict keyed by table name."""
    csv_map = {
        "patients": "patients.csv",
        "encounters": "encounters.csv",
        "observations": "observations.csv",
        "conditions": "conditions.csv",
        "medications": "medications.csv",
    }
    tables: dict[str, pd.DataFrame] = {}
    for name, fname in csv_map.items():
        fpath = config.processed_dir / fname
        if fpath.exists():
            tables[name] = pd.read_csv(fpath, low_memory=False)
            logger.debug("Loaded %s — %d rows", fname, len(tables[name]))
        else:
            logger.warning("CSV not found, skipping validation for: %s", fpath)
    return tables


# ── Relationship validation ───────────────────────────────────────────────────

def validate_relationships(tables: dict[str, pd.DataFrame]) -> dict[str, dict[str, Any]]:
    """Check referential integrity: every foreign key must resolve to a known ID.

    Checks:
      - Every child record's patient_id must exist in patients.patient_id
      - Every encounter_id reference must exist in encounters.encounter_id
      - No duplicate primary keys within a table

    Returns:
        Dict of per-table result dicts.
    """
    results: dict[str, dict[str, Any]] = {}

    valid_patient_ids: set[str] = set(
        tables["patients"]["patient_id"].dropna().astype(str)
    ) if "patients" in tables else set()

    valid_encounter_ids: set[str] = set(
        tables["encounters"]["encounter_id"].dropna().astype(str)
    ) if "encounters" in tables else set()

    # (table_name, pk_column, has_encounter_fk)
    child_configs = [
        ("encounters",   "encounter_id",   False),
        ("observations", "observation_id",  True),
        ("conditions",   "condition_id",    True),
        ("medications",  "medication_id",   True),
    ]

    for table_name, pk_col, has_enc_fk in child_configs:
        df = tables.get(table_name)
        if df is None:
            continue

        total = len(df)

        # Missing patient_id references
        bad_patients = int(
            (~df["patient_id"].astype(str).isin(valid_patient_ids)).sum()
        ) if "patient_id" in df.columns else 0

        # Missing encounter_id references (only where encounter_id is present and non-null)
        bad_encounters = 0
        if has_enc_fk and "encounter_id" in df.columns and valid_encounter_ids:
            mask = df["encounter_id"].notna() & (df["encounter_id"].astype(str) != "")
            bad_encounters = int(
                (mask & ~df["encounter_id"].astype(str).isin(valid_encounter_ids)).sum()
            )

        # Duplicate primary keys
        dupe_pks = int(df[pk_col].duplicated().sum()) if pk_col in df.columns else 0

        results[table_name] = {
            "total_rows": total,
            "missing_patient_refs": bad_patients,
            "missing_encounter_refs": bad_encounters,
            "duplicate_primary_keys": dupe_pks,
        }

        logger.info(
            "%-15s | rows=%6d | bad_patient_refs=%4d | bad_enc_refs=%4d | dupe_pks=%3d",
            table_name, total, bad_patients, bad_encounters, dupe_pks,
        )

    return results


# ── Temporal validation ───────────────────────────────────────────────────────

def validate_temporal(tables: dict[str, pd.DataFrame]) -> dict[str, dict[str, Any]]:
    """Verify chronological integrity: clinical events must follow birth date.

    Rules checked per table:
      - event_date >= patient birth_date
      - event_date <= today  (no future dates)
      - timestamp is not null/empty

    Returns:
        Dict of per-table result dicts.
    """
    results: dict[str, dict[str, Any]] = {}
    patients = tables.get("patients")
    if patients is None:
        logger.warning("patients table missing — skipping temporal validation")
        return results

    birth_map: dict[str, str] = (
        patients.set_index("patient_id")["birth_date"].dropna().to_dict()
    )
    today_str = datetime.today().strftime("%Y-%m-%d")

    # (table_name, pk_col, date_col)
    temporal_configs = [
        ("encounters",   "encounter_id",   "start_datetime"),
        ("observations", "observation_id", "effective_datetime"),
        ("conditions",   "condition_id",   "onset_datetime"),
        ("medications",  "medication_id",  "authored_on"),
    ]

    for table_name, pk_col, dt_col in temporal_configs:
        df = tables.get(table_name)
        if df is None or dt_col not in df.columns:
            continue

        total = len(df)
        missing_ts = int(df[dt_col].isna().sum() + (df[dt_col] == "").sum())

        # Work on rows that have both a timestamp and a known birth date
        work = df[["patient_id", dt_col]].copy()
        work["_birth"] = work["patient_id"].astype(str).map(birth_map)
        work = work.dropna(subset=[dt_col, "_birth"])
        work = work[work[dt_col].astype(str).str.strip() != ""]

        # Compare first 10 chars (YYYY-MM-DD prefix) — avoids timezone complexity
        date_col = work[dt_col].astype(str).str[:10]
        birth_col = work["_birth"].astype(str).str[:10]

        before_birth = int((date_col < birth_col).sum())
        future_dates = int((date_col > today_str).sum())

        results[table_name] = {
            "total_rows": total,
            "missing_timestamps": missing_ts,
            "events_before_birth": before_birth,
            "future_dates": future_dates,
        }

        logger.info(
            "%-15s | rows=%6d | missing_ts=%4d | before_birth=%4d | future=%4d",
            table_name, total, missing_ts, before_birth, future_dates,
        )

    return results


# ── Report writers ────────────────────────────────────────────────────────────

def _write_relationship_report(
    rel: dict[str, dict[str, Any]], config: Config
) -> None:
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    lines = [
        "# Relationship Validation Report\n\n",
        f"**Generated:** {ts}\n\n",
        "## Referential Integrity\n\n",
        "| Table | Total Rows | Missing Patient Refs | Missing Encounter Refs | Duplicate PKs |\n",
        "|-------|:----------:|:-------------------:|:---------------------:|:-------------:|\n",
    ]
    for table, stats in rel.items():
        ok_p = "✅" if stats["missing_patient_refs"] == 0 else "⚠️"
        ok_e = "✅" if stats["missing_encounter_refs"] == 0 else "⚠️"
        ok_d = "✅" if stats["duplicate_primary_keys"] == 0 else "⚠️"
        lines.append(
            f"| {table} "
            f"| {stats['total_rows']:,} "
            f"| {ok_p} {stats['missing_patient_refs']:,} "
            f"| {ok_e} {stats['missing_encounter_refs']:,} "
            f"| {ok_d} {stats['duplicate_primary_keys']:,} |\n"
        )

    all_clean = all(
        s["missing_patient_refs"] == 0
        and s["missing_encounter_refs"] == 0
        and s["duplicate_primary_keys"] == 0
        for s in rel.values()
    )
    lines.append(
        f"\n**Overall referential integrity:** {'✅ PASS' if all_clean else '⚠️  ISSUES FOUND'}\n"
    )

    path = config.reports_dir / "relationship_report.md"
    path.write_text("".join(lines), encoding="utf-8")
    logger.info("Saved: %s", path)


def _write_timeline_report(
    temp: dict[str, dict[str, Any]], config: Config
) -> None:
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    lines = [
        "# Temporal Validation Report\n\n",
        f"**Generated:** {ts}\n\n",
        "## Chronological Integrity\n\n",
        "| Table | Total Rows | Missing Timestamps | Before Birth | Future Dates |\n",
        "|-------|:----------:|:-----------------:|:------------:|:------------:|\n",
    ]
    for table, stats in temp.items():
        ok_m = "✅" if stats["missing_timestamps"] == 0 else "⚠️"
        ok_b = "✅" if stats["events_before_birth"] == 0 else "⚠️"
        ok_f = "✅" if stats["future_dates"] == 0 else "⚠️"
        lines.append(
            f"| {table} "
            f"| {stats['total_rows']:,} "
            f"| {ok_m} {stats['missing_timestamps']:,} "
            f"| {ok_b} {stats['events_before_birth']:,} "
            f"| {ok_f} {stats['future_dates']:,} |\n"
        )

    path = config.reports_dir / "timeline_report.md"
    path.write_text("".join(lines), encoding="utf-8")
    logger.info("Saved: %s", path)


# ── Public entry point ────────────────────────────────────────────────────────

def run(config: Config) -> None:
    """Run full validation pass and write both reports."""
    logger.info("Starting validation...")
    tables = _load_tables(config)

    rel_results = validate_relationships(tables)
    temp_results = validate_temporal(tables)

    _write_relationship_report(rel_results, config)
    _write_timeline_report(temp_results, config)

    logger.info("Validation complete.")
