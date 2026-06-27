"""SDV-compatible metadata generator for Phase 3.

Produces a ``metadata.json`` file that conforms to SDV v1.x
``MultiTableMetadata`` specification. This file is consumed directly by
SDV's HMASynthesizer (multi-table) and can also seed individual
SingleTableMetadata objects for CTGAN / PARSynthesizer.

SDV sdtype reference
--------------------
  id          : primary or foreign key column (UUIDs, identifiers)
  numerical   : continuous or discrete numeric values
  categorical : string or low-cardinality codes
  boolean     : 0/1 or True/False
  datetime    : ISO timestamp strings (not used here — converted to numerical)

Design decision: we convert all datetimes to ``days_since_birth`` (numerical)
in feature_engineering.py, so there are no ``datetime`` sdtypes in the
ready tables.  This removes timezone ambiguity and ensures SDV never tries to
parse mixed-offset ISO strings.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import pandas as pd

logger = logging.getLogger(__name__)

# ── Column-name heuristics ────────────────────────────────────────────────────

# Any column whose name contains one of these substrings is treated as an id.
_ID_SUFFIXES = ("_id",)

# Boolean column names in our schema
_BOOL_COLUMNS = {"is_deceased", "is_chronic", "is_active", "is_first_encounter"}

# Numerical columns that might look categorical (e.g. short integers)
_FORCE_NUMERICAL = {
    "age", "age_at_death", "days_since_birth", "encounter_duration_hours",
    "days_since_prev_encounter", "sequence_index", "duration_days",
    "encounter_count", "condition_count", "medication_count",
    "value_quantity", "onset_days_since_birth", "abatement_days_since_birth",
    "daly", "qaly",
}


def _infer_sdtype(col: pd.Series, col_name: str) -> dict:
    """Infer the SDV sdtype for a single column.

    Rules (applied in priority order):
    1. Column name ends with ``_id``        → ``id``
    2. Column name is in ``_BOOL_COLUMNS``  → ``boolean``
    3. Column name is in ``_FORCE_NUMERICAL``→ ``numerical``
    4. dtype is numeric                     → ``numerical``
    5. Unique-value ratio < 5 % or < 50 unique values → ``categorical``
    6. Fallback                             → ``categorical``
    """
    # Rule 1 — id columns
    if any(col_name.endswith(suffix) for suffix in _ID_SUFFIXES):
        return {"sdtype": "id"}

    # Rule 2 — boolean flags
    if col_name in _BOOL_COLUMNS:
        return {"sdtype": "boolean"}

    # Rule 3 — known numericals
    if col_name in _FORCE_NUMERICAL:
        return {"sdtype": "numerical"}

    # Rule 4 — numeric dtype
    if pd.api.types.is_numeric_dtype(col):
        n_unique = col.nunique(dropna=True)
        if n_unique <= 2:
            return {"sdtype": "boolean"}
        return {"sdtype": "numerical"}

    # Rule 5 — low cardinality object columns → categorical
    n_total = len(col.dropna())
    n_unique = col.nunique(dropna=True)
    if n_total == 0 or n_unique / max(n_total, 1) < 0.05 or n_unique < 50:
        return {"sdtype": "categorical"}

    # Fallback
    return {"sdtype": "categorical"}


def _build_table_metadata(
    df: pd.DataFrame,
    primary_key: str,
    id_foreign_keys: list[str] | None = None,
) -> dict:
    """Build the metadata dict for a single table.

    Args:
        df: The ready DataFrame for this table.
        primary_key: Name of the primary key column.
        id_foreign_keys: Additional columns that are foreign keys (also typed as ``id``).

    Returns:
        Dict conforming to SDV SingleTableMetadata / the ``tables`` sub-dict
        of MultiTableMetadata.
    """
    fk_set = set(id_foreign_keys or [])
    columns: dict[str, dict] = {}

    for col_name in df.columns:
        if col_name in fk_set:
            columns[col_name] = {"sdtype": "id"}
        else:
            columns[col_name] = _infer_sdtype(df[col_name], col_name)

    return {
        "primary_key": primary_key,
        "columns": columns,
    }


# ── Full multi-table metadata ─────────────────────────────────────────────────

def build_sdv_metadata(
    patients: pd.DataFrame,
    encounters: pd.DataFrame,
    observations: pd.DataFrame,
    conditions: pd.DataFrame,
    medications: pd.DataFrame,
) -> dict:
    """Build the complete SDV MultiTableMetadata dict for all five tables.

    The relationships section defines the parent → child FK links that
    HMASynthesizer uses to maintain referential integrity during synthesis.

    Returns:
        Dict ready to be serialised as ``metadata.json``.
    """
    tables = {
        "patients": _build_table_metadata(
            patients,
            primary_key="patient_id",
            id_foreign_keys=[],
        ),
        "encounters": _build_table_metadata(
            encounters,
            primary_key="encounter_id",
            id_foreign_keys=["patient_id"],
        ),
        "observations": _build_table_metadata(
            observations,
            primary_key="observation_id",
            id_foreign_keys=["patient_id", "encounter_id"],
        ),
        "conditions": _build_table_metadata(
            conditions,
            primary_key="condition_id",
            id_foreign_keys=["patient_id", "encounter_id"],
        ),
        "medications": _build_table_metadata(
            medications,
            primary_key="medication_id",
            id_foreign_keys=["patient_id", "encounter_id"],
        ),
    }

    relationships = [
        {
            "parent_table_name": "patients",
            "parent_primary_key": "patient_id",
            "child_table_name": "encounters",
            "child_foreign_key": "patient_id",
        },
        {
            "parent_table_name": "encounters",
            "parent_primary_key": "encounter_id",
            "child_table_name": "observations",
            "child_foreign_key": "encounter_id",
        },
        {
            "parent_table_name": "encounters",
            "parent_primary_key": "encounter_id",
            "child_table_name": "conditions",
            "child_foreign_key": "encounter_id",
        },
        {
            "parent_table_name": "encounters",
            "parent_primary_key": "encounter_id",
            "child_table_name": "medications",
            "child_foreign_key": "encounter_id",
        },
    ]

    metadata = {
        "METADATA_SPEC_VERSION": "V1",
        "tables": tables,
        "relationships": relationships,
    }

    # Log a summary
    logger.info("Metadata built for %d tables", len(tables))
    for tname, tmeta in tables.items():
        sdtypes = {}
        for col, info in tmeta["columns"].items():
            sdt = info["sdtype"]
            sdtypes[sdt] = sdtypes.get(sdt, 0) + 1
        logger.info("  %-15s pk=%-20s %s", tname, tmeta["primary_key"], sdtypes)

    return metadata


def save_metadata(metadata: dict, output_path: Path) -> None:
    """Serialise metadata dict to JSON and save to disk."""
    with open(output_path, "w", encoding="utf-8") as fh:
        json.dump(metadata, fh, indent=2)
    logger.info("Saved: %s", output_path)


def print_metadata_summary(metadata: dict) -> None:
    """Log a human-readable column-type breakdown for every table."""
    sep = "-" * 56
    logger.info(sep)
    logger.info("  SDV METADATA SUMMARY")
    logger.info(sep)
    for tname, tmeta in metadata["tables"].items():
        cols = tmeta["columns"]
        by_type: dict[str, list[str]] = {}
        for col, info in cols.items():
            by_type.setdefault(info["sdtype"], []).append(col)
        logger.info("  Table: %s  (pk=%s, %d columns)",
                    tname, tmeta["primary_key"], len(cols))
        for sdtype, col_list in sorted(by_type.items()):
            logger.info("    %-12s : %s", sdtype, ", ".join(col_list))
    logger.info(sep)
