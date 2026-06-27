"""Phase 3 preprocessing orchestrator.

Reads the five Phase 2 CSVs from ``data/processed/``, applies all feature
engineering transformations, validates relational integrity, generates SDV
metadata, and writes the ready tables to ``data/ready/``.

Outputs
-------
data/ready/
    patients_ready.csv
    encounters_ready.csv
    observations_ready.csv
    conditions_ready.csv
    medications_ready.csv
    metadata.json

outputs/reports/
    preprocessing_report.md
"""

from __future__ import annotations

import logging
import time
from datetime import datetime
from pathlib import Path

import pandas as pd

from .config_loader import Config
from .feature_engineering import (
    build_birth_map,
    clean_patients,
    add_patient_aggregates,
    engineer_encounters,
    engineer_observations,
    engineer_conditions,
    engineer_medications,
)
from .metadata_generator import (
    build_sdv_metadata,
    save_metadata,
    print_metadata_summary,
)

logger = logging.getLogger(__name__)


# ── Loaders ───────────────────────────────────────────────────────────────────

def _load_processed(config: Config) -> dict[str, pd.DataFrame]:
    """Load all five Phase 2 CSVs from data/processed/."""
    csv_map = {
        "patients":     "patients.csv",
        "encounters":   "encounters.csv",
        "observations": "observations.csv",
        "conditions":   "conditions.csv",
        "medications":  "medications.csv",
    }
    tables: dict[str, pd.DataFrame] = {}
    for name, fname in csv_map.items():
        path = config.processed_dir / fname
        if not path.exists():
            raise FileNotFoundError(
                f"Phase 2 output not found: {path}\n"
                "Run Phase 2 first: python run_phase2.py"
            )
        tables[name] = pd.read_csv(path, low_memory=False)
        logger.info("Loaded %-22s — %7d rows", fname, len(tables[name]))
    return tables


# ── Post-transform FK validation ──────────────────────────────────────────────

def _validate_foreign_keys(
    tables: dict[str, pd.DataFrame],
) -> dict[str, dict[str, int]]:
    """Check referential integrity on the ready tables.

    Returns a dict of {table_name: {check_name: orphan_count}}.
    """
    results: dict[str, dict[str, int]] = {}
    valid_pids = set(tables["patients"]["patient_id"].dropna().astype(str))
    valid_eids = set(tables["encounters"]["encounter_id"].dropna().astype(str))

    child_checks = {
        "encounters":   [("patient_id", valid_pids)],
        "observations": [("patient_id", valid_pids), ("encounter_id", valid_eids)],
        "conditions":   [("patient_id", valid_pids), ("encounter_id", valid_eids)],
        "medications":  [("patient_id", valid_pids), ("encounter_id", valid_eids)],
    }

    for tname, checks in child_checks.items():
        df = tables.get(tname)
        if df is None:
            continue
        results[tname] = {}
        for fk_col, valid_ids in checks:
            if fk_col not in df.columns:
                continue
            orphans = int((~df[fk_col].astype(str).isin(valid_ids)).sum())
            results[tname][fk_col] = orphans
            if orphans:
                logger.warning(
                    "FK violation: %s.%s has %d orphan records",
                    tname, fk_col, orphans,
                )
            else:
                logger.info("FK OK: %s.%s", tname, fk_col)

    return results


# ── Report writer ─────────────────────────────────────────────────────────────

def _write_report(
    raw_shapes: dict[str, tuple[int, int]],
    ready_shapes: dict[str, tuple[int, int]],
    fk_results: dict[str, dict[str, int]],
    missing_before: dict[str, dict[str, int]],
    missing_after: dict[str, dict[str, int]],
    config: Config,
) -> None:
    """Write outputs/reports/preprocessing_report.md."""

    def _missing_pct(counts: dict[str, int], total: int) -> str:
        if not counts:
            return "None"
        top = sorted(counts.items(), key=lambda x: -x[1])[:5]
        lines = [f"{col}: {n} ({100*n/max(total,1):.1f}%)" for col, n in top]
        return " | ".join(lines)

    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    lines = [
        "# Preprocessing Report\n\n",
        f"**Generated:** {now}\n\n",

        "## Row Count Summary\n\n",
        "| Table | Raw Rows | Raw Cols | Ready Rows | Ready Cols | Δ Rows |\n",
        "|-------|:--------:|:--------:|:----------:|:----------:|:------:|\n",
    ]
    for tname in raw_shapes:
        r_rows, r_cols = raw_shapes[tname]
        p_rows, p_cols = ready_shapes.get(tname, (0, 0))
        delta = p_rows - r_rows
        sign = "+" if delta >= 0 else ""
        lines.append(
            f"| {tname} | {r_rows:,} | {r_cols} "
            f"| {p_rows:,} | {p_cols} | {sign}{delta:,} |\n"
        )

    lines += [
        "\n## Missing Values (Top 5 columns per table)\n\n",
        "| Table | Before Preprocessing | After Preprocessing |\n",
        "|-------|---------------------|--------------------|\n",
    ]
    for tname in raw_shapes:
        before = _missing_pct(missing_before.get(tname, {}), raw_shapes[tname][0])
        after  = _missing_pct(missing_after.get(tname, {}),  ready_shapes.get(tname, (1, 0))[0])
        lines.append(f"| {tname} | {before} | {after} |\n")

    lines += [
        "\n## Foreign Key Validation (Post-Preprocessing)\n\n",
        "| Table | Foreign Key | Orphan Records |\n",
        "|-------|-------------|:--------------:|\n",
    ]
    for tname, checks in fk_results.items():
        for fk_col, orphans in checks.items():
            status = "✅ 0" if orphans == 0 else f"⚠️ {orphans:,}"
            lines.append(f"| {tname} | {fk_col} | {status} |\n")

    lines += [
        "\n## Transformations Applied\n\n",
        "### patients_ready.csv\n",
        "- Dropped PII columns: family_name, given_name, ssn, drivers_license, passport\n",
        "- Dropped zero-variance columns: country, postal_code\n",
        "- Computed: age (years), is_deceased (0/1), age_at_death\n",
        "- Added aggregates: encounter_count, condition_count, medication_count\n",
        "- Standardised categorical values to lowercase\n\n",
        "### encounters_ready.csv\n",
        "- Computed: days_since_birth, encounter_duration_hours\n",
        "- Added: sequence_index (per-patient chronological order)\n",
        "- Added: days_since_prev_encounter (inter-visit gap in days)\n",
        "- Dropped: type_code, practitioner_npi, reason_code\n\n",
        "### observations_ready.csv\n",
        f"- Filtered to top {config.top_n_loinc} LOINC codes (min freq {config.min_loinc_frequency})\n",
        "- Retained only value_type='quantity'; expanded blood pressure components\n",
        "- Computed: days_since_birth, sequence_index\n",
        "- Applied 3×IQR outlier capping per LOINC code\n\n",
        "### conditions_ready.csv\n",
        "- Computed: onset_days_since_birth, abatement_days_since_birth, duration_days\n",
        "- Added: is_chronic (1 = no abatement date), sequence_index\n",
        "- Dropped: snomed_code, verification_status, raw datetime columns\n\n",
        "### medications_ready.csv\n",
        "- Computed: days_since_birth, is_active (0/1), sequence_index\n",
        "- Dropped: rxnorm_code, requester_display, dosage_text, authored_on\n\n",
        "## Metadata\n\n",
        "- SDV MultiTableMetadata V1 written to `data/ready/metadata.json`\n",
        "- Column types inferred: id, numerical, categorical, boolean\n",
        "- All datetime columns converted to numerical (days_since_birth)\n",
    ]

    path = config.reports_dir / "preprocessing_report.md"
    path.write_text("".join(lines), encoding="utf-8")
    logger.info("Saved: %s", path)


# ── Summary printer ───────────────────────────────────────────────────────────

def _print_summary(
    ready_tables: dict[str, pd.DataFrame],
    fk_results: dict[str, dict[str, int]],
) -> None:
    all_ok = all(
        orphans == 0
        for checks in fk_results.values()
        for orphans in checks.values()
    )
    sep = "=" * 56
    logger.info(sep)
    logger.info("  PREPROCESSING SUMMARY")
    logger.info(sep)
    for tname, df in ready_tables.items():
        logger.info("  %-25s %7d rows  %2d cols",
                    f"{tname}_ready.csv", len(df), len(df.columns))
    logger.info(sep)
    logger.info("  Foreign Key Integrity: %s", "PASS" if all_ok else "ISSUES FOUND")
    logger.info("  metadata.json:         written")
    logger.info(sep)


# ── Public entry point ────────────────────────────────────────────────────────

def run(config: Config) -> None:
    """Execute the full Phase 3 preprocessing pipeline."""
    wall_start = time.time()
    sep = "=" * 56

    logger.info(sep)
    logger.info("  SYNTHETIC HEALTH — Phase 3: Preprocessing")
    logger.info(sep)

    # ── Load Phase 2 outputs ──────────────────────────────────────────────
    logger.info("[Step 1/6] Loading Phase 2 processed CSVs...")
    raw = _load_processed(config)

    # Record raw shapes and missing value counts for the report
    raw_shapes = {k: v.shape for k, v in raw.items()}
    missing_before = {
        k: {c: int(n) for c, n in v.isna().sum().items() if n > 0}
        for k, v in raw.items()
    }

    # ── Build birth map ───────────────────────────────────────────────────
    birth_map = build_birth_map(raw["patients"])

    # ── Feature engineering ───────────────────────────────────────────────
    logger.info("[Step 2/6] Feature engineering...")

    t0 = time.time()
    patients_ready = clean_patients(raw["patients"])
    logger.info("  patients done in %.1fs", time.time() - t0)

    t0 = time.time()
    encounters_ready = engineer_encounters(raw["encounters"], birth_map)
    logger.info("  encounters done in %.1fs", time.time() - t0)

    t0 = time.time()
    observations_ready = engineer_observations(
        raw["observations"], birth_map,
        top_n_loinc=config.top_n_loinc,
        min_frequency=config.min_loinc_frequency,
    )
    logger.info("  observations done in %.1fs", time.time() - t0)

    t0 = time.time()
    conditions_ready = engineer_conditions(raw["conditions"], birth_map)
    logger.info("  conditions done in %.1fs", time.time() - t0)

    t0 = time.time()
    medications_ready = engineer_medications(raw["medications"], birth_map)
    logger.info("  medications done in %.1fs", time.time() - t0)

    # ── Add patient aggregates (needs ready child tables) ─────────────────
    logger.info("[Step 3/6] Adding patient aggregate features...")
    patients_ready = add_patient_aggregates(
        patients_ready, encounters_ready, conditions_ready, medications_ready
    )

    ready: dict[str, pd.DataFrame] = {
        "patients":     patients_ready,
        "encounters":   encounters_ready,
        "observations": observations_ready,
        "conditions":   conditions_ready,
        "medications":  medications_ready,
    }

    # ── FK validation ─────────────────────────────────────────────────────
    logger.info("[Step 4/6] Validating foreign keys...")
    fk_results = _validate_foreign_keys(ready)

    # ── Write ready CSVs (checkpoint per table) ───────────────────────────
    logger.info("[Step 5/6] Writing ready CSVs...")
    ready_shapes: dict[str, tuple[int, int]] = {}
    missing_after: dict[str, dict[str, int]] = {}

    csv_names = {
        "patients":     "patients_ready.csv",
        "encounters":   "encounters_ready.csv",
        "observations": "observations_ready.csv",
        "conditions":   "conditions_ready.csv",
        "medications":  "medications_ready.csv",
    }
    for tname, csv_name in csv_names.items():
        df = ready[tname]
        out_path = config.ready_dir / csv_name
        df.to_csv(out_path, index=False)
        ready_shapes[tname] = df.shape
        missing_after[tname] = {
            c: int(n) for c, n in df.isna().sum().items() if n > 0
        }
        logger.info(
            "  [OK] %-24s %7d rows  %2d cols",
            csv_name, len(df), len(df.columns),
        )

    # ── Generate and save metadata ────────────────────────────────────────
    logger.info("[Step 6/6] Generating SDV metadata...")
    metadata = build_sdv_metadata(
        patients=patients_ready,
        encounters=encounters_ready,
        observations=observations_ready,
        conditions=conditions_ready,
        medications=medications_ready,
    )
    save_metadata(metadata, config.ready_dir / "metadata.json")
    print_metadata_summary(metadata)

    # ── Report + summary ──────────────────────────────────────────────────
    _write_report(
        raw_shapes=raw_shapes,
        ready_shapes=ready_shapes,
        fk_results=fk_results,
        missing_before=missing_before,
        missing_after=missing_after,
        config=config,
    )
    _print_summary(ready, fk_results)

    elapsed = time.time() - wall_start
    logger.info("Phase 3 complete in %.1f seconds (%.1f min).", elapsed, elapsed / 60)
