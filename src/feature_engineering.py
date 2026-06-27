"""Feature engineering for Phase 3 — transforms raw extracted CSVs into
model-ready DataFrames for SDV synthesis.

All functions are pure transformations: they receive a DataFrame and return
a new one. No I/O, no side effects — the orchestrator in preprocessor.py
handles reading and writing.

Temporal strategy
-----------------
All event timestamps are converted to ``days_since_birth`` (float).
This is the canonical temporal anchor throughout the project because:
  - It is timezone-independent (avoids +01:00 / +02:00 noise in Synthea data)
  - It preserves the *pattern* of a patient's clinical journey without leaking
    the absolute calendar year
  - Reconstruction is trivial: birth_date + timedelta(days=days_since_birth)
"""

from __future__ import annotations

import json
import logging

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# ── Datetime helpers ──────────────────────────────────────────────────────────

def _parse_dt(series: pd.Series) -> pd.Series:
    """Parse an ISO-8601 datetime series to timezone-naive UTC timestamps.

    Synthea emits mixed offsets (e.g. +01:00 in winter, +02:00 in summer).
    ``utc=True`` normalises everything to UTC; ``.dt.tz_localize(None)`` then
    drops the tzinfo so downstream arithmetic works without tz warnings.
    """
    return pd.to_datetime(series, utc=True, errors="coerce").dt.tz_localize(None)


def build_birth_map(patients: pd.DataFrame) -> dict[str, pd.Timestamp]:
    """Return {patient_id: birth_date_as_Timestamp} for all patients.

    Used as the temporal anchor when computing deltas on child tables.
    """
    birth_map: dict[str, pd.Timestamp] = {}
    for _, row in patients[["patient_id", "birth_date"]].dropna().iterrows():
        try:
            birth_map[str(row["patient_id"])] = pd.Timestamp(row["birth_date"])
        except Exception:
            pass
    logger.debug("birth_map built: %d patients", len(birth_map))
    return birth_map


def _days_delta(
    event_dates: pd.Series,
    patient_ids: pd.Series,
    birth_map: dict[str, pd.Timestamp],
) -> pd.Series:
    """Vectorised computation of (event_date − birth_date) in days.

    Returns a float Series aligned to the input index; NaN where birth date
    is unknown or event date is unparseable.
    """
    parsed = _parse_dt(event_dates)
    births = patient_ids.astype(str).map(birth_map)
    return (parsed - births).dt.days.astype(float).clip(lower=0)


# ── Table transformers ────────────────────────────────────────────────────────

def clean_patients(df: pd.DataFrame) -> pd.DataFrame:
    """Prepare the patients table for use as the context table in SDV.

    Transformations applied
    -----------------------
    - Drop PII columns (names, SSN, DL, passport)
    - Drop zero-variance columns (all patients are in MA, US)
    - Drop high-granularity geo columns (lat/lon, postal_code)
    - Compute ``age`` in years from birth_date
    - Compute ``is_deceased`` (0/1 integer flag)
    - Compute ``age_at_death`` for deceased patients
    - Standardise categorical values to lowercase stripped strings
    - Remove duplicate patient_ids
    """
    df = df.copy()

    # ── Drop PII and low-value columns ────────────────────────────────────
    drop_cols = [
        "family_name", "given_name",          # PII names
        "ssn", "drivers_license", "passport",  # PII identifiers
        "lat", "lon",                           # Too granular for synthesis
        "postal_code",                          # High cardinality, low value
        "country",                              # 100% US — zero variance
        "state",                                # 100% Massachusetts — zero variance
        "birth_place_city",                     # Very sparse
        "mothers_maiden_name",                  # PII (may not exist in all datasets)
    ]
    df = df.drop(columns=[c for c in drop_cols if c in df.columns])

    # ── Age features ──────────────────────────────────────────────────────
    today = pd.Timestamp.today().normalize()
    birth_dt = pd.to_datetime(df["birth_date"], errors="coerce")
    df["age"] = ((today - birth_dt).dt.days / 365.25).round(1)

    deceased_dt = _parse_dt(df["deceased_datetime"]) if "deceased_datetime" in df.columns else pd.NaT
    df["is_deceased"] = deceased_dt.notna().astype(int)
    df["age_at_death"] = np.where(
        deceased_dt.notna(),
        ((deceased_dt - birth_dt).dt.days / 365.25).round(1),
        np.nan,
    )

    # ── Drop raw date columns no longer needed ────────────────────────────
    df = df.drop(columns=["birth_date", "deceased_datetime"], errors="ignore")

    # ── Standardise categoricals ──────────────────────────────────────────
    cat_cols = ["gender", "race", "ethnicity", "marital_status",
                "state", "language", "birth_sex", "marital_code"]
    for col in cat_cols:
        if col in df.columns:
            df[col] = df[col].astype(str).str.strip().str.lower().replace("nan", pd.NA)

    # ── Deduplication ─────────────────────────────────────────────────────
    before = len(df)
    df = df.drop_duplicates(subset=["patient_id"], keep="first")
    dropped = before - len(df)
    if dropped:
        logger.warning("patients: dropped %d duplicate patient_ids", dropped)

    logger.info("patients_ready: %d rows, %d columns", len(df), len(df.columns))
    return df.reset_index(drop=True)


def add_patient_aggregates(
    patients: pd.DataFrame,
    encounters: pd.DataFrame,
    conditions: pd.DataFrame,
    medications: pd.DataFrame,
) -> pd.DataFrame:
    """Enrich the patients table with per-patient clinical volume counts.

    These aggregate features give PARSynthesizer richer context when
    generating synthetic encounter sequences — a patient with 200 encounters
    should produce a denser sequence than one with 5.
    """
    df = patients.copy()

    enc_counts  = encounters.groupby("patient_id").size().rename("encounter_count")
    cond_counts = conditions.groupby("patient_id").size().rename("condition_count")
    med_counts  = medications.groupby("patient_id").size().rename("medication_count")

    df = df.merge(enc_counts,  on="patient_id", how="left")
    df = df.merge(cond_counts, on="patient_id", how="left")
    df = df.merge(med_counts,  on="patient_id", how="left")

    for col in ("encounter_count", "condition_count", "medication_count"):
        df[col] = df[col].fillna(0).astype(int)

    logger.info("Patient aggregates added: encounter/condition/medication counts")
    return df


def engineer_encounters(
    df: pd.DataFrame,
    birth_map: dict[str, pd.Timestamp],
) -> pd.DataFrame:
    """Prepare the encounters table for sequential synthesis with PARSynthesizer.

    Key temporal features
    ---------------------
    - ``days_since_birth``          : position in patient's life timeline
    - ``encounter_duration_hours``  : length of visit
    - ``days_since_prev_encounter`` : inter-visit gap (NaN for first visit)
    - ``sequence_index``            : 0-based chronological order per patient

    These allow PARSynthesizer to learn realistic visit patterns:
    how long between visits, how long visits last, and how visit type
    evolves over a patient's life.
    """
    df = df.copy()

    # ── Parse datetimes ───────────────────────────────────────────────────
    start = _parse_dt(df["start_datetime"])
    end   = _parse_dt(df["end_datetime"])

    # ── Temporal features ─────────────────────────────────────────────────
    df["days_since_birth"] = _days_delta(df["start_datetime"], df["patient_id"], birth_map)
    df["encounter_duration_hours"] = (
        (end - start).dt.total_seconds() / 3600
    ).clip(lower=0).round(2)

    # ── Drop raw datetime columns ─────────────────────────────────────────
    df = df.drop(columns=["start_datetime", "end_datetime"], errors="ignore")

    # ── Sort chronologically per patient ──────────────────────────────────
    df = df.sort_values(["patient_id", "days_since_birth"]).reset_index(drop=True)

    # ── Sequence index and inter-visit gap ────────────────────────────────
    df["sequence_index"] = df.groupby("patient_id").cumcount()
    df["days_since_prev_encounter"] = (
        df.groupby("patient_id")["days_since_birth"].diff()
    )

    # ── Standardise categoricals ──────────────────────────────────────────
    cat_cols = ["status", "class_code", "type_display", "reason_display",
                "organization_display", "location_display", "discharge_disposition"]
    for col in cat_cols:
        if col in df.columns:
            df[col] = df[col].astype(str).str.strip().replace({"nan": pd.NA, "": pd.NA})

    # ── Drop constant and low-value columns ──────────────────────────────
    # status is always "finished" in Synthea; type_code redundant with display
    df = df.drop(
        columns=["type_code", "practitioner_npi", "reason_code", "status"],
        errors="ignore",
    )

    # ── Cap high-cardinality categoricals to top-50 + "other" ────────────
    for col in ("organization_display", "location_display"):
        if col in df.columns:
            top50 = df[col].value_counts().nlargest(50).index
            df[col] = df[col].where(df[col].isin(top50), other="other")

    # ── Deduplication ─────────────────────────────────────────────────────
    before = len(df)
    df = df.drop_duplicates(subset=["encounter_id"], keep="first")
    if before - len(df):
        logger.warning("encounters: dropped %d duplicate encounter_ids", before - len(df))

    logger.info("encounters_ready: %d rows, %d columns", len(df), len(df.columns))
    return df


def engineer_observations(
    df: pd.DataFrame,
    birth_map: dict[str, pd.Timestamp],
    top_n_loinc: int = 30,
    min_frequency: int = 100,
) -> pd.DataFrame:
    """Prepare observations for synthesis.

    Strategy
    --------
    1. Retain only ``value_type == 'quantity'`` rows (numerical lab/vital values).
       Categorical and string observations are dropped here; codeable_concept
       observations are too sparse per code to synthesise reliably.
    2. Expand blood pressure ``component`` rows into separate systolic and
       diastolic rows so they appear as numerical observations.
    3. Filter to the top ``top_n_loinc`` LOINC codes by frequency.
       Rare codes (<``min_frequency`` records) cannot be reliably synthesised.
    4. Add ``days_since_birth`` temporal anchor.
    5. Clean numerical value outliers (cap at 3 × IQR per LOINC code).

    The resulting table is suitable for either:
    - PARSynthesizer (sequential, conditioned on encounter context)
    - CTGAN with stratified sampling per LOINC code
    """
    df = df.copy()

    # ── Expand blood pressure components ──────────────────────────────────
    bp_rows = _expand_bp_components(df[df["value_type"] == "component"])

    # ── Keep only quantity observations ───────────────────────────────────
    df_qty = df[df["value_type"] == "quantity"].copy()

    # ── Merge expanded BP back in ─────────────────────────────────────────
    df_all = pd.concat([df_qty, bp_rows], ignore_index=True)
    logger.info("Observations after BP expansion: %d rows", len(df_all))

    # ── Filter to top LOINC codes ─────────────────────────────────────────
    loinc_counts = df_all["loinc_display"].value_counts()
    top_loinc = loinc_counts[loinc_counts >= min_frequency].head(top_n_loinc).index
    df_all = df_all[df_all["loinc_display"].isin(top_loinc)].copy()
    logger.info(
        "Observations after LOINC filter (top %d, min_freq=%d): %d rows",
        top_n_loinc, min_frequency, len(df_all),
    )

    # ── Temporal feature ──────────────────────────────────────────────────
    df_all["days_since_birth"] = _days_delta(
        df_all["effective_datetime"], df_all["patient_id"], birth_map
    )

    # ── Sort chronologically per patient ──────────────────────────────────
    df_all = df_all.sort_values(["patient_id", "days_since_birth"]).reset_index(drop=True)
    df_all["sequence_index"] = df_all.groupby("patient_id").cumcount()

    # ── Cap extreme outliers per LOINC code (3 × IQR) ────────────────────
    df_all = _cap_outliers(df_all, value_col="value_quantity", group_col="loinc_display")

    # ── Drop columns not needed for synthesis ─────────────────────────────
    # status is always "final" in Synthea observations — zero variance
    drop_cols = [
        "effective_datetime", "value_type", "status",
        "value_code", "value_display", "value_string", "component_json",
        "loinc_code",  # redundant with loinc_display
    ]
    df_all = df_all.drop(columns=[c for c in drop_cols if c in df_all.columns])

    # ── Standardise loinc_display and category ────────────────────────────
    df_all["loinc_display"] = df_all["loinc_display"].astype(str).str.strip()
    if "category" in df_all.columns:
        df_all["category"] = df_all["category"].astype(str).str.strip().str.lower()

    logger.info("observations_ready: %d rows, %d columns", len(df_all), len(df_all.columns))
    return df_all


def _expand_bp_components(df_comp: pd.DataFrame) -> pd.DataFrame:
    """Convert blood pressure panel component rows into individual quantity rows.

    Each BP observation has a ``component_json`` list with two entries:
    systolic (LOINC 8480-6) and diastolic (LOINC 8462-4).
    We emit one row per component, allowing the synthesiser to treat them
    as independent numerical observations.
    """
    expanded_rows: list[dict] = []
    base_cols = [c for c in df_comp.columns if c not in ("component_json", "value_type")]

    for _, row in df_comp.iterrows():
        try:
            components = json.loads(row["component_json"])
        except (TypeError, json.JSONDecodeError):
            continue

        for comp_idx, comp in enumerate(components):
            code_data = comp.get("code", {}).get("coding", [{}])[0]
            qty = comp.get("valueQuantity", {})
            value = qty.get("value")
            if value is None:
                continue
            new_row = {col: row[col] for col in base_cols if col in row}
            # Append component index to make observation_id unique after expansion
            new_row["observation_id"] = f"{row['observation_id']}_c{comp_idx}"
            new_row.update({
                "loinc_code":    code_data.get("code", ""),
                "loinc_display": code_data.get("display", "Blood Pressure Component"),
                "value_quantity": float(value),
                "value_unit":    qty.get("unit", "mm[Hg]"),
                "value_type":    "quantity",
            })
            expanded_rows.append(new_row)

    return pd.DataFrame(expanded_rows)


def _cap_outliers(
    df: pd.DataFrame,
    value_col: str,
    group_col: str,
    factor: float = 3.0,
) -> pd.DataFrame:
    """Cap extreme values at Q1 − factor×IQR and Q3 + factor×IQR per group.

    Prevents CTGAN/PAR from wasting model capacity on extreme outliers that
    likely represent data entry errors rather than true clinical values.
    """
    df = df.copy()
    for name, grp in df.groupby(group_col):
        q1 = grp[value_col].quantile(0.25)
        q3 = grp[value_col].quantile(0.75)
        iqr = q3 - q1
        lo, hi = q1 - factor * iqr, q3 + factor * iqr
        mask = df[group_col] == name
        df.loc[mask, value_col] = df.loc[mask, value_col].clip(lo, hi)
    return df


def engineer_conditions(
    df: pd.DataFrame,
    birth_map: dict[str, pd.Timestamp],
) -> pd.DataFrame:
    """Prepare conditions table with temporal and clinical features.

    New features
    ------------
    - ``onset_days_since_birth``     : temporal position of diagnosis
    - ``abatement_days_since_birth`` : when condition resolved (NaN = ongoing)
    - ``duration_days``              : condition duration (NaN = chronic/active)
    - ``is_chronic``                 : 1 if no abatement date (ongoing condition)
    - ``sequence_index``             : per-patient chronological order
    """
    df = df.copy()

    # ── Temporal features ─────────────────────────────────────────────────
    df["onset_days_since_birth"] = _days_delta(
        df["onset_datetime"], df["patient_id"], birth_map
    )
    df["abatement_days_since_birth"] = _days_delta(
        df["abatement_datetime"], df["patient_id"], birth_map
    )
    df["duration_days"] = (
        df["abatement_days_since_birth"] - df["onset_days_since_birth"]
    ).clip(lower=0)
    df["is_chronic"] = df["abatement_datetime"].isna().astype(int)

    # ── Drop raw datetime columns ─────────────────────────────────────────
    df = df.drop(
        columns=["onset_datetime", "abatement_datetime", "recorded_date"],
        errors="ignore",
    )

    # ── Sort and index ────────────────────────────────────────────────────
    df = df.sort_values(["patient_id", "onset_days_since_birth"]).reset_index(drop=True)
    df["sequence_index"] = df.groupby("patient_id").cumcount()

    # ── Standardise categoricals ──────────────────────────────────────────
    for col in ("clinical_status", "verification_status", "category", "snomed_display"):
        if col in df.columns:
            df[col] = df[col].astype(str).str.strip().replace({"nan": pd.NA, "": pd.NA})

    # ── Drop constant and redundant columns ───────────────────────────────
    # category is always "encounter" in Synthea; snomed_code redundant with display
    df = df.drop(
        columns=["snomed_code", "verification_status", "category"],
        errors="ignore",
    )

    # ── Deduplication ─────────────────────────────────────────────────────
    before = len(df)
    df = df.drop_duplicates(subset=["condition_id"], keep="first")
    if before - len(df):
        logger.warning("conditions: dropped %d duplicate condition_ids", before - len(df))

    logger.info("conditions_ready: %d rows, %d columns", len(df), len(df.columns))
    return df


def engineer_medications(
    df: pd.DataFrame,
    birth_map: dict[str, pd.Timestamp],
) -> pd.DataFrame:
    """Prepare medications table with temporal and clinical features.

    New features
    ------------
    - ``days_since_birth`` : when the prescription was written
    - ``is_active``        : 1 if status == 'active'
    - ``sequence_index``   : per-patient chronological order

    Dropped columns
    ---------------
    - ``rxnorm_code``       : redundant with rxnorm_display
    - ``requester_display`` : high cardinality identifier, no synthesis value
    - ``dosage_text``       : free text — not synthesisable with tabular models
    """
    df = df.copy()

    # ── Temporal feature ──────────────────────────────────────────────────
    df["days_since_birth"] = _days_delta(
        df["authored_on"], df["patient_id"], birth_map
    )
    df = df.drop(columns=["authored_on"], errors="ignore")

    # ── Clinical flag ─────────────────────────────────────────────────────
    df["is_active"] = (
        df["status"].astype(str).str.lower().str.strip() == "active"
    ).astype(int)

    # ── Sort and index ────────────────────────────────────────────────────
    df = df.sort_values(["patient_id", "days_since_birth"]).reset_index(drop=True)
    df["sequence_index"] = df.groupby("patient_id").cumcount()

    # ── Standardise categoricals ──────────────────────────────────────────
    for col in ("status", "intent", "category", "rxnorm_display", "reason_display"):
        if col in df.columns:
            df[col] = df[col].astype(str).str.strip().replace({"nan": pd.NA, "": pd.NA})

    # ── Drop constant and low-value columns ──────────────────────────────
    # intent is always "order" in Synthea; rxnorm_code redundant with display
    df = df.drop(
        columns=["rxnorm_code", "requester_display", "dosage_text", "intent"],
        errors="ignore",
    )

    # ── Deduplication ─────────────────────────────────────────────────────
    before = len(df)
    df = df.drop_duplicates(subset=["medication_id"], keep="first")
    if before - len(df):
        logger.warning("medications: dropped %d duplicate medication_ids", before - len(df))

    logger.info("medications_ready: %d rows, %d columns", len(df), len(df.columns))
    return df
