"""Distribution profiler — descriptive statistics for all numerical features.

Designed to be called twice:
  1. Before synthesis → ``data_profile_before_training.md``
  2. After synthesis  → ``data_profile_after_synthesis.md``

The two reports share an identical schema so columns align for diff / comparison.
"""

from __future__ import annotations

import logging
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# ── Vital-sign patterns (case-insensitive substring match on loinc_display) ───

_VITAL_PATTERNS: dict[str, list[str]] = {
    "BMI":             ["body mass index", "bmi [ratio]", "bmi"],
    "Height (cm)":     ["body height"],
    "Weight (kg)":     ["body weight"],
    "Systolic BP":     ["systolic blood pressure", "systolic bp"],
    "Diastolic BP":    ["diastolic blood pressure", "diastolic bp"],
    "Heart Rate":      ["heart rate"],
    "Respiratory Rate":["respiratory rate"],
    "Temperature":     ["body temperature", "oral temperature"],
    "O2 Saturation":   ["oxygen saturation", "spo2"],
    "Glucose":         ["glucose [mass", "glucose [moles"],
    "Total Cholesterol":["total cholesterol", "cholesterol [mass/volume]"],
    "HDL":             ["hdl", "cholesterol in hdl"],
    "LDL":             ["ldl", "cholesterol in ldl"],
    "Triglycerides":   ["triglycerides"],
    "eGFR":            ["glomerular filtration rate", "egfr"],
    "Hemoglobin":      ["hemoglobin [mass"],
}


# ── Core statistics function ──────────────────────────────────────────────────

def _stats(series: pd.Series, feature_name: str) -> dict | None:
    """Compute descriptive statistics for a numerical series.

    Returns None if the series is empty or has no finite values.
    """
    vals = pd.to_numeric(series, errors="coerce")
    vals = vals[np.isfinite(vals)].dropna()
    if len(vals) == 0:
        return None
    return {
        "feature":  feature_name,
        "n":        int(len(vals)),
        "mean":     round(float(vals.mean()),   3),
        "median":   round(float(vals.median()), 3),
        "std":      round(float(vals.std()),    3),
        "min":      round(float(vals.min()),    3),
        "max":      round(float(vals.max()),    3),
        "p5":       round(float(vals.quantile(0.05)), 3),
        "p25":      round(float(vals.quantile(0.25)), 3),
        "p75":      round(float(vals.quantile(0.75)), 3),
        "p95":      round(float(vals.quantile(0.95)), 3),
    }


def _profile_numerical_cols(
    df: pd.DataFrame,
    cols: list[str],
) -> list[dict]:
    """Profile all requested numerical columns from a table."""
    results = []
    for col in cols:
        if col not in df.columns:
            continue
        s = _stats(df[col], col)
        if s:
            results.append(s)
    return results


# ── Per-table profilers ───────────────────────────────────────────────────────

def profile_patients(df: pd.DataFrame) -> list[dict]:
    return _profile_numerical_cols(df, [
        "age", "age_at_death", "daly", "qaly",
        "encounter_count", "condition_count", "medication_count",
    ])


def profile_encounters(df: pd.DataFrame) -> list[dict]:
    return _profile_numerical_cols(df, [
        "days_since_birth",
        "encounter_duration_hours",
        "days_since_prev_encounter",
        "sequence_index",
    ])


def profile_conditions(df: pd.DataFrame) -> list[dict]:
    return _profile_numerical_cols(df, [
        "onset_days_since_birth",
        "abatement_days_since_birth",
        "duration_days",
        "sequence_index",
    ])


def profile_medications(df: pd.DataFrame) -> list[dict]:
    return _profile_numerical_cols(df, [
        "days_since_birth",
        "sequence_index",
    ])


def profile_vitals(df_obs: pd.DataFrame) -> list[dict]:
    """Extract and profile each vital sign from the observations table.

    Matches loinc_display against ``_VITAL_PATTERNS`` (case-insensitive
    substring). Falls back to profiling all loinc codes not caught by name
    patterns.
    """
    if "loinc_display" not in df_obs.columns or "value_quantity" not in df_obs.columns:
        return []

    loinc_lower = df_obs["loinc_display"].astype(str).str.lower()
    results: list[dict] = []
    matched_indices: set[int] = set()

    for vital_name, patterns in _VITAL_PATTERNS.items():
        mask = pd.Series(False, index=df_obs.index)
        for pat in patterns:
            mask |= loinc_lower.str.contains(pat, na=False, regex=False)
        subset = df_obs.loc[mask, "value_quantity"]
        matched_indices.update(subset.index.tolist())
        s = _stats(subset, vital_name)
        if s:
            results.append(s)

    # Profile any remaining LOINC codes not captured by named patterns
    remaining = df_obs.loc[~df_obs.index.isin(matched_indices)]
    for loinc_name, grp in remaining.groupby("loinc_display"):
        s = _stats(grp["value_quantity"], str(loinc_name))
        if s:
            results.append(s)

    return results


# ── Top-level builder ─────────────────────────────────────────────────────────

def build_distribution_profile(
    tables: dict[str, pd.DataFrame],
    label: str = "pre-synthesis",
) -> dict:
    """Build the full distribution profile for all five tables.

    Args:
        tables: Dict of {table_name: DataFrame} — the ready or synthetic tables.
        label:  Human-readable label embedded in the report ("pre-synthesis" or
                "post-synthesis"). Used to distinguish the two reports.

    Returns:
        Dict with keys: label, timestamp, patients, encounters, vitals,
        conditions, medications.
    """
    logger.info("Profiling distributions for: %s", label)
    return {
        "label":       label,
        "timestamp":   datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "patients":    profile_patients(tables["patients"]),
        "encounters":  profile_encounters(tables["encounters"]),
        "vitals":      profile_vitals(tables["observations"]),
        "conditions":  profile_conditions(tables["conditions"]),
        "medications": profile_medications(tables["medications"]),
    }


# ── Markdown report writer ────────────────────────────────────────────────────

_HEADER = "| Feature | N | Mean | Median | Std Dev | Min | Max | P5 | P95 |"
_DIVIDER = "|---------|--:|-----:|-------:|--------:|----:|----:|---:|---:|"


def _table_rows(stats_list: list[dict]) -> list[str]:
    rows = []
    for s in stats_list:
        rows.append(
            f"| {s['feature']} | {s['n']:,} "
            f"| {s['mean']:,.3f} | {s['median']:,.3f} | {s['std']:,.3f} "
            f"| {s['min']:,.3f} | {s['max']:,.3f} "
            f"| {s['p5']:,.3f} | {s['p95']:,.3f} |\n"
        )
    return rows


def save_distribution_report(profile: dict, output_path: Path) -> None:
    """Serialise the profile dict to a markdown file."""
    label     = profile["label"]
    timestamp = profile["timestamp"]

    lines = [
        f"# Data Distribution Profile — {label}\n\n",
        f"**Generated:** {timestamp}  \n",
        f"**Label:** `{label}`\n\n",
        "> This report is generated before **and** after synthesis to enable\n",
        "> side-by-side comparison of real vs. synthetic distributions.\n\n",

        "## Patients\n\n",
        _HEADER + "\n", _DIVIDER + "\n",
    ]
    lines += _table_rows(profile["patients"])

    lines += [
        "\n## Encounters\n\n",
        _HEADER + "\n", _DIVIDER + "\n",
    ]
    lines += _table_rows(profile["encounters"])

    lines += [
        "\n## Clinical Vitals (from Observations)\n\n",
        "> Values extracted from the `observations_ready` table by LOINC display name.\n\n",
        _HEADER + "\n", _DIVIDER + "\n",
    ]
    lines += _table_rows(profile["vitals"])

    lines += [
        "\n## Conditions\n\n",
        _HEADER + "\n", _DIVIDER + "\n",
    ]
    lines += _table_rows(profile["conditions"])

    lines += [
        "\n## Medications\n\n",
        _HEADER + "\n", _DIVIDER + "\n",
    ]
    lines += _table_rows(profile["medications"])

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("".join(lines), encoding="utf-8")
    logger.info("Saved: %s", output_path)


def print_distribution_summary(profile: dict) -> None:
    """Log a concise terminal summary of key vitals and patient stats."""
    sep  = "=" * 64
    thin = "-" * 64

    def _row(feat: str, stats: dict | None) -> None:
        if stats is None:
            logger.info("  %-28s  (no data)", feat)
            return
        logger.info(
            "  %-28s  mean=%8.2f  med=%8.2f  std=%7.2f",
            feat, stats["mean"], stats["median"], stats["std"],
        )

    def _find(stats_list: list[dict], name: str) -> dict | None:
        for s in stats_list:
            if s["feature"].lower() == name.lower():
                return s
        return None

    logger.info(sep)
    logger.info("  DISTRIBUTION SUMMARY — %s", profile["label"])
    logger.info(sep)

    # Patient stats
    logger.info("  PATIENTS")
    logger.info(thin)
    for feat in ("age", "age_at_death", "encounter_count", "condition_count"):
        _row(feat, _find(profile["patients"], feat))

    # Vitals
    logger.info(thin)
    logger.info("  KEY VITALS")
    logger.info(thin)
    for feat in ("BMI", "Height (cm)", "Weight (kg)",
                 "Systolic BP", "Diastolic BP", "Heart Rate"):
        _row(feat, _find(profile["vitals"], feat))

    # Encounter
    logger.info(thin)
    logger.info("  ENCOUNTERS")
    logger.info(thin)
    _row("encounter_duration_hours", _find(profile["encounters"], "encounter_duration_hours"))
    _row("days_since_prev_encounter", _find(profile["encounters"], "days_since_prev_encounter"))

    logger.info(sep)
