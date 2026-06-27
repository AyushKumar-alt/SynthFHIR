"""Dataset profiler — computes descriptive statistics from the extracted CSVs.

Outputs:
  outputs/reports/dataset_profile.json  — machine-readable full profile
  outputs/reports/dataset_profile.md    — human-readable summary report
"""

from __future__ import annotations

import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .config_loader import Config

logger = logging.getLogger(__name__)


# ── Age helper ────────────────────────────────────────────────────────────────

def _compute_ages(birth_date_series: pd.Series) -> pd.Series:
    """Return a Series of ages in years from a YYYY-MM-DD birth date Series."""
    today = pd.Timestamp.today()
    parsed = pd.to_datetime(birth_date_series, errors="coerce")
    return ((today - parsed).dt.days / 365.25).dropna()


# ── Core profile builder ──────────────────────────────────────────────────────

def build_profile(config: Config) -> dict[str, Any]:
    """Load all five CSVs and compute a comprehensive statistical profile.

    Args:
        config: Runtime configuration.

    Returns:
        Nested dict containing all profile metrics.
    """
    proc = config.processed_dir
    top_n = config.top_n

    patients    = pd.read_csv(proc / "patients.csv",    low_memory=False)
    encounters  = pd.read_csv(proc / "encounters.csv",  low_memory=False)
    observations= pd.read_csv(proc / "observations.csv",low_memory=False)
    conditions  = pd.read_csv(proc / "conditions.csv",  low_memory=False)
    medications = pd.read_csv(proc / "medications.csv", low_memory=False)

    ages = _compute_ages(patients["birth_date"])

    # Per-patient counts
    enc_per_pt  = encounters.groupby("patient_id").size()
    obs_per_pt  = observations.groupby("patient_id").size()
    cond_per_pt = conditions.groupby("patient_id").size()
    med_per_pt  = medications.groupby("patient_id").size()

    def dist(series: pd.Series) -> dict[str, int]:
        return series.value_counts().head(top_n).to_dict()

    def age_stats(s: pd.Series) -> dict[str, float]:
        return {
            "mean":   round(float(s.mean()), 1),
            "median": round(float(s.median()), 1),
            "std":    round(float(s.std()), 1),
            "min":    round(float(s.min()), 1),
            "max":    round(float(s.max()), 1),
            "p25":    round(float(np.percentile(s, 25)), 1),
            "p75":    round(float(np.percentile(s, 75)), 1),
            "p95":    round(float(np.percentile(s, 95)), 1),
        }

    def avg_stats(s: pd.Series) -> dict[str, float]:
        return {
            "mean":   round(float(s.mean()), 1),
            "median": round(float(s.median()), 1),
            "max":    int(s.max()),
            "p95":    round(float(np.percentile(s, 95)), 1),
        }

    # Missing value counts per table
    def missing(df: pd.DataFrame) -> dict[str, int]:
        counts = df.isna().sum()
        return {col: int(n) for col, n in counts.items() if n > 0}

    profile: dict[str, Any] = {
        "generated_at": datetime.now().isoformat(),
        "dataset": {
            "total_patients":   int(len(patients)),
            "deceased_patients":int(patients["deceased_datetime"].notna().sum()),
            "living_patients":  int(patients["deceased_datetime"].isna().sum()),
            "total_encounters": int(len(encounters)),
            "total_observations": int(len(observations)),
            "total_conditions": int(len(conditions)),
            "total_medications":int(len(medications)),
        },
        "age": age_stats(ages),
        "gender":    patients["gender"].value_counts().to_dict(),
        "race":      patients["race"].value_counts().to_dict(),
        "ethnicity": patients["ethnicity"].value_counts().to_dict(),
        "marital_status": patients["marital_status"].value_counts().to_dict(),
        "state": patients["state"].value_counts().to_dict(),
        "encounters_per_patient":    avg_stats(enc_per_pt),
        "observations_per_patient":  avg_stats(obs_per_pt),
        "conditions_per_patient":    avg_stats(cond_per_pt),
        "medications_per_patient":   avg_stats(med_per_pt),
        "top_conditions":      dist(conditions["snomed_display"]),
        "top_observations":    dist(observations["loinc_display"]),
        "top_medications":     dist(medications["rxnorm_display"]),
        "top_encounter_types": dist(encounters["type_display"]),
        "encounter_classes":   encounters["class_code"].value_counts().to_dict(),
        "observation_categories": observations["category"].value_counts().to_dict(),
        "observation_value_types":  observations["value_type"].value_counts().to_dict(),
        "condition_clinical_statuses": conditions["clinical_status"].value_counts().to_dict(),
        "medication_statuses":       medications["status"].value_counts().to_dict(),
        "missing_values": {
            "patients":     missing(patients),
            "encounters":   missing(encounters),
            "observations": missing(observations),
            "conditions":   missing(conditions),
            "medications":  missing(medications),
        },
    }

    logger.info(
        "Profile built — patients=%d | encounters=%d | observations=%d",
        profile["dataset"]["total_patients"],
        profile["dataset"]["total_encounters"],
        profile["dataset"]["total_observations"],
    )
    return profile


# ── Serialisation ─────────────────────────────────────────────────────────────

def _json_safe(obj: Any) -> Any:
    """Convert numpy scalars to native Python types for JSON serialisation."""
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.floating):
        return float(obj)
    raise TypeError(f"Cannot serialise {type(obj)}")


def save_profile(profile: dict[str, Any], config: Config) -> None:
    """Write dataset_profile.json and dataset_profile.md."""
    # ── JSON ──────────────────────────────────────────────────────────────
    json_path = config.reports_dir / "dataset_profile.json"
    with open(json_path, "w", encoding="utf-8") as fh:
        json.dump(profile, fh, indent=2, default=_json_safe)
    logger.info("Saved: %s", json_path)

    # ── Markdown ──────────────────────────────────────────────────────────
    d   = profile["dataset"]
    age = profile["age"]
    enc = profile["encounters_per_patient"]
    obs = profile["observations_per_patient"]
    cnd = profile["conditions_per_patient"]
    med = profile["medications_per_patient"]

    def _table(rows: list[tuple[str, Any]]) -> str:
        return (
            "| Metric | Value |\n|--------|-------|\n"
            + "".join(f"| {k} | {v} |\n" for k, v in rows)
        )

    def _ranked(d: dict) -> str:
        return "".join(f"{i}. {k} ({v:,})\n" for i, (k, v) in enumerate(d.items(), 1))

    md = "\n".join([
        "# Dataset Profile",
        f"\n**Generated:** {profile['generated_at']}\n",

        "## Summary\n",
        _table([
            ("Total Patients",    f"{d['total_patients']:,}"),
            ("Living",            f"{d['living_patients']:,}"),
            ("Deceased",          f"{d['deceased_patients']:,}"),
            ("Total Encounters",  f"{d['total_encounters']:,}"),
            ("Total Observations",f"{d['total_observations']:,}"),
            ("Total Conditions",  f"{d['total_conditions']:,}"),
            ("Total Medications", f"{d['total_medications']:,}"),
        ]),

        "\n## Age Distribution\n",
        _table([
            ("Mean",   f"{age['mean']} yrs"),
            ("Median", f"{age['median']} yrs"),
            ("Std Dev",f"{age['std']} yrs"),
            ("Min",    f"{age['min']} yrs"),
            ("Max",    f"{age['max']} yrs"),
            ("P25",    f"{age['p25']} yrs"),
            ("P75",    f"{age['p75']} yrs"),
            ("P95",    f"{age['p95']} yrs"),
        ]),

        "\n## Per-Patient Averages\n",
        _table([
            ("Avg Encounters",   f"{enc['mean']} (max {enc['max']})"),
            ("Avg Observations", f"{obs['mean']} (max {obs['max']})"),
            ("Avg Conditions",   f"{cnd['mean']} (max {cnd['max']})"),
            ("Avg Medications",  f"{med['mean']} (max {med['max']})"),
        ]),

        "\n## Gender Distribution\n",
        "".join(f"- {k}: {v:,}\n" for k, v in profile["gender"].items()),

        "\n## Race Distribution\n",
        "".join(f"- {k}: {v:,}\n" for k, v in profile["race"].items()),

        f"\n## Top {len(profile['top_conditions'])} Conditions\n",
        _ranked(profile["top_conditions"]),

        f"\n## Top {len(profile['top_medications'])} Medications\n",
        _ranked(profile["top_medications"]),

        f"\n## Top {len(profile['top_observations'])} Observations\n",
        _ranked(profile["top_observations"]),
    ])

    md_path = config.reports_dir / "dataset_profile.md"
    md_path.write_text(md + "\n", encoding="utf-8")
    logger.info("Saved: %s", md_path)


# ── Console summary ───────────────────────────────────────────────────────────

def print_summary(profile: dict[str, Any]) -> None:
    """Log the concise dataset summary block that confirms parser correctness."""
    d   = profile["dataset"]
    age = profile["age"]
    enc = profile["encounters_per_patient"]
    obs = profile["observations_per_patient"]
    cnd = profile["conditions_per_patient"]
    med = profile["medications_per_patient"]

    top_cond = next(iter(profile["top_conditions"]), "N/A")
    top_med  = next(iter(profile["top_medications"]), "N/A")

    sep = "=" * 56
    logger.info(sep)
    logger.info("  DATASET SUMMARY")
    logger.info(sep)
    logger.info("  Patients:                     %6d", d["total_patients"])
    logger.info("  Living / Deceased:            %6d / %d",
                d["living_patients"], d["deceased_patients"])
    logger.info("  Encounters:                   %6d", d["total_encounters"])
    logger.info("  Observations:                 %6d", d["total_observations"])
    logger.info("  Conditions:                   %6d", d["total_conditions"])
    logger.info("  Medications:                  %6d", d["total_medications"])
    logger.info(sep)
    logger.info("  Average Age:                  %.1f yrs", age["mean"])
    logger.info("  Avg Encounters / Patient:     %.1f  (max %d)", enc["mean"], enc["max"])
    logger.info("  Avg Observations / Patient:   %.1f  (max %d)", obs["mean"], obs["max"])
    logger.info("  Avg Conditions / Patient:     %.1f  (max %d)", cnd["mean"], cnd["max"])
    logger.info("  Avg Medications / Patient:    %.1f  (max %d)", med["mean"], med["max"])
    logger.info(sep)
    logger.info("  Most Common Condition:        %s", top_cond)
    logger.info("  Most Common Medication:       %s", top_med)
    logger.info(sep)


# ── Public entry point ────────────────────────────────────────────────────────

def run(config: Config) -> None:
    """Build profile, save reports, and print summary."""
    logger.info("Starting dataset profiling...")
    profile = build_profile(config)
    save_profile(profile, config)
    print_summary(profile)
    logger.info("Profiling complete.")
