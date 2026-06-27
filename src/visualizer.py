"""Chart generation for Phase 2 dataset exploration.

All charts are saved as PNG files to outputs/figures/.
Uses the non-interactive Agg backend so it runs without a display.
"""

from __future__ import annotations

import logging
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd

from .config_loader import Config

logger = logging.getLogger(__name__)

# ── Shared style ──────────────────────────────────────────────────────────────

PALETTE = {
    "blue":   "#2196F3",
    "green":  "#4CAF50",
    "red":    "#F44336",
    "orange": "#FF9800",
    "purple": "#9C27B0",
    "teal":   "#00BCD4",
    "pink":   "#E91E63",
    "slate":  "#607D8B",
    "indigo": "#3F51B5",
}

RC = {
    "axes.spines.top":   False,
    "axes.spines.right": False,
    "font.size":         11,
    "axes.titlesize":    13,
    "axes.titleweight":  "bold",
}


def _save(fig: plt.Figure, out_dir: Path, filename: str) -> None:
    """Save figure and close it immediately to free memory."""
    fpath = out_dir / filename
    fig.savefig(fpath, dpi=120, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved chart: %s", fpath)


# ── Individual chart functions ────────────────────────────────────────────────

def plot_resource_distribution(resource_csv: Path, out_dir: Path) -> None:
    """Horizontal bar chart of all FHIR resource type counts."""
    df = pd.read_csv(resource_csv)
    df = df.sort_values("total_records").tail(20)

    with plt.rc_context(RC):
        fig, ax = plt.subplots(figsize=(10, 7))
        bars = ax.barh(df["resource_type"], df["total_records"],
                       color=PALETTE["blue"], edgecolor="white")
        ax.set_xlabel("Total Records")
        ax.set_title("FHIR Resource Distribution")
        ax.xaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{x:,.0f}"))
        for bar in bars:
            width = bar.get_width()
            ax.text(width * 1.01, bar.get_y() + bar.get_height() / 2,
                    f"{int(width):,}", va="center", fontsize=9)
        _save(fig, out_dir, "resource_distribution.png")


def plot_age_histogram(patients_csv: Path, out_dir: Path) -> None:
    """Histogram of patient ages."""
    patients = pd.read_csv(patients_csv)
    today = pd.Timestamp.today()
    ages = ((today - pd.to_datetime(patients["birth_date"], errors="coerce")).dt.days / 365.25).dropna()

    with plt.rc_context(RC):
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.hist(ages, bins=30, color=PALETTE["green"], edgecolor="white", alpha=0.85)
        ax.axvline(ages.mean(), color="red", linestyle="--", linewidth=1.5,
                   label=f"Mean: {ages.mean():.1f} yrs")
        ax.axvline(ages.median(), color="orange", linestyle="--", linewidth=1.5,
                   label=f"Median: {ages.median():.1f} yrs")
        ax.set_xlabel("Age (years)")
        ax.set_ylabel("Number of Patients")
        ax.set_title("Patient Age Distribution")
        ax.legend()
        _save(fig, out_dir, "age_histogram.png")


def plot_gender_distribution(patients_csv: Path, out_dir: Path) -> None:
    """Pie chart of patient gender split."""
    patients = pd.read_csv(patients_csv)
    counts = patients["gender"].value_counts()

    with plt.rc_context(RC):
        fig, ax = plt.subplots(figsize=(6, 6))
        ax.pie(
            counts,
            labels=[f"{k}\n({v:,})" for k, v in counts.items()],
            autopct="%1.1f%%",
            colors=[PALETTE["blue"], PALETTE["red"], PALETTE["purple"]],
            startangle=90,
            wedgeprops={"edgecolor": "white", "linewidth": 2},
        )
        ax.set_title("Gender Distribution")
        _save(fig, out_dir, "gender_distribution.png")


def plot_race_distribution(patients_csv: Path, out_dir: Path) -> None:
    """Horizontal bar chart of race distribution."""
    patients = pd.read_csv(patients_csv)
    counts = patients["race"].value_counts()

    with plt.rc_context(RC):
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.barh(counts.index, counts.values, color=PALETTE["orange"], edgecolor="white")
        ax.set_xlabel("Count")
        ax.set_title("Race Distribution")
        for i, (label, val) in enumerate(zip(counts.index, counts.values)):
            ax.text(val * 1.01, i, f"{val:,}", va="center", fontsize=9)
        _save(fig, out_dir, "race_distribution.png")


def plot_top_conditions(conditions_csv: Path, out_dir: Path, top_n: int = 20) -> None:
    """Horizontal bar chart of top N conditions by frequency."""
    df = pd.read_csv(conditions_csv)
    counts = df["snomed_display"].value_counts().head(top_n).sort_values()

    with plt.rc_context(RC):
        fig, ax = plt.subplots(figsize=(11, 9))
        ax.barh(counts.index, counts.values, color=PALETTE["pink"], edgecolor="white")
        ax.set_xlabel("Count")
        ax.set_title(f"Top {top_n} Conditions (SNOMED)")
        _save(fig, out_dir, "top_conditions.png")


def plot_top_medications(medications_csv: Path, out_dir: Path, top_n: int = 20) -> None:
    """Horizontal bar chart of top N medications by prescription frequency."""
    df = pd.read_csv(medications_csv)
    counts = df["rxnorm_display"].value_counts().head(top_n).sort_values()

    with plt.rc_context(RC):
        fig, ax = plt.subplots(figsize=(11, 9))
        ax.barh(counts.index, counts.values, color=PALETTE["purple"], edgecolor="white")
        ax.set_xlabel("Count")
        ax.set_title(f"Top {top_n} Medications (RxNorm)")
        _save(fig, out_dir, "top_medications.png")


def plot_top_observations(observations_csv: Path, out_dir: Path, top_n: int = 20) -> None:
    """Horizontal bar chart of top N LOINC observation codes."""
    df = pd.read_csv(observations_csv, usecols=["loinc_display"])
    counts = df["loinc_display"].value_counts().head(top_n).sort_values()

    with plt.rc_context(RC):
        fig, ax = plt.subplots(figsize=(11, 9))
        ax.barh(counts.index, counts.values, color=PALETTE["teal"], edgecolor="white")
        ax.set_xlabel("Count")
        ax.set_title(f"Top {top_n} Observations (LOINC)")
        _save(fig, out_dir, "top_observations.png")


def plot_encounter_class(encounters_csv: Path, out_dir: Path) -> None:
    """Bar chart of encounter class distribution (AMB / EMER / IMP / HH / VR)."""
    df = pd.read_csv(encounters_csv, usecols=["class_code"])
    labels_map = {
        "AMB": "Ambulatory", "EMER": "Emergency",
        "IMP": "Inpatient", "HH": "Home Health", "VR": "Virtual",
    }
    counts = df["class_code"].map(labels_map).fillna(df["class_code"]).value_counts()
    colors = [PALETTE["blue"], PALETTE["red"], PALETTE["orange"],
              PALETTE["green"], PALETTE["purple"]]

    with plt.rc_context(RC):
        fig, ax = plt.subplots(figsize=(8, 5))
        bars = ax.bar(counts.index, counts.values,
                      color=colors[:len(counts)], edgecolor="white")
        ax.set_ylabel("Count")
        ax.set_title("Encounter Class Distribution")
        ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{x:,.0f}"))
        for bar in bars:
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() * 1.01,
                    f"{int(bar.get_height()):,}", ha="center", fontsize=9)
        _save(fig, out_dir, "encounter_class.png")


def plot_encounters_per_patient(encounters_csv: Path, out_dir: Path) -> None:
    """Histogram showing distribution of encounter counts per patient."""
    df = pd.read_csv(encounters_csv, usecols=["patient_id"])
    counts = df.groupby("patient_id").size()

    with plt.rc_context(RC):
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.hist(counts, bins=40, color=PALETTE["slate"], edgecolor="white", alpha=0.85)
        ax.axvline(counts.mean(), color="red", linestyle="--", linewidth=1.5,
                   label=f"Mean: {counts.mean():.1f}")
        ax.axvline(counts.median(), color="orange", linestyle="--", linewidth=1.5,
                   label=f"Median: {counts.median():.0f}")
        ax.set_xlabel("Encounters per Patient")
        ax.set_ylabel("Number of Patients")
        ax.set_title("Encounters per Patient Distribution")
        ax.legend()
        _save(fig, out_dir, "encounters_per_patient.png")


def plot_observations_timeline(observations_csv: Path, out_dir: Path) -> None:
    """Area chart of observation volume by calendar year."""
    df = pd.read_csv(observations_csv, usecols=["effective_datetime"])
    df["year"] = pd.to_datetime(df["effective_datetime"], errors="coerce", utc=True).dt.year
    year_counts = (
        df["year"]
        .value_counts()
        .sort_index()
        .loc[lambda s: (s.index >= 1940) & (s.index <= 2025)]
    )

    with plt.rc_context(RC):
        fig, ax = plt.subplots(figsize=(12, 5))
        ax.fill_between(year_counts.index, year_counts.values,
                        color=PALETTE["indigo"], alpha=0.75)
        ax.plot(year_counts.index, year_counts.values,
                color=PALETTE["indigo"], linewidth=1)
        ax.set_xlabel("Year")
        ax.set_ylabel("Observation Count")
        ax.set_title("Observations Over Time")
        ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{x:,.0f}"))
        _save(fig, out_dir, "observations_timeline.png")


# ── Public entry point ────────────────────────────────────────────────────────

def run(config: Config) -> None:
    """Generate all ten charts and save them to outputs/figures/."""
    logger.info("Generating visualizations...")
    out   = config.figures_dir
    proc  = config.processed_dir
    rep   = config.reports_dir
    top_n = config.top_n

    plot_resource_distribution(rep  / "resource_summary.csv",   out)
    plot_age_histogram         (proc / "patients.csv",           out)
    plot_gender_distribution   (proc / "patients.csv",           out)
    plot_race_distribution     (proc / "patients.csv",           out)
    plot_top_conditions        (proc / "conditions.csv",         out, top_n)
    plot_top_medications       (proc / "medications.csv",        out, top_n)
    plot_top_observations      (proc / "observations.csv",       out, top_n)
    plot_encounter_class       (proc / "encounters.csv",         out)
    plot_encounters_per_patient(proc / "encounters.csv",         out)
    plot_observations_timeline (proc / "observations.csv",       out)

    logger.info("All charts saved to: %s", out)
