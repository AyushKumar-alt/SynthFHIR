"""Module B — Numeric evaluation: descriptive stats, KS test, histograms, boxplots."""

from __future__ import annotations

import logging
import re
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

from .config import EXCLUDED_COLUMNS
from .loader import TablePair

logger = logging.getLogger(__name__)

# Columns whose uniqueness ratio approaches 1.0 are IDs — skip evaluation.
_ID_UNIQUENESS_THRESHOLD = 0.95

# Minimum non-null values needed to compute meaningful statistics.
_MIN_VALUES = 10


def run_numeric_evaluation(
    pairs:     dict[str, TablePair],
    plots_dir: Path,
) -> dict[str, list[dict]]:
    """Run numeric evaluation for every table.

    Returns a dict keyed by table name; values are lists of per-column
    result dicts.
    """
    all_results: dict[str, list[dict]] = {}
    for table_name, (original, synthetic) in pairs.items():
        if original is None or synthetic is None:
            logger.warning("[NUMERIC] %s — missing data, skipped", table_name)
            all_results[table_name] = []
            continue
        results = _evaluate_table(original, synthetic, table_name, plots_dir / "numeric")
        logger.info("[NUMERIC] %s — evaluated %d numeric columns", table_name, len(results))
        all_results[table_name] = results
    return all_results


# ── Per-table logic ───────────────────────────────────────────────────────────

def _evaluate_table(
    original:   pd.DataFrame,
    synthetic:  pd.DataFrame,
    table_name: str,
    plots_dir:  Path,
) -> list[dict]:
    num_cols = _numeric_columns(original, synthetic)
    results  = []

    for col in num_cols:
        real_vals  = pd.to_numeric(original[col],  errors="coerce").dropna()
        synth_vals = pd.to_numeric(synthetic[col], errors="coerce").dropna()

        if len(real_vals) < _MIN_VALUES or len(synth_vals) < _MIN_VALUES:
            continue

        ks_stat, ks_p = stats.ks_2samp(real_vals.values, synth_vals.values)

        result = {
            "table":      table_name,
            "column":     col,
            "ks_stat":    round(float(ks_stat), 4),
            "ks_p_value": round(float(ks_p), 4),
            "similarity": round(1.0 - float(ks_stat), 4),
            "real":       _desc_stats(real_vals),
            "synthetic":  _desc_stats(synth_vals),
        }

        safe = _safe_name(col)
        _plot_histogram(real_vals, synth_vals, col, table_name, plots_dir, safe)
        _plot_boxplot(  real_vals, synth_vals, col, table_name, plots_dir, safe)

        results.append(result)

    return results


# ── Column selection ──────────────────────────────────────────────────────────

def _numeric_columns(original: pd.DataFrame, synthetic: pd.DataFrame) -> list[str]:
    """Return numeric columns present in both frames, excluding IDs."""
    num = [
        c for c in original.select_dtypes(include="number").columns
        if c in synthetic.columns
        and c not in EXCLUDED_COLUMNS
        and not _is_id_like(original[c])
    ]
    return num


def _is_id_like(series: pd.Series) -> bool:
    n_total  = len(series.dropna())
    n_unique = series.nunique(dropna=True)
    if n_total == 0:
        return False
    return (n_unique / n_total) >= _ID_UNIQUENESS_THRESHOLD


# ── Descriptive statistics ────────────────────────────────────────────────────

def _desc_stats(s: pd.Series) -> dict:
    return {
        "count":  int(len(s)),
        "mean":   _r(s.mean()),
        "std":    _r(s.std()),
        "min":    _r(s.min()),
        "p25":    _r(s.quantile(0.25)),
        "median": _r(s.median()),
        "p75":    _r(s.quantile(0.75)),
        "max":    _r(s.max()),
    }


def _r(v) -> float:
    return round(float(v), 4)


# ── Plots ─────────────────────────────────────────────────────────────────────

def _plot_histogram(
    real:       pd.Series,
    synth:      pd.Series,
    col:        str,
    table_name: str,
    plots_dir:  Path,
    safe_col:   str,
) -> None:
    try:
        plots_dir.mkdir(parents=True, exist_ok=True)
        combined = pd.concat([real, synth])
        lo, hi   = combined.quantile(0.01), combined.quantile(0.99)
        if lo == hi:
            hi = lo + 1

        fig, ax = plt.subplots(figsize=(8, 4))
        ax.hist(
            real.clip(lo, hi),
            bins=40, alpha=0.55, density=True,
            label="Original",  color="#2196F3",
        )
        ax.hist(
            synth.clip(lo, hi),
            bins=40, alpha=0.55, density=True,
            label="Synthetic", color="#FF5722",
        )
        ax.set_title(f"{table_name} — {col} (Distribution)", fontsize=11)
        ax.set_xlabel(col)
        ax.set_ylabel("Density")
        ax.legend(framealpha=0.8)
        fig.tight_layout()
        fig.savefig(plots_dir / f"{table_name}_{safe_col}_hist.png", dpi=100)
        plt.close(fig)
    except Exception as exc:
        logger.debug("Histogram skipped for %s.%s: %s", table_name, col, exc)


def _plot_boxplot(
    real:       pd.Series,
    synth:      pd.Series,
    col:        str,
    table_name: str,
    plots_dir:  Path,
    safe_col:   str,
) -> None:
    try:
        plots_dir.mkdir(parents=True, exist_ok=True)
        fig, ax = plt.subplots(figsize=(6, 4))
        bp = ax.boxplot(
            [real.values, synth.values],
            labels=["Original", "Synthetic"],
            patch_artist=True,
            notch=False,
            flierprops=dict(marker=".", markersize=3, alpha=0.4),
        )
        bp["boxes"][0].set_facecolor("#BBDEFB")
        bp["boxes"][1].set_facecolor("#FFCCBC")
        ax.set_title(f"{table_name} — {col} (Boxplot)", fontsize=11)
        ax.set_ylabel(col)
        fig.tight_layout()
        fig.savefig(plots_dir / f"{table_name}_{safe_col}_box.png", dpi=100)
        plt.close(fig)
    except Exception as exc:
        logger.debug("Boxplot skipped for %s.%s: %s", table_name, col, exc)


def _safe_name(s: str) -> str:
    return re.sub(r"[^\w\-]", "_", s)[:60]
