"""Module G — KS Test evaluation (dedicated standalone output).

Produces ks_results.csv and ks_results.json with per-column
Kolmogorov-Smirnov statistics across every numeric column.

KS statistic interpretation:
  0.0 — identical CDFs
  1.0 — completely disjoint distributions
  similarity = 1 − ks_stat  (higher is better)
"""

from __future__ import annotations

import json
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

_MIN_VALUES            = 10
_ID_UNIQUENESS_THRESHOLD = 0.95


def run_ks_evaluation(
    pairs:      dict[str, TablePair],
    output_dir: Path,
    plots_dir:  Path,
) -> dict[str, dict]:
    """Run dedicated KS evaluation for all tables.

    Parameters
    ----------
    pairs       : table-name → (original_df, synthetic_df)
    output_dir  : outputs/evaluation/  — where CSV/JSON are written
    plots_dir   : outputs/evaluation/plots/ — parent of plots/ks/

    Returns
    -------
    dict keyed by table name::

        {
            "status":          "ok" | "missing_data",
            "columns":         [per-column dicts],
            "mean_ks_stat":    float | None,
            "mean_similarity": float | None,
        }

    Side-effects
    -----------
    Writes output_dir/ks_results.csv and output_dir/ks_results.json.
    Writes KDE PNGs under plots_dir/ks/.
    """
    all_results: dict[str, dict] = {}
    flat_rows:   list[dict]      = []

    for table_name, (original, synthetic) in pairs.items():
        if original is None or synthetic is None:
            logger.warning("[KS EVAL] %s — missing data, skipped", table_name)
            all_results[table_name] = {"status": "missing_data", "columns": []}
            continue

        cols        = _numeric_columns(original, synthetic)
        col_results = []

        for col in cols:
            real_s  = _safe_numeric(pd.to_numeric(original[col],  errors="coerce").dropna())
            synth_s = _safe_numeric(pd.to_numeric(synthetic[col], errors="coerce").dropna())

            if len(real_s) < _MIN_VALUES or len(synth_s) < _MIN_VALUES:
                continue

            ks_stat, ks_p = stats.ks_2samp(real_s.values, synth_s.values)

            row = {
                "table":       table_name,
                "column":      col,
                "ks_stat":     round(float(ks_stat), 4),
                "ks_p_value":  round(float(ks_p),    6),
                "similarity":  round(1.0 - float(ks_stat), 4),
                "n_real":      int(len(real_s)),
                "n_synthetic": int(len(synth_s)),
                "mean_real":   round(float(real_s.mean()),  4),
                "mean_synth":  round(float(synth_s.mean()), 4),
                "std_real":    round(float(real_s.std()),   4),
                "std_synth":   round(float(synth_s.std()),  4),
            }
            col_results.append(row)
            flat_rows.append(row)

            _plot_kde(real_s, synth_s, col, table_name,
                      plots_dir / "ks", _safe_name(col))

        ks_vals  = [r["ks_stat"]    for r in col_results]
        sim_vals = [r["similarity"] for r in col_results]

        mean_ks  = round(float(np.mean(ks_vals)),  4) if ks_vals  else None
        mean_sim = round(float(np.mean(sim_vals)), 4) if sim_vals else None

        all_results[table_name] = {
            "status":          "ok",
            "columns":         col_results,
            "mean_ks_stat":    mean_ks,
            "mean_similarity": mean_sim,
        }
        logger.info(
            "[KS EVAL] %-14s  columns=%d  mean_ks=%s  mean_sim=%s",
            table_name,
            len(col_results),
            f"{mean_ks:.4f}" if mean_ks is not None else "N/A",
            f"{mean_sim:.4f}" if mean_sim is not None else "N/A",
        )

    _save_csv(output_dir, flat_rows)
    _save_json(output_dir, all_results)
    return all_results


# ── Column selection ──────────────────────────────────────────────────────────

def _numeric_columns(original: pd.DataFrame, synthetic: pd.DataFrame) -> list[str]:
    return [
        c for c in original.select_dtypes(include="number").columns
        if c in synthetic.columns
        and c not in EXCLUDED_COLUMNS
        and not pd.api.types.is_bool_dtype(original[c])
        and not pd.api.types.is_bool_dtype(synthetic[c])
        and not _is_id_like(original[c])
    ]


def _is_id_like(series: pd.Series) -> bool:
    n_total  = len(series.dropna())
    n_unique = series.nunique(dropna=True)
    return (n_unique / n_total) >= _ID_UNIQUENESS_THRESHOLD if n_total > 0 else False


def _safe_numeric(s: pd.Series) -> pd.Series:
    if pd.api.types.is_bool_dtype(s):
        return s.astype(np.int8)
    return s


# ── KDE plot ──────────────────────────────────────────────────────────────────

def _plot_kde(
    real:       pd.Series,
    synth:      pd.Series,
    col:        str,
    table_name: str,
    plots_dir:  Path,
    safe_col:   str,
) -> None:
    try:
        plots_dir.mkdir(parents=True, exist_ok=True)
        fig, ax = plt.subplots(figsize=(8, 4))
        real.plot.kde(ax=ax,  label="Original",  color="#2196F3", linewidth=2)
        synth.plot.kde(ax=ax, label="Synthetic", color="#FF5722", linewidth=2,
                       linestyle="--")
        ax.set_title(f"{table_name} — {col} (KDE Comparison)", fontsize=11)
        ax.set_xlabel(col)
        ax.set_ylabel("Density")
        ax.legend(framealpha=0.8)
        fig.tight_layout()
        fig.savefig(plots_dir / f"{table_name}_{safe_col}_kde.png", dpi=100)
        plt.close(fig)
    except Exception as exc:
        logger.debug("KDE plot skipped for %s.%s: %s", table_name, col, exc)


# ── File I/O ──────────────────────────────────────────────────────────────────

def _save_csv(output_dir: Path, rows: list[dict]) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    cols = [
        "table", "column", "ks_stat", "ks_p_value", "similarity",
        "n_real", "n_synthetic", "mean_real", "mean_synth", "std_real", "std_synth",
    ]
    df   = pd.DataFrame(rows, columns=cols) if rows else pd.DataFrame(columns=cols)
    path = output_dir / "ks_results.csv"
    df.to_csv(path, index=False)
    logger.info("Saved: %s", path)


def _save_json(output_dir: Path, results: dict) -> None:
    path = output_dir / "ks_results.json"
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(results, fh, indent=2, default=str)
    logger.info("Saved: %s", path)


def _safe_name(s: str) -> str:
    return re.sub(r"[^\w\-]", "_", s)[:60]
