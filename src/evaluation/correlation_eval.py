"""Module D — Correlation evaluation: Pearson matrices, heatmaps, diff matrix."""

from __future__ import annotations

import logging
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .config import EXCLUDED_COLUMNS
from .loader import TablePair

logger = logging.getLogger(__name__)

# Require at least this many numeric columns to compute a correlation matrix.
_MIN_COLS = 2


def run_correlation_evaluation(
    pairs:     dict[str, TablePair],
    plots_dir: Path,
) -> dict[str, dict]:
    """Compute Pearson correlation matrices and absolute difference for all tables."""
    all_results: dict[str, dict] = {}
    for table_name, (original, synthetic) in pairs.items():
        if original is None or synthetic is None:
            logger.warning("[CORRELATION] %s — missing data, skipped", table_name)
            all_results[table_name] = {"table": table_name, "status": "missing_data"}
            continue
        result = _evaluate_table(original, synthetic, table_name, plots_dir / "correlation")
        _log_result(result)
        all_results[table_name] = result
    return all_results


# ── Per-table logic ───────────────────────────────────────────────────────────

def _evaluate_table(
    original:   pd.DataFrame,
    synthetic:  pd.DataFrame,
    table_name: str,
    plots_dir:  Path,
) -> dict:
    corr_real  = _corr_matrix(original)
    corr_synth = _corr_matrix(synthetic)

    if corr_real.empty:
        return {"table": table_name, "status": "insufficient_numeric_columns_original"}
    if corr_synth.empty:
        return {"table": table_name, "status": "insufficient_numeric_columns_synthetic"}

    common = sorted(set(corr_real.columns) & set(corr_synth.columns))
    if len(common) < _MIN_COLS:
        return {"table": table_name, "status": "insufficient_common_numeric_columns"}

    cr   = corr_real.loc[common, common]
    cs   = corr_synth.loc[common, common]
    diff = (cr - cs).abs()

    # Upper-triangle values only (exclude diagonal)
    tri_idx    = np.triu_indices_from(diff.values, k=1)
    tri_vals   = diff.values[tri_idx]
    mean_diff  = float(np.mean(tri_vals)) if len(tri_vals) else 0.0
    max_diff   = float(np.max(tri_vals))  if len(tri_vals) else 0.0

    result = {
        "table":                  table_name,
        "status":                 "ok",
        "n_columns":              len(common),
        "columns":                common,
        "mean_correlation_diff":  round(mean_diff, 4),
        "max_correlation_diff":   round(max_diff, 4),
        "similarity":             round(max(0.0, 1.0 - mean_diff), 4),
        # Store as list-of-lists for JSON serialisation
        "corr_real_matrix":  cr.round(4).values.tolist(),
        "corr_synth_matrix": cs.round(4).values.tolist(),
        "corr_diff_matrix":  diff.round(4).values.tolist(),
    }

    _plot_heatmaps(cr, cs, diff, table_name, common, plots_dir)
    return result


# ── Correlation matrix ────────────────────────────────────────────────────────

def _corr_matrix(df: pd.DataFrame) -> pd.DataFrame:
    """Pearson correlation over numeric columns with nonzero variance."""
    num = df.select_dtypes(include="number").drop(
        columns=[c for c in EXCLUDED_COLUMNS if c in df.columns],
        errors="ignore",
    )
    # Drop zero-variance columns (correlation is undefined for constants)
    num = num.loc[:, num.std(ddof=0) > 0]
    if num.shape[1] < _MIN_COLS:
        return pd.DataFrame()
    return num.corr(method="pearson")


# ── Heatmap plot ──────────────────────────────────────────────────────────────

def _plot_heatmaps(
    cr:         pd.DataFrame,
    cs:         pd.DataFrame,
    diff:       pd.DataFrame,
    table_name: str,
    columns:    list[str],
    plots_dir:  Path,
) -> None:
    try:
        plots_dir.mkdir(parents=True, exist_ok=True)
        n        = len(columns)
        cell_sz  = max(0.55, min(1.0, 8.0 / n))
        fig_size = n * cell_sz

        fig, axes = plt.subplots(1, 3, figsize=(fig_size * 3 + 2, fig_size + 1))

        def _draw(ax, data, title, vmin=-1.0, vmax=1.0, cmap="coolwarm"):
            im = ax.imshow(data.values, vmin=vmin, vmax=vmax, cmap=cmap, aspect="auto")
            ax.set_xticks(range(n))
            ax.set_yticks(range(n))
            tick_fs = max(5, min(9, 80 // n))
            ax.set_xticklabels(columns, rotation=45, ha="right", fontsize=tick_fs)
            ax.set_yticklabels(columns, fontsize=tick_fs)
            ax.set_title(title, fontsize=10, pad=6)
            fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

        _draw(axes[0], cr,   f"{table_name}\nOriginal Correlation")
        _draw(axes[1], cs,   f"{table_name}\nSynthetic Correlation")
        _draw(axes[2], diff, f"{table_name}\nAbsolute Difference", vmin=0.0, vmax=1.0, cmap="Reds")

        fig.tight_layout()
        fig.savefig(plots_dir / f"{table_name}_correlation.png", dpi=100)
        plt.close(fig)
    except Exception as exc:
        logger.debug("Correlation heatmap skipped for %s: %s", table_name, exc)


def _log_result(result: dict) -> None:
    if result.get("status") != "ok":
        logger.info("[CORRELATION] %s — %s", result["table"], result["status"])
        return
    logger.info(
        "[CORRELATION] %-14s  columns=%d  mean_diff=%.4f  max_diff=%.4f  similarity=%.4f",
        result["table"],
        result["n_columns"],
        result["mean_correlation_diff"],
        result["max_correlation_diff"],
        result["similarity"],
    )
