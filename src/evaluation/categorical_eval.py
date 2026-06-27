"""Module C — Categorical evaluation: TVD, frequency comparison, bar charts."""

from __future__ import annotations

import logging
import re
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from .config import EXCLUDED_COLUMNS
from .loader import TablePair

logger = logging.getLogger(__name__)

# Maximum unique-value count to treat a column as categorical.
# High-cardinality object columns (e.g. free-text) are skipped.
_MAX_CARDINALITY = 50

# Minimum unique values — single-value columns carry no information.
_MIN_UNIQUE = 2

# Maximum categories shown per bar chart.
_TOP_N_CHART = 15

# Maximum categories stored in the top-N frequency dict (JSON output).
_TOP_N_JSON = 10


def run_categorical_evaluation(
    pairs:     dict[str, TablePair],
    plots_dir: Path,
) -> dict[str, list[dict]]:
    """Run categorical evaluation for every table."""
    all_results: dict[str, list[dict]] = {}
    for table_name, (original, synthetic) in pairs.items():
        if original is None or synthetic is None:
            logger.warning("[CATEGORICAL] %s — missing data, skipped", table_name)
            all_results[table_name] = []
            continue
        results = _evaluate_table(original, synthetic, table_name, plots_dir / "categorical")
        logger.info(
            "[CATEGORICAL] %s — evaluated %d categorical columns", table_name, len(results)
        )
        all_results[table_name] = results
    return all_results


# ── Per-table logic ───────────────────────────────────────────────────────────

def _evaluate_table(
    original:   pd.DataFrame,
    synthetic:  pd.DataFrame,
    table_name: str,
    plots_dir:  Path,
) -> list[dict]:
    cat_cols = _categorical_columns(original, synthetic)
    results  = []

    for col in cat_cols:
        real_s  = original[col].astype(str).fillna("__NULL__")
        synth_s = synthetic[col].astype(str).fillna("__NULL__")

        real_freq  = real_s.value_counts(normalize=True).sort_index()
        synth_freq = synth_s.value_counts(normalize=True).sort_index()

        tvd = _tvd(real_freq, synth_freq)

        result = {
            "table":              table_name,
            "column":             col,
            "tvd":                round(tvd, 4),
            "similarity":         round(1.0 - tvd, 4),
            "n_unique_real":      int(original[col].nunique(dropna=True)),
            "n_unique_synthetic": int(synthetic[col].nunique(dropna=True)),
            "top10_real":         (
                real_s.value_counts(normalize=True)
                .head(_TOP_N_JSON)
                .round(4)
                .to_dict()
            ),
            "top10_synthetic":    (
                synth_s.value_counts(normalize=True)
                .head(_TOP_N_JSON)
                .round(4)
                .to_dict()
            ),
        }

        safe = _safe_name(col)
        _plot_bar(real_freq, synth_freq, col, table_name, plots_dir, safe)
        results.append(result)

    return results


# ── Column selection ──────────────────────────────────────────────────────────

def _categorical_columns(
    original:  pd.DataFrame,
    synthetic: pd.DataFrame,
) -> list[str]:
    """Return columns to evaluate categorically."""
    shared = [c for c in original.columns if c in synthetic.columns]
    result = []
    for col in shared:
        if col in EXCLUDED_COLUMNS:
            continue
        n_unique = original[col].nunique(dropna=True)
        if n_unique < _MIN_UNIQUE or n_unique > _MAX_CARDINALITY:
            continue
        dtype = original[col].dtype
        is_cat = (
            dtype == object
            or str(dtype).startswith("bool")
            or (not str(dtype).startswith("float") and n_unique <= _MAX_CARDINALITY)
        )
        if is_cat:
            result.append(col)
    return result


# ── TVD ───────────────────────────────────────────────────────────────────────

def _tvd(real_freq: pd.Series, synth_freq: pd.Series) -> float:
    """Total Variation Distance: 0.5 × Σ|p_real − p_synth|.

    0 = identical distributions, 1 = completely disjoint.
    """
    all_cats = real_freq.index.union(synth_freq.index)
    p = real_freq.reindex(all_cats, fill_value=0.0)
    q = synth_freq.reindex(all_cats, fill_value=0.0)
    return float(0.5 * (p - q).abs().sum())


# ── Plot ──────────────────────────────────────────────────────────────────────

def _plot_bar(
    real_freq:  pd.Series,
    synth_freq: pd.Series,
    col:        str,
    table_name: str,
    plots_dir:  Path,
    safe_col:   str,
) -> None:
    try:
        plots_dir.mkdir(parents=True, exist_ok=True)

        # Show top-N categories by real-data frequency
        top_cats = real_freq.nlargest(_TOP_N_CHART).index
        r = real_freq.reindex(top_cats, fill_value=0.0)
        s = synth_freq.reindex(top_cats, fill_value=0.0)

        n   = len(top_cats)
        x   = range(n)
        w   = 0.38
        fig_w = max(7, n * 0.65)

        fig, ax = plt.subplots(figsize=(fig_w, 4))
        ax.bar(
            [i - w / 2 for i in x], r.values,
            width=w, label="Original",  color="#2196F3", alpha=0.85,
        )
        ax.bar(
            [i + w / 2 for i in x], s.values,
            width=w, label="Synthetic", color="#FF5722", alpha=0.85,
        )
        ax.set_xticks(list(x))
        ax.set_xticklabels(
            [str(c) for c in top_cats],
            rotation=40, ha="right", fontsize=8,
        )
        ax.set_title(f"{table_name} — {col} (Category Frequencies)", fontsize=11)
        ax.set_ylabel("Proportion")
        ax.legend(framealpha=0.8)
        fig.tight_layout()
        fig.savefig(plots_dir / f"{table_name}_{safe_col}_bar.png", dpi=100)
        plt.close(fig)
    except Exception as exc:
        logger.debug("Bar plot skipped for %s.%s: %s", table_name, col, exc)


def _safe_name(s: str) -> str:
    return re.sub(r"[^\w\-]", "_", s)[:60]
