"""Sequence length control layer for PAR training — GPU memory safety.

WHY THIS MODULE EXISTS
----------------------
PARSynthesizer (PyTorch-based) pads every sequence in a mini-batch to the
length of the *longest* sequence in that batch.  Without clipping:

    Patient A: 420 conditions  →  420-step padded tensor allocated for
    Patient B:   3 conditions     BOTH patients, even the one with 3.

With 998 patients, one outlier with 420 events forces the GPU to hold
420 × batch_size × hidden_dim × dtype_bytes in VRAM simultaneously.
At float32, hidden=128, batch_size=32:
    420 × 32 × 128 × 4 bytes ≈ 6.9 MB  per layer per forward pass
    Multiply by 2–4 PAR LSTM layers + backward gradients → OOM.

THE FIX
-------
Clip each patient's history to the most-recent ``max_seq_len`` events
BEFORE the data reaches PARSynthesizer.fit().  This bounds GPU allocation
to O(max_seq_len × batch_size) regardless of outlier patients.

DESIGN DECISIONS
----------------
Keep most-recent events (not oldest, not random).
    Recent events are more predictive of current clinical state.  A
    patient's last 50 encounters contain richer signal for a generative
    model than their first 50 encounters 20 years ago.

sequence_index is the canonical sort column.
    Phase 3 feature engineering writes a 0-based per-patient chronological
    index (sequence_index) into every PAR table.  Sorting by it is always
    correct, even if underlying temporal columns have NaN values.

No-op fast path.
    If every sequence already has ≤ max_seq_len events, the original
    DataFrame is returned without copying.  The profiling log still runs
    so operators can verify the assumption.

HARD RULE FOR ALL FUTURE MODELS
---------------------------------
Any sequence-based model in this project MUST pass data through
``sanitize_sequences()`` before fitting.  Memory must never be the
limiting factor.  Sequence size is controlled here, not inside the model.
"""

from __future__ import annotations

import logging
from typing import Optional

import pandas as pd

logger = logging.getLogger(__name__)

DEFAULT_MAX_SEQ_LEN: int = 50
_SEQ_INDEX_COL: str = "sequence_index"
_SEQUENCE_KEY: str = "patient_id"


# ── Profiling ─────────────────────────────────────────────────────────────────

def profile_sequences(
    df: pd.DataFrame,
    sequence_key: str,
    max_seq_len: Optional[int] = None,
) -> dict:
    """Compute per-sequence length statistics.

    Args:
        df           : Training DataFrame (one row per event).
        sequence_key : Column that groups rows into sequences (e.g. 'patient_id').
        max_seq_len  : Clipping threshold used to count outliers.  Pass None to
                       skip the outlier count.

    Returns:
        Dict with keys: n_sequences, max_length, min_length, mean_length,
        p50_length, p95_length, p99_length, and optionally outliers.
    """
    lengths = df.groupby(sequence_key, sort=False).size()
    profile: dict = {
        "n_sequences": int(len(lengths)),
        "max_length":  int(lengths.max()),
        "min_length":  int(lengths.min()),
        "mean_length": round(float(lengths.mean()), 1),
        "p50_length":  int(lengths.quantile(0.50)),
        "p95_length":  int(lengths.quantile(0.95)),
        "p99_length":  int(lengths.quantile(0.99)),
    }
    if max_seq_len is not None:
        profile["outliers"] = int((lengths > max_seq_len).sum())
    return profile


def log_sequence_profile(
    table_name: str,
    profile: dict,
    max_seq_len: int,
) -> None:
    """Emit a structured sequence-length profile at INFO level."""
    sep = "-" * 54
    n_out = profile.get("outliers", 0)
    pct   = 100 * n_out / max(profile["n_sequences"], 1)
    logger.info(sep)
    logger.info("  [SEQUENCE PROFILE] %s", table_name.upper())
    logger.info(sep)
    logger.info("  sequences    : %d", profile["n_sequences"])
    logger.info("  max_length   : %d", profile["max_length"])
    logger.info("  mean_length  : %.1f", profile["mean_length"])
    logger.info("  p50_length   : %d",  profile["p50_length"])
    logger.info("  p95_length   : %d",  profile["p95_length"])
    logger.info("  p99_length   : %d",  profile["p99_length"])
    logger.info("  max_seq_len  : %d  ← configured GPU safety limit", max_seq_len)
    logger.info(
        "  outliers     : %d sequences clipped  (%.1f%%)",
        n_out, pct,
    )
    logger.info(sep)


# ── Clipping ──────────────────────────────────────────────────────────────────

def sanitize_sequences(
    df: pd.DataFrame,
    sequence_key: str,
    max_seq_len: int,
    sort_col: str = _SEQ_INDEX_COL,
    table_name: str = "",
) -> tuple[pd.DataFrame, dict]:
    """Clip each sequence to the ``max_seq_len`` most-recent events.

    This is the mandatory entry point for all PAR training data.  Call it
    after loading the ready CSV and before passing the DataFrame to
    PARSynthesizer.fit().

    Strategy: sort by ``sort_col`` within each ``sequence_key`` group, then
    keep the **tail** (most recent ``max_seq_len`` events).  A patient with
    420 encounters will retain only their 421-370 = last 50 encounters.

    Args:
        df           : Training DataFrame (one row per event).
        sequence_key : Grouping column (``'patient_id'`` for all PAR tables).
        max_seq_len  : Maximum events to keep per sequence.  Must be > 0.
        sort_col     : Column used to establish chronological order within
                       each sequence.  Defaults to ``'sequence_index'``,
                       which is guaranteed to exist for all Phase-3 PAR tables.
                       If not found, natural row order is used instead.
        table_name   : Used in log messages only.

    Returns:
        ``(trimmed_df, stats_dict)``

        ``trimmed_df``  — DataFrame with ≤ max_seq_len rows per sequence.
                          Equals the original df (not a copy) when no clipping
                          was needed.
        ``stats_dict``  — Dict of before/after metrics, merged into the
                          training_time.json manifest by the pipeline.
    """
    if max_seq_len <= 0:
        raise ValueError(f"max_seq_len must be > 0, got {max_seq_len!r}")

    label = table_name or "table"
    profile = profile_sequences(df, sequence_key, max_seq_len)
    log_sequence_profile(label, profile, max_seq_len)

    rows_before = len(df)
    n_clipped   = profile.get("outliers", 0)

    # Fast-path: nothing to clip
    if profile["max_length"] <= max_seq_len:
        logger.info(
            "[SEQ-CLIP] %s: max_length=%d ≤ limit=%d — no clipping applied.",
            label, profile["max_length"], max_seq_len,
        )
        return df, _stats(max_seq_len, 0, rows_before, rows_before, profile)

    # Determine sort column — fall back gracefully if missing
    effective_sort = sort_col if sort_col in df.columns else None
    if effective_sort is None:
        logger.warning(
            "[SEQ-CLIP] %s: sort column '%s' not found — using natural row "
            "order.  Ensure Phase 3 has run to populate 'sequence_index'.",
            label, sort_col,
        )

    # Sort chronologically within each sequence, then keep tail
    if effective_sort:
        df_sorted = df.sort_values([sequence_key, effective_sort])
    else:
        df_sorted = df

    df_clipped = (
        df_sorted
        .groupby(sequence_key, sort=False)
        .tail(max_seq_len)
        .reset_index(drop=True)
    )

    rows_after = len(df_clipped)
    logger.info(
        "[SEQ-CLIP] %s: clipped %d/%d sequences — rows %d → %d  "
        "(%d rows removed, %.1f%% reduction)",
        label,
        n_clipped, profile["n_sequences"],
        rows_before, rows_after,
        rows_before - rows_after,
        100 * (rows_before - rows_after) / max(rows_before, 1),
    )

    return df_clipped, _stats(max_seq_len, n_clipped, rows_before, rows_after, profile)


def _stats(
    max_seq_len: int,
    n_clipped: int,
    rows_before: int,
    rows_after: int,
    profile: dict,
) -> dict:
    """Build the stats sub-dict that the pipeline merges into the manifest."""
    return {
        "seq_max_len_cfg":  max_seq_len,
        "seq_n_clipped":    n_clipped,
        "seq_rows_before":  rows_before,
        "seq_rows_after":   rows_after,
        "seq_max_before":   profile["max_length"],
        "seq_mean_before":  profile["mean_length"],
        "seq_p95_before":   profile["p95_length"],
    }
