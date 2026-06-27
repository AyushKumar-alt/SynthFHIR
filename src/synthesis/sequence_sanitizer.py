"""Sequence memory-safety layer for PAR training.

WHY THIS MODULE EXISTS
----------------------
PARSynthesizer (PyTorch-based) allocates a padded tensor of shape

    [max_seq_len_in_batch, n_sequences, hidden_dim]

before each forward pass.  Two independent dimensions drive VRAM:

  (A) n_sequences  — number of patients in the training set.
  (B) max_seq_len  — longest event history among those patients.

Either dimension alone can cause CUDA OOM:

    998 patients × 50 events × 128 hidden × float32 × grad overhead ≈ OOM
    400 patients × 420 events × 128 hidden × float32 × grad overhead ≈ OOM

Both must be bounded before fit() is called.

TWO-STAGE CONTROL
-----------------
Stage 1 — ``sample_par_patients()``
    Randomly samples a reproducible subset of ``par_max_patients`` patients.
    All events for each sampled patient are retained (complete histories).
    This shrinks the n_sequences axis.

Stage 2 — ``sanitize_sequences()``
    For each remaining patient, keeps only the most-recent ``max_seq_len``
    events.  This shrinks the max_seq_len axis.

Together they bound peak VRAM to:
    O(par_max_patients × max_seq_len × batch_size × hidden_dim)

DESIGN DECISIONS
----------------
Patient sampling uses fixed seed → reproducible across runs.

Keep most-recent events (Stage 2).
    Recent events are more clinically relevant than older ones.

sequence_index is the canonical sort column.
    Phase 3 writes a 0-based per-patient chronological index into every
    PAR table.  Sorting by it is correct even when temporal columns have NaN.

No-op fast paths.
    If n_patients ≤ par_max_patients, no sampling is applied.
    If max_seq_len_in_table ≤ max_seq_len, no clipping is applied.

HARD RULE FOR ALL FUTURE MODELS
---------------------------------
Any sequence-based model in this project MUST pass data through
``sample_par_patients()`` then ``sanitize_sequences()`` before fitting.
Memory must NEVER be the limiting factor.  Sequence count and sequence
length are controlled here, not inside the model.
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


# ── Patient sampling (Stage 1) ────────────────────────────────────────────────

def sample_par_patients(
    df: pd.DataFrame,
    sequence_key: str,
    max_patients: int,
    seed: int = 42,
    table_name: str = "",
) -> tuple[pd.DataFrame, dict]:
    """Sample a reproducible subset of patients to bound PAR sequence count.

    PAR VRAM scales with n_sequences.  Sampling ``max_patients`` patients
    from the full training set is the coarsest but most effective lever for
    reducing peak GPU memory when sequence-length clipping alone is insufficient.

    All events for each selected patient are retained in full — histories are
    complete, just fewer patients.  The same ``seed`` always produces the same
    patient subset across runs.

    Args:
        df           : Training DataFrame (one row per event).
        sequence_key : Grouping column (``'patient_id'`` for all PAR tables).
        max_patients : Maximum number of patients to retain.  Must be > 0.
        seed         : Random seed passed to pandas sample().  Use
                       ``cfg.seed`` from SynthesisConfig for reproducibility.
        table_name   : Used in log messages only.

    Returns:
        ``(sampled_df, stats_dict)``

        ``sampled_df``  — DataFrame containing only rows for the sampled
                          patients, with index reset.  Equals the original df
                          (not a copy) when n_patients ≤ max_patients.
        ``stats_dict``  — Dict of before/after metrics, merged into the
                          training_time.json manifest by the pipeline.
    """
    if max_patients <= 0:
        raise ValueError(f"max_patients must be > 0, got {max_patients!r}")

    label = table_name or "table"
    all_patients = df[sequence_key].unique()
    n_patients   = int(len(all_patients))

    logger.info(
        "[PAT-SAMPLE] %s: n_patients=%d  limit=%d  seed=%d",
        label, n_patients, max_patients, seed,
    )

    base_stats = {
        "pat_max_patients_cfg": max_patients,
        "pat_n_before":         n_patients,
    }

    # Fast-path: already within limit
    if n_patients <= max_patients:
        logger.info(
            "[PAT-SAMPLE] %s: n_patients=%d ≤ limit=%d — no sampling applied.",
            label, n_patients, max_patients,
        )
        return df, {**base_stats, "pat_n_after": n_patients, "pat_n_dropped": 0}

    # Reproducible random sample of patient IDs (no replacement)
    sampled_ids = set(
        pd.Series(all_patients).sample(n=max_patients, random_state=seed, replace=False)
    )
    df_sampled = df[df[sequence_key].isin(sampled_ids)].reset_index(drop=True)

    n_dropped   = n_patients - max_patients
    rows_before = len(df)
    rows_after  = len(df_sampled)

    logger.info(
        "[PAT-SAMPLE] %s: sampled %d/%d patients  "
        "(dropped %d, %.1f%%)  rows %d → %d",
        label,
        max_patients, n_patients,
        n_dropped, 100 * n_dropped / max(n_patients, 1),
        rows_before, rows_after,
    )

    return df_sampled, {
        **base_stats,
        "pat_n_after":   max_patients,
        "pat_n_dropped": n_dropped,
        "pat_rows_before": rows_before,
        "pat_rows_after":  rows_after,
    }


# ── Sequence clipping (Stage 2) ───────────────────────────────────────────────

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
