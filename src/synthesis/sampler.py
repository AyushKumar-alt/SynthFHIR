"""Sample generation from trained SDV synthesizers.

Two sampling modes
------------------
sample_patients(synth, n)
    Single-table sampling (CTGAN / TVAE). Returns n synthetic patient rows.

sample_sequences(synth, n_sequences)
    Sequential sampling (PAR). Generates n_sequences patient timelines and
    returns all rows concatenated into one DataFrame.

After generation, ``assign_synthetic_ids`` replaces the model-generated id
column with fresh UUIDs to guarantee uniqueness across runs.
"""

from __future__ import annotations

import logging
import uuid
from pathlib import Path

import pandas as pd

logger = logging.getLogger(__name__)


def sample_patients(synth, n_rows: int, seed: int | None = None) -> pd.DataFrame:
    """Generate synthetic patient rows from a single-table synthesizer."""
    logger.info("Sampling %d synthetic patients ...", n_rows)
    df = synth.sample(num_rows=n_rows)
    logger.info("Sampled %d rows x %d cols.", len(df), len(df.columns))
    return df


def sample_sequences(synth, n_sequences: int, seed: int | None = None) -> pd.DataFrame:
    """Generate synthetic event sequences from a PARSynthesizer.

    Args:
        synth       : Fitted PARSynthesizer.
        n_sequences : Number of patient sequences to generate.

    Returns:
        DataFrame with all generated rows (multiple rows per sequence).
    """
    logger.info("Sampling %d synthetic sequences ...", n_sequences)
    df = synth.sample(num_sequences=n_sequences)
    logger.info("Sampled %d rows for %d sequences.", len(df), n_sequences)
    return df


def assign_synthetic_ids(
    df: pd.DataFrame,
    id_column: str,
    prefix: str = "syn",
) -> pd.DataFrame:
    """Replace the id column with fresh UUIDs prefixed with ``prefix``.

    The prefix makes synthetic IDs visually distinct from real UUIDs in
    validation outputs.

    Args:
        df        : DataFrame whose id column will be replaced.
        id_column : Name of the primary key column.
        prefix    : Short string prepended to each new UUID.

    Returns:
        DataFrame with replaced id column; original index preserved.
    """
    df = df.copy()
    new_ids = [f"{prefix}-{uuid.uuid4()}" for _ in range(len(df))]
    df[id_column] = new_ids
    return df


def reconnect_patient_ids(
    patients_df: pd.DataFrame,
    child_df: pd.DataFrame,
    real_patient_ids: list[str],
) -> pd.DataFrame:
    """Reassign patient_id in a child table to match synthetic patient IDs.

    PARSynthesizer preserves the original patient_id values from training.
    After generating synthetic patients with new IDs, we need to remap.

    Strategy: sort synthetic patients by their generated order and map
    the original patient_ids (from training data) to the new synthetic IDs
    in the same order.

    Args:
        patients_df      : Synthetic patients DataFrame (already has new IDs).
        child_df         : Synthetic child table (still has training patient_ids).
        real_patient_ids : Ordered list of patient_ids from the training data.

    Returns:
        child_df with patient_id remapped to synthetic patient IDs.
    """
    child_df = child_df.copy()
    synthetic_pids = patients_df["patient_id"].tolist()
    n_real   = len(real_patient_ids)
    n_synth  = len(synthetic_pids)

    # Build a mapping from training patient_id → synthetic patient_id
    # If counts differ, wrap using modulo so every real ID maps to something.
    pid_map = {
        real_pid: synthetic_pids[i % n_synth]
        for i, real_pid in enumerate(real_patient_ids[:n_real])
    }

    child_df["patient_id"] = child_df["patient_id"].astype(str).map(pid_map)
    unmapped = child_df["patient_id"].isna().sum()
    if unmapped:
        logger.warning("%d child rows could not be remapped to synthetic patient IDs.", unmapped)
    return child_df
