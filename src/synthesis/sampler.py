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

    Legacy helper kept for backwards compatibility. Use
    ``remap_to_synthetic_patients`` for Phase 4B.
    """
    child_df = child_df.copy()
    synthetic_pids = patients_df["patient_id"].tolist()
    n_real  = len(real_patient_ids)
    n_synth = len(synthetic_pids)
    pid_map = {
        real_pid: synthetic_pids[i % n_synth]
        for i, real_pid in enumerate(real_patient_ids[:n_real])
    }
    child_df["patient_id"] = child_df["patient_id"].astype(str).map(pid_map)
    unmapped = child_df["patient_id"].isna().sum()
    if unmapped:
        logger.warning("%d child rows could not be remapped to synthetic patient IDs.", unmapped)
    return child_df


def remap_to_synthetic_patients(
    child_df: pd.DataFrame,
    synthetic_patients: pd.DataFrame,
) -> pd.DataFrame:
    """Map generated patient_ids in a child table to synthetic patient_ids.

    Works for both PAR-generated and CTGAN-generated child tables.
    Generated patient_ids may be original training IDs (PAR) or random
    SDV-generated IDs (CTGAN). In both cases we remap unique generated IDs
    to synthetic patient IDs via round-robin, preserving within-patient
    grouping structure.

    Args:
        child_df           : Generated child table (still has generated patient_ids).
        synthetic_patients : Synthetic patients DataFrame with final patient_ids.

    Returns:
        child_df with patient_id column replaced by synthetic patient IDs.
    """
    df = child_df.copy()
    # Unique generated patient IDs in order of first appearance
    gen_pids = list(dict.fromkeys(df["patient_id"].astype(str)))
    syn_pids = synthetic_patients["patient_id"].tolist()

    if not syn_pids:
        logger.warning("synthetic_patients is empty — patient_id not remapped.")
        return df

    pid_map = {
        gen_pid: syn_pids[i % len(syn_pids)]
        for i, gen_pid in enumerate(gen_pids)
    }
    df["patient_id"] = df["patient_id"].astype(str).map(pid_map)
    unmapped = df["patient_id"].isna().sum()
    if unmapped:
        logger.warning("%d rows could not be remapped to synthetic patient IDs.", unmapped)
    return df
