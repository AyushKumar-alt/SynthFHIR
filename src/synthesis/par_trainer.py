"""PARSynthesizer trainer for sequential clinical event tables.

PARSynthesizer models within-patient event sequences. Each patient is one
sequence; ``sequence_key`` groups rows by patient; ``sequence_index`` orders
events within that patient chronologically.

Applied tables
--------------
encounters   : sequence_key=patient_id, sequence_index=sequence_index
observations : sequence_key=patient_id, sequence_index=sequence_index
conditions   : sequence_key=patient_id, sequence_index=sequence_index
medications  : sequence_key=patient_id, sequence_index=sequence_index

Design note
-----------
PARSynthesizer can accept ``context_columns`` — attributes that are constant
within a sequence (same for every row of a patient). We pass an empty list
here because patient demographics live in the separate patients table.
For Phase 4B, patient-level context can be merged in to condition the
sequence generation on age, gender, etc.
"""

from __future__ import annotations

import logging
import time
from pathlib import Path

import pandas as pd

from .config import SynthesisConfig
from .ctgan_trainer import build_single_table_metadata

logger = logging.getLogger(__name__)

_SEQUENCE_KEY   = "patient_id"
_SEQUENCE_INDEX = "sequence_index"


class PARTrainer:
    """Train, save, and load a PARSynthesizer for one sequential table."""

    def __init__(self, synth_config: SynthesisConfig) -> None:
        self.cfg = synth_config

    def train(
        self,
        df: pd.DataFrame,
        table_meta_dict: dict,
        epochs: int | None = None,
        verbose: bool = True,
    ):
        """Fit a PARSynthesizer on ``df``.

        The table must contain ``patient_id`` and ``sequence_index`` columns.

        Args:
            df             : Training DataFrame (a ready table with sequences).
            table_meta_dict: Table sub-dict from metadata.json.
            epochs         : Override config epochs.
            verbose        : Whether to print SDV progress.

        Returns:
            Fitted PARSynthesizer.
        """
        from sdv.sequential import PARSynthesizer

        if _SEQUENCE_KEY not in df.columns:
            raise ValueError(f"PAR training requires '{_SEQUENCE_KEY}' column.")
        if _SEQUENCE_INDEX not in df.columns:
            raise ValueError(f"PAR training requires '{_SEQUENCE_INDEX}' column.")

        metadata = build_single_table_metadata(table_meta_dict)
        n_epochs = epochs if epochs is not None else self.cfg.epochs

        logger.info(
            "Training PARSynthesizer on %d rows x %d cols for %d epochs ...",
            len(df), len(df.columns), n_epochs,
        )
        t0 = time.time()

        synth = PARSynthesizer(
            metadata,
            context_columns=[],
            sequence_key=_SEQUENCE_KEY,
            sequence_index=_SEQUENCE_INDEX,
            epochs=n_epochs,
            verbose=verbose,
        )
        synth.fit(df)
        elapsed = time.time() - t0
        logger.info("PAR training complete in %.1f s (%.1f min).", elapsed, elapsed / 60)
        return synth

    def save(self, synth, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        synth.save(str(path))
        logger.info("PAR model saved: %s", path)

    @staticmethod
    def load(path: Path):
        from sdv.sequential import PARSynthesizer
        return PARSynthesizer.load(str(path))
