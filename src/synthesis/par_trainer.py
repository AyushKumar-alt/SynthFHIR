"""PARSynthesizer trainer for sequential clinical event tables.

SDV 1.37.2 API — critical notes
--------------------------------
PARSynthesizer.__init__(metadata, context_columns, segment_size,
                        epochs, sample_size, cuda, verbose)

``sequence_key`` and ``sequence_index`` are NOT constructor arguments.
They must be embedded in the metadata object BEFORE construction:

    SingleTableMetadata.load_from_dict({
        ...
        "sequence_key":   "patient_id",
        "sequence_index": "sequence_index",
    })

PARSynthesizer reads them internally from
    self._get_table_metadata().sequence_key
    self._get_table_metadata().sequence_index

Using the new ``Metadata`` class returned by
``Metadata.get_table_metadata()`` does NOT work here because that
class exposes ``set_sequence_key`` but not ``.sequence_key`` as a
plain attribute — PAR's internal read fails.

Resolution: ``build_par_metadata()`` injects the sequence keys
into a ``SingleTableMetadata`` dict before loading.

Applied tables
--------------
encounters   : sequence_key=patient_id, sequence_index=sequence_index
conditions   : sequence_key=patient_id, sequence_index=sequence_index
medications  : sequence_key=patient_id, sequence_index=sequence_index

(observations uses CTGANSynthesizer — 303K rows × 300 epochs exceeds
 practical PAR training time even on T4 GPU.)
"""

from __future__ import annotations

import logging
import time
import warnings
from pathlib import Path

import pandas as pd

from .config import SynthesisConfig
from .ctgan_trainer import _coerce_booleans

logger = logging.getLogger(__name__)

_SEQUENCE_KEY   = "patient_id"
_SEQUENCE_INDEX = "sequence_index"


# ── Metadata builder (PAR-specific) ──────────────────────────────────────────

def build_par_metadata(table_meta_dict: dict):
    """Build a SingleTableMetadata with sequence_key and sequence_index set.

    PAR reads sequence_key and sequence_index from the metadata object,
    not from its constructor. We inject them into the dict before loading
    so that PARSynthesizer can find them via _get_table_metadata().

    Args:
        table_meta_dict: One table entry from metadata.json
                         (keys: primary_key, columns, …).

    Returns:
        SingleTableMetadata instance with .sequence_key and .sequence_index set.
    """
    from sdv.metadata import SingleTableMetadata

    st_dict = {
        **table_meta_dict,
        "METADATA_SPEC_VERSION": "V1",
        "sequence_key":   _SEQUENCE_KEY,
        "sequence_index": _SEQUENCE_INDEX,
    }
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=FutureWarning)
        meta = SingleTableMetadata.load_from_dict(st_dict)

    return meta


# ── Trainer ────────────────────────────────────────────────────────────────────

class PARTrainer:
    """Train, save, and load a PARSynthesizer for one sequential table.

    The training DataFrame must contain columns ``patient_id`` (sequence
    grouping key) and ``sequence_index`` (within-patient sort order).
    Both are produced by Phase 3 feature engineering for all child tables.
    """

    def __init__(self, synth_config: SynthesisConfig) -> None:
        self.cfg = synth_config

    def train(
        self,
        df: pd.DataFrame,
        table_meta_dict: dict,
        epochs: int | None = None,
        verbose: bool = True,
        cuda: bool = False,
    ):
        """Fit a PARSynthesizer on ``df``.

        Args:
            df             : Training DataFrame with patient_id + sequence_index.
            table_meta_dict: Table entry from metadata.json (columns, primary_key).
            epochs         : Epoch count override; falls back to cfg.epochs.
            verbose        : Whether to print SDV's per-epoch progress.
            cuda           : True to use GPU (passed directly to PARSynthesizer).

        Returns:
            Fitted PARSynthesizer.

        Raises:
            ValueError if patient_id or sequence_index columns are missing.
        """
        from sdv.sequential import PARSynthesizer

        if _SEQUENCE_KEY not in df.columns:
            raise ValueError(
                f"PAR training requires column '{_SEQUENCE_KEY}'. "
                f"Available: {list(df.columns)}"
            )
        if _SEQUENCE_INDEX not in df.columns:
            raise ValueError(
                f"PAR training requires column '{_SEQUENCE_INDEX}'. "
                f"Available: {list(df.columns)}"
            )

        n_epochs  = epochs if epochs is not None else self.cfg.epochs
        n_seqs    = df[_SEQUENCE_KEY].nunique()
        metadata  = build_par_metadata(table_meta_dict)

        logger.info(
            "Training PARSynthesizer on %d rows / %d sequences x %d cols "
            "for %d epochs (cuda=%s) ...",
            len(df), n_seqs, len(df.columns), n_epochs, cuda,
        )
        t0 = time.time()

        # sequence_key and sequence_index come from metadata, NOT from the
        # constructor.  context_columns=[] because patient demographics live
        # in the separate patients table (not merged here).
        synth = PARSynthesizer(
            metadata,
            context_columns=[],
            segment_size=None,
            epochs=n_epochs,
            sample_size=1,
            cuda=cuda,
            verbose=verbose,
        )
        # Coerce boolean columns (is_chronic, is_active, etc.) from 0/1
        # integers or "True"/"False" strings to Python True/False.
        # SDV 1.x rejects any other representation for boolean-typed columns.
        df_prepared = _coerce_booleans(df, table_meta_dict)
        synth.fit(df_prepared)

        elapsed = time.time() - t0
        logger.info(
            "PAR training complete in %.1f s (%.1f min).", elapsed, elapsed / 60
        )
        return synth

    def save(self, synth, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        synth.save(str(path))
        logger.info("PAR model saved: %s", path)

    @staticmethod
    def load(path: Path):
        from sdv.sequential import PARSynthesizer
        return PARSynthesizer.load(str(path))
