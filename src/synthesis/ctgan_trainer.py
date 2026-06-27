"""CTGAN (and other single-table model) trainer.

Factory pattern: ``create_single_table_synthesizer(model_name, ...)``
returns the right SDV synthesizer class so callers never import SDV directly.
Swapping CTGAN → TVAE requires only a config string change.

Supported models
----------------
ctgan             : CTGANSynthesizer  (default for patients)
tvae              : TVAESynthesizer
gaussian_copula   : GaussianCopulaSynthesizer
"""

from __future__ import annotations

import logging
import time
import warnings
from pathlib import Path

import pandas as pd

from .config import SynthesisConfig

logger = logging.getLogger(__name__)

_SUPPORTED_MODELS = ("ctgan", "tvae", "gaussian_copula")


# ── Factory ───────────────────────────────────────────────────────────────────

def create_single_table_synthesizer(model_name: str, metadata, **kwargs):
    """Instantiate an SDV single-table synthesizer by name.

    Args:
        model_name : One of ``_SUPPORTED_MODELS``.
        metadata   : SDV ``SingleTableMetadata`` instance.
        **kwargs   : Passed directly to the synthesizer constructor
                     (e.g. ``epochs``, ``batch_size``, ``cuda``).

    Returns:
        Configured (but not yet fitted) SDV synthesizer instance.
    """
    model_name = model_name.lower().strip()
    if model_name == "ctgan":
        from sdv.single_table import CTGANSynthesizer
        return CTGANSynthesizer(metadata, **kwargs)
    if model_name == "tvae":
        from sdv.single_table import TVAESynthesizer
        return TVAESynthesizer(metadata, **kwargs)
    if model_name == "gaussian_copula":
        from sdv.single_table import GaussianCopulaSynthesizer
        return GaussianCopulaSynthesizer(metadata, **kwargs)
    raise ValueError(
        f"Unknown model '{model_name}'. Choose from: {_SUPPORTED_MODELS}"
    )


# ── Boolean coercion ──────────────────────────────────────────────────────────

def _coerce_booleans(df: pd.DataFrame, table_meta_dict: dict) -> pd.DataFrame:
    """Convert 0/1 integer columns typed as boolean to Python True/False.

    SDV 1.37+ validates boolean columns strictly: values must be True/False,
    not 0/1 integers. We stored them as int for CSV compatibility; convert
    them here before passing to SDV.
    """
    bool_cols = [
        col for col, info in table_meta_dict.get("columns", {}).items()
        if info.get("sdtype") == "boolean" and col in df.columns
    ]
    if not bool_cols:
        return df
    df = df.copy()
    for col in bool_cols:
        df[col] = df[col].map(
            lambda x: None if pd.isna(x) else bool(int(x))
        )
    return df


# ── Metadata builder ──────────────────────────────────────────────────────────

def build_single_table_metadata(table_meta_dict: dict):
    """Convert one entry from our metadata.json into an SDV Metadata object.

    SDV 1.37+ deprecates SingleTableMetadata in favour of the unified Metadata
    class. We build a minimal multi-table dict with a single table entry and
    extract the single-table metadata from it.
    """
    from sdv.metadata import Metadata

    multi_dict = {
        "METADATA_SPEC_VERSION": "MULTI_TABLE_V1",
        "tables": {"_table": table_meta_dict},
        "relationships": [],
    }
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=FutureWarning)
        meta = Metadata.load_from_dict(multi_dict)

    # Return the single-table metadata object for use with single-table synthesizers
    return meta.get_table_metadata("_table")


# ── Trainer class ─────────────────────────────────────────────────────────────

class CTGANTrainer:
    """Train, save, and load a single-table SDV synthesizer."""

    def __init__(self, synth_config: SynthesisConfig) -> None:
        self.cfg = synth_config

    def train(
        self,
        df: pd.DataFrame,
        table_meta_dict: dict,
        epochs: int | None = None,
        verbose: bool = True,
    ):
        """Fit a synthesizer on ``df``.

        Args:
            df             : Training DataFrame (the ready table).
            table_meta_dict: The table's sub-dict from metadata.json.
            epochs         : Override config epochs (useful for smoke test).
            verbose        : Whether to print SDV's training progress.

        Returns:
            Fitted SDV synthesizer.
        """
        metadata = build_single_table_metadata(table_meta_dict)
        n_epochs = epochs if epochs is not None else self.cfg.epochs

        logger.info(
            "Training %s on %d rows x %d cols for %d epochs ...",
            self.cfg.patient_model, len(df), len(df.columns), n_epochs,
        )
        t0 = time.time()

        df_prepared = _coerce_booleans(df, table_meta_dict)
        synth = create_single_table_synthesizer(
            self.cfg.patient_model,
            metadata,
            epochs=n_epochs,
            batch_size=self.cfg.batch_size,
            verbose=verbose,
        )
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=FutureWarning)
            synth.fit(df_prepared)
        elapsed = time.time() - t0
        logger.info("Training complete in %.1f s (%.1f min).", elapsed, elapsed / 60)
        return synth

    def save(self, synth, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        synth.save(str(path))
        logger.info("Model saved: %s", path)

    @staticmethod
    def load(model_name: str, path: Path):
        """Load a previously saved synthesizer."""
        model_name = model_name.lower().strip()
        if model_name == "ctgan":
            from sdv.single_table import CTGANSynthesizer
            return CTGANSynthesizer.load(str(path))
        if model_name == "tvae":
            from sdv.single_table import TVAESynthesizer
            return TVAESynthesizer.load(str(path))
        if model_name == "gaussian_copula":
            from sdv.single_table import GaussianCopulaSynthesizer
            return GaussianCopulaSynthesizer.load(str(path))
        raise ValueError(f"Unknown model '{model_name}'")
