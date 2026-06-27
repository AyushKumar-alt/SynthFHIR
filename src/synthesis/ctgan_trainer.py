"""CTGAN (and other single-table model) trainer.

Factory pattern: ``create_single_table_synthesizer(model_name, ...)``
returns the right SDV synthesizer class so callers never import SDV directly.
Swapping CTGAN → TVAE requires only a config string change.

Supported models
----------------
ctgan             : CTGANSynthesizer  (default for patients)
tvae              : TVAESynthesizer
gaussian_copula   : GaussianCopulaSynthesizer

Kwarg filtering
---------------
Each synthesizer accepts a different set of constructor parameters.
CTGANTrainer.train() passes a uniform set of kwargs (epochs, batch_size,
verbose, cuda) to create_single_table_synthesizer() regardless of model type.
The factory filters to the per-synthesizer whitelist defined in _*_KWARGS
constants, logs any dropped kwargs at WARNING level, and never forwards
unsupported arguments.  This prevents TypeError on GaussianCopula fallback.
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

# Per-synthesizer constructor parameter whitelists (SDV 1.x public API).
# Only kwargs present in the relevant frozenset are forwarded.
# Everything else is logged at WARNING and silently dropped.
_CTGAN_KWARGS: frozenset[str] = frozenset({
    "enforce_min_max_values", "enforce_rounding", "locales",
    "epochs", "log_frequency", "embedding_dim",
    "generator_dim", "discriminator_dim",
    "generator_lr", "generator_decay",
    "discriminator_lr", "discriminator_decay",
    "batch_size", "discriminator_steps",
    "verbose", "cuda", "pac",
})

_TVAE_KWARGS: frozenset[str] = frozenset({
    "enforce_min_max_values", "enforce_rounding", "locales",
    "epochs", "batch_size", "embedding_dim",
    "compress_dims", "decompress_dims",
    "l2scale", "loss_factor",
    "cuda", "verbose",
})

# GaussianCopula is a statistical copula — no training loop.
# It accepts NO epochs, batch_size, verbose, or cuda.
_GAUSSIAN_COPULA_KWARGS: frozenset[str] = frozenset({
    "enforce_min_max_values", "enforce_rounding", "locales",
    "default_distribution", "numerical_distributions",
})


def _filter_kwargs(display_name: str, allowed: frozenset[str], kwargs: dict) -> dict:
    """Return only the kwargs in ``allowed``, logging any that are dropped."""
    dropped = sorted(k for k in kwargs if k not in allowed)
    if dropped:
        logger.warning(
            "Ignoring unsupported kwargs for %s: %s",
            display_name,
            ", ".join(dropped),
        )
    return {k: v for k, v in kwargs.items() if k in allowed}


# ── Factory ───────────────────────────────────────────────────────────────────

def create_single_table_synthesizer(model_name: str, metadata, **kwargs):
    """Instantiate an SDV single-table synthesizer by name.

    Each synthesizer receives ONLY the constructor parameters it supports.
    Unsupported kwargs are logged at WARNING level and silently dropped, so
    callers can pass a uniform kwarg set regardless of which model is
    ultimately constructed — including the GaussianCopula OOM fallback path.

    Args:
        model_name : One of ``_SUPPORTED_MODELS``.
        metadata   : SDV ``SingleTableMetadata`` instance.
        **kwargs   : Candidate parameters — filtered per synthesizer whitelist.

    Returns:
        Configured (but not yet fitted) SDV synthesizer instance.
    """
    model_name = model_name.lower().strip()
    if model_name == "ctgan":
        from sdv.single_table import CTGANSynthesizer
        return CTGANSynthesizer(metadata, **_filter_kwargs("CTGANSynthesizer", _CTGAN_KWARGS, kwargs))
    if model_name == "tvae":
        from sdv.single_table import TVAESynthesizer
        return TVAESynthesizer(metadata, **_filter_kwargs("TVAESynthesizer", _TVAE_KWARGS, kwargs))
    if model_name == "gaussian_copula":
        from sdv.single_table import GaussianCopulaSynthesizer
        return GaussianCopulaSynthesizer(metadata, **_filter_kwargs("GaussianCopulaSynthesizer", _GAUSSIAN_COPULA_KWARGS, kwargs))
    raise ValueError(
        f"Unknown model '{model_name}'. Choose from: {_SUPPORTED_MODELS}"
    )


# ── Boolean coercion ──────────────────────────────────────────────────────────

def _coerce_booleans(df: pd.DataFrame, table_meta_dict: dict) -> pd.DataFrame:
    """Coerce all boolean-typed columns in df to Python True/False.

    Reads which columns are boolean from table_meta_dict (sdtype='boolean'),
    then delegates to enforce_boolean_columns() for the actual conversion.
    Handles 0/1 integers, "True"/"False" strings from CSV round-trips, and
    Python booleans — all representations that arise in this pipeline.
    """
    from ..feature_engineering import enforce_boolean_columns
    bool_cols = [
        col for col, info in table_meta_dict.get("columns", {}).items()
        if info.get("sdtype") == "boolean" and col in df.columns
    ]
    return enforce_boolean_columns(df, bool_cols)


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
        cuda: bool = False,
        model_name: str | None = None,
    ):
        """Fit a synthesizer on ``df``.

        Args:
            df             : Training DataFrame (the ready table).
            table_meta_dict: The table's sub-dict from metadata.json.
            epochs         : Override config epochs (useful for smoke test).
            verbose        : Whether to print SDV's training progress.
            cuda           : Pass True to enable GPU acceleration.
            model_name     : Override the model type (default: cfg.patient_model).

        Returns:
            Fitted SDV synthesizer.
        """
        metadata  = build_single_table_metadata(table_meta_dict)
        n_epochs  = epochs if epochs is not None else self.cfg.epochs
        model     = model_name or self.cfg.patient_model

        logger.info(
            "Training %s on %d rows x %d cols for %d epochs (cuda=%s) ...",
            model, len(df), len(df.columns), n_epochs, cuda,
        )
        t0 = time.time()

        df_prepared = _coerce_booleans(df, table_meta_dict)
        synth = create_single_table_synthesizer(
            model,
            metadata,
            epochs=n_epochs,
            batch_size=self.cfg.batch_size,
            verbose=verbose,
            cuda=cuda,
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
