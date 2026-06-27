"""Load (original, synthetic) CSV pairs for each table."""

from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd

from .config import EvaluationConfig

logger = logging.getLogger(__name__)

# Type alias used throughout the evaluation package
TablePair = tuple[pd.DataFrame | None, pd.DataFrame | None]


def load_table_pair(table_name: str, cfg: EvaluationConfig) -> TablePair:
    """Load the original and synthetic CSV for one table.

    Returns (original_df, synthetic_df).  Either can be None when the
    corresponding file does not exist — callers must handle this.
    """
    original_path  = cfg.ready_dir     / f"{table_name}_ready.csv"
    synthetic_path = cfg.synthetic_dir / f"synthetic_{table_name}.csv"

    original  = _read(original_path,  label=f"{table_name} original")
    synthetic = _read(synthetic_path, label=f"{table_name} synthetic")

    if original is not None and synthetic is not None:
        logger.info(
            "[LOAD] %s — original: %d rows × %d cols | synthetic: %d rows × %d cols",
            table_name, len(original), len(original.columns),
            len(synthetic), len(synthetic.columns),
        )
    elif original is not None:
        logger.warning(
            "[LOAD] %s — original: %d rows | synthetic: NOT FOUND (%s)",
            table_name, len(original), synthetic_path,
        )
    elif synthetic is not None:
        logger.warning(
            "[LOAD] %s — original: NOT FOUND (%s) | synthetic: %d rows",
            table_name, original_path, len(synthetic),
        )
    else:
        logger.warning("[LOAD] %s — both files missing, skipping table", table_name)

    return original, synthetic


def load_all_pairs(cfg: EvaluationConfig) -> dict[str, TablePair]:
    """Load all tables defined in cfg.tables.

    Returns a dict keyed by table name.  Missing files produce None
    entries; the pipeline continues gracefully.
    """
    return {table: load_table_pair(table, cfg) for table in cfg.tables}


def _read(path: Path, label: str) -> pd.DataFrame | None:
    if not path.exists():
        return None
    try:
        df = pd.read_csv(path, low_memory=False)
        return df
    except Exception as exc:
        logger.error("Failed to read %s (%s): %s", label, path, exc)
        return None
