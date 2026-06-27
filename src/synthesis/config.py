"""Synthesis configuration — loaded from the ``synthesis:`` block in settings.yaml."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import yaml


_DEFAULT_TABLE_MODELS = {
    "patients":     "ctgan",
    "encounters":   "par",
    "observations": "ctgan",
    "conditions":   "par",
    "medications":  "par",
}


@dataclass(frozen=True)
class SynthesisConfig:
    # Model selection (factory keys)
    patient_model: str    # "ctgan" | "tvae" | "gaussian_copula"
    sequence_model: str   # "par"   | "ctgan"

    # Per-table model override (set in settings.yaml synthesis.table_models)
    table_models: dict    # {"patients": "ctgan", "encounters": "par", ...}

    # Full training params
    epochs: int
    batch_size: int
    seed: int
    n_synthetic_patients: int

    # Smoke test params
    smoke_test_epochs: int
    smoke_test_n_rows: int

    # Sequence length control (PAR GPU memory safety)
    # All PAR tables are clipped to this many events per patient before training.
    # Prevents CUDA OOM from outlier patients with hundreds of encounters.
    max_seq_len: int

    # Output directories (absolute)
    synthetic_dir: Path
    model_dir: Path


def load_synthesis_config(settings_path: str | Path) -> SynthesisConfig:
    """Parse the ``synthesis:`` block from settings.yaml.

    All relative paths are resolved against the project root (parent of
    the ``config/`` folder), matching the same convention as ConfigLoader.
    """
    settings_path = Path(settings_path).resolve()
    with open(settings_path, encoding="utf-8") as fh:
        raw = yaml.safe_load(fh)

    project_root = settings_path.parent.parent
    syn = raw.get("synthesis", {})

    def resolve(p: str) -> Path:
        path = Path(p)
        return path if path.is_absolute() else (project_root / path).resolve()

    table_models = {**_DEFAULT_TABLE_MODELS, **syn.get("table_models", {})}

    return SynthesisConfig(
        patient_model=syn.get("patient_model", "ctgan"),
        sequence_model=syn.get("sequence_model", "par"),
        table_models=table_models,
        epochs=int(syn.get("epochs", 300)),
        batch_size=int(syn.get("batch_size", 500)),
        seed=int(syn.get("seed", 42)),
        n_synthetic_patients=int(syn.get("n_synthetic_patients", 1000)),
        smoke_test_epochs=int(syn.get("smoke_test_epochs", 5)),
        smoke_test_n_rows=int(syn.get("smoke_test_n_rows", 20)),
        max_seq_len=int(syn.get("max_seq_len", 50)),
        synthetic_dir=resolve(syn.get("synthetic_dir", "outputs/synthetic")),
        model_dir=resolve(syn.get("model_dir", "outputs/models")),
    )
