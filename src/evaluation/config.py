"""Evaluation configuration — loaded from settings.yaml + defaults."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import yaml

# Mirrors pipeline.py TABLE_SEQUENCE (never import from synthesis to keep
# evaluation fully independent of training code).
TABLE_SEQUENCE: list[tuple[str, str]] = [
    ("patients",     "patient_id"),
    ("encounters",   "encounter_id"),
    ("observations", "observation_id"),
    ("conditions",   "condition_id"),
    ("medications",  "medication_id"),
]

# Columns excluded from numeric / categorical evaluation.
# UUIDs and monotonic counters have no distribution worth comparing.
EXCLUDED_COLUMNS: frozenset[str] = frozenset({
    "patient_id", "encounter_id", "observation_id",
    "condition_id", "medication_id", "sequence_index",
})


@dataclass(frozen=True)
class EvaluationConfig:
    ready_dir:     Path          # data/ready/    — original _ready.csv files
    synthetic_dir: Path          # outputs/synthetic/ — synthetic_*.csv files
    output_dir:    Path          # outputs/evaluation/
    plots_dir:     Path          # outputs/evaluation/plots/
    metadata_path: Path          # outputs/synthetic/metadata.json
    tables:        tuple[str, ...]
    pk_map:        dict[str, str]  # {"patients": "patient_id", ...}


def load_evaluation_config(
    settings_path: str | Path,
    project_root: Path | None = None,
) -> EvaluationConfig:
    """Parse settings.yaml and build EvaluationConfig.

    Resolves all relative paths against the project root (parent of
    config/).  Matches the same convention as all other loaders in
    this project.
    """
    settings_path = Path(settings_path).resolve()
    with open(settings_path, encoding="utf-8") as fh:
        raw = yaml.safe_load(fh)

    root = project_root or settings_path.parent.parent

    def resolve(p: str) -> Path:
        path = Path(p)
        return path if path.is_absolute() else (root / path).resolve()

    paths = raw.get("paths", {})
    syn   = raw.get("synthesis", {})
    ev    = raw.get("evaluation", {})

    ready_dir     = resolve(paths.get("ready_dir", "data/ready"))
    synthetic_dir = resolve(syn.get("synthetic_dir", "outputs/synthetic"))
    output_dir    = resolve(ev.get("output_dir", "outputs/evaluation"))

    return EvaluationConfig(
        ready_dir     = ready_dir,
        synthetic_dir = synthetic_dir,
        output_dir    = output_dir,
        plots_dir     = output_dir / "plots",
        metadata_path = synthetic_dir / "metadata.json",
        tables        = tuple(t for t, _ in TABLE_SEQUENCE),
        pk_map        = {t: pk for t, pk in TABLE_SEQUENCE},
    )
