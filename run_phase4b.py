"""Phase 4B — Full Synthetic Data Generation.

Trains all 5 tables and saves synthetic CSVs + model files.
Checkpointing: any table whose synthetic CSV already exists is skipped,
so re-running after a Colab disconnect continues from the next table.

Usage — local (Windows):
    python run_phase4b.py

Usage — Google Colab (T4 GPU):
    !PROJECT_ROOT=/content/SynthFHIR python run_phase4b.py
    # or:
    import os; os.environ["PROJECT_ROOT"] = "/content/SynthFHIR"
    !python run_phase4b.py

The single variable PROJECT_ROOT controls all paths.
No other edits are needed when switching environments.
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
from pathlib import Path


# ── PROJECT_ROOT resolution ────────────────────────────────────────────────────
# Priority: CLI --project-root  >  env var PROJECT_ROOT  >  directory of this file
def _resolve_project_root(cli_value: str | None) -> Path:
    if cli_value:
        return Path(cli_value).resolve()
    env = os.environ.get("PROJECT_ROOT")
    if env:
        return Path(env).resolve()
    return Path(__file__).parent.resolve()


# ── GPU detection ──────────────────────────────────────────────────────────────
def detect_device() -> tuple[bool, str]:
    """Return (use_cuda, display_name) based on torch availability."""
    try:
        import torch
        if torch.cuda.is_available():
            return True, torch.cuda.get_device_name(0)
    except ImportError:
        pass
    return False, "CPU"


# ── Dependency guard ───────────────────────────────────────────────────────────
def _require_sdv() -> None:
    try:
        import sdv  # noqa: F401
    except ImportError:
        print(
            "\n[ERROR] SDV is not installed.\n"
            "Install with:  pip install -r requirements.txt\n"
        )
        sys.exit(1)


# ── Logging setup ──────────────────────────────────────────────────────────────
def setup_logging(log_file: Path, level: str) -> None:
    log_file.parent.mkdir(parents=True, exist_ok=True)
    numeric_level = getattr(logging, level.upper(), logging.INFO)
    fmt = logging.Formatter(
        fmt="%(asctime)s | %(levelname)-8s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    root = logging.getLogger()
    root.setLevel(numeric_level)
    for handler in [
        logging.FileHandler(log_file, encoding="utf-8", mode="a"),
        logging.StreamHandler(sys.stdout),
    ]:
        handler.setFormatter(fmt)
        root.addHandler(handler)


# ── Main ───────────────────────────────────────────────────────────────────────
def main() -> None:
    _require_sdv()

    parser = argparse.ArgumentParser(
        description="Synthetic Health — Phase 4B: Full Training"
    )
    parser.add_argument(
        "--project-root", "-r",
        default=None,
        help=(
            "Project root directory.  "
            "Overrides the PROJECT_ROOT environment variable.  "
            "Default: directory containing this script."
        ),
    )
    parser.add_argument(
        "--config",
        default=None,
        help=(
            "Path to settings.yaml.  "
            "Default: <project-root>/config/settings.yaml"
        ),
    )
    args = parser.parse_args()

    project_root = _resolve_project_root(args.project_root)
    config_path  = Path(args.config).resolve() if args.config else (
        project_root / "config" / "settings.yaml"
    )

    # Ensure imports work regardless of where the script is invoked from
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))

    from src.config_loader import load_config, ensure_directories
    from src.synthesis.config import load_synthesis_config
    from src.synthesis.pipeline import SynthesisPipeline

    config       = load_config(str(config_path))
    synth_config = load_synthesis_config(config_path)

    # Resolve log path inside project so it works on any OS
    log_file = project_root / "outputs" / "logs" / "phase4b.log"
    setup_logging(log_file, config.log_level)

    logger = logging.getLogger(__name__)

    # ── Device detection ───────────────────────────────────────────────────────
    use_cuda, device_name = detect_device()
    if use_cuda:
        logger.info("Running on %s", device_name)
    else:
        logger.info("Running on CPU")

    # ── Config summary ─────────────────────────────────────────────────────────
    logger.info("Project root:  %s", project_root)
    logger.info("Config:        %s", config_path)
    logger.info("Ready dir:     %s", config.ready_dir)
    logger.info("Synthetic dir: %s", synth_config.synthetic_dir)
    logger.info("Model dir:     %s", synth_config.model_dir)
    logger.info("Epochs:        %d", synth_config.epochs)
    logger.info("N patients:    %d", synth_config.n_synthetic_patients)
    logger.info("Table models:  %s", synth_config.table_models)

    # ── Directory setup ────────────────────────────────────────────────────────
    ensure_directories(config)
    synth_config.synthetic_dir.mkdir(parents=True, exist_ok=True)
    synth_config.model_dir.mkdir(parents=True, exist_ok=True)

    # ── Run full pipeline ──────────────────────────────────────────────────────
    pipeline = SynthesisPipeline(
        synth_config=synth_config,
        ready_dir=config.ready_dir,
        reports_dir=config.reports_dir,
    )

    pipeline.run_full(use_cuda=use_cuda, device_name=device_name)

    logger.info("Phase 4B complete.")
    logger.info("Synthetic CSVs: %s", synth_config.synthetic_dir)
    logger.info("Models:         %s", synth_config.model_dir)
    logger.info("Stats:          %s", synth_config.model_dir / "training_time.json")
    logger.info("Run log:        %s", synth_config.model_dir / "training_log.txt")


if __name__ == "__main__":
    main()
