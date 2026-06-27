"""Phase 4B — Full Synthetic Data Generation.

Production ML pipeline with:
  - Startup banner with hardware summary
  - Dataset row-count summary
  - Estimated runtime per table
  - Per-table training headers (table N/5, model, rows, start time)
  - Per-epoch progress (console every 10 epochs, file every epoch)
  - Table completion blocks with synthetic row counts and file paths
  - Overall progress tracker after every table
  - GPU memory/utilisation/temperature monitor (every 60 s)
  - Statistics files: training_statistics.json/csv, epoch_times.csv, gpu_usage.csv
  - Final summary with totals and output paths
  - Automatic ZIP of all outputs (SynthFHIR_Outputs.zip)
  - Kaggle /kaggle/working/ copy for one-click download
  - Checkpoint recovery: skips tables whose synthetic CSV already exists

Usage — local (Windows, CPU):
    python run_phase4b.py

Usage — Google Colab (T4 GPU):
    !PROJECT_ROOT=/content/SynthFHIR python run_phase4b.py
    # or set env var then run:
    import os; os.environ["PROJECT_ROOT"] = "/content/SynthFHIR"
    !python run_phase4b.py

All paths derive from PROJECT_ROOT — no other edits needed across environments.
"""

from __future__ import annotations

import argparse
import datetime
import logging
import os
import sys
from pathlib import Path


# ── PROJECT_ROOT ───────────────────────────────────────────────────────────────

def _resolve_project_root(cli_value: str | None) -> Path:
    if cli_value:
        return Path(cli_value).resolve()
    env = os.environ.get("PROJECT_ROOT")
    if env:
        return Path(env).resolve()
    return Path(__file__).parent.resolve()


# ── GPU detection ──────────────────────────────────────────────────────────────

def detect_device() -> tuple[bool, str]:
    """Return (use_cuda, display_name)."""
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


# ── Logging: tee to file ───────────────────────────────────────────────────────

class _TeeStream:
    """Writes to both the original stream and a log file with timestamps."""

    def __init__(self, original, log_file: Path) -> None:
        self._orig   = original
        self._log    = open(log_file, "a", encoding="utf-8")   # noqa: SIM115

    def write(self, text: str) -> int:
        n = self._orig.write(text)
        if text.strip():
            ts = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            self._log.write(f"[{ts}] {text}" if not text.startswith("[") else text)
            self._log.flush()
        return n

    def flush(self) -> None:
        self._orig.flush()

    def fileno(self) -> int:
        return self._orig.fileno()

    def isatty(self) -> bool:
        return False

    def __getattr__(self, name: str):
        return getattr(self._orig, name)


def setup_logging(log_dir: Path, level: str) -> Path:
    log_dir.mkdir(parents=True, exist_ok=True)
    log_file = log_dir / "training.log"

    numeric = getattr(logging, level.upper(), logging.INFO)
    fmt     = logging.Formatter(
        fmt="%(asctime)s | %(levelname)-8s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    root = logging.getLogger()
    root.setLevel(numeric)
    for handler in [
        logging.FileHandler(log_file, encoding="utf-8", mode="a"),
        logging.StreamHandler(sys.stdout),
    ]:
        handler.setFormatter(fmt)
        root.addHandler(handler)

    # Tee stdout so print() calls also land in the log file
    sys.stdout = _TeeStream(sys.stdout, log_file)   # type: ignore[assignment]
    return log_file


# ── Main ───────────────────────────────────────────────────────────────────────

def main() -> None:
    _require_sdv()

    parser = argparse.ArgumentParser(
        description="SynthFHIR — Phase 4B: Full Synthetic Data Generation"
    )
    parser.add_argument(
        "--project-root", "-r",
        default=None,
        help=(
            "Project root directory. "
            "Overrides the PROJECT_ROOT env var. "
            "Default: directory containing this script."
        ),
    )
    parser.add_argument(
        "--config",
        default=None,
        help="Path to settings.yaml. Default: <project-root>/config/settings.yaml",
    )
    args = parser.parse_args()

    project_root = _resolve_project_root(args.project_root)
    config_path  = (
        Path(args.config).resolve()
        if args.config
        else project_root / "config" / "settings.yaml"
    )

    # Ensure project-root is importable
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))

    from src.config_loader import load_config, ensure_directories
    from src.synthesis.config import load_synthesis_config
    from src.synthesis.pipeline import SynthesisPipeline
    from src.synthesis.progress import ProgressTracker

    config       = load_config(str(config_path))
    synth_config = load_synthesis_config(config_path)

    outputs_dir = project_root / "outputs"
    log_dir     = outputs_dir / "logs"
    log_file    = setup_logging(log_dir, config.log_level)

    logger = logging.getLogger(__name__)

    # ── Device detection ───────────────────────────────────────────────────────
    use_cuda, device_name = detect_device()
    if use_cuda:
        logger.info("Running on %s", device_name)
    else:
        logger.info("Running on CPU")

    # ── Tracker setup ──────────────────────────────────────────────────────────
    tracker = ProgressTracker(
        epochs    = synth_config.epochs,
        log_dir   = log_dir,
        synth_dir = synth_config.synthetic_dir,
        model_dir = synth_config.model_dir,
    )

    tracker.print_banner(
        device_name  = device_name,
        use_cuda     = use_cuda,
        epochs       = synth_config.epochs,
        n_patients   = synth_config.n_synthetic_patients,
        project_root = project_root,
    )

    # ── Directory setup ────────────────────────────────────────────────────────
    ensure_directories(config)
    synth_config.synthetic_dir.mkdir(parents=True, exist_ok=True)
    synth_config.model_dir.mkdir(parents=True, exist_ok=True)

    # ── Pipeline init (loads metadata) ────────────────────────────────────────
    pipeline = SynthesisPipeline(
        synth_config = synth_config,
        ready_dir    = config.ready_dir,
        reports_dir  = config.reports_dir,
    )

    # ── Dataset summary ────────────────────────────────────────────────────────
    import pandas as pd
    TABLE_NAMES = ["patients", "encounters", "observations", "conditions", "medications"]
    table_sizes: dict[str, int] = {}
    for name in TABLE_NAMES:
        csv_path = config.ready_dir / f"{name}_ready.csv"
        if csv_path.exists():
            table_sizes[name] = sum(1 for _ in open(csv_path, encoding="utf-8")) - 1
        else:
            table_sizes[name] = 0
    tracker.print_data_summary(table_sizes)

    # ── Runtime estimate (skip already-checkpointed tables) ───────────────────
    completed = [
        name for name in TABLE_NAMES
        if (synth_config.synthetic_dir / f"synthetic_{name}.csv").exists()
    ]
    tracker.print_runtime_estimate(
        use_cuda         = use_cuda,
        epochs           = synth_config.epochs,
        completed_tables = completed,
    )

    # ── Full training ──────────────────────────────────────────────────────────
    pipeline.run_full(
        use_cuda     = use_cuda,
        device_name  = device_name,
        tracker      = tracker,
        project_root = project_root,
        outputs_dir  = outputs_dir,
    )

    logger.info("Phase 4B complete.")
    logger.info("Synthetic CSVs : %s", synth_config.synthetic_dir)
    logger.info("Models         : %s", synth_config.model_dir)
    logger.info("Statistics     : %s", log_dir)
    logger.info("Log            : %s", log_file)


if __name__ == "__main__":
    main()
