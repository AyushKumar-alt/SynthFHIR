"""Phase 4B — Fault-Tolerant Synthetic Data Generation.

CRITICAL: The first block of code sets environment variables that limit joblib
worker processes. These MUST be set before any import of SDV, sklearn, rdt,
joblib, or numpy (which may link against OpenMP). Moving or delaying this block
will re-enable parallel DataTransformer workers and reproduce the SIGKILL crash.

Why this works
--------------
SDV's RDT DataTransformer calls joblib.Parallel(n_jobs=-1) to transform columns.
On Kaggle (4-core CPU), that spawns 4 child processes. Each receives a full copy
of the input DataFrame. For observations (303,696 rows, ~140 MB), this creates
4 × 140 MB = 560 MB in workers before any transformation output — exhausting the
~13 GB Kaggle RAM budget alongside CUDA memory and Python overhead, causing the
OS to issue SIGKILL(-9), which is not catchable and wipes /kaggle/working.

LOKY_MAX_CPU_COUNT=1 limits joblib's loky backend to 1 worker.
run_full() additionally wraps every synth.fit() call in
joblib.parallel_backend('sequential'), which eliminates even that 1 worker and
runs DataTransformer entirely in the main process. An OOM at that point raises
Python MemoryError (catchable) instead of SIGKILL (not catchable), allowing the
GaussianCopula fallback to activate.

Usage — train all tables (default)
    python run_phase4b.py

Usage — train one table
    python run_phase4b.py --table observations
    python run_phase4b.py --table patients
    python run_phase4b.py --table encounters
    python run_phase4b.py --table conditions
    python run_phase4b.py --table medications

    After training, outputs/table_<name>.zip is created. On Kaggle it is
    automatically copied to /kaggle/working/ for one-click download.

Usage — retrain even if checkpoint exists
    python run_phase4b.py --table observations --force

Usage — Kaggle (T4 GPU)
    import os; os.environ["PROJECT_ROOT"] = "/kaggle/working/SynthFHIR"
    !python /kaggle/working/SynthFHIR/run_phase4b.py --table observations

Usage — Google Colab (T4 GPU)
    from google.colab import drive; drive.mount('/content/drive')
    import os; os.environ["PROJECT_ROOT"] = "/content/SynthFHIR"
    !python /content/SynthFHIR/run_phase4b.py --table observations

Recovery after Kaggle session wipe
    See detailed instructions in src/synthesis/backup.py module docstring.
"""

from __future__ import annotations

# =============================================================================
# MUST come before any import of SDV / sklearn / rdt / joblib / torch / numpy.
# Caps the joblib loky worker pool to 1, preventing DataTransformer from
# forking copies of the DataFrame into parallel worker processes.
# Using setdefault() so an explicit env override from the caller still wins.
# =============================================================================
import os
os.environ.setdefault("LOKY_MAX_CPU_COUNT",  "1")
os.environ.setdefault("OMP_NUM_THREADS",     "1")
os.environ.setdefault("MKL_NUM_THREADS",     "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS","1")
# =============================================================================

import argparse
import datetime
import json
import logging
import sys
from pathlib import Path

_VALID_TABLES = ["patients", "encounters", "observations", "conditions", "medications"]


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


# ── Logging: tee stdout to file ───────────────────────────────────────────────

class _TeeStream:
    """Writes to both the original stream and a log file with timestamps."""

    def __init__(self, original, log_file: Path) -> None:
        self._orig = original
        self._log  = open(log_file, "a", encoding="utf-8")  # noqa: SIM115

    def write(self, text: str) -> int:
        n = self._orig.write(text)
        if text.strip():
            ts = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            line = f"[{ts}] {text}" if not text.startswith("[") else text
            self._log.write(line)
            self._log.flush()
        return n

    def flush(self)       -> None: self._orig.flush()
    def fileno(self)      -> int:  return self._orig.fileno()
    def isatty(self)      -> bool: return False
    def __getattr__(self, name): return getattr(self._orig, name)


def setup_logging(log_dir: Path, level: str) -> Path:
    log_dir.mkdir(parents=True, exist_ok=True)
    log_file = log_dir / "training.log"

    fmt  = logging.Formatter(
        fmt="%(asctime)s | %(levelname)-8s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    root = logging.getLogger()
    root.setLevel(getattr(logging, level.upper(), logging.INFO))
    for handler in [
        logging.FileHandler(log_file, encoding="utf-8", mode="a"),
        logging.StreamHandler(sys.stdout),
    ]:
        handler.setFormatter(fmt)
        root.addHandler(handler)

    sys.stdout = _TeeStream(sys.stdout, log_file)   # type: ignore[assignment]
    return log_file


# ── Single-table banners ───────────────────────────────────────────────────────

def _print_table_training_banner(
    table_name:  str,
    model_type:  str,
    epochs:      int,
    n_rows:      int,
    device_name: str,
) -> None:
    """Startup banner for --table mode (requirement 8)."""
    w = 42
    sep = "=" * w
    print()
    print(sep)
    print("  TABLE TRAINING MODE")
    print(sep)
    print(f"  Table  : {table_name}")
    print(f"  Model  : {model_type.upper()}")
    print(f"  Epochs : {epochs:,}")
    print(f"  Rows   : {n_rows:,}")
    print(f"  GPU    : {device_name}")
    print(sep)
    print()


def _print_table_finish_banner(
    table_name: str,
    stats:      dict,
    zip_path:   Path,
) -> None:
    """Finish banner for --table mode (requirement 8)."""
    w = 42
    sep = "=" * w

    elapsed_s = stats.get("training_duration_s", 0)
    h, rem   = divmod(int(elapsed_s), 3600)
    m, s     = divmod(rem, 60)
    if h:
        time_str = f"{h} hr {m:02d} min {s:02d} sec"
    elif m:
        time_str = f"{m} min {s:02d} sec"
    else:
        time_str = f"{s} sec"

    model_path  = stats.get("model_path", "n/a")
    syn_rows    = stats.get("synthetic_rows", 0)
    model_size  = stats.get("model_size_mb", 0)
    zip_size_mb = zip_path.stat().st_size / 1024 ** 2 if zip_path.exists() else 0.0

    print()
    print(sep)
    print("  Training Complete")
    print(sep)
    print(f"  Time          : {time_str}")
    print(f"  Table         : {table_name}")
    print(f"  Model         : {model_path}")
    print(f"  Synthetic Rows: {syn_rows:,}")
    print(f"  Model Size    : {model_size:.1f} MB")
    print(f"  ZIP           : {zip_path}  ({zip_size_mb:.1f} MB)")
    print(sep)
    print()


# ── Recovery banner (full-run mode) ───────────────────────────────────────────

def _print_recovery_status(completed: list[str], pending: list[str]) -> None:
    if not completed:
        return
    print()
    print("  [RECOVERY] Resuming from checkpoint:")
    for t in completed:
        print(f"    [DONE] {t}")
    for t in pending:
        print(f"    [TODO] {t}")
    print()


# ── Resume prompt ──────────────────────────────────────────────────────────────

def _ask_retrain(table_name: str) -> bool:
    """Ask the user whether to retrain a table that already has a checkpoint.

    Returns True → retrain.  Returns False → skip.
    """
    print()
    print(f'  Checkpoint found for "{table_name}".')
    print("  The model and synthetic CSV already exist.")
    try:
        answer = input("  Retrain? (y/N): ").strip().lower()
    except (EOFError, KeyboardInterrupt):
        answer = "n"
    return answer == "y"


# ── Single-table driver ────────────────────────────────────────────────────────

def _run_single_table(
    *,
    table_name:     str,
    force:          bool,
    pipeline,
    checkpoint_mgr,
    backup_mgr,
    tracker,
    use_cuda:       bool,
    device_name:    str,
    synth_config,
    config,
    outputs_dir:    Path,
    project_root:   Path,
    log_file:       Path,
) -> None:
    """Drive the full lifecycle for a single --table run.

    Steps:
    1. Check for existing checkpoint → prompt (unless --force)
    2. Count rows (for banner)
    3. Print TABLE TRAINING MODE banner
    4. Run pipeline.run_full(single_table=table_name)
    5. Create outputs/table_{name}.zip  (req 4)
    6. Copy to /kaggle/working/ + print DOWNLOAD READY banner  (req 5)
    7. Print Training Complete banner  (req 8)
    """
    logger = logging.getLogger(__name__)

    # ── Checkpoint / resume prompt (req 6) ────────────────────────────────
    if not force and checkpoint_mgr.is_complete(table_name):
        if not _ask_retrain(table_name):
            # Re-create ZIP from existing files and exit
            print(f"\n  Skipping retraining. Re-creating ZIP for {table_name} ...")
            zip_path = backup_mgr.create_single_table_zip(table_name)
            backup_mgr.print_kaggle_download_banner(zip_path)
            logger.info("Single-table mode: re-zipped existing checkpoint for %s", table_name)
            return

    # ── Row count for banner ───────────────────────────────────────────────
    ready_csv = config.ready_dir / f"{table_name}_ready.csv"
    if ready_csv.exists():
        n_rows = sum(1 for _ in open(ready_csv, encoding="utf-8")) - 1
    else:
        n_rows = 0

    model_type = synth_config.table_models.get(table_name, "ctgan")

    # ── Startup banner (req 8) ────────────────────────────────────────────
    _print_table_training_banner(table_name, model_type, synth_config.epochs, n_rows, device_name)

    # ── Train ─────────────────────────────────────────────────────────────
    stats = pipeline.run_full(
        use_cuda       = use_cuda,
        device_name    = device_name,
        tracker        = tracker,
        project_root   = project_root,
        outputs_dir    = outputs_dir,
        checkpoint_mgr = checkpoint_mgr,
        backup_mgr     = backup_mgr,
        single_table   = table_name,
    )

    if stats is None:
        # Table was skipped (checkpoint exists and was not marked for retrain).
        # This branch is hit when --force was NOT set and the checkpoint was
        # valid but the user confirmed retrain above, yet the table ended up
        # skipped internally — e.g., manifest is_complete() disagreed.
        print(f"\n  [INFO] {table_name}: no training performed (checkpoint up-to-date).")
        logger.info("Single-table mode: %s skipped by pipeline", table_name)

    # ── Create single-table ZIP (req 4) ───────────────────────────────────
    # disk sync + gc already done inside run_single_table() before we get here
    zip_path = backup_mgr.create_single_table_zip(table_name)

    # ── Kaggle download banner (req 5) ────────────────────────────────────
    backup_mgr.print_kaggle_download_banner(zip_path)

    # ── Finish banner (req 8) ─────────────────────────────────────────────
    if stats is not None:
        _print_table_finish_banner(table_name, stats, zip_path)

    logger.info("Single-table run complete: %s", table_name)
    logger.info("ZIP : %s", zip_path)
    logger.info("Log : %s", log_file)


# ── Main ───────────────────────────────────────────────────────────────────────

def main() -> None:
    _require_sdv()

    parser = argparse.ArgumentParser(
        description="SynthFHIR — Phase 4B: Fault-Tolerant Synthetic Data Generation"
    )
    parser.add_argument(
        "--table",
        choices=_VALID_TABLES + ["all"],
        default="all",
        metavar="TABLE",
        help=(
            "Train a single table instead of all five. "
            "Choices: " + " | ".join(_VALID_TABLES) + " | all  "
            "(default: all). "
            "Creates outputs/table_<TABLE>.zip when done."
        ),
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help=(
            "When --table is specified and a checkpoint already exists, "
            "retrain without asking. Ignored in --table all mode."
        ),
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
    parser.add_argument(
        "--gdrive-dir",
        default=None,
        help=(
            "Google Drive directory for backup ZIPs on Colab. "
            "Example: /content/drive/MyDrive/SynthFHIR"
        ),
    )
    args = parser.parse_args()

    project_root = _resolve_project_root(args.project_root)
    config_path  = (
        Path(args.config).resolve()
        if args.config
        else project_root / "config" / "settings.yaml"
    )
    gdrive_dir = Path(args.gdrive_dir) if args.gdrive_dir else None

    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))

    from src.config_loader import load_config, ensure_directories
    from src.synthesis.config import load_synthesis_config
    from src.synthesis.pipeline import SynthesisPipeline, TABLE_SEQUENCE, TABLE_PK
    from src.synthesis.progress import ProgressTracker
    from src.synthesis.checkpoint import CheckpointManager
    from src.synthesis.backup import BackupManager, auto_recover_from_zip
    from src.synthesis import memory as mem

    config       = load_config(str(config_path))
    synth_config = load_synthesis_config(config_path)

    outputs_dir = project_root / "outputs"
    log_dir     = outputs_dir / "logs"
    log_file    = setup_logging(log_dir, config.log_level)

    logger = logging.getLogger(__name__)
    logger.info(
        "joblib workers: LOKY_MAX_CPU_COUNT=%s  OMP_NUM_THREADS=%s",
        os.environ.get("LOKY_MAX_CPU_COUNT", "?"),
        os.environ.get("OMP_NUM_THREADS", "?"),
    )
    if args.table != "all":
        logger.info("Single-table mode: --table %s  --force %s", args.table, args.force)

    # ── Auto-recovery from ZIP ────────────────────────────────────────────
    recovered = auto_recover_from_zip(project_root, outputs_dir)
    if recovered:
        logger.info("Session recovered from checkpoint ZIP")

    # ── System info ───────────────────────────────────────────────────────
    system_info   = mem.collect_system_info()
    sys_info_path = log_dir / "system_info.json"
    log_dir.mkdir(parents=True, exist_ok=True)
    with open(sys_info_path, "w", encoding="utf-8") as f:
        json.dump(system_info, f, indent=2)
    logger.info(
        "Platform: %s  GPU: %s  RAM: %.1f GB",
        system_info.get("platform"),
        system_info.get("gpu_name", "none"),
        system_info.get("total_ram_gb", 0),
    )

    # ── CheckpointManager ─────────────────────────────────────────────────
    checkpoint_mgr = CheckpointManager(log_dir)
    checkpoint_mgr.update_system_info(system_info)
    checkpoint_mgr.update_config_snapshot(
        {
            "epochs":               synth_config.epochs,
            "batch_size":           synth_config.batch_size,
            "n_synthetic_patients": synth_config.n_synthetic_patients,
            "table_models":         synth_config.table_models,
        }
    )

    # ── BackupManager ─────────────────────────────────────────────────────
    backup_mgr = BackupManager(
        project_root = project_root,
        outputs_dir  = outputs_dir,
        gdrive_dir   = gdrive_dir,
    )

    # ── Device detection ──────────────────────────────────────────────────
    use_cuda, device_name = detect_device()
    if use_cuda:
        logger.info("GPU: %s", device_name)
    else:
        logger.info("Running on CPU")

    # ── ProgressTracker ───────────────────────────────────────────────────
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

    # ── Directory setup ───────────────────────────────────────────────────
    ensure_directories(config)
    synth_config.synthetic_dir.mkdir(parents=True, exist_ok=True)
    synth_config.model_dir.mkdir(parents=True, exist_ok=True)

    # ── Pipeline init ─────────────────────────────────────────────────────
    pipeline = SynthesisPipeline(
        synth_config = synth_config,
        ready_dir    = config.ready_dir,
        reports_dir  = config.reports_dir,
    )

    # ── Dataset summary ───────────────────────────────────────────────────
    TABLE_NAMES = [t for t, _ in TABLE_SEQUENCE]
    table_sizes: dict[str, int] = {}
    for name in TABLE_NAMES:
        csv_path = config.ready_dir / f"{name}_ready.csv"
        if csv_path.exists():
            table_sizes[name] = sum(1 for _ in open(csv_path, encoding="utf-8")) - 1
        else:
            table_sizes[name] = 0
    tracker.print_data_summary(table_sizes)

    # ── Dispatch: single-table vs full run ────────────────────────────────
    if args.table != "all":
        # Single-table mode
        _run_single_table(
            table_name     = args.table,
            force          = args.force,
            pipeline       = pipeline,
            checkpoint_mgr = checkpoint_mgr,
            backup_mgr     = backup_mgr,
            tracker        = tracker,
            use_cuda       = use_cuda,
            device_name    = device_name,
            synth_config   = synth_config,
            config         = config,
            outputs_dir    = outputs_dir,
            project_root   = project_root,
            log_file       = log_file,
        )
    else:
        # Full training mode — existing behaviour, all five tables
        completed = checkpoint_mgr.get_completed()
        pending   = checkpoint_mgr.get_pending()
        _print_recovery_status(completed, pending)
        backup_mgr.print_workflow()

        tracker.print_runtime_estimate(
            use_cuda         = use_cuda,
            epochs           = synth_config.epochs,
            completed_tables = completed,
        )

        pipeline.run_full(
            use_cuda       = use_cuda,
            device_name    = device_name,
            tracker        = tracker,
            project_root   = project_root,
            outputs_dir    = outputs_dir,
            checkpoint_mgr = checkpoint_mgr,
            backup_mgr     = backup_mgr,
        )

        logger.info("Phase 4B complete.")
        logger.info("Synthetic CSVs: %s", synth_config.synthetic_dir)
        logger.info("Models        : %s", synth_config.model_dir)
        logger.info("Manifest      : %s", log_dir / "manifest.json")
        logger.info("Log           : %s", log_file)


if __name__ == "__main__":
    main()
