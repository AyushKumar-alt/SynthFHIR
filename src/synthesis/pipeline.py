"""Synthesis pipeline — orchestrates smoke test (4A) and full run (4B).

smoke_test()
    5 epochs on patients_ready.csv → 20 synthetic rows → 8 sanity checks.
    Blocks full training on any FAIL.

run_single_table()
    Train exactly one table. Called by run_full() in a loop and directly by
    run_phase4b.py when --table is specified. Returns a stats dict so the
    caller can build banners and ZIP files without duplicating query logic.

run_full()
    Fault-tolerant full training (--table all, the default). Calls
    run_single_table() for each table in TABLE_SEQUENCE, filtered by the
    optional single_table argument. After each table in full mode: creates
    a rolling checkpoint ZIP. In single-table mode: skips the rolling ZIP
    so the caller can create a table-specific ZIP instead.

Key design decisions
--------------------
joblib sequential backend for all training calls.
    SDV's RDT DataTransformer internally calls joblib.Parallel(n_jobs=-1) to
    transform columns. The parallel_backend('sequential') context manager forces
    those calls to run in the main process, serialising them and preventing the
    worker-process RAM multiplication that caused SIGKILL on observations (303K
    rows, 140 MB). This is applied to EVERY table, not just observations, because
    the overhead of a sequential context manager is negligible and the protection
    is valuable everywhere.

CTGAN for observations (not TVAE or GaussianCopula).
    TVAE uses the same RDT DataTransformer as CTGAN, so it would have the same
    OOM issue before our joblib fix — it does not help with the root cause.
    GaussianCopula cannot model conditional distributions: in observations,
    loinc_display determines what value_quantity means (BMI ≈ 25, heart rate
    ≈ 75, etc.), and GaussianCopula would blend all 30 LOINC distributions into
    one marginal, destroying utility. CTGAN with joblib limited to sequential
    mode is the correct choice: it handles mixed categorical/numerical tables and
    can capture the loinc_display-conditioned value_quantity distributions.

MemoryError → GaussianCopula fallback.
    With joblib sequential, an OOM now raises Python MemoryError (catchable)
    rather than SIGKILL(-9) (not catchable). If CTGAN OOMs even in sequential
    mode, we fall back to GaussianCopula, log a warning about reduced utility,
    and record the actual model type in the manifest. GaussianCopula is still
    better than losing the table entirely.

Disk sync before ZIP.
    After saving model + CSV, os.sync() is called (POSIX) or silently skipped
    (Windows) before creating ZIPs, ensuring the kernel buffer is flushed to
    disk before we package the files. This prevents corrupted ZIPs on sudden
    power loss or Kaggle kernel kill.

n_real_patients / synthetic_patients auto-loading.
    run_single_table() loads these from disk when not supplied by the caller.
    This makes it safe to call standalone (--table X) without the caller
    needing to know about inter-table dependencies.
"""

from __future__ import annotations

import csv
import dataclasses
import datetime
import gc
import json
import logging
import os
import time
from pathlib import Path
from typing import Any

import pandas as pd

from .backup import BackupManager
from .checkpoint import CheckpointManager
from .config import SynthesisConfig
from .ctgan_trainer import CTGANTrainer
from .par_trainer import PARTrainer
from .progress import EpochSniffer, ProgressTracker
from .sampler import (
    assign_synthetic_ids,
    remap_to_synthetic_patients,
    sample_patients,
    sample_sequences,
)
from .evaluator import (
    print_smoke_results,
    run_smoke_test,
    save_smoke_report,
)
from . import memory as mem

logger = logging.getLogger(__name__)

TABLE_SEQUENCE: list[tuple[str, str]] = [
    ("patients",     "patient_id"),
    ("encounters",   "encounter_id"),
    ("observations", "observation_id"),
    ("conditions",   "condition_id"),
    ("medications",  "medication_id"),
]

# Convenience lookup — used by run_phase4b when dispatching --table
TABLE_PK: dict[str, str] = {name: pk for name, pk in TABLE_SEQUENCE}


class SynthesisPipeline:
    """Top-level pipeline for synthetic data generation."""

    def __init__(
        self,
        synth_config: SynthesisConfig,
        ready_dir:    Path,
        reports_dir:  Path,
    ) -> None:
        self.cfg         = synth_config
        self.ready_dir   = ready_dir
        self.reports_dir = reports_dir
        self._meta       = self._load_metadata()

    def _load_metadata(self) -> dict:
        path = self.ready_dir / "metadata.json"
        with open(path, encoding="utf-8") as fh:
            return json.load(fh)

    def _load_table(self, name: str) -> pd.DataFrame:
        return pd.read_csv(
            self.ready_dir / f"{name}_ready.csv", low_memory=False
        )

    # ── Phase 4A: Smoke test ──────────────────────────────────────────────

    def smoke_test(self) -> bool:
        """Train CTGAN for smoke_test_epochs and generate smoke_test_n_rows.

        Returns True if all smoke checks pass, False otherwise.
        """
        wall = time.time()
        sep  = "=" * 60

        logger.info(sep)
        logger.info("  PHASE 4A — SMOKE TEST")
        logger.info(
            "  epochs=%d  rows=%d  model=%s",
            self.cfg.smoke_test_epochs,
            self.cfg.smoke_test_n_rows,
            self.cfg.patient_model,
        )
        logger.info(sep)

        patients_df   = self._load_table("patients")
        patients_meta = self._meta["tables"]["patients"]

        trainer = CTGANTrainer(self.cfg)
        synth   = trainer.train(
            patients_df,
            patients_meta,
            epochs=self.cfg.smoke_test_epochs,
            verbose=True,
        )

        synthetic = sample_patients(synth, self.cfg.smoke_test_n_rows, seed=self.cfg.seed)
        synthetic = assign_synthetic_ids(synthetic, "patient_id", prefix="smoke")

        checks = run_smoke_test(
            synthetic=synthetic,
            real=patients_df,
            table_name="patients",
            pk_col="patient_id",
            expected_rows=self.cfg.smoke_test_n_rows,
        )
        print_smoke_results(checks, "patients", synthetic)

        report_path = self.reports_dir / "smoke_test_report.md"
        save_smoke_report(checks, "patients", synthetic, report_path)

        sample_path = self.cfg.synthetic_dir / "smoke_patients_sample.csv"
        sample_path.parent.mkdir(parents=True, exist_ok=True)
        synthetic.to_csv(sample_path, index=False)
        logger.info("Sample saved: %s", sample_path)

        elapsed = time.time() - wall
        n_fail  = sum(1 for c in checks if c.status == "FAIL")
        passed  = n_fail == 0
        logger.info("Smoke test %s in %.1f s.", "PASSED" if passed else "FAILED", elapsed)
        return passed

    # ── Phase 4B core: single-table training ─────────────────────────────

    def run_single_table(
        self,
        table_name:         str,
        pk_col:             str,
        use_cuda:           bool,
        device_name:        str,
        tracker:            ProgressTracker | None,
        checkpoint_mgr:     CheckpointManager | None,
        table_index:        int = 1,
        n_tables:           int = 1,
        n_real_patients:    int | None = None,
        synthetic_patients: pd.DataFrame | None = None,
    ) -> dict | None:
        """Train one table end-to-end.

        Handles: skip check, load, train (joblib sequential), sample, save,
        disk sync, checkpoint update, memory cleanup.

        Returns a stats dict on success, or None if the table was skipped.
        The stats dict is used by run_full() and run_phase4b.py to build
        finish banners and ZIP files.

        n_real_patients and synthetic_patients are loaded from disk when not
        supplied, so this method is fully self-contained when called directly
        from run_phase4b.py --table mode.
        """
        synth_path = self.cfg.synthetic_dir / f"synthetic_{table_name}.csv"
        model_path = self.cfg.model_dir    / f"{table_name}_model.pkl"
        model_type = self.cfg.table_models.get(table_name, "ctgan")

        # ── Skip if already complete ───────────────────────────────────────
        if checkpoint_mgr and checkpoint_mgr.is_complete(table_name):
            logger.info("[SKIP] %s: verified in manifest", table_name)
            if tracker:
                tracker.skip_table(table_name)
            return None

        if not checkpoint_mgr and synth_path.exists():
            logger.info("[SKIP] %s: synthetic CSV exists", table_name)
            if tracker:
                tracker.skip_table(table_name)
            return None

        # ── Resolve inter-table dependencies ──────────────────────────────
        # n_real_patients is needed to compute proportional output row count.
        # synthetic_patients is needed for FK remapping in child tables.
        # Both are loaded from disk when the caller is run_phase4b --table X
        # (standalone mode) rather than run_full() (which caches them).
        if n_real_patients is None:
            patients_df     = self._load_table("patients")
            n_real_patients = len(patients_df)
            del patients_df

        if synthetic_patients is None and table_name != "patients":
            sp_path = self.cfg.synthetic_dir / "synthetic_patients.csv"
            if sp_path.exists():
                synthetic_patients = pd.read_csv(sp_path)
            else:
                logger.warning(
                    "[WARN] synthetic_patients.csv not found — "
                    "FK remapping SKIPPED for %s. "
                    "Run --table patients first to generate it.",
                    table_name,
                )

        # ── Load real table ────────────────────────────────────────────────
        df   = self._load_table(table_name)
        meta = self._meta["tables"][table_name]
        n_rows = len(df)

        logger.info("[START] %s  model=%s  rows=%d", table_name, model_type, n_rows)

        if checkpoint_mgr:
            checkpoint_mgr.mark_started(table_name, model_type, n_rows)

        if tracker:
            tracker.begin_table(
                table_name, model_type, n_rows,
                self.cfg.epochs, table_index, n_tables,
            )

        # ── Train with joblib sequential backend ───────────────────────────
        t0 = time.time()
        actual_model_type = model_type
        epoch_cb = tracker.record_epoch if tracker else lambda *_: None

        try:
            from joblib import parallel_backend as _pb
            _has_pb = True
        except ImportError:
            _has_pb = False

        try:
            with EpochSniffer(on_epoch=epoch_cb, model_type=model_type):
                if _has_pb:
                    with _pb("sequential"):
                        synth, actual_model_type = self._train_table(
                            table_name, model_type, df, meta,
                            use_cuda, n_rows, n_real_patients,
                        )
                else:
                    synth, actual_model_type = self._train_table(
                        table_name, model_type, df, meta,
                        use_cuda, n_rows, n_real_patients,
                    )
        except Exception as e:
            logger.error("[FAIL] %s: training failed: %s", table_name, e)
            if checkpoint_mgr:
                checkpoint_mgr.mark_failed(table_name, str(e))
            mem.cleanup(table_name)
            del df
            gc.collect()
            raise

        # ── Sample ────────────────────────────────────────────────────────
        if actual_model_type == "par":
            synthetic = sample_sequences(synth, self.cfg.n_synthetic_patients)
            if synthetic_patients is not None:
                synthetic = remap_to_synthetic_patients(synthetic, synthetic_patients)
        else:
            if table_name == "patients":
                out_rows = self.cfg.n_synthetic_patients
            else:
                ratio    = self.cfg.n_synthetic_patients / max(n_real_patients, 1)
                out_rows = max(1, int(n_rows * ratio))
            synthetic = sample_patients(synth, out_rows)
            if table_name != "patients" and synthetic_patients is not None:
                synthetic = remap_to_synthetic_patients(synthetic, synthetic_patients)

        synthetic = assign_synthetic_ids(synthetic, pk_col, prefix="syn")
        elapsed   = time.time() - t0

        # ── Flush to disk ─────────────────────────────────────────────────
        # Write model and CSV before sync so the OS has something to flush.
        self.cfg.model_dir.mkdir(parents=True, exist_ok=True)
        self.cfg.synthetic_dir.mkdir(parents=True, exist_ok=True)
        _save_trainer(synth, model_path, actual_model_type, self.cfg)
        synthetic.to_csv(synth_path, index=False)

        # Flush kernel buffers to disk before creating ZIP
        try:
            os.sync()
        except AttributeError:
            pass  # Windows — writes are synchronous enough

        gc.collect()
        gc.collect()
        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.synchronize()
        except ImportError:
            pass

        logger.info(
            "[DONE] %s: %d synthetic rows in %.1f s",
            table_name, len(synthetic), elapsed,
        )

        # ── Stats dict ────────────────────────────────────────────────────
        table_stats: dict[str, Any] = {
            "model_type":          actual_model_type,
            "epochs":              self.cfg.epochs,
            "rows_processed":      n_rows,
            "synthetic_rows":      len(synthetic),
            "training_duration_s": round(elapsed, 1),
            "model_size_mb":       round(model_path.stat().st_size / 1024 ** 2, 3),
            "csv_size_mb":         round(synth_path.stat().st_size  / 1024 ** 2, 3),
            "model_path":          str(model_path),
            "csv_path":            str(synth_path),
        }

        # ── Checkpoint update ─────────────────────────────────────────────
        if checkpoint_mgr:
            checkpoint_mgr.mark_complete(
                table_name, model_path, synth_path, table_stats
            )

        if tracker:
            tracker.complete_table(
                table_name, elapsed, len(synthetic), model_path, synth_path
            )

        # ── Memory cleanup ────────────────────────────────────────────────
        del synth, df, synthetic
        cleanup_stats = mem.cleanup(table_name)
        logger.info(
            "Memory after cleanup: RAM=%.0f MB freed=%.0f MB",
            cleanup_stats["ram_after_mb"],
            cleanup_stats["ram_freed_mb"],
        )

        return table_stats

    # ── Phase 4B: Full / single-table dispatch ────────────────────────────

    def run_full(
        self,
        use_cuda:       bool             = False,
        device_name:    str              = "CPU",
        tracker:        ProgressTracker | None = None,
        project_root:   Path | None      = None,
        outputs_dir:    Path | None      = None,
        checkpoint_mgr: CheckpointManager | None = None,
        backup_mgr:     BackupManager | None = None,
        single_table:   str | None       = None,
    ) -> dict | None:
        """Orchestrate training for all tables or a single table.

        Args:
            single_table: If set, only that table is trained and this method
                          returns its stats dict. The rolling checkpoint ZIP
                          is NOT created — the caller (run_phase4b.py) creates
                          a table-specific ZIP instead.
                          If None (default), all tables are trained in
                          TABLE_SEQUENCE order and rolling checkpoint ZIPs
                          are created after each completed table.

        Returns:
            dict  : stats for the trained table (single_table mode)
            None  : full mode (all tables)
        """
        self.cfg.model_dir.mkdir(parents=True, exist_ok=True)
        self.cfg.synthetic_dir.mkdir(parents=True, exist_ok=True)

        log_dir    = outputs_dir / "logs" if outputs_dir else self.cfg.model_dir
        mem_csv    = log_dir / "memory_usage.csv"
        stats_path = self.cfg.model_dir / "training_time.json"
        stats      = _load_stats(stats_path, device_name)
        start_time = datetime.datetime.now()

        log_dir.mkdir(parents=True, exist_ok=True)

        if tracker:
            tracker.start_gpu_monitor(interval_s=60)

        mem_writer = _MemCSVWriter(mem_csv)

        # Pre-load n_real_patients once — it's the sampling ratio denominator.
        # (998 rows; load is negligible even if we already have it cached.)
        real_patients_df = self._load_table("patients")
        n_real_patients  = len(real_patients_df)
        del real_patients_df

        # Pre-cache synthetic_patients for FK remapping in child tables.
        synthetic_patients: pd.DataFrame | None = None
        sp_path = self.cfg.synthetic_dir / "synthetic_patients.csv"
        if sp_path.exists():
            synthetic_patients = pd.read_csv(sp_path)

        # Decide which tables to train
        if single_table is not None:
            tables_to_process = [
                (name, pk) for name, pk in TABLE_SEQUENCE if name == single_table
            ]
            n_tables = 1
            # Single-table mode: don't print the full-run banner
        else:
            tables_to_process = list(TABLE_SEQUENCE)
            n_tables = len(TABLE_SEQUENCE)
            logger.info("=" * 60)
            logger.info("  PHASE 4B — FULL TRAINING (FAULT-TOLERANT)")
            logger.info(
                "  device=%s  epochs=%d  n_patients=%d",
                device_name, self.cfg.epochs, self.cfg.n_synthetic_patients,
            )
            logger.info("=" * 60)

        for idx, (table_name, pk_col) in enumerate(tables_to_process, start=1):

            mem_writer.write(mem.snapshot(f"{table_name}_before"))

            result = self.run_single_table(
                table_name         = table_name,
                pk_col             = pk_col,
                use_cuda           = use_cuda,
                device_name        = device_name,
                tracker            = tracker,
                checkpoint_mgr     = checkpoint_mgr,
                table_index        = idx,
                n_tables           = n_tables,
                n_real_patients    = n_real_patients,
                synthetic_patients = synthetic_patients,
            )

            if result is not None:
                stats["tables"][table_name] = result
                _save_stats(stats_path, stats)

                # Update synthetic_patients cache so child tables get correct PIDs
                if table_name == "patients":
                    sp_path = self.cfg.synthetic_dir / "synthetic_patients.csv"
                    if sp_path.exists():
                        synthetic_patients = pd.read_csv(sp_path)

                # Rolling checkpoint ZIP — full mode only.
                # Single-table mode skips this; run_phase4b creates table_{name}.zip
                if single_table is None and backup_mgr and checkpoint_mgr:
                    backup_mgr.backup_after_table(
                        table_name,
                        checkpoint_mgr.get_completed(),
                        checkpoint_mgr,
                    )

            mem_writer.write(mem.snapshot(f"{table_name}_after"))

        # ── Wrap up ────────────────────────────────────────────────────────
        total_s = sum(
            t.get("training_duration_s", 0) for t in stats["tables"].values()
        )
        stats["total_duration_s"] = round(total_s, 1)
        _save_stats(stats_path, stats)

        mem_writer.close()

        if tracker:
            tracker.stop_gpu_monitor()
            if single_table is None:
                # Full-mode wrap-up: aggregate stats + final ZIP + final banner
                tracker.save_statistics()
                if project_root and outputs_dir:
                    tracker.create_zip(project_root, outputs_dir)
                cuda_ver = "n/a"
                try:
                    import torch
                    cuda_ver = torch.version.cuda or "n/a"
                except ImportError:
                    pass
                tracker.print_final_summary(device_name, cuda_ver, start_time)

        if single_table is not None:
            return stats.get("tables", {}).get(single_table)
        return None

    # ── Training dispatch ─────────────────────────────────────────────────

    def _train_table(
        self,
        table_name:      str,
        model_type:      str,
        df:              pd.DataFrame,
        meta:            dict,
        use_cuda:        bool,
        n_rows:          int,
        n_real_patients: int,
    ) -> tuple:
        """Train the table. Returns (synth, actual_model_type).

        Falls back to GaussianCopulaSynthesizer if CTGAN raises MemoryError.
        With joblib sequential mode, MemoryError is catchable (unlike SIGKILL).
        GaussianCopula does not capture LOINC-conditioned distributions as
        accurately as CTGAN, so it is a utility-degraded fallback, not a
        first choice.
        """
        if model_type == "par":
            trainer = PARTrainer(self.cfg)
            synth   = trainer.train(
                df, meta, epochs=self.cfg.epochs, verbose=True, cuda=use_cuda
            )
            return synth, "par"

        # CTGAN / TVAE / gaussian_copula
        cfg_for = dataclasses.replace(self.cfg, patient_model=model_type)
        trainer = CTGANTrainer(cfg_for)
        try:
            synth = trainer.train(
                df, meta,
                epochs=self.cfg.epochs, verbose=True,
                cuda=use_cuda, model_name=model_type,
            )
            return synth, model_type

        except MemoryError:
            logger.warning(
                "[OOM] %s: MemoryError during CTGAN fit on %d rows. "
                "Falling back to GaussianCopulaSynthesizer. "
                "NOTE: GaussianCopula cannot model conditional distributions "
                "(value_quantity | loinc_display), so utility for observations "
                "will be reduced. Consider lowering batch_size or reducing "
                "top_n_loinc in settings.yaml to avoid this fallback.",
                table_name, n_rows,
            )
            mem.cleanup(f"{table_name}_oom_fallback")
            gc_cfg = dataclasses.replace(self.cfg, patient_model="gaussian_copula")
            gc_trainer = CTGANTrainer(gc_cfg)
            synth = gc_trainer.train(
                df, meta,
                epochs=self.cfg.epochs, verbose=True,
                cuda=False, model_name="gaussian_copula",
            )
            return synth, "gaussian_copula"


# ── Checkpoint helpers ────────────────────────────────────────────────────────

def _load_stats(stats_path: Path, device_name: str = "CPU") -> dict[str, Any]:
    if stats_path.exists():
        with open(stats_path, encoding="utf-8") as fh:
            return json.load(fh)
    try:
        import torch
        cuda_ver  = torch.version.cuda or "n/a"
        torch_ver = torch.__version__
    except ImportError:
        cuda_ver  = "n/a"
        torch_ver = "n/a"
    return {
        "run_timestamp":    datetime.datetime.now().isoformat(timespec="seconds"),
        "device":           device_name,
        "cuda_version":     cuda_ver,
        "torch_version":    torch_ver,
        "tables":           {},
        "total_duration_s": 0,
    }


def _save_stats(stats_path: Path, stats: dict) -> None:
    stats_path.parent.mkdir(parents=True, exist_ok=True)
    with open(stats_path, "w", encoding="utf-8") as fh:
        json.dump(stats, fh, indent=2)


def _save_trainer(synth, model_path: Path, actual_model_type: str, cfg: SynthesisConfig) -> None:
    """Save model via the appropriate trainer's save() method."""
    if actual_model_type == "par":
        PARTrainer(cfg).save(synth, model_path)
    else:
        cfg_for = dataclasses.replace(cfg, patient_model=actual_model_type)
        CTGANTrainer(cfg_for).save(synth, model_path)


# ── Memory CSV writer ─────────────────────────────────────────────────────────

class _MemCSVWriter:
    """Appends rows to memory_usage.csv after each training phase."""

    _FIELDS = [
        "timestamp", "label", "ram_mb", "avail_ram_gb",
        "gpu_alloc_mb", "gpu_total_mb", "gpu_pct",
    ]

    def __init__(self, path: Path) -> None:
        self._path   = path
        self._is_new = not path.exists()
        self._fh     = open(path, "a", newline="", encoding="utf-8")  # noqa: SIM115
        self._writer = csv.DictWriter(self._fh, fieldnames=self._FIELDS)
        if self._is_new:
            self._writer.writeheader()
            self._fh.flush()

    def write(self, snap: dict) -> None:
        row = {k: snap.get(k, "") for k in self._FIELDS}
        self._writer.writerow(row)
        self._fh.flush()

    def close(self) -> None:
        try:
            self._fh.close()
        except Exception:
            pass
