"""Synthesis pipeline — orchestrates smoke test (4A) and full run (4B).

smoke_test()
    5 epochs on patients_ready.csv → 20 synthetic rows → 8 sanity checks.
    Blocks full training on any FAIL.

run_full()
    Full training: CTGAN on patients, PAR on all child tables.
    Generates and saves all 5 synthetic CSVs + model files.
    Called by run_phase4b.py (implemented in Phase 4B).
"""

from __future__ import annotations

import dataclasses
import datetime
import gc
import json
import logging
import time
from pathlib import Path
from typing import Any

import pandas as pd
from tqdm import tqdm

from .config import SynthesisConfig
from .ctgan_trainer import CTGANTrainer
from .par_trainer import PARTrainer
from .progress import EpochSniffer, ProgressTracker
from .sampler import sample_patients, sample_sequences, assign_synthetic_ids, remap_to_synthetic_patients
from .evaluator import (
    run_smoke_test,
    print_smoke_results,
    save_smoke_report,
    quick_sanity_report,
)

logger = logging.getLogger(__name__)


class SynthesisPipeline:
    """Top-level pipeline for synthetic data generation."""

    def __init__(
        self,
        synth_config: SynthesisConfig,
        ready_dir: Path,
        reports_dir: Path,
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
        return pd.read_csv(self.ready_dir / f"{name}_ready.csv", low_memory=False)

    # ── Phase 4A: Smoke test ──────────────────────────────────────────────

    def smoke_test(self) -> bool:
        """Train CTGAN for smoke_test_epochs and generate smoke_test_n_rows.

        Returns True if all smoke checks pass, False otherwise.
        Gate: caller should exit(1) on False to block full training.
        """
        wall = time.time()
        sep = "=" * 60

        logger.info(sep)
        logger.info("  PHASE 4A — SMOKE TEST")
        logger.info("  epochs=%d  rows=%d  model=%s",
                    self.cfg.smoke_test_epochs,
                    self.cfg.smoke_test_n_rows,
                    self.cfg.patient_model)
        logger.info(sep)

        # Load patients
        patients_df   = self._load_table("patients")
        patients_meta = self._meta["tables"]["patients"]

        # Train (short run)
        trainer = CTGANTrainer(self.cfg)
        synth   = trainer.train(
            patients_df,
            patients_meta,
            epochs=self.cfg.smoke_test_epochs,
            verbose=True,
        )

        # Sample
        synthetic = sample_patients(synth, self.cfg.smoke_test_n_rows, seed=self.cfg.seed)
        synthetic = assign_synthetic_ids(synthetic, "patient_id", prefix="smoke")

        # Evaluate
        checks = run_smoke_test(
            synthetic=synthetic,
            real=patients_df,
            table_name="patients",
            pk_col="patient_id",
            expected_rows=self.cfg.smoke_test_n_rows,
        )
        print_smoke_results(checks, "patients", synthetic)

        # Save report
        report_path = self.reports_dir / "smoke_test_report.md"
        save_smoke_report(checks, "patients", synthetic, report_path)

        # Save sample CSV for inspection
        sample_path = self.cfg.synthetic_dir / "smoke_patients_sample.csv"
        sample_path.parent.mkdir(parents=True, exist_ok=True)
        synthetic.to_csv(sample_path, index=False)
        logger.info("Sample saved: %s", sample_path)

        elapsed = time.time() - wall
        n_fail  = sum(1 for c in checks if c.status == "FAIL")
        passed  = n_fail == 0

        logger.info("Smoke test %s in %.1f s.", "PASSED" if passed else "FAILED", elapsed)
        return passed

    # ── Phase 4B: Full training ───────────────────────────────────────────

    def run_full(
        self,
        use_cuda:     bool = False,
        device_name:  str  = "CPU",
        tracker:      ProgressTracker | None = None,
        project_root: Path | None = None,
        outputs_dir:  Path | None = None,
    ) -> None:
        """Train all tables and generate the complete synthetic dataset.

        Resumes automatically: tables whose synthetic CSV already exists are
        skipped, so re-running after a disconnect continues from the next table.

        Args:
            use_cuda     : Enable GPU acceleration (torch.cuda).
            device_name  : Human-readable device label for display/stats.
            tracker      : Optional ProgressTracker for production-grade UI.
                           If None, falls back to plain logger output.
            project_root : Project root directory (needed for ZIP creation).
            outputs_dir  : Parent of models/, synthetic/, logs/ directories.
        """
        TABLE_SEQUENCE = [
            ("patients",     "patient_id"),
            ("encounters",   "encounter_id"),
            ("observations", "observation_id"),
            ("conditions",   "condition_id"),
            ("medications",  "medication_id"),
        ]
        n_tables = len(TABLE_SEQUENCE)

        self.cfg.model_dir.mkdir(parents=True, exist_ok=True)
        self.cfg.synthetic_dir.mkdir(parents=True, exist_ok=True)

        stats_path = self.cfg.model_dir / "training_time.json"
        log_path   = self.cfg.model_dir / "training_log.txt"
        stats      = _load_stats(stats_path, device_name)
        start_time = datetime.datetime.now()

        # GPU monitoring thread (no-op on CPU)
        if tracker:
            tracker.start_gpu_monitor(interval_s=60)

        # Cache synthetic patients for FK remapping in child tables
        synthetic_patients: pd.DataFrame | None = None
        sp_path = self.cfg.synthetic_dir / "synthetic_patients.csv"
        if sp_path.exists():
            synthetic_patients = pd.read_csv(sp_path)

        # Real patient count for proportional CTGAN row sampling
        real_patients_df = self._load_table("patients")
        n_real_patients  = len(real_patients_df)

        logger.info("=" * 60)
        logger.info("  PHASE 4B — FULL TRAINING")
        logger.info("  device=%s  epochs=%d  n_patients=%d",
                    device_name, self.cfg.epochs, self.cfg.n_synthetic_patients)
        logger.info("=" * 60)

        _log_event(log_path,
                   f"START Phase 4B | device={device_name} | "
                   f"epochs={self.cfg.epochs} | "
                   f"n_patients={self.cfg.n_synthetic_patients}")

        for idx, (table_name, pk_col) in enumerate(TABLE_SEQUENCE, start=1):
            synth_path = self.cfg.synthetic_dir / f"synthetic_{table_name}.csv"
            model_path = self.cfg.model_dir    / f"{table_name}_model.pkl"
            model_type = self.cfg.table_models.get(table_name, "ctgan")

            # ── Checkpoint: skip already-completed tables ──────────────────
            if synth_path.exists():
                logger.info("[SKIP] %s: checkpoint found", table_name)
                _log_event(log_path, f"SKIP {table_name} | checkpoint exists")
                if table_name == "patients" and synthetic_patients is None:
                    synthetic_patients = pd.read_csv(synth_path)
                if tracker:
                    tracker.skip_table(table_name)
                continue

            # ── Load real table ────────────────────────────────────────────
            df   = self._load_table(table_name)
            meta = self._meta["tables"][table_name]

            logger.info("[START] %s  model=%s  rows=%d", table_name, model_type, len(df))

            if tracker:
                tracker.begin_table(table_name, model_type, len(df),
                                    self.cfg.epochs, idx, n_tables)
            t0 = time.time()

            # ── Train (EpochSniffer intercepts tqdm for per-epoch metrics) ─
            epoch_cb = tracker.record_epoch if tracker else lambda *_: None
            with EpochSniffer(on_epoch=epoch_cb, model_type=model_type):
                if model_type == "par":
                    trainer   = PARTrainer(self.cfg)
                    synth     = trainer.train(df, meta,
                                              epochs=self.cfg.epochs,
                                              verbose=True,
                                              cuda=use_cuda)
                else:
                    if table_name == "patients":
                        n_rows = self.cfg.n_synthetic_patients
                    else:
                        ratio  = self.cfg.n_synthetic_patients / max(n_real_patients, 1)
                        n_rows = max(1, int(len(df) * ratio))

                    cfg_for_table = dataclasses.replace(self.cfg, patient_model=model_type)
                    trainer       = CTGANTrainer(cfg_for_table)
                    synth         = trainer.train(df, meta,
                                                  epochs=self.cfg.epochs,
                                                  verbose=True,
                                                  cuda=use_cuda,
                                                  model_name=model_type)

            # ── Sample ─────────────────────────────────────────────────────
            if model_type == "par":
                synthetic = sample_sequences(synth, self.cfg.n_synthetic_patients)
                if synthetic_patients is not None:
                    synthetic = remap_to_synthetic_patients(synthetic, synthetic_patients)
            else:
                synthetic = sample_patients(synth, n_rows)
                if table_name != "patients" and synthetic_patients is not None:
                    synthetic = remap_to_synthetic_patients(synthetic, synthetic_patients)

            synthetic = assign_synthetic_ids(synthetic, pk_col, prefix="syn")
            elapsed   = time.time() - t0

            # ── Save model and CSV ─────────────────────────────────────────
            trainer.save(synth, model_path)
            synthetic.to_csv(synth_path, index=False)
            logger.info("[DONE] %s: %d synthetic rows in %.1f s",
                        table_name, len(synthetic), elapsed)

            if table_name == "patients":
                synthetic_patients = synthetic

            # ── Checkpoint log and stats ───────────────────────────────────
            _log_event(log_path,
                       f"DONE {table_name} | duration={elapsed:.1f}s | "
                       f"synthetic_rows={len(synthetic)} | model={model_type}")

            stats["tables"][table_name] = {
                "model":               model_type,
                "epochs":              self.cfg.epochs,
                "rows_processed":      len(df),
                "synthetic_rows":      len(synthetic),
                "training_duration_s": round(elapsed, 1),
                "model_file_size_mb":  round(model_path.stat().st_size / 1024 / 1024, 2),
            }
            _save_stats(stats_path, stats)

            if tracker:
                tracker.complete_table(table_name, elapsed, len(synthetic),
                                       model_path, synth_path)

            # ── Memory cleanup ─────────────────────────────────────────────
            del synth, df
            gc.collect()
            try:
                import torch
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            except ImportError:
                pass

        # ── Wrap up ────────────────────────────────────────────────────────
        total_s = sum(
            t.get("training_duration_s", 0) for t in stats["tables"].values()
        )
        stats["total_duration_s"] = round(total_s, 1)
        _save_stats(stats_path, stats)
        _log_event(log_path,
                   f"COMPLETE Phase 4B | total={total_s:.1f}s ({total_s/60:.1f} min)")
        logger.info("Phase 4B complete. Total time: %.1f min", total_s / 60)

        if tracker:
            tracker.stop_gpu_monitor()
            tracker.save_statistics()
            if project_root and outputs_dir:
                tracker.create_zip(project_root, outputs_dir)

            # CUDA version for final summary
            cuda_ver = "n/a"
            try:
                import torch
                cuda_ver = torch.version.cuda or "n/a"
            except ImportError:
                pass
            tracker.print_final_summary(device_name, cuda_ver, start_time)


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
        "run_timestamp": datetime.datetime.now().isoformat(timespec="seconds"),
        "device":        device_name,
        "cuda_version":  cuda_ver,
        "torch_version": torch_ver,
        "tables":        {},
        "total_duration_s": 0,
    }


def _save_stats(stats_path: Path, stats: dict) -> None:
    stats_path.parent.mkdir(parents=True, exist_ok=True)
    with open(stats_path, "w", encoding="utf-8") as fh:
        json.dump(stats, fh, indent=2)


def _log_event(log_path: Path, message: str) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    ts = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    with open(log_path, "a", encoding="utf-8") as fh:
        fh.write(f"[{ts}] {message}\n")
