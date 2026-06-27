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

import json
import logging
import time
from pathlib import Path

import pandas as pd

from .config import SynthesisConfig
from .ctgan_trainer import CTGANTrainer
from .par_trainer import PARTrainer
from .sampler import sample_patients, assign_synthetic_ids
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

    # ── Phase 4B: Full training (stub — implemented in Phase 4B) ─────────

    def run_full(self) -> None:
        """Full training and generation. Implemented in Phase 4B."""
        raise NotImplementedError(
            "Full training is implemented in Phase 4B. "
            "Run the smoke test (Phase 4A) first."
        )
