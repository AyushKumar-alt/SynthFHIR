"""Phase 4B — SDV 1.37.2 Compatibility Smoke Test.

Tests CTGAN (patients) and PARSynthesizer (encounters) end-to-end:
    train → sample → save → load → sample → write CSV

Every step is independently timed and PASS/FAIL reported.
If all steps pass, the full 300-epoch Phase 4B training can proceed safely.

Usage:
    python run_smoke_4b.py
"""

from __future__ import annotations

import json
import sys
import time
import warnings
from pathlib import Path

# ── ANSI colour is unreliable on Windows terminals; use plain ASCII labels
_PASS = "[PASS]"
_FAIL = "[FAIL]"
_SKIP = "[SKIP]"


# ── Helpers ───────────────────────────────────────────────────────────────────

class Step:
    """Tracks step results so the summary table is always complete."""

    def __init__(self):
        self._results: list[tuple[str, str, float, str]] = []

    def run(self, label: str, fn):
        """Execute fn(), record PASS/FAIL, return the function's return value."""
        t0 = time.time()
        try:
            result = fn()
            elapsed = time.time() - t0
            self._results.append((label, _PASS, elapsed, ""))
            print(f"  {_PASS}  {label}  ({elapsed:.2f}s)")
            return result
        except Exception as exc:
            elapsed = time.time() - t0
            self._results.append((label, _FAIL, elapsed, str(exc)))
            print(f"  {_FAIL}  {label}  ({elapsed:.2f}s)")
            print(f"         Error: {exc}")
            return None

    def summary(self) -> bool:
        """Print the summary table. Returns True if all steps passed."""
        width = max(len(r[0]) for r in self._results) + 2
        sep   = "-" * (width + 26)
        print()
        print(sep)
        print(f"  {'Step':<{width}}  {'Result':<6}  {'Time':>7}")
        print(sep)
        n_fail = 0
        for label, status, elapsed, _ in self._results:
            print(f"  {label:<{width}}  {status:<6}  {elapsed:>6.2f}s")
            if status == _FAIL:
                n_fail += 1
        print(sep)
        total = sum(r[2] for r in self._results)
        overall = _PASS if n_fail == 0 else _FAIL
        print(f"  {'OVERALL':<{width}}  {overall:<6}  {total:>6.2f}s")
        print(sep)
        if n_fail:
            print(f"\n  {n_fail} step(s) FAILED — fix before running full Phase 4B.")
        else:
            print("\n  All steps PASSED — SDV 1.37.2 compatible. Run: python run_phase4b.py")
        print()
        return n_fail == 0


def _header(title: str) -> None:
    print()
    print("=" * 60)
    print(f"  {title}")
    print("=" * 60)


# ── Paths ─────────────────────────────────────────────────────────────────────

PROJECT_ROOT = Path(__file__).parent
READY_DIR    = PROJECT_ROOT / "data" / "ready"
OUT_DIR      = PROJECT_ROOT / "outputs" / "smoke_4b"
OUT_DIR.mkdir(parents=True, exist_ok=True)


# ── Load shared fixtures ──────────────────────────────────────────────────────

def _load_fixtures() -> tuple:
    import pandas as pd

    with open(READY_DIR / "metadata.json", encoding="utf-8") as fh:
        meta_all = json.load(fh)

    patients_df  = pd.read_csv(READY_DIR / "patients_ready.csv",  low_memory=False)
    encounters_df = pd.read_csv(READY_DIR / "encounters_ready.csv", low_memory=False)

    patients_meta  = meta_all["tables"]["patients"]
    encounters_meta = meta_all["tables"]["encounters"]

    return patients_df, patients_meta, encounters_df, encounters_meta


# ── CTGAN smoke (patients) ─────────────────────────────────────────────────────

def smoke_ctgan(patients_df, patients_meta, tracker: Step):
    """Train CTGANSynthesizer for 2 epochs, sample 10 rows, save/load/resample."""
    from sdv.single_table import CTGANSynthesizer
    from sdv.metadata import Metadata
    import warnings

    # ── Build metadata (same as ctgan_trainer.build_single_table_metadata) ──
    def _build_meta():
        multi_dict = {
            "METADATA_SPEC_VERSION": "MULTI_TABLE_V1",
            "tables": {"_table": patients_meta},
            "relationships": [],
        }
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=FutureWarning)
            m = Metadata.load_from_dict(multi_dict)
        return m.get_table_metadata("_table")

    # ── Coerce boolean columns (0/1 → True/False) ──────────────────────────
    def _coerce_booleans(df):
        import pandas as pd
        bool_cols = [
            col for col, info in patients_meta.get("columns", {}).items()
            if info.get("sdtype") == "boolean" and col in df.columns
        ]
        if not bool_cols:
            return df
        df = df.copy()
        for col in bool_cols:
            df[col] = df[col].map(lambda x: None if pd.isna(x) else bool(int(x)))
        return df

    st_meta = tracker.run("CTGAN: build metadata", _build_meta)
    if st_meta is None:
        return None

    df_ready = _coerce_booleans(patients_df)

    # ── Train ───────────────────────────────────────────────────────────────
    def _train():
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=FutureWarning)
            synth = CTGANSynthesizer(st_meta, epochs=2, batch_size=500, verbose=True)
            synth.fit(df_ready)
        return synth

    synth = tracker.run("CTGAN: train 2 epochs on patients (998 rows)", _train)
    if synth is None:
        return None

    # ── Sample ──────────────────────────────────────────────────────────────
    def _sample():
        out = synth.sample(num_rows=10)
        assert len(out) == 10, f"Expected 10 rows, got {len(out)}"
        assert "patient_id" in out.columns, "Missing patient_id column"
        return out

    synthetic = tracker.run("CTGAN: sample 10 rows", _sample)

    # ── Save ────────────────────────────────────────────────────────────────
    model_path = OUT_DIR / "smoke_patients_ctgan.pkl"

    def _save():
        synth.save(str(model_path))
        assert model_path.exists(), f"Model file not found: {model_path}"
        size_kb = model_path.stat().st_size / 1024
        print(f"         Model size: {size_kb:.1f} KB")

    tracker.run("CTGAN: save model", _save)

    # ── Load ────────────────────────────────────────────────────────────────
    def _load():
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=FutureWarning)
            loaded = CTGANSynthesizer.load(str(model_path))
        return loaded

    loaded_synth = tracker.run("CTGAN: load model", _load)

    # ── Sample from loaded model ─────────────────────────────────────────────
    def _resample():
        out = loaded_synth.sample(num_rows=10)
        assert len(out) == 10, f"Expected 10 rows, got {len(out)}"
        return out

    resampled = tracker.run("CTGAN: sample 10 rows from loaded model", _resample)

    # ── Write CSV ────────────────────────────────────────────────────────────
    csv_path = OUT_DIR / "smoke_synthetic_patients.csv"

    def _write_csv():
        df = resampled if resampled is not None else synthetic
        if df is None:
            raise RuntimeError("No synthetic data to write (prior step failed)")
        df.to_csv(csv_path, index=False)
        assert csv_path.exists(), f"CSV not found: {csv_path}"
        print(f"         Rows: {len(df)}   Path: {csv_path.name}")

    tracker.run("CTGAN: write synthetic_patients CSV", _write_csv)
    return resampled if resampled is not None else synthetic


# ── PAR smoke (encounters) ─────────────────────────────────────────────────────

def smoke_par(encounters_df, encounters_meta, tracker: Step):
    """Train PARSynthesizer for 2 epochs, sample 3 sequences, save/load/resample."""
    from sdv.sequential import PARSynthesizer
    from sdv.metadata import SingleTableMetadata
    import warnings

    # ── Build PAR-specific metadata (sequence_key must be in the dict) ──────
    def _build_meta():
        st_dict = {
            **encounters_meta,
            "METADATA_SPEC_VERSION": "V1",
            "sequence_key":   "patient_id",
            "sequence_index": "sequence_index",
        }
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=FutureWarning)
            m = SingleTableMetadata.load_from_dict(st_dict)
        assert m.sequence_key   == "patient_id",    f"Bad sequence_key: {m.sequence_key}"
        assert m.sequence_index == "sequence_index", f"Bad sequence_index: {m.sequence_index}"
        return m

    st_meta = tracker.run("PAR:  build metadata (sequence_key in SingleTableMetadata)", _build_meta)
    if st_meta is None:
        return None

    # ── Train on a 10-patient slice (keeps test fast, ~30 rows) ────────────
    sample_pids = encounters_df["patient_id"].unique()[:10]
    tiny_df     = encounters_df[encounters_df["patient_id"].isin(sample_pids)].copy()
    n_rows, n_seqs = len(tiny_df), tiny_df["patient_id"].nunique()

    def _train():
        synth = PARSynthesizer(
            st_meta,
            context_columns=[],
            segment_size=None,
            epochs=2,
            sample_size=1,
            cuda=False,
            verbose=True,
        )
        synth.fit(tiny_df)
        return synth

    label = f"PAR:  train 2 epochs on encounters ({n_rows} rows / {n_seqs} patients)"
    synth = tracker.run(label, _train)
    if synth is None:
        return None

    # ── Sample 3 sequences ───────────────────────────────────────────────────
    # PAR samples by num_sequences (each sequence = one patient's events)
    def _sample():
        out = synth.sample(num_sequences=3)
        assert len(out) > 0,           "Sample returned 0 rows"
        assert "patient_id" in out.columns, "Missing patient_id column"
        assert out["patient_id"].nunique() <= 3, "Too many unique patients in sample"
        print(f"         Generated {len(out)} rows for 3 sequences")
        return out

    synthetic = tracker.run("PAR:  sample 3 sequences (variable row count)", _sample)

    # ── Save ─────────────────────────────────────────────────────────────────
    model_path = OUT_DIR / "smoke_encounters_par.pkl"

    def _save():
        synth.save(str(model_path))
        assert model_path.exists(), f"Model file not found: {model_path}"
        size_kb = model_path.stat().st_size / 1024
        print(f"         Model size: {size_kb:.1f} KB")

    tracker.run("PAR:  save model", _save)

    # ── Load ─────────────────────────────────────────────────────────────────
    def _load():
        loaded = PARSynthesizer.load(str(model_path))
        return loaded

    loaded_synth = tracker.run("PAR:  load model", _load)

    # ── Sample from loaded model ─────────────────────────────────────────────
    def _resample():
        out = loaded_synth.sample(num_sequences=3)
        assert len(out) > 0, "Loaded model sample returned 0 rows"
        print(f"         Generated {len(out)} rows for 3 sequences")
        return out

    resampled = tracker.run("PAR:  sample 3 sequences from loaded model", _resample)

    # ── Write CSV ─────────────────────────────────────────────────────────────
    csv_path = OUT_DIR / "smoke_synthetic_encounters.csv"

    def _write_csv():
        df = resampled if resampled is not None else synthetic
        if df is None:
            raise RuntimeError("No synthetic data to write (prior step failed)")
        df.to_csv(csv_path, index=False)
        assert csv_path.exists(), f"CSV not found: {csv_path}"
        print(f"         Rows: {len(df)}   Path: {csv_path.name}")

    tracker.run("PAR:  write synthetic_encounters CSV", _write_csv)
    return resampled if resampled is not None else synthetic


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    warnings.filterwarnings("ignore", category=FutureWarning)

    print()
    print("=" * 60)
    print("  Phase 4B — SDV 1.37.2 Compatibility Smoke Test")
    print("  CTGAN (patients) + PARSynthesizer (encounters)")
    print("  2 epochs each | 10 sample rows | save/load cycle")
    print("=" * 60)

    # ── Dependency check ──────────────────────────────────────────────────
    try:
        import sdv
        import pandas as pd
        print(f"\n  SDV version:    {sdv.__version__}")
        print(f"  Python version: {sys.version.split()[0]}")
        try:
            import torch
            cuda_str = f"CUDA {torch.version.cuda}" if torch.cuda.is_available() else "CPU only"
            print(f"  PyTorch:        {torch.__version__} ({cuda_str})")
        except ImportError:
            print("  PyTorch:        not installed (CPU training)")
    except ImportError:
        print("\n  [ERROR] SDV is not installed. Run: pip install -r requirements.txt")
        sys.exit(1)

    # ── Load data ────────────────────────────────────────────────────────
    print()
    print("Loading ready tables ...")
    try:
        patients_df, patients_meta, encounters_df, encounters_meta = _load_fixtures()
        print(f"  patients:  {patients_df.shape[0]:,} rows x {patients_df.shape[1]} cols")
        print(f"  encounters: {encounters_df.shape[0]:,} rows x {encounters_df.shape[1]} cols")
    except FileNotFoundError as exc:
        print(f"\n  [ERROR] Ready tables not found: {exc}")
        print("  Run Phase 3 first: python run_phase3.py")
        sys.exit(1)

    tracker = Step()

    # ── CTGAN smoke ───────────────────────────────────────────────────────
    _header("CTGAN — patients table")
    smoke_ctgan(patients_df, patients_meta, tracker)

    # ── PAR smoke ─────────────────────────────────────────────────────────
    _header("PAR — encounters table (10-patient slice)")
    smoke_par(encounters_df, encounters_meta, tracker)

    # ── Output location ───────────────────────────────────────────────────
    _header("Output files")
    for f in sorted(OUT_DIR.iterdir()):
        size_kb = f.stat().st_size / 1024
        print(f"  {f.name:<45}  {size_kb:>7.1f} KB")

    # ── Final summary ─────────────────────────────────────────────────────
    _header("Summary")
    passed = tracker.summary()
    sys.exit(0 if passed else 1)


if __name__ == "__main__":
    main()
