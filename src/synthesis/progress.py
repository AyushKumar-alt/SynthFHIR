"""Production-grade monitoring, logging, and reporting for Phase 4B training.

Components
----------
EpochSniffer    : Context manager wrapping sys.stderr to capture SDV tqdm epoch
                  updates without modifying synthesis code.  Fires an on_epoch
                  callback once per completed epoch.
GPUMonitor      : Daemon thread that polls GPU stats every interval_s seconds
                  via torch.cuda and nvidia-smi.
ProgressTracker : Central state machine for startup banner, per-table headers,
                  epoch progress, table completion blocks, overall status,
                  statistics files, ZIP packaging, and final summary.

Wiring
------
1. run_phase4b.py creates a ProgressTracker and prints banner / data summary.
2. pipeline.run_full() receives the tracker, wraps each synth.fit() call with
   EpochSniffer(on_epoch=tracker.record_epoch), and calls begin/complete hooks.
3. Tracker saves stats files and ZIP at the end of run_full().
"""

from __future__ import annotations

import csv
import datetime
import json
import re
import shutil
import subprocess
import sys
import threading
import time
import zipfile
from pathlib import Path
from typing import Callable


# ── Encoding-safe symbols ──────────────────────────────────────────────────────
# Windows PowerShell may not display Unicode; detect and fall back to ASCII.
def _unicode_ok() -> bool:
    enc = getattr(sys.stdout, "encoding", "") or ""
    return enc.lower().replace("-", "") in ("utf8", "utf16")


if _unicode_ok():
    _SYM_DONE    = "✓"   # ✓
    _SYM_RUN     = "→"   # →
    _SYM_PENDING = "·"   # ·
    _SYM_SKIP    = "✓"   # ✓
else:
    _SYM_DONE    = "[OK]"
    _SYM_RUN     = "[>>]"
    _SYM_PENDING = "[  ]"
    _SYM_SKIP    = "[SK]"


# ── Format helpers ─────────────────────────────────────────────────────────────

def _fmt_dur(seconds: float) -> str:
    s = int(max(0.0, seconds))
    if s < 60:
        return f"{s} sec"
    if s < 3600:
        return f"{s // 60} min {s % 60:02d} sec"
    h, rem = divmod(s, 3600)
    return f"{h} hr {rem // 60:02d} min"


def _fmt_ts(dt: datetime.datetime | None = None) -> str:
    return (dt or datetime.datetime.now()).strftime("%Y-%m-%d %H:%M:%S")


def _fmt_clock(dt: datetime.datetime | None = None) -> str:
    return (dt or datetime.datetime.now()).strftime("%I:%M:%S %p")


# ── Per-epoch runtime estimates (CPU seconds/epoch, and GPU speedup factor) ────
# Calibrated to: SDV 1.37.2, Python 3.12, typical i7/i9, 998-patient dataset.
_EPOCH_EST_CPU_S: dict[str, float] = {
    "patients":     13.5,   # CTGAN,  998 rows
    "encounters":   27.0,   # PAR,    998 patients, avg 57.8 rows/patient
    "observations": 11.0,   # CTGAN,  303 K rows
    "conditions":   14.0,   # PAR,    998 patients, avg 37.9 rows/patient
    "medications":  17.0,   # PAR,    998 patients, avg 46.8 rows/patient
}
_GPU_SPEEDUP: dict[str, float] = {
    "patients":     12.0,
    "encounters":    6.0,
    "observations": 12.0,
    "conditions":    6.0,
    "medications":   6.0,
}

_TABLE_ORDER  = ["patients", "encounters", "observations", "conditions", "medications"]
_TABLE_LABELS = {
    "patients":     "Patients    ",
    "encounters":   "Encounters  ",
    "observations": "Observations",
    "conditions":   "Conditions  ",
    "medications":  "Medications ",
}
_MODEL_LABELS = {
    "ctgan": "CTGANSynthesizer",
    "tvae":  "TVAESynthesizer",
    "par":   "PARSynthesizer",
}


# ── EpochSniffer ───────────────────────────────────────────────────────────────

class _SnifferStream:
    """Minimal file-like proxy: passes writes through to original and to callback."""

    def __init__(self, original, callback: Callable[[str], None]) -> None:
        self._orig = original
        self._cb   = callback

    def write(self, text: str) -> int:
        n = self._orig.write(text)
        try:
            self._cb(text)
        except Exception:
            pass  # monitoring must never crash training
        return n

    def flush(self) -> None:
        self._orig.flush()

    def fileno(self) -> int:
        return self._orig.fileno()

    def isatty(self) -> bool:
        return False

    def __getattr__(self, name: str):
        return getattr(self._orig, name)


class EpochSniffer:
    """Context manager that intercepts sys.stderr to parse SDV tqdm epoch lines.

    SDV writes tqdm progress to stderr as:
        CTGAN : ``\\rGen. (+02.13) | Discrim. (-00.04):  50%|█| 1/300 [...]``
        PAR   : ``\\rLoss (+06.69): 100%|█| 150/300 [...]``

    The on_epoch callback receives (epoch: int, total: int, loss: float) and
    is fired ONCE per new completed epoch count.  Logging every epoch to a
    file while printing to console every CONSOLE_EVERY epochs is handled by
    ProgressTracker.record_epoch(), not here.

    This class does NOT modify any synthesis logic — it wraps sys.stderr as a
    side channel only.
    """

    # Parse CTGAN line: capture gen_loss, current, total
    _RE_CTGAN = re.compile(
        r"Gen\.\s+\(([+-]?\d+\.\d+)\).*?(\d+)/(\d+)\s+\["
    )
    # Parse PAR line: capture loss, current, total
    _RE_PAR = re.compile(
        r"Loss\s+\(([+-]?\d+\.\d+)\).*?(\d+)/(\d+)\s+\["
    )

    def __init__(self, on_epoch: Callable[[int, int, float], None], model_type: str) -> None:
        self._on_epoch = on_epoch
        self._pattern  = (
            self._RE_CTGAN
            if model_type in ("ctgan", "tvae", "gaussian_copula")
            else self._RE_PAR
        )
        self._last_epoch = -1
        self._orig: object = None

    def __enter__(self) -> "EpochSniffer":
        self._orig  = sys.stderr
        sys.stderr  = _SnifferStream(self._orig, self._dispatch)
        return self

    def __exit__(self, *_) -> None:
        sys.stderr = self._orig

    def _dispatch(self, text: str) -> None:
        # Strip ANSI colour codes and carriage returns before matching
        clean = re.sub(r"\x1b\[[0-9;]*[mK]|\r", "", text)
        m = self._pattern.search(clean)
        if not m:
            return
        try:
            loss  = float(m.group(1))
            epoch = int(m.group(2))
            total = int(m.group(3))
        except (ValueError, IndexError):
            return
        if epoch > self._last_epoch:   # fire once per new epoch count
            self._last_epoch = epoch
            self._on_epoch(epoch, total, loss)


# ── GPUMonitor ─────────────────────────────────────────────────────────────────

class GPUMonitor:
    """Daemon thread that polls GPU stats every interval_s seconds.

    Collects: allocated memory, utilisation, temperature, power draw.
    Prints a compact status line and accumulates records for gpu_usage.csv.
    """

    def __init__(self, interval_s: int = 60) -> None:
        self._interval = interval_s
        self._stop     = threading.Event()
        self._thread   = threading.Thread(target=self._run, daemon=True, name="gpu-monitor")
        self.records:  list[dict] = []

    def start(self) -> None:
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        self._thread.join(timeout=max(5, self._interval // 2))

    def _run(self) -> None:
        while not self._stop.wait(self._interval):
            rec = self._poll()
            if rec:
                self.records.append(rec)
                self._print(rec)

    def _poll(self) -> dict | None:
        try:
            import torch
            if not torch.cuda.is_available():
                return None
            alloc  = torch.cuda.memory_allocated(0) / 1024 ** 2
            total  = torch.cuda.get_device_properties(0).total_memory / 1024 ** 2
            rec: dict = {
                "timestamp":    _fmt_ts(),
                "mem_used_mb":  round(alloc, 1),
                "mem_total_mb": round(total, 1),
                "mem_pct":      round(alloc / total * 100, 1),
            }
            try:
                out = subprocess.check_output(
                    [
                        "nvidia-smi",
                        "--query-gpu=utilization.gpu,temperature.gpu,power.draw",
                        "--format=csv,noheader,nounits",
                    ],
                    timeout=5,
                    stderr=subprocess.DEVNULL,
                    text=True,
                )
                parts = out.strip().split(",")
                if len(parts) == 3:
                    rec["gpu_util_pct"] = float(parts[0].strip())
                    rec["temp_c"]       = float(parts[1].strip())
                    rec["power_w"]      = float(parts[2].strip())
            except Exception:
                pass
            return rec
        except ImportError:
            return None

    @staticmethod
    def _print(rec: dict) -> None:
        parts = [
            f"GPU Mem: {rec['mem_used_mb']:.0f}/{rec['mem_total_mb']:.0f} MB"
            f" ({rec['mem_pct']:.1f}%)"
        ]
        if "gpu_util_pct" in rec:
            parts.append(f"Util: {rec['gpu_util_pct']:.0f}%")
        if "temp_c" in rec:
            parts.append(f"Temp: {rec['temp_c']:.0f}C")
        if "power_w" in rec:
            parts.append(f"Power: {rec['power_w']:.0f}W")
        print(f"\n  [GPU] {' | '.join(parts)}", flush=True)


# ── ProgressTracker ────────────────────────────────────────────────────────────

class ProgressTracker:
    """Central tracker for all training progress, display, and statistics.

    Usage pattern (handled automatically by pipeline.run_full):

        tracker.print_banner(...)
        tracker.print_data_summary(...)
        tracker.print_runtime_estimate(...)
        tracker.start_gpu_monitor()

        for table, pk in TABLE_SEQUENCE:
            if checkpoint_exists:
                tracker.skip_table(table)
                continue
            tracker.begin_table(table, model, n_rows, epochs, idx, total)
            with EpochSniffer(on_epoch=tracker.record_epoch, model_type=model):
                synth = trainer.train(...)
            tracker.complete_table(table, elapsed, n_synthetic, model_path, csv_path)

        tracker.stop_gpu_monitor()
        tracker.save_statistics()
        tracker.create_zip(project_root, outputs_dir)
        tracker.print_final_summary(device_name, cuda_version, start_time)
    """

    # Print epoch summary to console every N epochs; log to file every epoch
    CONSOLE_EVERY: int = 10

    def __init__(
        self,
        epochs:    int,
        log_dir:   Path,
        synth_dir: Path,
        model_dir: Path,
    ) -> None:
        self.epochs    = epochs
        self.log_dir   = log_dir
        self.synth_dir = synth_dir
        self.model_dir = model_dir

        self._statuses:    dict[str, str]  = {t: "pending" for t in _TABLE_ORDER}
        self._table_stats: dict[str, dict] = {}
        self._epoch_log:   list[dict]      = []
        self._gpu:         GPUMonitor | None = None

        # Reset per table
        self._cur_table: str   = ""
        self._cur_model: str   = ""
        self._cur_t0:    float = 0.0

    # ── Startup display ────────────────────────────────────────────────────────

    def print_banner(
        self,
        device_name:  str,
        use_cuda:     bool,
        epochs:       int,
        n_patients:   int,
        project_root: Path,
    ) -> None:
        w   = 62
        bar = "=" * w
        print(f"\n{bar}")
        print(f"{'SynthFHIR Phase 4B':^{w}}")
        print(f"{'Synthetic Healthcare Data Generation Pipeline':^{w}}")
        print(bar)
        for k, v in [
            ("GPU",          device_name),
            ("CUDA",         str(use_cuda)),
            ("Epochs",       str(epochs)),
            ("Patients",     str(n_patients)),
            ("Project Root", str(project_root)),
            ("Started At",   _fmt_ts()),
        ]:
            print(f"  {k:<16}: {v}")
        print(bar)

    def print_data_summary(self, table_sizes: dict[str, int]) -> None:
        total = sum(table_sizes.values())
        sep   = "-" * 52
        print(f"\n{sep}")
        print("  DATA SUMMARY")
        print(sep)
        for name in _TABLE_ORDER:
            n = table_sizes.get(name, 0)
            print(f"  {_TABLE_LABELS[name]} : {n:>10,} rows")
        print(sep)
        print(f"  {'Total Rows':<14} : {total:>10,}")
        print(sep)

    def print_runtime_estimate(
        self,
        use_cuda:         bool,
        epochs:           int,
        completed_tables: list[str],
    ) -> None:
        sep = "-" * 52
        print(f"\n{sep}")
        print("  ESTIMATED RUNTIME  (approximate)")
        print(sep)
        total_s = 0.0
        for name in _TABLE_ORDER:
            label = _TABLE_LABELS[name]
            if name in completed_tables:
                print(f"  {label} : (checkpoint — will skip)")
                continue
            cpu_s = _EPOCH_EST_CPU_S.get(name, 10.0)
            if use_cuda:
                est_s = cpu_s / _GPU_SPEEDUP.get(name, 8.0) * epochs
            else:
                est_s = cpu_s * epochs
            total_s += est_s
            print(f"  {label} : {_fmt_dur(est_s)}")
        print(sep)
        print(f"  {'Estimated Total':<14} : {_fmt_dur(total_s)}")
        print(f"  (Rough estimate; actual time varies by hardware)")
        print(sep)

    # ── Per-table display ──────────────────────────────────────────────────────

    def begin_table(
        self,
        table_name: str,
        model_type: str,
        n_rows:     int,
        n_epochs:   int,
        idx:        int,
        total:      int,
    ) -> None:
        self._statuses[table_name] = "running"
        self._cur_table = table_name
        self._cur_model = model_type
        self._cur_t0    = time.time()

        model_label = _MODEL_LABELS.get(model_type, model_type.upper())
        sep = "=" * 51
        print(f"\n{sep}")
        print(f"  Training Table {idx} / {total}")
        print()
        print(f"  Table  : {table_name}")
        print(f"  Rows   : {n_rows:,}")
        print(f"  Model  : {model_label}")
        print(f"  Epochs : {n_epochs}")
        print(f"  Started: {_fmt_clock()}")
        print(sep)

    def record_epoch(self, epoch: int, total: int, loss: float) -> None:
        """Called by EpochSniffer on each newly completed epoch."""
        elapsed = time.time() - self._cur_t0
        avg_s   = elapsed / max(epoch, 1)
        rem_s   = avg_s * max(total - epoch, 0)

        # Always log to file
        self._epoch_log.append({
            "table":       self._cur_table,
            "model":       self._cur_model,
            "epoch":       epoch,
            "total":       total,
            "loss":        round(loss, 4),
            "elapsed_s":   round(elapsed, 2),
            "avg_epoch_s": round(avg_s, 2),
            "remaining_s": round(rem_s, 2),
            "timestamp":   _fmt_ts(),
        })

        # Console: every CONSOLE_EVERY epochs and on final epoch
        if epoch % self.CONSOLE_EVERY == 0 or epoch == total:
            print(
                f"\n  Epoch {epoch}/{total}"
                f"  |  Elapsed: {_fmt_dur(elapsed)}"
                f"  |  Avg/epoch: {avg_s:.1f}s"
                f"  |  Remaining: {_fmt_dur(rem_s)}"
                f"  |  Loss: {loss:+.4f}",
                flush=True,
            )

    def complete_table(
        self,
        table_name:  str,
        elapsed_s:   float,
        n_synthetic: int,
        model_path:  Path,
        csv_path:    Path,
    ) -> None:
        self._statuses[table_name] = "done"
        self._table_stats[table_name] = {
            "elapsed_s":   round(elapsed_s, 1),
            "n_synthetic": n_synthetic,
            "model_path":  str(model_path),
            "csv_path":    str(csv_path),
        }

        def _rel(p: Path) -> str:
            try:
                return str(p.relative_to(p.parents[2]))
            except (ValueError, IndexError):
                return str(p)

        sep = "-" * 52
        print(f"\n{sep}")
        print(f"  Table Completed : {table_name}")
        print(f"  Training Time   : {_fmt_dur(elapsed_s)}")
        print(f"  Synthetic Rows  : {n_synthetic:,}")
        print(f"  Model Saved     : {_rel(model_path)}")
        print(f"  CSV Saved       : {_rel(csv_path)}")
        print(sep)
        self.print_overall_progress()

    def skip_table(self, table_name: str) -> None:
        self._statuses[table_name] = "skipped"
        print(f"\n  [CHECKPOINT] {table_name}: already complete — skipping.")
        self.print_overall_progress()

    def print_overall_progress(self) -> None:
        sym = {
            "done":    _SYM_DONE,
            "running": _SYM_RUN,
            "pending": _SYM_PENDING,
            "skipped": _SYM_SKIP,
        }
        txt = {
            "done":    "",
            "running": "Running",
            "pending": "Pending",
            "skipped": "",
        }
        print()
        print("  Overall Progress:")
        for name in _TABLE_ORDER:
            label  = _TABLE_LABELS[name]
            status = self._statuses.get(name, "pending")
            print(f"    {sym.get(status, ' ')}  {label}  {txt.get(status, '')}")

    # ── GPU monitoring ─────────────────────────────────────────────────────────

    def start_gpu_monitor(self, interval_s: int = 60) -> None:
        try:
            import torch
            if not torch.cuda.is_available():
                return
        except ImportError:
            return
        self._gpu = GPUMonitor(interval_s=interval_s)
        self._gpu.start()

    def stop_gpu_monitor(self) -> None:
        if self._gpu:
            self._gpu.stop()

    # ── Statistics files ───────────────────────────────────────────────────────

    def save_statistics(self) -> None:
        self.log_dir.mkdir(parents=True, exist_ok=True)

        # training_statistics.json
        stats_json = {
            "tables":    self._table_stats,
            "epochs":    self.epochs,
            "generated": _fmt_ts(),
        }
        (self.log_dir / "training_statistics.json").write_text(
            json.dumps(stats_json, indent=2), encoding="utf-8"
        )

        # training_statistics.csv
        if self._table_stats:
            _write_csv(
                self.log_dir / "training_statistics.csv",
                ["table", "elapsed_s", "n_synthetic", "model_path", "csv_path"],
                [{"table": k, **v} for k, v in self._table_stats.items()],
            )

        # epoch_times.csv
        if self._epoch_log:
            _write_csv(
                self.log_dir / "epoch_times.csv",
                list(self._epoch_log[0].keys()),
                self._epoch_log,
            )

        # gpu_usage.csv
        if self._gpu and self._gpu.records:
            _write_csv(
                self.log_dir / "gpu_usage.csv",
                list(self._gpu.records[0].keys()),
                self._gpu.records,
            )

        print(f"\n  Statistics saved to: {self.log_dir}")

    # ── ZIP packaging ──────────────────────────────────────────────────────────

    def create_zip(self, project_root: Path, outputs_dir: Path) -> Path:
        """Bundle outputs into SynthFHIR_Outputs.zip."""
        zip_path = project_root / "SynthFHIR_Outputs.zip"

        dirs_to_include = [
            outputs_dir / "models",
            outputs_dir / "synthetic",
            outputs_dir / "logs",
        ]
        extras = [
            project_root / "data" / "ready" / "metadata.json",
        ]

        with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
            for d in dirs_to_include:
                if d.exists():
                    for f in sorted(d.rglob("*")):
                        if f.is_file():
                            zf.write(f, f.relative_to(project_root))
            for e in extras:
                if e.exists():
                    zf.write(e, e.relative_to(project_root))

        size_mb = zip_path.stat().st_size / 1024 / 1024
        print(f"\n  ZIP created: {zip_path.name}  ({size_mb:.1f} MB)")

        # Kaggle: copy to /kaggle/working/ so it appears in the Output panel
        kaggle_dir = Path("/kaggle/working")
        if kaggle_dir.exists():
            dest = kaggle_dir / "SynthFHIR_Outputs.zip"
            shutil.copy2(zip_path, dest)
            bar = "=" * 60
            print(f"\n{bar}")
            print(f"{'Download Ready':^60}")
            print(bar)
            print(f"  File : {dest}")
            print("  Click the file in the Kaggle Output panel to download.")
            print(bar)

        return zip_path

    # ── Final summary ──────────────────────────────────────────────────────────

    def print_final_summary(
        self,
        device_name:  str,
        cuda_version: str,
        start_time:   datetime.datetime,
    ) -> None:
        total_s      = sum(v.get("elapsed_s", 0) for v in self._table_stats.values())
        total_syn    = sum(v.get("n_synthetic", 0) for v in self._table_stats.values())
        n_models     = sum(1 for v in self._table_stats.values() if v.get("model_path"))
        sym_ok       = {
            "done": _SYM_DONE, "skipped": _SYM_SKIP,
            "pending": "!", "running": "?",
        }

        w   = 56
        bar = "=" * w
        print(f"\n{bar}")
        print(f"{'TRAINING COMPLETED':^{w}}")
        print(bar)
        for name in _TABLE_ORDER:
            label  = _TABLE_LABELS[name]
            status = self._statuses.get(name, "pending")
            elapsed = self._table_stats.get(name, {}).get("elapsed_s", 0)
            rows    = self._table_stats.get(name, {}).get("n_synthetic", 0)
            sym_chr = sym_ok.get(status, " ")
            if status in ("done", "skipped"):
                detail = f"{rows:,} rows  {_fmt_dur(elapsed)}"
            else:
                detail = status.upper()
            print(f"  {sym_chr}  {label}  {detail}")
        print()
        print(f"  Total Runtime      : {_fmt_dur(total_s)}")
        print(f"  GPU                : {device_name}")
        print(f"  CUDA Version       : {cuda_version}")
        print(f"  Total Syn. Records : {total_syn:,}")
        print(f"  Models Saved       : {n_models}")
        print()
        print(f"  Outputs:")
        print(f"    {self.model_dir}")
        print(f"    {self.synth_dir}")
        print(f"    {self.log_dir}")
        print(bar)


# ── CSV helper ─────────────────────────────────────────────────────────────────

def _write_csv(path: Path, fields: list[str], rows: list[dict]) -> None:
    with open(path, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=fields, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)
