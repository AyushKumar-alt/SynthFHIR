"""Memory management for fault-tolerant training.

Design decisions
----------------
configure_joblib()
    SDV's RDT DataTransformer calls joblib.Parallel(n_jobs=-1) to fit column
    transformers in parallel. On a 4-core Kaggle instance with default settings,
    that spawns 4 child processes — each receives a full copy of the DataFrame.
    For observations (303K rows, 140 MB), that is 4 × 140 MB = 560 MB in workers
    alone before any transformation overhead, triggering SIGKILL(-9) before
    epoch 1. Setting LOKY_MAX_CPU_COUNT=1 limits the pool to a single loky worker,
    halving the copy overhead. Combined with parallel_backend('sequential') in
    the training call, no worker processes are forked at all — the transform runs
    in the main process, and any OOM raises Python MemoryError (catchable) instead
    of SIGKILL (not catchable).

    MUST be called before importing SDV, sklearn, rdt, or joblib.

cleanup()
    Two passes of gc.collect() are deliberate: the first pass reclaims objects
    whose refcount dropped to zero; objects with __del__ methods may become
    eligible only after that first pass, so the second pass cleans them.

snapshot() / get_ram_mb()
    psutil is optional. If absent, returns 0.0 — the rest of the pipeline
    still works; only memory-usage CSV rows will show zero.
"""

from __future__ import annotations

import gc
import logging
import os
from pathlib import Path

logger = logging.getLogger(__name__)


def configure_joblib() -> None:
    """Limit parallel workers BEFORE any SDV/sklearn/joblib import.

    Call this at the very top of run_phase4b.py before other imports.
    When called from inside an already-loaded module it is still useful as a
    belt-and-suspenders measure, but the env vars may arrive too late for
    worker pools that were already created.
    """
    os.environ.setdefault("LOKY_MAX_CPU_COUNT",   "1")
    os.environ.setdefault("OMP_NUM_THREADS",       "1")
    os.environ.setdefault("MKL_NUM_THREADS",       "1")
    os.environ.setdefault("NUMEXPR_NUM_THREADS",   "1")
    os.environ.setdefault("OPENBLAS_NUM_THREADS",  "1")
    logger.info(
        "memory: joblib workers capped at 1  "
        "(prevents DataTransformer OOM on large tables)"
    )


def cleanup(label: str = "") -> dict:
    """Aggressive memory cleanup. Returns stats dict with freed amounts."""
    before_ram = get_ram_mb()

    gc.collect()
    gc.collect()

    gpu_freed = 0.0
    try:
        import torch
        if torch.cuda.is_available():
            before_gpu = torch.cuda.memory_allocated(0) / 1024 ** 2
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
            gpu_freed = before_gpu - torch.cuda.memory_allocated(0) / 1024 ** 2
    except ImportError:
        pass

    after_ram = get_ram_mb()
    ram_freed = before_ram - after_ram

    tag = f" [{label}]" if label else ""
    logger.info(
        "cleanup%s: RAM freed=%.0f MB  GPU freed=%.0f MB",
        tag, ram_freed, gpu_freed,
    )
    return {
        "ram_before_mb": round(before_ram, 1),
        "ram_after_mb":  round(after_ram, 1),
        "ram_freed_mb":  round(ram_freed, 1),
        "gpu_freed_mb":  round(gpu_freed, 1),
    }


def get_ram_mb() -> float:
    """Current process RSS in MB. Returns 0.0 if psutil is not installed."""
    try:
        import psutil
        return psutil.Process().memory_info().rss / 1024 ** 2
    except ImportError:
        return 0.0


def get_available_ram_gb() -> float:
    """System-wide available RAM in GB. Returns 999.0 if psutil absent."""
    try:
        import psutil
        return psutil.virtual_memory().available / 1024 ** 3
    except ImportError:
        return 999.0


def get_gpu_mb() -> tuple[float, float]:
    """Return (allocated_MB, total_MB) for GPU 0, or (0.0, 0.0)."""
    try:
        import torch
        if torch.cuda.is_available():
            alloc = torch.cuda.memory_allocated(0) / 1024 ** 2
            total = torch.cuda.get_device_properties(0).total_memory / 1024 ** 2
            return alloc, total
    except ImportError:
        pass
    return 0.0, 0.0


def get_disk_free_gb(path: Path) -> float:
    """Free disk space in GB for the partition containing path."""
    try:
        import shutil
        return shutil.disk_usage(str(path)).free / 1024 ** 3
    except Exception:
        return 0.0


def snapshot(label: str = "") -> dict:
    """Full memory snapshot suitable for memory_usage.csv rows."""
    import datetime
    ram        = get_ram_mb()
    avail_ram  = get_available_ram_gb()
    gpu_a, gpu_t = get_gpu_mb()
    return {
        "timestamp":    datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "label":        label,
        "ram_mb":       round(ram, 1),
        "avail_ram_gb": round(avail_ram, 2),
        "gpu_alloc_mb": round(gpu_a, 1),
        "gpu_total_mb": round(gpu_t, 1),
        "gpu_pct":      round(gpu_a / gpu_t * 100, 1) if gpu_t else 0.0,
    }


def collect_system_info() -> dict:
    """One-time system metadata captured at startup for system_info.json."""
    import datetime
    import platform
    import sys

    info: dict = {
        "timestamp":      datetime.datetime.now().isoformat(timespec="seconds"),
        "python_version": sys.version.split()[0],
        "platform":       _detect_platform(),
        "hostname":       platform.node(),
        "cpu_count":      os.cpu_count() or 0,
    }

    # RAM
    try:
        import psutil
        vm = psutil.virtual_memory()
        info["total_ram_gb"]     = round(vm.total    / 1024 ** 3, 2)
        info["available_ram_gb"] = round(vm.available / 1024 ** 3, 2)
    except ImportError:
        info["total_ram_gb"]     = 0.0
        info["available_ram_gb"] = 0.0

    # GPU
    try:
        import torch
        info["torch_version"] = torch.__version__
        if torch.cuda.is_available():
            info["gpu_name"]      = torch.cuda.get_device_name(0)
            info["cuda_version"]  = torch.version.cuda or "unknown"
            props = torch.cuda.get_device_properties(0)
            info["gpu_total_mb"]  = round(props.total_memory / 1024 ** 2, 0)
        else:
            info["gpu_name"]     = "none"
            info["cuda_version"] = "n/a"
            info["gpu_total_mb"] = 0.0
    except ImportError:
        info["torch_version"] = "not installed"
        info["gpu_name"]      = "none"

    # SDV
    try:
        import sdv
        info["sdv_version"] = sdv.__version__
    except ImportError:
        info["sdv_version"] = "not installed"

    return info


def _detect_platform() -> str:
    if Path("/kaggle").exists():
        return "kaggle"
    if Path("/content").exists():
        return "colab"
    return "local"
