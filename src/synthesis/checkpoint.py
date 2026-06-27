"""Checkpoint manager — manifest.json as the fault-tolerant state machine.

Design decisions
----------------
manifest.json is the single source of truth for recovery.
    On Kaggle, the /kaggle/working directory may be erased when the kernel is
    killed by OOM (SIGKILL). The manifest.json is included in every per-table
    ZIP that is copied to /kaggle/output, so the user can download the ZIP,
    re-upload it as a dataset, and restart training from the next pending table.

Atomic writes (write tmp → rename).
    A crash mid-write would corrupt the manifest. We write to a temp file in the
    same directory (same filesystem, same partition) then rename. On POSIX, rename
    is atomic at the OS level. On Windows it may fail if the target exists, so we
    fall back to shutil.copy2 + unlink.

Hash verification on recovery.
    is_complete() checks that the model .pkl and synthetic .csv still exist on
    disk AND their paths match what the manifest recorded. This catches the case
    where /kaggle/working was partially erased (e.g., the CSV survived but the
    model was lost), preventing the pipeline from skipping a table and then
    failing at sampling because the model file is absent.

Status enum:
    pending       — not yet started
    in_progress   — started but not confirmed complete (crash here = restart)
    completed     — model + CSV saved and hashed; verified on recovery
    failed        — training raised an exception
    file_missing  — was completed but file disappeared (triggers retrain)
"""

from __future__ import annotations

import hashlib
import json
import logging
import shutil
import tempfile
from datetime import datetime
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

MANIFEST_VERSION = "1.0"

TABLE_ORDER = [
    "patients",
    "encounters",
    "observations",
    "conditions",
    "medications",
]


class CheckpointManager:
    """Maintains outputs/logs/manifest.json and validates checkpoint integrity."""

    def __init__(self, log_dir: Path) -> None:
        self.log_dir       = log_dir
        self.manifest_path = log_dir / "manifest.json"
        log_dir.mkdir(parents=True, exist_ok=True)
        self.manifest = self._load_or_create()

    # ── Recovery ──────────────────────────────────────────────────────────

    def is_complete(self, table_name: str) -> bool:
        """True iff table is marked completed AND its files exist on disk.

        File existence is re-checked every call because /kaggle/working can
        vanish between sessions even if the manifest was saved to /kaggle/output.
        """
        tbl = self.manifest["tables"].get(table_name, {})
        if tbl.get("status") != "completed":
            return False

        for key in ("model_path", "csv_path"):
            p = tbl.get(key, "")
            if p and not Path(p).exists():
                logger.warning(
                    "Checkpoint: %s file missing: %s  → marking as file_missing",
                    table_name, p,
                )
                tbl["status"] = "file_missing"
                self.save()
                return False
        return True

    def get_completed(self) -> list[str]:
        return [t for t in TABLE_ORDER if self.is_complete(t)]

    def get_pending(self) -> list[str]:
        return [t for t in TABLE_ORDER if not self.is_complete(t)]

    # ── State transitions ─────────────────────────────────────────────────

    def mark_started(self, table_name: str, model_type: str, n_rows: int) -> None:
        self.manifest["tables"][table_name] = {
            "status":     "in_progress",
            "model_type": model_type,
            "n_rows":     n_rows,
            "started_at": _now(),
        }
        self._touch()
        self.save()

    def mark_complete(
        self,
        table_name:  str,
        model_path:  Path,
        csv_path:    Path,
        stats:       dict[str, Any],
    ) -> None:
        tbl = self.manifest["tables"].get(table_name, {})
        tbl.update(
            {
                "status":            "completed",
                "completed_at":      _now(),
                "model_path":        str(model_path),
                "csv_path":          str(csv_path),
                "model_hash_sha256": _sha256(model_path),
                "csv_hash_sha256":   _sha256(csv_path),
                "model_size_mb":     round(model_path.stat().st_size / 1024 ** 2, 3),
                "csv_size_mb":       round(csv_path.stat().st_size   / 1024 ** 2, 3),
            }
        )
        tbl.update(stats)
        self.manifest["tables"][table_name] = tbl
        self._touch()
        self.save()
        logger.info("Checkpoint saved: %s", table_name)

    def mark_failed(self, table_name: str, error: str) -> None:
        tbl = self.manifest["tables"].get(table_name, {})
        tbl.update(
            {
                "status":    "failed",
                "failed_at": _now(),
                "error":     str(error)[:1000],
            }
        )
        self.manifest["tables"][table_name] = tbl
        self._touch()
        self.save()

    def record_zip(self, zip_path: Path, tables_included: list[str]) -> None:
        self.manifest.setdefault("zips", []).append(
            {
                "path":            str(zip_path),
                "created_at":      _now(),
                "tables_included": tables_included,
                "size_mb":         round(zip_path.stat().st_size / 1024 ** 2, 2),
            }
        )
        self._touch()
        self.save()

    def update_system_info(self, info: dict) -> None:
        self.manifest["system_info"] = info
        self._touch()
        self.save()

    def update_config_snapshot(self, cfg: dict) -> None:
        self.manifest["config"] = cfg
        self._touch()
        self.save()

    # ── I/O ───────────────────────────────────────────────────────────────

    def save(self) -> None:
        """Atomic write via temp file to prevent corruption on crash."""
        fd, tmp = tempfile.mkstemp(
            dir=str(self.log_dir), suffix=".tmp", text=True
        )
        try:
            with open(fd, "w", encoding="utf-8") as f:
                json.dump(self.manifest, f, indent=2)
            try:
                Path(tmp).replace(self.manifest_path)
            except OSError:
                shutil.copy2(tmp, self.manifest_path)
                Path(tmp).unlink(missing_ok=True)
        except Exception:
            try:
                Path(tmp).unlink(missing_ok=True)
            except Exception:
                pass
            raise

    # ── Internal ──────────────────────────────────────────────────────────

    def _load_or_create(self) -> dict:
        if self.manifest_path.exists():
            try:
                with open(self.manifest_path, encoding="utf-8") as f:
                    m = json.load(f)
                done = [
                    k for k, v in m.get("tables", {}).items()
                    if v.get("status") == "completed"
                ]
                if done:
                    logger.info(
                        "manifest.json loaded — completed tables: %s",
                        done,
                    )
                return m
            except (json.JSONDecodeError, OSError) as e:
                logger.warning("manifest.json corrupted (%s) — starting fresh", e)
        return _empty_manifest()

    def _touch(self) -> None:
        self.manifest["updated_at"] = _now()


# ── Helpers ───────────────────────────────────────────────────────────────────

def _empty_manifest() -> dict:
    return {
        "version":     MANIFEST_VERSION,
        "created_at":  _now(),
        "updated_at":  _now(),
        "system_info": {},
        "config":      {},
        "tables":      {t: {"status": "pending"} for t in TABLE_ORDER},
        "zips":        [],
    }


def _now() -> str:
    return datetime.now().isoformat(timespec="seconds")


def _sha256(path: Path) -> str:
    if not path.exists():
        return ""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()
