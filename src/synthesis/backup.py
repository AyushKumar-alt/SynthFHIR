"""Multi-destination backup manager.

After each table completes, this manager:
  1. Creates a rolling per-table ZIP: SynthFHIR_checkpoint_after_{table}.zip
  2. Copies it to all detected destinations in parallel (sequential copy, not
     threaded — copies are fast and we want each to complete before moving on)

Design decisions
----------------
Per-table ZIPs (not one final ZIP).
    If the kernel dies during observations training, the ZIP from after_encounters
    is already sitting in /kaggle/output. The user downloads it, re-uploads as a
    dataset, and restarts — the manifest inside the ZIP tells the pipeline to skip
    patients and encounters and start from observations.

Destination auto-detection.
    /kaggle/output   — Kaggle persists this directory as a "dataset output" when
                       the kernel run completes or is saved. It also survives
                       within-session kernel restarts, which is the common OOM
                       crash scenario.
    /kaggle/working  — primary working dir; mirrored here so local paths work.
    /content/drive   — Google Drive on Colab; only if the drive is mounted.
    local            — project_root itself; always available.

ZIP contents.
    outputs/models/      — all .pkl files trained so far
    outputs/synthetic/   — all synthetic CSVs generated so far
    outputs/logs/        — training.log, manifest.json, statistics files
    data/ready/metadata.json — needed to resume sampling in a fresh session

What is NOT zipped.
    data/ready/*.csv — real patient data; we don't ship it in the backup ZIP
                       (privacy) and it's already available on Kaggle/Colab.
    outputs/reports/ and outputs/figures/ — large, re-creatable.

Recovery workflow (Kaggle session expired).
    1. Download SynthFHIR_checkpoint_after_<last_table>.zip from Kaggle Output.
    2. Upload the ZIP as a new Kaggle dataset named "synthfhir-checkpoint".
    3. In the next notebook run, add that dataset as input.
    4. At the top of the notebook, before running run_phase4b.py:
           import zipfile, os
           with zipfile.ZipFile('/kaggle/input/synthfhir-checkpoint/<zipname>', 'r') as z:
               z.extractall('/kaggle/working/SynthFHIR')
       Then:
           os.environ['PROJECT_ROOT'] = '/kaggle/working/SynthFHIR'
           !python /kaggle/working/SynthFHIR/run_phase4b.py
    5. The pipeline reads manifest.json, sees completed tables, and skips them.

Google Colab auto-recovery.
    Mount Google Drive before running:
        from google.colab import drive
        drive.mount('/content/drive')
    ZIPs are written to /content/drive/MyDrive/SynthFHIR/ automatically.
    On reconnect, unzip from Drive to /content/SynthFHIR and re-run.
"""

from __future__ import annotations

import logging
import shutil
import zipfile
from pathlib import Path

logger = logging.getLogger(__name__)

# Items to include in every ZIP, relative to project_root
_ZIP_DIRS  = ["outputs/models", "outputs/synthetic", "outputs/logs"]
_ZIP_FILES = ["data/ready/metadata.json"]


class BackupManager:
    """Creates per-table ZIP checkpoints and distributes them to all destinations."""

    def __init__(
        self,
        project_root: Path,
        outputs_dir:  Path,
        gdrive_dir:   Path | None = None,
    ) -> None:
        self.project_root = project_root
        self.outputs_dir  = outputs_dir
        self.gdrive_dir   = gdrive_dir
        self.destinations = self._detect_destinations()
        if self.destinations:
            logger.info(
                "Backup destinations: %s",
                [str(d) for d in self.destinations],
            )

    # ── Public API ────────────────────────────────────────────────────────

    def backup_after_table(
        self,
        table_name:        str,
        completed_tables:  list[str],
        checkpoint_mgr:    object,  # CheckpointManager — avoids circular import
    ) -> Path | None:
        """Build ZIP for completed_tables and copy to all destinations.

        Returns the local ZIP path, or None if ZIP creation failed.
        The checkpoint manager records the ZIP entry in manifest.json.
        """
        zip_name = f"SynthFHIR_checkpoint_after_{table_name}.zip"
        zip_path = self.project_root / zip_name

        try:
            self._create_zip(zip_path)
        except Exception as e:
            logger.error("ZIP creation failed after %s: %s", table_name, e)
            return None

        # Record in manifest before copying (so manifest inside ZIP is current)
        try:
            checkpoint_mgr.record_zip(zip_path, list(completed_tables))
            # Re-create ZIP to include the updated manifest
            self._create_zip(zip_path)
        except Exception as e:
            logger.warning("Could not update manifest in ZIP: %s", e)

        # Distribute to backup destinations
        for dest_dir in self.destinations:
            self._copy_to(zip_path, dest_dir, zip_name)

        size_mb = zip_path.stat().st_size / 1024 ** 2
        logger.info("ZIP ready: %s  (%.1f MB)", zip_name, size_mb)
        return zip_path

    def print_workflow(self) -> None:
        """Print environment-specific recovery instructions at startup."""
        env = _detect_environment()
        if env == "kaggle":
            print("\n  [KAGGLE] After each table: ZIP saved to /kaggle/output/")
            print("           If kernel dies: see ZIP recovery steps in backup.py docstring")
        elif env == "colab":
            if any(str(d).startswith("/content/drive") for d in self.destinations):
                print("\n  [COLAB] After each table: ZIP saved to Google Drive")
            else:
                print("\n  [COLAB] Google Drive not mounted — ZIPs saved locally only")
                print("          Mount Drive for persistent backups:")
                print("              from google.colab import drive")
                print("              drive.mount('/content/drive')")
        else:
            print(f"\n  [LOCAL] ZIPs saved to: {self.project_root}")

    # ── Internal ──────────────────────────────────────────────────────────

    def _create_zip(self, zip_path: Path) -> None:
        """Bundle models, CSVs, logs, and metadata into zip_path."""
        with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED, compresslevel=6) as zf:
            for rel in _ZIP_DIRS:
                d = self.project_root / rel
                if not d.exists():
                    continue
                for f in sorted(d.rglob("*")):
                    if f.is_file():
                        arcname = _safe_arcname(f, self.project_root)
                        try:
                            zf.write(f, arcname)
                        except (OSError, PermissionError) as e:
                            logger.warning("ZIP: skipping %s: %s", f.name, e)

            for rel in _ZIP_FILES:
                f = self.project_root / rel
                if f.exists():
                    arcname = _safe_arcname(f, self.project_root)
                    zf.write(f, arcname)

    def _copy_to(self, zip_path: Path, dest_dir: Path, zip_name: str) -> None:
        try:
            dest_dir.mkdir(parents=True, exist_ok=True)
            dest = dest_dir / zip_name
            shutil.copy2(zip_path, dest)
            logger.info("Backup → %s", dest)
            size_mb = dest.stat().st_size / 1024 ** 2
            print(f"       → {dest}  ({size_mb:.1f} MB)")
        except Exception as e:
            logger.warning("Backup to %s failed: %s", dest_dir, e)

    def _detect_destinations(self) -> list[Path]:
        dests: list[Path] = []

        for p in ("/kaggle/output", "/kaggle/working"):
            d = Path(p)
            if d.exists() and d != self.project_root:
                dests.append(d)

        if self.gdrive_dir:
            dests.append(Path(self.gdrive_dir))
        elif Path("/content/drive").exists():
            candidate = Path("/content/drive/MyDrive/SynthFHIR")
            dests.append(candidate)

        return dests

    # ── Single-table ZIP ─────────────────────────────────────────────────

    def create_single_table_zip(self, table_name: str) -> Path:
        """Create outputs/table_{table_name}.zip for single-table download.

        Named differently from the rolling checkpoint ZIPs
        (SynthFHIR_checkpoint_after_*.zip) so the user can easily identify
        which file to download after a --table run.

        Includes everything the rolling ZIP includes: models/, synthetic/,
        logs/, metadata.json. The user gets a self-contained archive with
        both the model and the synthetic CSV for that table.
        """
        zip_path = self.outputs_dir / f"table_{table_name}.zip"
        self._create_zip(zip_path)
        size_mb = zip_path.stat().st_size / 1024 ** 2
        logger.info("Single-table ZIP: %s  (%.1f MB)", zip_path.name, size_mb)
        return zip_path

    def copy_to_kaggle_working(self, zip_path: Path) -> Path | None:
        """Copy zip_path to /kaggle/working/ if on Kaggle. Returns dest path."""
        kaggle_working = Path("/kaggle/working")
        if not kaggle_working.exists():
            return None
        dest = kaggle_working / zip_path.name
        if dest.resolve() == zip_path.resolve():
            # ZIP already is inside /kaggle/working — nothing to copy
            return dest
        try:
            shutil.copy2(zip_path, dest)
            logger.info("Copied to /kaggle/working: %s", dest.name)
            return dest
        except Exception as e:
            logger.warning("Could not copy to /kaggle/working: %s", e)
            return None

    def print_kaggle_download_banner(self, zip_path: Path) -> None:
        """Print the DOWNLOAD READY banner and copy ZIP to /kaggle/working/.

        Only prints on Kaggle. On Colab or local, prints a simpler message.
        Always copies to /kaggle/working/ when available so the file appears
        in the Kaggle Output panel for one-click download.
        """
        kaggle_dest = self.copy_to_kaggle_working(zip_path)

        if kaggle_dest:
            print()
            print("=" * 42)
            print("  DOWNLOAD READY")
            print()
            print(f"  {kaggle_dest}")
            print()
            print("  Kaggle Output panel → click the file to download.")
            print("=" * 42)
            print()
        else:
            print()
            print(f"  ZIP ready: {zip_path}")
            print()

    @property
    def is_kaggle(self) -> bool:
        return Path("/kaggle").exists()

    @property
    def is_colab(self) -> bool:
        return Path("/content").exists()


# ── Standalone recovery helper ────────────────────────────────────────────────

def auto_recover_from_zip(project_root: Path, outputs_dir: Path) -> bool:
    """Scan for checkpoint ZIPs and extract the most recent one.

    Called at startup before the manifest is loaded. If /kaggle/working was
    wiped but the previous run saved a ZIP to /kaggle/output, this function
    extracts it and returns True so the pipeline can read the recovered manifest
    and skip already-completed tables.

    Does nothing if the outputs directory already has files (not a fresh start).
    """
    # Don't overwrite an existing working directory
    manifest = outputs_dir / "logs" / "manifest.json"
    if manifest.exists():
        return False

    search_dirs = [
        Path("/kaggle/output"),
        Path("/kaggle/input"),
        project_root,
    ]

    candidates: list[Path] = []
    for d in search_dirs:
        if d.exists():
            candidates.extend(d.glob("SynthFHIR_checkpoint_after_*.zip"))

    if not candidates:
        return False

    # Pick the ZIP that covers the most tables
    _order = {t: i for i, t in enumerate(
        ["patients", "encounters", "observations", "conditions", "medications"]
    )}
    def _rank(p: Path) -> int:
        for t, i in sorted(_order.items(), key=lambda x: -x[1]):
            if t in p.name:
                return i
        return -1

    best = max(candidates, key=_rank)
    print(f"\n  [RECOVERY] Found checkpoint ZIP: {best}")
    print(f"  [RECOVERY] Extracting to: {project_root}")

    with zipfile.ZipFile(best, "r") as zf:
        zf.extractall(project_root)

    print("  [RECOVERY] Extraction complete. Resuming from checkpoint.\n")
    logger.info("Auto-recovery from %s", best)
    return True


# ── Helpers ───────────────────────────────────────────────────────────────────

def _safe_arcname(path: Path, root: Path) -> str:
    try:
        return str(path.relative_to(root))
    except ValueError:
        return path.name


def _detect_environment() -> str:
    if Path("/kaggle").exists():
        return "kaggle"
    if Path("/content").exists():
        return "colab"
    return "local"
