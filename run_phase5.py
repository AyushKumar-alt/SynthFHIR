"""Phase 5 — Evaluation (Lite): SynthFHIR synthetic data quality assessment.

Reads:
  data/ready/{table}_ready.csv          — original training data
  outputs/synthetic/synthetic_{table}.csv — generated synthetic data

Writes:
  outputs/evaluation/evaluation_summary.json  — full nested results
  outputs/evaluation/evaluation_summary.csv   — per-table score table
  outputs/evaluation/plots/                   — all PNG charts
  outputs/evaluation/report.html             — self-contained HTML report

Usage
-----
  python run_phase5.py                   # evaluate all tables
  python run_phase5.py --tables patients encounters
  python run_phase5.py --no-plots        # skip plot generation (faster)
  python run_phase5.py --project-root /path/to/SynthFHIR

Phase 5.2 (not yet implemented)
---------------------------------
  TSTR, Anonymeter singling-out / linkability / inference risk,
  membership inference, temporal pattern evaluation.
  See the TODO section at the bottom of outputs/evaluation/report.html.
"""

from __future__ import annotations

import argparse
import datetime
import json
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


# ── Logging ───────────────────────────────────────────────────────────────────

def _setup_logging(log_dir: Path, level: str) -> Path:
    log_dir.mkdir(parents=True, exist_ok=True)
    log_file = log_dir / "phase5_evaluation.log"

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

    return log_file


# ── Banner ────────────────────────────────────────────────────────────────────

def _print_banner(tables: list[str], project_root: Path) -> None:
    w   = 56
    sep = "=" * w
    print()
    print(sep)
    print("  SYNTHFHIR — PHASE 5 EVALUATION (LITE)")
    print(sep)
    print(f"  Project : {project_root}")
    print(f"  Tables  : {', '.join(tables)}")
    print(f"  Time    : {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(sep)
    print()


def _print_finish_banner(
    output_dir: Path,
    global_score: float | None,
    elapsed_s: float,
) -> None:
    m, s = divmod(int(elapsed_s), 60)
    w    = 56
    sep  = "=" * w
    score_str = f"{global_score:.3f}" if global_score is not None else "N/A"
    verdict   = (
        "GOOD" if global_score is not None and global_score >= 0.80 else
        "FAIR" if global_score is not None and global_score >= 0.60 else
        "POOR" if global_score is not None else "N/A"
    )
    print()
    print(sep)
    print("  Phase 5 Evaluation Complete")
    print(sep)
    print(f"  Overall Score : {score_str}  ({verdict})")
    print(f"  Elapsed       : {m} min {s:02d} sec")
    print(f"  Outputs       : {output_dir}")
    print(f"    · evaluation_summary.json")
    print(f"    · evaluation_summary.csv")
    print(f"    · ks_results.csv / .json")
    print(f"    · kanonymity.csv / .json")
    print(f"    · tstr_results.csv / .json")
    print(f"    · plots/")
    print(f"    · report.html")
    print(sep)
    print()


# ── Main ───────────────────────────────────────────────────────────────────────

def main() -> None:
    import time

    parser = argparse.ArgumentParser(
        description="SynthFHIR — Phase 5: Synthetic Data Evaluation (Lite)"
    )
    parser.add_argument(
        "--tables", nargs="+", default=None,
        metavar="TABLE",
        help=(
            "Evaluate only these tables. Default: all five tables. "
            "Example: --tables patients encounters conditions"
        ),
    )
    parser.add_argument(
        "--no-plots", action="store_true",
        help="Skip plot generation. Faster but report.html will have no charts.",
    )
    parser.add_argument(
        "--project-root", "-r", default=None,
        help=(
            "Project root directory. "
            "Overrides the PROJECT_ROOT env var. "
            "Default: directory containing this script."
        ),
    )
    parser.add_argument(
        "--config", default=None,
        help="Path to settings.yaml. Default: <project-root>/config/settings.yaml",
    )
    args = parser.parse_args()

    t0           = time.time()
    project_root = _resolve_project_root(args.project_root)
    config_path  = (
        Path(args.config).resolve()
        if args.config
        else project_root / "config" / "settings.yaml"
    )

    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))

    # ── Imports (after sys.path is set) ──────────────────────────────────
    from src.evaluation.config  import load_evaluation_config, TABLE_SEQUENCE
    from src.evaluation.loader  import load_all_pairs
    from src.evaluation.dataset_summary     import run_dataset_summary
    from src.evaluation.numeric_eval        import run_numeric_evaluation
    from src.evaluation.categorical_eval    import run_categorical_evaluation
    from src.evaluation.correlation_eval    import run_correlation_evaluation
    from src.evaluation.privacy_eval        import run_privacy_evaluation
    from src.evaluation.sdv_quality         import run_sdv_quality
    from src.evaluation.ks_eval             import run_ks_evaluation
    from src.evaluation.privacy_k_anonymity import run_k_anonymity
    from src.evaluation.tstr_eval           import run_tstr_evaluation
    from src.evaluation.report import (
        compute_scores,
        save_summary_json,
        save_summary_csv,
        generate_html_report,
    )

    # ── Config ────────────────────────────────────────────────────────────
    cfg = load_evaluation_config(config_path, project_root=project_root)

    # ── Table filter ──────────────────────────────────────────────────────
    valid_tables = [t for t, _ in TABLE_SEQUENCE]
    if args.tables:
        unknown = [t for t in args.tables if t not in valid_tables]
        if unknown:
            print(f"[ERROR] Unknown tables: {unknown}. Valid: {valid_tables}")
            sys.exit(1)
        tables = args.tables
    else:
        tables = valid_tables

    # ── Logging ───────────────────────────────────────────────────────────
    log_file = _setup_logging(project_root / "outputs" / "logs", "INFO")
    logger   = logging.getLogger(__name__)

    _print_banner(tables, project_root)
    logger.info("Config  : %s", config_path)
    logger.info("Ready   : %s", cfg.ready_dir)
    logger.info("Synth   : %s", cfg.synthetic_dir)
    logger.info("Output  : %s", cfg.output_dir)
    logger.info("Tables  : %s", tables)
    logger.info("Plots   : %s", "disabled" if args.no_plots else "enabled")

    # ── Output directories ────────────────────────────────────────────────
    cfg.output_dir.mkdir(parents=True, exist_ok=True)
    if not args.no_plots:
        (cfg.plots_dir / "numeric").mkdir(parents=True, exist_ok=True)
        (cfg.plots_dir / "categorical").mkdir(parents=True, exist_ok=True)
        (cfg.plots_dir / "correlation").mkdir(parents=True, exist_ok=True)
        (cfg.plots_dir / "ks").mkdir(parents=True, exist_ok=True)

    # ── Load CSVs ────────────────────────────────────────────────────────
    logger.info("=" * 56)
    logger.info("  Loading datasets")
    logger.info("=" * 56)
    all_pairs = load_all_pairs(cfg)
    # Filter to requested tables
    pairs = {t: all_pairs[t] for t in tables if t in all_pairs}

    # ── A: Dataset summary ────────────────────────────────────────────────
    logger.info("=" * 56)
    logger.info("  A. Dataset Summary")
    logger.info("=" * 56)
    dataset_summary = run_dataset_summary(pairs)

    # ── B: Numeric evaluation ─────────────────────────────────────────────
    logger.info("=" * 56)
    logger.info("  B. Numeric Evaluation")
    logger.info("=" * 56)
    plots_dir = cfg.plots_dir if not args.no_plots else Path("/dev/null")
    numeric_results = run_numeric_evaluation(pairs, plots_dir)

    # ── C: Categorical evaluation ─────────────────────────────────────────
    logger.info("=" * 56)
    logger.info("  C. Categorical Evaluation")
    logger.info("=" * 56)
    categorical_results = run_categorical_evaluation(pairs, plots_dir)

    # ── D: Correlation evaluation ─────────────────────────────────────────
    logger.info("=" * 56)
    logger.info("  D. Correlation Evaluation")
    logger.info("=" * 56)
    correlation_results = run_correlation_evaluation(pairs, plots_dir)

    # ── E: Privacy checks ─────────────────────────────────────────────────
    logger.info("=" * 56)
    logger.info("  E. Privacy Checks")
    logger.info("=" * 56)
    privacy_results = run_privacy_evaluation(pairs, cfg.pk_map)

    # ── F: SDV quality ────────────────────────────────────────────────────
    logger.info("=" * 56)
    logger.info("  F. SDV Quality")
    logger.info("=" * 56)
    sdv_quality_results = run_sdv_quality(pairs, cfg.metadata_path)

    # ── G: KS Test ───────────────────────────────────────────────────────────
    logger.info("=" * 56)
    logger.info("  G. KS Test Evaluation")
    logger.info("=" * 56)
    ks_results = run_ks_evaluation(pairs, cfg.output_dir, plots_dir)

    # ── H: k-Anonymity ───────────────────────────────────────────────────────
    logger.info("=" * 56)
    logger.info("  H. k-Anonymity")
    logger.info("=" * 56)
    kanon_results = run_k_anonymity(pairs, cfg.output_dir)

    # ── I: TSTR ──────────────────────────────────────────────────────────────
    logger.info("=" * 56)
    logger.info("  I. Train on Synthetic, Test on Real (TSTR)")
    logger.info("=" * 56)
    tstr_results = run_tstr_evaluation(pairs, cfg.output_dir)

    # ── Compute scores ────────────────────────────────────────────────────
    scores = compute_scores(
        numeric_results,
        categorical_results,
        correlation_results,
        privacy_results,
        sdv_quality_results,
        dataset_summary,
    )
    global_score = scores.get("global_score")

    # ── Save outputs ──────────────────────────────────────────────────────
    logger.info("=" * 56)
    logger.info("  Saving outputs")
    logger.info("=" * 56)

    all_results = {
        "generated_at":          datetime.datetime.now().isoformat(),
        "project_root":          str(project_root),
        "tables_evaluated":      tables,
        "global_score":          global_score,
        "scores":                scores,
        "dataset_summary":       dataset_summary,
        "numeric_results":       numeric_results,
        "categorical_results":   categorical_results,
        "correlation_results":   correlation_results,
        "privacy_results":       privacy_results,
        "sdv_quality_results":   sdv_quality_results,
        "ks_results":            ks_results,
        "kanon_results":         kanon_results,
        "tstr_results":          tstr_results,
    }

    save_summary_json(cfg.output_dir, all_results)
    save_summary_csv(
        cfg.output_dir, scores,
        ks_results    = ks_results,
        kanon_results = kanon_results,
        tstr_results  = tstr_results,
    )

    generate_html_report(
        output_dir          = cfg.output_dir,
        plots_dir           = cfg.plots_dir,
        scores              = scores,
        dataset_summary     = dataset_summary,
        numeric_results     = numeric_results,
        categorical_results = categorical_results,
        correlation_results = correlation_results,
        privacy_results     = privacy_results,
        sdv_quality_results = sdv_quality_results,
        ks_results          = ks_results,
        kanon_results       = kanon_results,
        tstr_results        = tstr_results,
    )

    # ── Console score summary ─────────────────────────────────────────────
    logger.info("=" * 56)
    logger.info("  Score Summary")
    logger.info("=" * 56)
    for table, s in scores.get("tables", {}).items():
        logger.info(
            "  %-14s  numeric=%-6s  cat=%-6s  corr=%-6s  priv=%-6s  overall=%s",
            table,
            f"{s['numeric_similarity']:.3f}" if s.get("numeric_similarity") is not None else "N/A",
            f"{s['categorical_similarity']:.3f}" if s.get("categorical_similarity") is not None else "N/A",
            f"{s['correlation_preservation']:.3f}" if s.get("correlation_preservation") is not None else "N/A",
            f"{s['privacy_score']:.3f}" if s.get("privacy_score") is not None else "N/A",
            f"{s['overall']:.3f}" if s.get("overall") is not None else "N/A",
        )
    logger.info("-" * 56)
    logger.info(
        "  GLOBAL SCORE: %s",
        f"{global_score:.3f}" if global_score is not None else "N/A",
    )
    logger.info("=" * 56)

    _print_finish_banner(cfg.output_dir, global_score, time.time() - t0)
    logger.info("Log: %s", log_file)


if __name__ == "__main__":
    main()
