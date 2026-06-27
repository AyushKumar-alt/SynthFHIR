"""Phase 5 evaluation package for SynthFHIR.

Modules
-------
config           — EvaluationConfig dataclass and settings loader.
loader           — Load (original, synthetic) CSV pairs per table.
dataset_summary  — A: row counts, column schema, missing-value summary.
numeric_eval     — B: KS statistics, histograms, boxplots.
categorical_eval — C: TVD, frequency comparison, bar charts.
correlation_eval — D: Pearson correlation matrices and heatmaps.
privacy_eval     — E: duplicate rows, exact overlap, uniqueness.
sdv_quality      — F: SDV evaluate_quality() wrapper (graceful skip).
report           — Produce evaluation_summary.json/.csv and report.html.
"""
