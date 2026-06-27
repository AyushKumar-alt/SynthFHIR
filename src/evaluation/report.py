"""Phase 5 report generator.

Outputs
-------
outputs/evaluation/evaluation_summary.json  — full nested results
outputs/evaluation/evaluation_summary.csv   — flat per-table score table
outputs/evaluation/report.html              — self-contained HTML report
"""

from __future__ import annotations

import base64
import json
import logging
from datetime import datetime
from pathlib import Path

import pandas as pd

logger = logging.getLogger(__name__)

# Columns shown in the CSV summary (order matters)
_CSV_COLUMNS = [
    "table",
    "original_rows",
    "synthetic_rows",
    "numeric_similarity",
    "categorical_similarity",
    "correlation_preservation",
    "privacy_score",
    "sdv_quality_score",
    "overall_score",
]

# Weights for the per-table overall score
_WEIGHTS = {
    "numeric_similarity":       3,
    "categorical_similarity":   3,
    "correlation_preservation": 2,
    "privacy_score":            2,
}

# Top-N plots embedded in HTML per evaluation type per table
_TOP_N_EMBED_NUMERIC      = 5
_TOP_N_EMBED_CATEGORICAL  = 5


# ── Score computation ─────────────────────────────────────────────────────────

def compute_scores(
    numeric_results:     dict[str, list[dict]],
    categorical_results: dict[str, list[dict]],
    correlation_results: dict[str, dict],
    privacy_results:     dict[str, dict],
    sdv_quality_results: dict[str, dict],
    dataset_summary:     list[dict],
) -> dict:
    """Compute per-table and global quality scores.

    Returns a dict with keys ``tables`` (per-table breakdown) and
    ``global_score`` (mean of all table overall scores).
    """
    summary_by_table = {s["table"]: s for s in dataset_summary}
    all_tables       = list(numeric_results.keys())
    table_scores: dict[str, dict] = {}

    for table in all_tables:
        scores: dict = {}

        # Numeric similarity: mean(1 − KS) over all evaluated columns
        num_sims = [r["similarity"] for r in numeric_results.get(table, [])
                    if r.get("similarity") is not None]
        scores["numeric_similarity"] = _mean(num_sims)

        # Categorical similarity: mean(1 − TVD)
        cat_sims = [r["similarity"] for r in categorical_results.get(table, [])
                    if r.get("similarity") is not None]
        scores["categorical_similarity"] = _mean(cat_sims)

        # Correlation preservation: 1 − mean(|corr_diff|)
        corr = correlation_results.get(table, {})
        scores["correlation_preservation"] = (
            corr.get("similarity") if corr.get("status") == "ok" else None
        )

        # Privacy score
        priv = privacy_results.get(table, {})
        scores["privacy_score"] = priv.get("privacy_score")

        # SDV quality
        sdv = sdv_quality_results.get(table, {})
        scores["sdv_quality_score"] = (
            sdv.get("overall_score") if sdv.get("status") == "ok" else None
        )

        # Row counts (from dataset summary)
        ds = summary_by_table.get(table, {})
        scores["original_rows"]  = ds.get("original_rows")
        scores["synthetic_rows"] = ds.get("synthetic_rows")

        # Weighted overall (excludes SDV quality to avoid double-counting)
        weighted_sum  = 0.0
        weight_total  = 0.0
        for key, w in _WEIGHTS.items():
            v = scores.get(key)
            if v is not None:
                weighted_sum  += v * w
                weight_total  += w
        scores["overall"] = round(weighted_sum / weight_total, 4) if weight_total > 0 else None

        table_scores[table] = scores

    overalls = [s["overall"] for s in table_scores.values() if s.get("overall") is not None]
    global_score = round(sum(overalls) / len(overalls), 4) if overalls else None

    return {"tables": table_scores, "global_score": global_score}


# ── JSON output ───────────────────────────────────────────────────────────────

def save_summary_json(output_dir: Path, all_results: dict) -> Path:
    path = output_dir / "evaluation_summary.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(all_results, fh, indent=2, default=str)
    logger.info("Saved: %s", path)
    return path


# ── CSV output ────────────────────────────────────────────────────────────────

def save_summary_csv(output_dir: Path, scores: dict) -> Path:
    path = output_dir / "evaluation_summary.csv"
    rows = []
    for table, s in scores["tables"].items():
        rows.append({
            "table":                    table,
            "original_rows":            s.get("original_rows"),
            "synthetic_rows":           s.get("synthetic_rows"),
            "numeric_similarity":       s.get("numeric_similarity"),
            "categorical_similarity":   s.get("categorical_similarity"),
            "correlation_preservation": s.get("correlation_preservation"),
            "privacy_score":            s.get("privacy_score"),
            "sdv_quality_score":        s.get("sdv_quality_score"),
            "overall_score":            s.get("overall"),
        })
    df = pd.DataFrame(rows, columns=_CSV_COLUMNS)
    df.to_csv(path, index=False)
    logger.info("Saved: %s", path)
    return path


# ── HTML output ───────────────────────────────────────────────────────────────

def generate_html_report(
    output_dir:          Path,
    plots_dir:           Path,
    scores:              dict,
    dataset_summary:     list[dict],
    numeric_results:     dict[str, list[dict]],
    categorical_results: dict[str, list[dict]],
    correlation_results: dict[str, dict],
    privacy_results:     dict[str, dict],
    sdv_quality_results: dict[str, dict],
) -> Path:
    path = output_dir / "report.html"
    html = _build_html(
        plots_dir, scores, dataset_summary,
        numeric_results, categorical_results,
        correlation_results, privacy_results, sdv_quality_results,
    )
    path.write_text(html, encoding="utf-8")
    logger.info("Saved: %s", path)
    return path


# ── HTML builder ──────────────────────────────────────────────────────────────

def _build_html(
    plots_dir:           Path,
    scores:              dict,
    dataset_summary:     list[dict],
    numeric_results:     dict[str, list[dict]],
    categorical_results: dict[str, list[dict]],
    correlation_results: dict[str, dict],
    privacy_results:     dict[str, dict],
    sdv_quality_results: dict[str, dict],
) -> str:
    now          = datetime.now().strftime("%Y-%m-%d %H:%M")
    global_score = scores.get("global_score")
    score_color  = _score_color(global_score)

    sections = [
        _html_head(),
        "<body>",
        f'<div class="container">',
        # Title
        f'<h1>SynthFHIR — Phase 5 Evaluation Report</h1>',
        f'<p class="meta">Generated: {now}</p>',
        # Global score banner
        _banner_html(global_score, score_color),
        # Overall score table
        _score_table_html(scores),
        # Dataset summary
        _dataset_summary_html(dataset_summary),
        # Per-table sections
        *[
            _table_section_html(
                table, scores, plots_dir,
                numeric_results, categorical_results,
                correlation_results, privacy_results, sdv_quality_results,
            )
            for table in scores.get("tables", {})
        ],
        # Phase 5.2 TODO
        _todo_section_html(),
        "</div>",
        "</body></html>",
    ]
    return "\n".join(sections)


def _html_head() -> str:
    return """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>SynthFHIR — Phase 5 Evaluation Report</title>
<style>
  body { font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
         background: #f5f7fa; color: #1a202c; margin: 0; padding: 0; }
  .container { max-width: 1200px; margin: 0 auto; padding: 2rem; }
  h1 { color: #1a202c; border-bottom: 3px solid #4299e1; padding-bottom: .5rem; }
  h2 { color: #2d3748; border-left: 4px solid #4299e1; padding-left: .75rem; margin-top: 2.5rem; }
  h3 { color: #4a5568; margin-top: 1.5rem; }
  .meta { color: #718096; font-size: .9rem; }
  .banner { background: #fff; border-radius: 12px; padding: 1.5rem 2rem;
            box-shadow: 0 2px 8px rgba(0,0,0,.08); margin: 1.5rem 0;
            display: flex; align-items: center; gap: 2rem; }
  .score-big { font-size: 3.5rem; font-weight: 700; }
  .score-label { font-size: 1rem; color: #718096; }
  table { border-collapse: collapse; width: 100%; background: #fff;
          border-radius: 8px; overflow: hidden;
          box-shadow: 0 1px 4px rgba(0,0,0,.07); margin-bottom: 1.5rem; }
  th { background: #edf2f7; padding: .6rem .9rem; text-align: left;
       font-size: .85rem; color: #4a5568; }
  td { padding: .55rem .9rem; border-bottom: 1px solid #f0f4f8;
       font-size: .88rem; }
  tr:last-child td { border-bottom: none; }
  tr:hover td { background: #f7fafc; }
  .good  { color: #276749; font-weight: 600; }
  .warn  { color: #744210; font-weight: 600; }
  .bad   { color: #742a2a; font-weight: 600; }
  .na    { color: #a0aec0; }
  .card  { background: #fff; border-radius: 10px; padding: 1.25rem 1.5rem;
           box-shadow: 0 1px 4px rgba(0,0,0,.07); margin-bottom: 1.25rem; }
  .plots { display: flex; flex-wrap: wrap; gap: 1rem; margin-top: 1rem; }
  .plot  { flex: 1 1 320px; max-width: 520px; }
  .plot img { width: 100%; border-radius: 6px;
              box-shadow: 0 1px 3px rgba(0,0,0,.12); }
  .plot-caption { font-size: .78rem; color: #718096; text-align: center; margin-top: .3rem; }
  .todo-box { background: #fffaf0; border: 1.5px solid #f6ad55;
              border-radius: 8px; padding: 1rem 1.25rem; margin-top: 1rem; }
  .todo-box li { margin: .35rem 0; }
  .pill { display: inline-block; padding: .15rem .55rem; border-radius: 9999px;
          font-size: .78rem; font-weight: 600; }
  .pill-good { background: #c6f6d5; color: #276749; }
  .pill-warn { background: #fefcbf; color: #744210; }
  .pill-bad  { background: #fed7d7; color: #742a2a; }
  .pill-na   { background: #edf2f7; color: #718096; }
  section { margin-bottom: 2.5rem; }
</style>
</head>"""


def _banner_html(global_score: float | None, color: str) -> str:
    score_str = f"{global_score:.3f}" if global_score is not None else "N/A"
    verdict   = _verdict(global_score)
    return f"""
<div class="banner">
  <div>
    <div class="score-big" style="color:{color}">{score_str}</div>
    <div class="score-label">Overall Quality Score (0–1)</div>
  </div>
  <div>
    <div style="font-size:1.5rem;font-weight:700;color:{color}">{verdict}</div>
    <div class="score-label">Weighted mean across numeric · categorical · correlation · privacy</div>
  </div>
</div>"""


def _score_table_html(scores: dict) -> str:
    rows = ""
    for table, s in scores.get("tables", {}).items():
        def _cell(v):
            if v is None:
                return '<td class="na">N/A</td>'
            pct = _pill(v)
            return f"<td>{pct}</td>"

        orig  = s.get("original_rows")
        synth = s.get("synthetic_rows")
        orig_str  = f"{orig:,}"  if orig  is not None else "N/A"
        synth_str = f"{synth:,}" if synth is not None else "N/A"
        rows += f"""
<tr>
  <td><b>{table}</b></td>
  <td>{orig_str}</td>
  <td>{synth_str}</td>
  {_cell(s.get('numeric_similarity'))}
  {_cell(s.get('categorical_similarity'))}
  {_cell(s.get('correlation_preservation'))}
  {_cell(s.get('privacy_score'))}
  {_cell(s.get('sdv_quality_score'))}
  {_cell(s.get('overall'))}
</tr>"""

    return f"""
<section>
<h2>Overall Scores by Table</h2>
<table>
<tr>
  <th>Table</th><th>Orig Rows</th><th>Synth Rows</th>
  <th>Numeric Sim</th><th>Categorical Sim</th><th>Correlation</th>
  <th>Privacy</th><th>SDV Quality</th><th>Overall</th>
</tr>
{rows}
</table>
</section>"""


def _dataset_summary_html(dataset_summary: list[dict]) -> str:
    rows = ""
    for s in dataset_summary:
        table      = s["table"]
        orig_rows  = s.get("original_rows",        "N/A")
        synth_rows = s.get("synthetic_rows",       "N/A")
        orig_cols  = s.get("original_cols",        "N/A")
        synth_cols = s.get("synthetic_cols",       "N/A")
        orig_miss  = s.get("original_missing_pct", "N/A")
        synth_miss = s.get("synthetic_missing_pct","N/A")
        col_match  = s.get("column_count_match")
        n_mismatch = len(s.get("dtype_mismatches", {}))
        match_str  = (
            '<span class="good">✓</span>' if col_match is True else
            '<span class="bad">✗</span>'  if col_match is False else
            '<span class="na">N/A</span>'
        )
        orig_rows_str  = f"{orig_rows:,}"  if isinstance(orig_rows,  int) else str(orig_rows)
        synth_rows_str = f"{synth_rows:,}" if isinstance(synth_rows, int) else str(synth_rows)
        rows += f"""
<tr>
  <td><b>{table}</b></td>
  <td>{orig_rows_str}</td>
  <td>{synth_rows_str}</td>
  <td>{orig_cols}</td><td>{synth_cols}</td>
  <td>{match_str}</td>
  <td>{orig_miss}%</td><td>{synth_miss}%</td>
  <td>{'<span class="bad">' + str(n_mismatch) + '</span>' if n_mismatch else '<span class="good">0</span>'}</td>
</tr>"""

    return f"""
<section>
<h2>A. Dataset Summary</h2>
<table>
<tr>
  <th>Table</th><th>Orig Rows</th><th>Synth Rows</th>
  <th>Orig Cols</th><th>Synth Cols</th><th>Cols Match</th>
  <th>Orig Missing %</th><th>Synth Missing %</th><th>Dtype Mismatches</th>
</tr>
{rows}
</table>
</section>"""


def _table_section_html(
    table:               str,
    scores:              dict,
    plots_dir:           Path,
    numeric_results:     dict,
    categorical_results: dict,
    correlation_results: dict,
    privacy_results:     dict,
    sdv_quality_results: dict,
) -> str:
    s      = scores.get("tables", {}).get(table, {})
    overall = s.get("overall")

    parts = [
        f'<section id="{table}">',
        f'<h2>{table.upper()} &nbsp; {_pill(overall)}</h2>',
        _numeric_section_html(table, numeric_results.get(table, []), plots_dir),
        _categorical_section_html(table, categorical_results.get(table, []), plots_dir),
        _correlation_section_html(table, correlation_results.get(table, {}), plots_dir),
        _privacy_section_html(table, privacy_results.get(table, {})),
        _sdv_section_html(table, sdv_quality_results.get(table, {})),
        "</section>",
    ]
    return "\n".join(parts)


def _numeric_section_html(table: str, results: list[dict], plots_dir: Path) -> str:
    if not results:
        return '<div class="card"><h3>B. Numeric Evaluation</h3><p class="na">No numeric columns evaluated.</p></div>'

    # Sort by KS stat (worst first = most interesting)
    sorted_r = sorted(results, key=lambda r: r.get("ks_stat", 0), reverse=True)
    avg_sim   = _mean([r["similarity"] for r in results if r.get("similarity") is not None])
    sim_pill  = _pill(avg_sim) if avg_sim is not None else '<span class="na">N/A</span>'

    rows = ""
    for r in sorted_r:
        col  = r["column"]
        real = r.get("real", {})
        syn  = r.get("synthetic", {})
        rows += f"""
<tr>
  <td>{col}</td>
  <td>{r.get('ks_stat','N/A')}</td>
  <td>{_pill(r.get('similarity'))}</td>
  <td>{real.get('mean','')}</td><td>{syn.get('mean','')}</td>
  <td>{real.get('std','')}</td><td>{syn.get('std','')}</td>
  <td>{real.get('min','')}</td><td>{syn.get('min','')}</td>
  <td>{real.get('max','')}</td><td>{syn.get('max','')}</td>
</tr>"""

    # Embed top-N histograms (highest KS stat)
    plots_html = _embed_plots(
        plots_dir / "numeric",
        [f"{table}_{_safe_name(r['column'])}_hist.png" for r in sorted_r[:_TOP_N_EMBED_NUMERIC]],
        [f"{r['column']} — Histogram (KS={r.get('ks_stat','?')})" for r in sorted_r[:_TOP_N_EMBED_NUMERIC]],
    )

    return f"""
<div class="card">
<h3>B. Numeric Evaluation &nbsp; {sim_pill}</h3>
<table>
<tr>
  <th>Column</th><th>KS Stat</th><th>Similarity</th>
  <th>Mean (Real)</th><th>Mean (Synth)</th>
  <th>Std (Real)</th><th>Std (Synth)</th>
  <th>Min (Real)</th><th>Min (Synth)</th>
  <th>Max (Real)</th><th>Max (Synth)</th>
</tr>
{rows}
</table>
{plots_html}
</div>"""


def _categorical_section_html(table: str, results: list[dict], plots_dir: Path) -> str:
    if not results:
        return '<div class="card"><h3>C. Categorical Evaluation</h3><p class="na">No categorical columns evaluated.</p></div>'

    sorted_r = sorted(results, key=lambda r: r.get("tvd", 0), reverse=True)
    avg_sim  = _mean([r["similarity"] for r in results if r.get("similarity") is not None])
    sim_pill = _pill(avg_sim) if avg_sim is not None else '<span class="na">N/A</span>'

    rows = ""
    for r in sorted_r:
        col = r["column"]
        rows += f"""
<tr>
  <td>{col}</td>
  <td>{r.get('tvd','N/A')}</td>
  <td>{_pill(r.get('similarity'))}</td>
  <td>{r.get('n_unique_real','')}</td>
  <td>{r.get('n_unique_synthetic','')}</td>
</tr>"""

    plots_html = _embed_plots(
        plots_dir / "categorical",
        [f"{table}_{_safe_name(r['column'])}_bar.png" for r in sorted_r[:_TOP_N_EMBED_CATEGORICAL]],
        [f"{r['column']} — Frequencies (TVD={r.get('tvd','?')})" for r in sorted_r[:_TOP_N_EMBED_CATEGORICAL]],
    )

    return f"""
<div class="card">
<h3>C. Categorical Evaluation &nbsp; {sim_pill}</h3>
<table>
<tr>
  <th>Column</th><th>TVD</th><th>Similarity</th>
  <th>Unique (Real)</th><th>Unique (Synth)</th>
</tr>
{rows}
</table>
{plots_html}
</div>"""


def _correlation_section_html(table: str, result: dict, plots_dir: Path) -> str:
    status = result.get("status", "missing")
    if status != "ok":
        return f'<div class="card"><h3>D. Correlation</h3><p class="na">{status}</p></div>'

    sim  = result.get("similarity")
    pill = _pill(sim) if sim is not None else '<span class="na">N/A</span>'

    heatmap_path = plots_dir / "correlation" / f"{table}_correlation.png"
    img_html     = _embed_image(heatmap_path, f"{table} — Correlation Heatmaps")

    return f"""
<div class="card">
<h3>D. Correlation Preservation &nbsp; {pill}</h3>
<p>
  Columns: {result.get('n_columns','?')} &nbsp;|&nbsp;
  Mean diff: <b>{result.get('mean_correlation_diff','N/A')}</b> &nbsp;|&nbsp;
  Max diff: <b>{result.get('max_correlation_diff','N/A')}</b>
</p>
{img_html}
</div>"""


def _privacy_section_html(table: str, result: dict) -> str:
    status = result.get("status", "missing")
    if status != "ok":
        return f'<div class="card"><h3>E. Privacy</h3><p class="na">{status}</p></div>'

    pill = _pill(result.get("privacy_score"))
    return f"""
<div class="card">
<h3>E. Privacy Checks &nbsp; {pill}</h3>
<table>
<tr><th>Metric</th><th>Value</th></tr>
<tr><td>Synthetic unique rows</td><td>{result.get('synthetic_unique_rows','N/A')} ({result.get('synthetic_unique_pct','N/A')}%)</td></tr>
<tr><td>Synthetic duplicate rows</td><td>{result.get('synthetic_duplicate_rows','N/A')} ({result.get('synthetic_duplicate_pct','N/A')}%)</td></tr>
<tr><td>Exact row overlap with original</td><td>{result.get('exact_row_overlap','N/A')} ({result.get('exact_overlap_pct','N/A')}%)</td></tr>
<tr><td>Privacy score</td><td>{pill}</td></tr>
</table>
</div>"""


def _sdv_section_html(table: str, result: dict) -> str:
    status = result.get("status", "missing")
    if status == "sdv_unavailable":
        return '<div class="card"><h3>F. SDV Quality</h3><p class="na">sdv.evaluation not available — skipped.</p></div>'
    if status not in ("ok",):
        msg = result.get("error", status)
        return f'<div class="card"><h3>F. SDV Quality</h3><p class="na">{msg}</p></div>'

    score = result.get("overall_score")
    pill  = _pill(score) if score is not None else '<span class="na">N/A</span>'
    props = result.get("properties", [])

    prop_rows = ""
    for p in props:
        prop_name  = p.get("Property", p.get("property", "?"))
        prop_score = p.get("Score", p.get("score", "?"))
        prop_rows += f"<tr><td>{prop_name}</td><td>{prop_score}</td></tr>"

    return f"""
<div class="card">
<h3>F. SDV Quality &nbsp; {pill}</h3>
<table>
<tr><th>Property</th><th>Score</th></tr>
{prop_rows}
</table>
</div>"""


def _todo_section_html() -> str:
    return """
<section>
<h2>Phase 5.2 — Advanced Evaluation (TODO)</h2>
<div class="todo-box">
  <p><b>The following evaluations are scoped for Phase 5.2 and not yet implemented:</b></p>
  <ul>
    <li><b>TSTR</b> — Train on Synthetic, Test on Real: ML efficacy benchmark
        (e.g. mortality prediction AUC on real vs synthetic training data).</li>
    <li><b>Anonymeter: Singling-out risk</b> — probability that an attacker can
        uniquely identify a real record from the synthetic dataset.</li>
    <li><b>Anonymeter: Linkability risk</b> — probability that an attacker can
        link two synthetic records to the same real individual.</li>
    <li><b>Anonymeter: Inference risk</b> — probability that an attacker can
        infer a sensitive attribute from the synthetic data.</li>
    <li><b>Membership inference attacks</b> — determine whether a specific
        real record was in the training set.</li>
    <li><b>Temporal pattern evaluation</b> — inter-visit gap distributions,
        temporal autocorrelation of lab values, ICD code co-occurrence over time.</li>
  </ul>
  <p style="color:#744210;font-size:.9rem">
    Install <code>anonymeter</code> and uncomment the dependency in
    <code>requirements.txt</code> when Phase 5.2 is ready.
  </p>
</div>
</section>"""


# ── Image embedding ───────────────────────────────────────────────────────────

def _embed_plots(plots_dir: Path, filenames: list[str], captions: list[str]) -> str:
    items = []
    for fname, caption in zip(filenames, captions):
        path = plots_dir / fname
        img  = _embed_image(path, caption)
        if img:
            items.append(img)
    if not items:
        return ""
    return '<div class="plots">' + "\n".join(items) + "</div>"


def _embed_image(path: Path, caption: str) -> str:
    if not path.exists():
        return ""
    try:
        with open(path, "rb") as fh:
            b64 = base64.b64encode(fh.read()).decode("ascii")
        return (
            f'<div class="plot">'
            f'<img src="data:image/png;base64,{b64}" alt="{caption}">'
            f'<div class="plot-caption">{caption}</div>'
            f"</div>"
        )
    except Exception:
        return ""


# ── Score utilities ───────────────────────────────────────────────────────────

def _mean(values: list) -> float | None:
    clean = [v for v in values if v is not None]
    return round(sum(clean) / len(clean), 4) if clean else None


def _score_color(score: float | None) -> str:
    if score is None:
        return "#a0aec0"
    if score >= 0.80:
        return "#276749"
    if score >= 0.60:
        return "#744210"
    return "#742a2a"


def _verdict(score: float | None) -> str:
    if score is None:
        return "N/A"
    if score >= 0.80:
        return "GOOD"
    if score >= 0.60:
        return "FAIR"
    return "POOR"


def _pill(score: float | None) -> str:
    if score is None:
        return '<span class="pill pill-na">N/A</span>'
    css = "pill-good" if score >= 0.80 else "pill-warn" if score >= 0.60 else "pill-bad"
    return f'<span class="pill {css}">{score:.3f}</span>'


def _safe_name(s: str) -> str:
    import re
    return re.sub(r"[^\w\-]", "_", s)[:60]
