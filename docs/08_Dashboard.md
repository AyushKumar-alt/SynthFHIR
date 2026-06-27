# 08 — Dashboard

**Version:** 1.0  
**Last Updated:** 2026-06-28  
**Phase Coverage:** Phase 7 (Planned)  
**Status:** Design complete — implementation pending

---

## Table of Contents

1. [Why a Dashboard](#1-why-a-dashboard)
2. [Technology Choice: Streamlit](#2-technology-choice-streamlit)
3. [Dashboard Sections](#3-dashboard-sections)
4. [Implementation Plan](#4-implementation-plan)

---

## 1. Why a Dashboard

The evaluation outputs from Phase 5 are JSON and HTML files. These are useful for
technical users but opaque to non-technical stakeholders: clinical researchers, hospital
administrators, or policy makers who want to understand the quality of the synthetic data
without reading evaluation metrics.

A dashboard provides:

- **Visual comparison** of real and synthetic distributions side-by-side
- **At-a-glance quality scores** without requiring knowledge of statistics
- **Drill-down capability** — click on a distribution to see details
- **Privacy risk indicators** — clear traffic-light indicators for each table
- **Downloadable** synthetic data directly from the interface

---

## 2. Technology Choice: Streamlit

**Streamlit** was selected as the dashboard framework because:

- **Python-native:** No JavaScript, HTML, or CSS required — the same language used
  for the rest of the pipeline
- **Already in requirements.txt:** No new dependency to add
- **Automatic reactivity:** Sliders and dropdowns update charts without callbacks
- **Fast prototyping:** A functional chart can be added in 3–5 lines of code
- **Kaggle/Colab compatible:** Can be tunnelled through `ngrok` or `localtunnel`
  to expose a public URL from a Kaggle/Colab session

Alternatives considered:

| Framework | Rejected because |
|---|---|
| Dash (Plotly) | Requires defining callbacks and layout in a more complex way; slower iteration |
| Gradio | Better suited for ML model demos; limited charting |
| Tableau | Commercial; not Python-native |
| Jupyter widgets | Tied to notebook interface; not a standalone app |

---

## 3. Dashboard Sections

### Section 1: Overview

```
SynthFHIR — Synthetic Health Data Dashboard
────────────────────────────────────────────
Dataset:      1,000 synthetic patients  (from 998 real patients)
Trained on:   2026-06-28
Overall Quality Score:  0.87 / 1.00  [████████░░]

Table      | Model | Real Rows | Synthetic Rows | Quality | Privacy
-----------|-------|-----------|----------------|---------|--------
patients   | CTGAN |       998 |          1,000 |   0.91  |   Safe
encounters | PAR   |    57,667 |         58,012 |   0.85  |   Safe
observations | CTGAN | 303,696 |        301,224 |   0.88  |   Safe
conditions | PAR   |    37,835 |         38,441 |   0.84  |   Safe
medications | PAR  |    46,734 |         46,890 |   0.86  |   Safe
```

### Section 2: Per-Table Distribution Comparison

A dropdown selects which table to inspect. For each column in the selected table,
a side-by-side chart compares the real and synthetic distributions.

**For numerical columns:** Overlapping histogram with KDE (kernel density estimate).
Displays: KS statistic, real mean, synthetic mean.

**For categorical columns:** Side-by-side bar chart showing percentage frequency
for each category. Displays: TVD (Total Variation Distance).

### Section 3: Correlation Analysis

For numerical columns, display a correlation matrix heatmap:
- Left panel: real data correlation matrix
- Right panel: synthetic data correlation matrix
- Difference plot: absolute difference per cell

### Section 4: Temporal Pattern Analysis

For sequential tables (encounters, conditions, medications):
- Inter-event gap distribution: real vs synthetic histogram
- Sequence length distribution: real vs synthetic histogram
- Timeline scatter plot: one row per patient, X=days_since_birth,
  colour=event type

### Section 5: Privacy Dashboard

For each privacy attack type:

```
Privacy Risk Assessment
───────────────────────
Singling-Out Risk:  LOW  ● ●○○
  Control rate: 2.3%  |  Synthetic rate: 2.7%  |  Δ: +0.4% (within tolerance)

Linkability Risk:   LOW  ● ●○○
  Control rate: 8.1%  |  Synthetic rate: 8.9%  |  Δ: +0.8% (within tolerance)

Inference Risk:     LOW  ● ●○○
  Most at-risk attribute: ethnicity
  Baseline inference accuracy: 74.2%
  Synthetic-assisted accuracy: 75.1%
  Δ: +0.9% (within tolerance)
```

### Section 6: Download

- Download individual synthetic CSVs
- Download full synthetic dataset as ZIP
- Download FHIR bundles (Phase 6 output)
- Download evaluation reports

---

## 4. Implementation Plan

Phase 7 will add:

```
dashboard.py              ← Streamlit app entry point
src/dashboard/
├── data_loader.py        ← Loads real and synthetic data; caches with @st.cache_data
├── charts.py             ← Reusable Plotly chart functions
├── metrics.py            ← Formats evaluation metrics for display
└── layout.py             ← Page layout and navigation

Run:  streamlit run dashboard.py
```

On Kaggle, expose with:
```python
!pip install pyngrok
from pyngrok import ngrok
public_url = ngrok.connect(8501)
print(f"Dashboard: {public_url}")
!streamlit run dashboard.py &
```
