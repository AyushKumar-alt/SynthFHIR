# 06 — Evaluation

**Version:** 1.0  
**Last Updated:** 2026-06-28  
**Phase Coverage:** Phase 5 (Planned)  
**Status:** Design complete — implementation pending

---

## Table of Contents

1. [Why Evaluation Is Necessary](#1-why-evaluation-is-necessary)
2. [Two Dimensions of Evaluation](#2-two-dimensions-of-evaluation)
3. [Utility Evaluation: Statistical Fidelity](#3-utility-evaluation-statistical-fidelity)
4. [Privacy Evaluation: Re-identification Risk](#4-privacy-evaluation-re-identification-risk)
5. [The Privacy-Utility Tradeoff](#5-the-privacy-utility-tradeoff)
6. [Planned Evaluation Metrics](#6-planned-evaluation-metrics)
7. [Implementation Plan](#7-implementation-plan)
8. [Expected Outputs](#8-expected-outputs)

---

## 1. Why Evaluation Is Necessary

Generating synthetic data is not sufficient on its own. Without evaluation, you cannot
know whether:

- The synthetic data actually resembles the real data statistically
- The model memorised specific patients from the training data (privacy failure)
- The synthetic data is useful for machine learning (a model trained on it would
  perform well on real data)
- Rare events (rare conditions, rare medications) are preserved in the synthetic data

A synthetic dataset that passes evaluation is a synthetic dataset you can distribute
with confidence. One that fails evaluation reveals which aspects of the generative model
need improvement.

---

## 2. Two Dimensions of Evaluation

Synthetic health data evaluation has two fundamental dimensions that exist in tension
with each other:

```
Utility                              Privacy
(How realistic?)      ←──────────→  (How safe?)

High utility =                       High privacy =
synthetic data closely               synthetic data does not
resembles real data                  leak individual records

These goals partially conflict:
A synthetic record that perfectly copies a real patient's record
has perfect utility but zero privacy.
```

A good synthetic dataset sits in the upper-right of this trade-off space:
high utility AND high privacy. Evaluation measures both.

---

## 3. Utility Evaluation: Statistical Fidelity

**Definition:** Does the synthetic data have the same statistical properties as the real data?

### 3.1 Column-Shape Metrics

For each column in each table, compare the distribution of values between real and
synthetic data.

**Numerical columns (age, value_quantity, days_since_birth):**
- **KS test (Kolmogorov-Smirnov):** Tests whether two samples come from the same distribution.
  KS statistic ranges from 0 (identical) to 1 (completely different). Target: KS < 0.1
- **Wasserstein distance (Earth Mover's Distance):** Measures how much distribution
  mass must be moved to transform one distribution into another.
- **Mean, median, standard deviation comparison:** Basic summary statistics should match.

**Categorical columns (gender, race, loinc_display):**
- **Total Variation Distance (TVD):** Measures the difference between two categorical
  probability distributions. TVD = 0.5 × Σ|P(x) - Q(x)|. Target: TVD < 0.1
- **Chi-squared test:** Tests whether observed frequencies in synthetic data are
  consistent with expected frequencies from real data.

### 3.2 Pairwise Correlation Metrics

For numerical columns, compute the Pearson correlation matrix for both real and synthetic
data, then compare:

```
CorrelationDifference = |corr_real[i,j] - corr_synth[i,j]|

Mean correlation error across all column pairs: target < 0.05
```

This detects whether the generative model preserved the relationships between columns.
Example: if BMI and age are positively correlated in real data, they should be in
synthetic data too.

### 3.3 Machine Learning Efficacy (Train on Synthetic, Test on Real)

The most powerful utility test: train a predictive model on synthetic data and test
it on real data. If the synthetic data is statistically faithful, the model should
perform similarly to one trained on real data.

**Protocol:**
1. Define a prediction task: "Predict whether a patient will be diagnosed with
   Type 2 Diabetes"
2. Extract features from the clinical tables (age, BMI, HbA1c history, family history)
3. Train Classifier A on **real** data, evaluate on a held-out **real** test set → AUC_real
4. Train Classifier B on **synthetic** data, evaluate on the same **real** test set → AUC_synth
5. Compare: if `AUC_synth ≈ AUC_real`, the synthetic data is faithful

Target: `|AUC_real - AUC_synth| < 0.05`

### 3.4 Temporal Pattern Evaluation (Encounters/Conditions/Medications)

For sequential tables, evaluate whether the temporal structure is preserved:

- Distribution of inter-visit gaps (`days_since_prev_encounter`): histogram comparison
- Distribution of sequence lengths (number of encounters per patient): KS test
- Distribution of condition onset ages: KS test by condition type
- Medication adherence patterns: comparison of active duration distributions

---

## 4. Privacy Evaluation: Re-identification Risk

**Definition:** Can an adversary use the synthetic data to learn information about
specific real individuals?

SDV and the broader privacy research community define three types of privacy attack:

### 4.1 Singling-Out Attack

**Question:** "Can the attacker find a unique combination of attributes in the synthetic
data that corresponds to only one real individual?"

A patient with extremely rare characteristics (the only 108-year-old Asian female with
Cushing's syndrome in the dataset) might have a unique combination of values in both
the real and synthetic datasets. If a synthetic record matches this combination, an
attacker knows something about a real person.

**Metric:** Singling-Out Rate — the fraction of synthetic records that uniquely match
a real record on a combination of 2–3 columns.

**Tool:** `anonymeter` library's `SinglingOutEvaluator`

**Target:** Singling-Out Rate not significantly higher than the same rate on a
randomly generated control dataset.

### 4.2 Linkability Attack

**Question:** "Given one record in dataset A and one in dataset B (both derived from
the same real patient), can the attacker link them?"

In healthcare: can an attacker use a synthetic encounter record to identify which real
patient it corresponds to?

**Metric:** Linkability Rate — success rate of a trained linkage classifier.

**Target:** Linkability Rate ≈ baseline (no better than random guessing)

### 4.3 Inference Attack

**Question:** "Given everything in the synthetic data about a patient except one attribute,
can the attacker infer that attribute for the corresponding real patient?"

Example: The attacker knows a real patient's age, gender, and race. Can they use
correlations in the synthetic data to infer that patient's HIV status?

**Metric:** Inference Risk — how much better than baseline can the attacker predict
a sensitive attribute using correlations learned from the synthetic data?

**Tool:** `anonymeter` library's `InferenceEvaluator`

**Target:** Inference Risk ≈ baseline (correlations in synthetic data do not reveal
more than publicly known correlations)

---

## 5. The Privacy-Utility Tradeoff

High utility means the synthetic data closely mirrors the real data. High privacy means
the synthetic data cannot be used to re-identify real individuals. These goals conflict
because perfect mirroring would reveal individual records.

```
             │ High Utility
             │
      A      │         B
(poor synth) │   (ideal synthetic)
             │
─────────────┼──────────────────── (Privacy threshold)
             │
      C      │         D
 (safe but   │   (memorised real
  useless)   │    data - dangerous)
             │
             │ Low Utility
             Low Privacy        High Privacy
```

The goal is quadrant B: high utility AND high privacy.

Differential privacy (adding calibrated noise to model weights) can move the system
from B toward A (reducing utility to increase privacy guarantees). Phase 5 will measure
where the trained models sit on this spectrum without adding differential privacy noise,
and report the privacy-utility tradeoff quantitatively.

---

## 6. Planned Evaluation Metrics

| Category | Metric | Tool | Target |
|---|---|---|---|
| Column shapes | KS statistic (numerical) | scipy.stats.ks_2samp | < 0.1 per column |
| Column shapes | TVD (categorical) | custom | < 0.1 per column |
| Correlations | Pairwise correlation error | numpy | Mean < 0.05 |
| ML efficacy | AUC difference (TSTR) | sklearn | Difference < 0.05 |
| Privacy | Singling-out rate | anonymeter | ≈ control baseline |
| Privacy | Linkability rate | anonymeter | ≈ control baseline |
| Privacy | Inference risk | anonymeter | ≈ control baseline |
| Temporal | Inter-visit gap distribution | scipy.stats.ks_2samp | KS < 0.1 |
| Temporal | Sequence length distribution | scipy.stats.ks_2samp | KS < 0.1 |

SDV also provides a built-in `evaluate_quality()` report that produces an overall
"Quality Score" (0–1) and per-column scores. This will be run as a first-pass check.

---

## 7. Implementation Plan

Phase 5 will add a new entry point `run_phase5.py` and a new source module
`src/evaluation/`:

```
src/evaluation/
├── utility.py      — column-shape, correlation, ML efficacy evaluators
├── privacy.py      — singling-out, linkability, inference attack evaluators
├── reporter.py     — generates HTML and JSON evaluation reports
└── visualiser.py   — comparison plots (real vs synthetic distributions)
```

The evaluation will be run automatically after Phase 4B completes, using the
real data in `data/ready/` and the synthetic data in `outputs/synthetic/`.

---

## 8. Expected Outputs

After Phase 5 completes:

```
outputs/evaluation/
├── quality_report.html          — SDV built-in quality report
├── utility_report.json          — per-column KS, TVD, correlation errors
├── privacy_report.json          — singling-out, linkability, inference scores
├── ml_efficacy_report.json      — TSTR AUC comparison per prediction task
└── figures/
    ├── age_distribution_comparison.png
    ├── encounter_gap_comparison.png
    ├── loinc_value_comparison.png
    └── correlation_matrix_comparison.png
```

These outputs will feed into the Phase 7 dashboard as interactive visualisations.
