# SynthFHIR

**Production-grade synthetic healthcare data generation from FHIR bundles**

SynthFHIR ingests Synthea FHIR JSON bundles, applies clinical preprocessing, and trains
deep generative models (CTGAN + PAR + GaussianCopula) to produce statistically faithful
synthetic patient records — preserving clinical distributions and relational structure
while containing no real patient information.

---

## Current Status

| Phase | Description | Status |
|---|---|---|
| Phase 1 | Synthea data generation — 1,000 patients, FHIR JSON | ✅ Complete |
| Phase 2 | FHIR bundle parsing → 5 structured CSVs | ✅ Complete |
| Phase 3 | Clinical preprocessing + SDV metadata | ✅ Complete |
| Phase 4A | Smoke test — 5 epochs, 20 rows per table | ✅ Complete |
| Phase 4B | Fault-tolerant full training (CTGAN + PAR) | ✅ Complete |
| Phase 5 | Utility + privacy evaluation (KS, k-Anonymity, TSTR) | ✅ Complete |
| Phase 6 | FHIR bundle reconstruction from synthetic data | 🔲 Planned |
| Phase 7 | Interactive dashboard | ✅ Complete |

**Global evaluation score: 0.776 (FAIR)**

---

## Quick Navigation

| I want to… | Go to |
|---|---|
| Understand what this project does | [docs/00_Project_Overview.md](docs/00_Project_Overview.md) |
| Learn FHIR and healthcare background | [docs/01_Background.md](docs/01_Background.md) |
| Understand the dataset | [docs/02_Dataset.md](docs/02_Dataset.md) |
| Understand preprocessing decisions | [docs/03_Preprocessing.md](docs/03_Preprocessing.md) |
| Understand model selection | [docs/04_Model_Architecture.md](docs/04_Model_Architecture.md) |
| Understand the training pipeline | [docs/05_Training_Pipeline.md](docs/05_Training_Pipeline.md) |
| See evaluation methodology | [docs/06_Evaluation.md](docs/06_Evaluation.md) |
| See the engineering incident log | [docs/09_Engineering_Log.md](docs/09_Engineering_Log.md) |
| Troubleshoot or read FAQ | [docs/10_FAQ.md](docs/10_FAQ.md) |

---

## Setup

```bash
# Clone and enter the project
cd synthetic_health

# Install dependencies
pip install -r requirements.txt

# GPU (CUDA 12.1) — install PyTorch first, then requirements
pip install torch --index-url https://download.pytorch.org/whl/cu121
pip install -r requirements.txt
```

---

## Running the Pipeline

```bash
# Phase 2: Parse FHIR bundles → data/processed/
python run_phase2.py

# Phase 3: Preprocess → data/ready/
python run_phase3.py

# Phase 4A: Smoke test (fast, 5 epochs, 20 rows per table)
python run_smoke_4b.py

# Phase 4B: Full training — all tables
python run_phase4b.py

# Phase 4B: Train one table only
python run_phase4b.py --table patients
python run_phase4b.py --table encounters

# Force retrain even if checkpoint exists
python run_phase4b.py --table patients --force

# Enable GPU
python run_phase4b.py --gpu

# Google Drive backup (Colab/Kaggle)
python run_phase4b.py --gdrive-dir /content/drive/MyDrive/SynthFHIR

# Phase 5: Evaluate all tables
python run_phase5.py

# Phase 5: Evaluate specific tables, skip plots
python run_phase5.py --tables patients conditions --no-plots

# Launch dashboard
streamlit run app.py
```

---

## Repository Structure

```
synthetic_health/
├── README.md
├── requirements.txt
├── config/
│   └── settings.yaml              ← All hyperparameters and model assignments
├── data/
│   ├── raw/                       ← 1,000 Synthea FHIR JSON files (input)
│   ├── processed/                 ← Phase 2: parsed CSVs
│   └── ready/                     ← Phase 3: preprocessed training tables
│       └── metadata.json          ← SDV MultiTableMetadata
├── src/
│   ├── parsers/                   ← Phase 2: FHIR resource extractors
│   │   ├── patient.py
│   │   ├── encounter.py
│   │   ├── observation.py
│   │   ├── condition.py
│   │   └── medication.py
│   ├── feature_engineering.py     ← Phase 3: all transformations
│   ├── preprocessor.py            ← Phase 3: orchestrator
│   ├── metadata_generator.py      ← Phase 3: SDV metadata builder
│   ├── evaluation/                ← Phase 5: evaluation modules
│   │   ├── config.py
│   │   ├── loader.py
│   │   ├── dataset_summary.py
│   │   ├── numeric_eval.py
│   │   ├── categorical_eval.py
│   │   ├── correlation_eval.py
│   │   ├── privacy_eval.py
│   │   ├── sdv_quality.py
│   │   ├── ks_eval.py             ← KS Test (Kolmogorov-Smirnov)
│   │   ├── privacy_k_anonymity.py ← k-Anonymity
│   │   ├── tstr_eval.py           ← TSTR (Train-on-Synthetic/Test-on-Real)
│   │   └── report.py              ← HTML + CSV report generation
│   └── synthesis/                 ← Phase 4: synthesis engine
│       ├── pipeline.py            ← Training orchestrator
│       ├── ctgan_trainer.py       ← CTGAN / GaussianCopula wrapper
│       ├── par_trainer.py         ← PAR sequential model wrapper
│       ├── sampler.py             ← Post-training synthetic data sampler
│       ├── checkpoint.py          ← manifest.json + fault tolerance
│       ├── backup.py              ← Per-table ZIPs + multi-destination backup
│       ├── memory.py              ← RAM/GPU monitoring + joblib safety
│       └── config.py              ← SynthesisConfig dataclass
├── outputs/
│   ├── models/                    ← Trained .pkl files + training_time.json
│   ├── synthetic/                 ← Synthetic CSV outputs
│   ├── evaluation/                ← Phase 5 results (CSV, JSON, HTML, plots)
│   ├── logs/                      ← manifest.json, training logs
│   └── reports/                   ← Preprocessing and readiness reports
├── app.py                         ← Streamlit dashboard
├── run_phase2.py                  ← CLI: FHIR parsing
├── run_phase3.py                  ← CLI: preprocessing
├── run_phase4b.py                 ← CLI: fault-tolerant full training
├── run_phase5.py                  ← CLI: evaluation
├── run_smoke_4b.py                ← CLI: smoke test
└── docs/                          ← Full engineering documentation
```

---

## Model Architecture

Each table uses a different synthesizer based on its data structure:

| Table | Model | Reason |
|---|---|---|
| patients | CTGANSynthesizer | One row per patient — purely cross-sectional tabular data |
| encounters | PARSynthesizer | Multiple encounters per patient — temporal sequences |
| observations | GaussianCopulaSynthesizer* | CTGAN configured; fell back due to MemoryError on 303k rows |
| conditions | PARSynthesizer | Multiple diagnoses per patient — temporal sequences |
| medications | PARSynthesizer | Multiple prescriptions per patient — temporal sequences |

\* *Configured as CTGAN; GaussianCopula is the automatic MemoryError fallback.
The actual model used is recorded in `outputs/models/training_time.json`.*

### Why PAR instead of CTGAN for sequential tables?

CTGAN treats every row as independent. A patient's 47 encounters are generated as 47
unrelated random rows with no memory of each other. PAR (Probabilistic Auto-Regressive)
models each patient's sequence explicitly — learning "given this patient had encounter A,
what comes next" — preserving temporal dependencies and ensuring that the generated
sequence is clinically plausible for that patient's profile.

### Automatic GaussianCopula fallback

If CTGAN training raises `MemoryError`, the pipeline silently downgrades to
`GaussianCopulaSynthesizer`, logs a warning, records the actual model type in
`manifest.json`, and continues. GaussianCopula is statistically weaker (it cannot
model LOINC-conditioned distributions in observations) but always fits in memory.

---

## Training Results

| Table | Model | Original Rows | Synthetic Rows | Training Time |
|---|---|---|---|---|
| patients | CTGAN | 998 | 1,000 | 12 s |
| encounters | PAR | 57,667 | 63,866 | 90 min 59 s |
| observations | GaussianCopula | 303,696 | 304,304 | 2 min 37 s |
| conditions | PAR | 37,835 | 32,196 | 62 min 6 s |
| medications | PAR | 46,734 | 25,100 | 34 min 39 s |

---

## Evaluation Results (Phase 5)

**Global score: 0.776 / 1.0**

### Per-Table Utility Scores

| Table | Numeric Sim | Categorical Sim | Correlation | Privacy | Overall |
|---|---|---|---|---|---|
| patients | 0.574 | 0.805 | 0.522 | 1.000 | **0.718** |
| encounters | 0.275 | 0.978 | 0.940 | 1.000 | **0.764** |
| observations | 0.733 | 0.998 | 0.980 | 1.000 | **0.915** |
| conditions | 0.707 | 0.474 | 0.701 | 1.000 | **0.695** |
| medications | 0.815 | 0.614 | — | 1.000 | **0.786** |

### KS Test (Kolmogorov-Smirnov)

KS similarity = 1 − KS statistic; higher is better. Measures distributional fidelity
column by column across all numeric features.

| Table | Columns Tested | Mean KS Stat | Mean KS Similarity |
|---|---|---|---|
| patients | 7 | 0.427 | 0.574 |
| encounters | 3 | 0.725 | 0.275 |
| observations | 2 | 0.268 | 0.733 |
| conditions | 3 | 0.293 | 0.707 |
| medications | 1 | 0.186 | 0.815 |

### k-Anonymity

Computed on synthetic data only. QI columns detected by keyword matching
(age, gender, race, zip, state, marital, etc.) with cardinality ≤ 200.

| Table | Result |
|---|---|
| patients | min-k = 1, 1.6% of records in groups ≥ 5 — warrants review |
| encounters | No QI columns found (no demographic attributes in this table) |
| observations | No QI columns found |
| conditions | No QI columns found |
| medications | No QI columns found |

### TSTR — Train on Synthetic, Test on Real

A `RandomForestClassifier` (100 estimators, depth 8) is trained on synthetic data and
evaluated on the original held-out data. ROC-AUC > 0.5 confirms the synthetic data
preserves predictive signal. F1 = 0.0 across all tables is expected: targets are
highly imbalanced (e.g., < 5% deceased) so the classifier defaults to the majority class,
achieving high accuracy but zero F1.

| Table | Target | Accuracy | F1 | ROC-AUC |
|---|---|---|---|---|
| patients | is_deceased | 0.870 | 0.00 | **0.744** |
| encounters | — | no suitable binary target | — | — |
| observations | — | no suitable binary target | — | — |
| conditions | is_chronic | 0.724 | 0.00 | **0.645** |
| medications | is_active | 0.926 | 0.00 | **0.460** |

---

## Key Design Principles

- **Never lose work.** Every table completion triggers an immediate ZIP checkpoint.
- **Resume from crash.** `manifest.json` records which tables completed; the pipeline skips them on restart.
- **No SIGKILL.** `LOKY_MAX_CPU_COUNT=1` + `parallel_backend('sequential')` prevents joblib from forking child processes that the OS kills under memory pressure.
- **Privacy by design.** Names, SSN, dates, addresses are stripped before any model sees data.
- **Swap models with config.** Changing `patient_model: ctgan` → `gaussian_copula` in `settings.yaml` requires zero code changes.
- **Robust path resolution.** The dashboard uses `Path(__file__).resolve().parent` — never `Path.cwd()` — so it works on Streamlit Cloud without env-var configuration.

---

## Dashboard

```bash
streamlit run app.py
```

The dashboard provides five pages:

- **Dashboard** — dataset KPIs, row counts, per-table quality scores, global score gauge
- **Dataset** — per-table schema explorer: original vs synthetic counts, column types, missingness
- **Analytics** — distribution charts: patient demographics, encounter types, LOINC observations, conditions, medications, cross-table comparisons
- **Evaluation** — KS similarity, k-Anonymity, TSTR metrics, interactive HTML report
- **Model Cards** — per-table synthesizer type, training duration, hyperparameters
- **About** — architecture overview, phase roadmap, technology stack

---

## Technology Stack

| Component | Library / Tool |
|---|---|
| FHIR parsing | `fhir.resources` 7.x |
| Synthetic data generation | `sdv` 1.37.2 |
| Tabular synthesis (cross-sectional) | `CTGANSynthesizer` (GAN-based) |
| Sequential synthesis | `PARSynthesizer` (autoregressive) |
| Fallback synthesis | `GaussianCopulaSynthesizer` |
| Deep learning backend | `PyTorch` 2.x |
| Statistical evaluation | `scipy.stats` (KS test) |
| ML evaluation (TSTR) | `scikit-learn` RandomForestClassifier |
| Data processing | `pandas` 2.x, `numpy` |
| Dashboard | `streamlit` 1.28+ |
| Visualisation | `plotly`, `matplotlib` |
| Config | `pyyaml` |

---

## Documentation

Full engineering handbook → **[docs/](docs/)**

| Document | Contents |
|---|---|
| [00_Project_Overview.md](docs/00_Project_Overview.md) | Problem statement, why synthetic data matters, project scope |
| [01_Background.md](docs/01_Background.md) | FHIR standard, HIPAA, Synthea, SDV library |
| [02_Dataset.md](docs/02_Dataset.md) | Table schemas, row counts, column descriptions |
| [03_Preprocessing.md](docs/03_Preprocessing.md) | Every transformation applied in Phase 3 with rationale |
| [04_Model_Architecture.md](docs/04_Model_Architecture.md) | CTGAN vs PAR vs GaussianCopula — when and why |
| [05_Training_Pipeline.md](docs/05_Training_Pipeline.md) | Fault tolerance, checkpointing, memory management |
| [06_Evaluation.md](docs/06_Evaluation.md) | Evaluation methodology and metric definitions |
| [09_Engineering_Log.md](docs/09_Engineering_Log.md) | Chronological incident log — every problem and fix |
| [10_FAQ.md](docs/10_FAQ.md) | Common questions and troubleshooting |
