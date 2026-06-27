# SynthFHIR

**Realistic Synthetic Healthcare Data Generation from FHIR Bundles**

SynthFHIR is a production-grade pipeline that ingests Synthea FHIR JSON bundles, applies
clinical preprocessing, and trains deep learning models (CTGAN + PAR) to generate
statistically faithful synthetic patient records — preserving clinical distributions and
relational structure while containing no real patient information.

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

## Current Status

| Phase | Description | Status |
|---|---|---|
| Phase 1 | Synthea data generation (1,000 patients, FHIR JSON) | ✅ Complete |
| Phase 2 | FHIR bundle parsing → 5 structured CSVs | ✅ Complete |
| Phase 3 | Clinical preprocessing + SDV metadata | ✅ Complete |
| Phase 4A | Smoke test (5 epochs, 20 rows per table) | ✅ Complete |
| Phase 4B | Fault-tolerant full training (CTGAN + PAR) | ✅ Complete |
| Phase 5 | Utility + privacy evaluation | 🔲 Planned |
| Phase 6 | FHIR bundle reconstruction from synthetic data | 🔲 Planned |
| Phase 7 | Interactive dashboard | 🔲 Planned |

---

## Setup

```bash
# Clone and enter the project
cd synthetic_health

# Install dependencies
pip install -r requirements.txt

# If using GPU (CUDA 12.1), install PyTorch first:
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

# Phase 4A: Smoke test (fast, 5 epochs)
python run_smoke_4b.py

# Phase 4B: Full training — all tables
python run_phase4b.py

# Phase 4B: Train one table only
python run_phase4b.py --table patients
python run_phase4b.py --table observations
python run_phase4b.py --table encounters

# Force retrain even if checkpoint exists
python run_phase4b.py --table patients --force

# Enable GPU
python run_phase4b.py --gpu

# Google Drive backup (Colab)
python run_phase4b.py --gdrive-dir /content/drive/MyDrive/SynthFHIR
```

---

## Repository Structure

```
synthetic_health/
├── README.md                    ← You are here
├── requirements.txt
├── config/
│   └── settings.yaml            ← All hyperparameters and paths
├── data/
│   ├── raw/                     ← 1000 Synthea FHIR JSON files (input)
│   ├── processed/               ← Phase 2 output: parsed CSVs
│   └── ready/                   ← Phase 3 output: preprocessed training tables
│       └── metadata.json        ← SDV MultiTableMetadata
├── src/
│   ├── parsers/                 ← Phase 2: FHIR resource extractors
│   ├── feature_engineering.py   ← Phase 3: all transformations
│   ├── preprocessor.py          ← Phase 3: orchestrator
│   ├── metadata_generator.py    ← Phase 3: SDV metadata builder
│   └── synthesis/               ← Phase 4: synthesis engine
│       ├── pipeline.py          ← Training orchestrator
│       ├── ctgan_trainer.py     ← CTGAN/TVAE/GaussianCopula wrapper
│       ├── par_trainer.py       ← PAR sequential model wrapper
│       ├── sampler.py           ← Post-training synthetic data sampler
│       ├── checkpoint.py        ← manifest.json + fault tolerance
│       ├── backup.py            ← Per-table ZIPs + multi-destination backup
│       ├── memory.py            ← RAM/GPU monitoring + joblib safety
│       └── config.py            ← SynthesisConfig dataclass
├── outputs/
│   ├── models/                  ← Trained .pkl model files
│   ├── synthetic/               ← Synthetic CSV outputs
│   ├── logs/                    ← manifest.json, training logs, statistics
│   ├── reports/                 ← Preprocessing and readiness reports
│   └── figures/                 ← EDA visualisations
├── run_phase2.py                ← CLI: FHIR parsing
├── run_phase3.py                ← CLI: preprocessing
├── run_phase4b.py               ← CLI: fault-tolerant full training
├── run_smoke_4b.py              ← CLI: smoke test
└── docs/                        ← Full engineering documentation ← READ THIS
```

---

## Key Design Principles

- **Never lose work.** Every table completion triggers an immediate ZIP checkpoint.
- **Resume from crash.** `manifest.json` records exactly which tables completed; the pipeline skips them on restart.
- **No SIGKILL.** `LOKY_MAX_CPU_COUNT=1` + `parallel_backend('sequential')` prevents joblib from forking child processes that the OS kills under memory pressure.
- **Privacy by design.** Names, SSN, dates, addresses are removed before any model sees data.
- **Swap models with config.** Changing `patient_model: ctgan` → `gaussian_copula` in `settings.yaml` requires no code changes.

---

## Documentation

Full engineering handbook → **[docs/](docs/)**

For questions about specific design decisions, see [docs/04_Model_Architecture.md](docs/04_Model_Architecture.md)
and [docs/09_Engineering_Log.md](docs/09_Engineering_Log.md).
