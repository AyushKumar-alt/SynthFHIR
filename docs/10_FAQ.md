# 10 — FAQ, Glossary, and Troubleshooting

**Version:** 1.0  
**Last Updated:** 2026-06-28

---

## Table of Contents

1. [Frequently Asked Questions](#1-frequently-asked-questions)
2. [Troubleshooting Guide](#2-troubleshooting-guide)
3. [Glossary: Healthcare Terms](#3-glossary-healthcare-terms)
4. [Glossary: Machine Learning Terms](#4-glossary-machine-learning-terms)
5. [Glossary: Software and Infrastructure Terms](#5-glossary-software-and-infrastructure-terms)
6. [Useful Commands Reference](#6-useful-commands-reference)
7. [Repository Structure Reference](#7-repository-structure-reference)
8. [References and Further Reading](#8-references-and-further-reading)

---

## 1. Frequently Asked Questions

### Q: How is this different from just using Synthea directly?

Synthea generates patient records using pre-programmed clinical rules. SynthFHIR trains
deep learning models on Synthea-generated data and uses those models to generate a new
population. The key difference:

- **Synthea:** A rule-based simulator. Output reflects encoded clinical guidelines.
- **SynthFHIR:** A data-driven generator. Output reflects statistical patterns learned
  from the data, not hardcoded rules.

When pointed at real hospital data (not Synthea), SynthFHIR would produce synthetic data
reflecting that specific hospital's patient population — something Synthea cannot do.

### Q: Is the synthetic data private?

For this project: yes, trivially — because the training data (Synthea output) contains
no real people. There is nobody to re-identify.

For a future version using real EHR data: the synthetic data would be evaluated using
formal privacy metrics (singling-out, linkability, inference attacks) in Phase 5. The
goal is synthetic data that passes these evaluations, meaning it cannot be used to
infer information about specific real individuals.

### Q: Why not use differential privacy?

Differential privacy (DP) adds calibrated Gaussian noise to the model's gradients or
weights during training. It provides mathematically provable privacy guarantees.

DP was not applied in this version because:
1. Phase 5 evaluation first quantifies the current privacy risk — if the models already
   produce safe synthetic data without DP, adding DP (which reduces utility) is unnecessary
2. SDV 1.37.2 does not have built-in DP support for CTGAN or PAR
3. Implementing DP correctly is complex and out of scope for this phase

DP will be considered for Phase 5 if the privacy evaluation reveals unacceptable risk.

### Q: Can this pipeline work with real hospital EHR data?

Yes, with modifications:

1. **Input format:** The parsers expect FHIR R4 JSON bundles. Any EHR system that
   exports FHIR R4 (Epic, Cerner, Azure Health Data Services) can feed this pipeline.
2. **PII removal:** Additional preprocessing steps are required for real data —
   see [03_Preprocessing.md § 13](03_Preprocessing.md#13-what-would-change-for-real-ehr-data)
3. **Data governance:** A formal data access agreement and ethics approval would be required
4. **Scale:** Real hospital datasets can have millions of patients. The current pipeline
   would need distributed training or chunked processing for datasets above ~50,000 patients

### Q: What is the quality of the synthetic data?

Phase 5 (evaluation) will provide quantitative answers. From visual inspection after Phase 4B:

- Patient demographic distributions (gender, race, age) match the training data
- Encounter class distributions (AMB vs EMER vs IMP) match
- Vital sign value distributions per LOINC code appear realistic
- Temporal patterns (encounter frequency, inter-visit gaps) appear realistic

Formal evaluation metrics (KS statistics, TVD, correlation matrices, TSTR AUC) will
be reported after Phase 5.

### Q: Why are there only 998 patients when Synthea was run for 1000?

Two patients in the Synthea output had duplicate `patient_id` values (a known edge case
in batch Synthea runs). The Phase 3 preprocessing deduplication step (`drop_duplicates`
on `patient_id`) removes the duplicate, leaving 998.

### Q: Why is the `city` column still in the patients table? Isn't that PII?

For Synthea data: no. Synthea assigns cities from Massachusetts census data — they are
not real patient addresses. "Arlington" in the patients table means "this patient was
generated with Massachusetts demographics that Synthea assigned to Arlington," not "a
real person in Arlington."

For real EHR data: city is a quasi-identifier that could contribute to re-identification.
See [03_Preprocessing.md § 13](03_Preprocessing.md#13-what-would-change-for-real-ehr-data).

### Q: Why train tables separately instead of using SDV's multi-table HMA?

See [04_Model_Architecture.md § 10](04_Model_Architecture.md#10-hma--hierarchical-modelling-algorithm)
for the full analysis. Short answer: HMA in SDV 1.37.2 has API instability issues,
does not support PAR as the child table model (losing sequential modelling for encounters),
and loads all tables into memory simultaneously (creating memory pressure on large datasets).

### Q: How long does full training take?

Approximate training times on a Kaggle T4 GPU (16 GB VRAM):

| Table | Model | Rows | Time |
|---|---|---|---|
| patients | CTGAN | 998 | ~4 min |
| encounters | PAR | 57,667 | ~25 min |
| observations | CTGAN | 303,696 | ~2–3 hr |
| conditions | PAR | 37,835 | ~20 min |
| medications | PAR | 46,734 | ~20 min |
| **Total** | | **446,930** | **~3.5–4 hr** |

On CPU (no GPU): approximately 8–10× slower. Observations alone would take ~20+ hours on CPU.

### Q: The kernel crashed. How do I recover?

If you are on Kaggle:

1. Check `/kaggle/output` for `SynthFHIR_checkpoint_after_*.zip` files
2. Download the most recent ZIP (the one with the latest table name)
3. Start a new Kaggle session
4. Add the ZIP as a Kaggle input dataset (named "synthfhir-checkpoint")
5. In the notebook, before running the pipeline:
   ```python
   import zipfile
   with zipfile.ZipFile('/kaggle/input/synthfhir-checkpoint/<zipname>', 'r') as z:
       z.extractall('/kaggle/working/SynthFHIR')
   ```
6. Run `python run_phase4b.py` — it will auto-detect the checkpoint and resume

If the ZIP is not in `/kaggle/output`, it may be in the session output panel. Look
for files named `SynthFHIR_checkpoint_after_*.zip`.

### Q: Can I train just one table?

Yes:

```bash
python run_phase4b.py --table patients
python run_phase4b.py --table encounters
python run_phase4b.py --table observations
python run_phase4b.py --table conditions
python run_phase4b.py --table medications
```

Note: training encounters before observations, conditions, and medications is important
because those tables reference encounter IDs. Training patients must come first.

---

## 2. Troubleshooting Guide

### Error: `SyntaxError: from __future__ imports must occur at the beginning of the file`

**Cause:** Something is imported before `from __future__ import annotations` in `run_phase4b.py`.

**Fix:** Ensure the file starts exactly with:
```python
from __future__ import annotations

import os
os.environ.setdefault(...)
```

No imports before `from __future__ import annotations`.

---

### Error: `TerminatedWorkerError: exit code -9 [SIGKILL]`

**Cause:** The joblib multiprocessing fix is not active. Environment variables were not
set before SDV was imported.

**Fix:**
1. Verify `run_phase4b.py` sets `LOKY_MAX_CPU_COUNT=1` before all other imports
2. Verify `synth.fit()` is wrapped in `with parallel_backend("sequential"):`
3. Check that `LOKY_MAX_CPU_COUNT` is actually in `os.environ` when training starts:
   ```python
   import os; print(os.environ.get("LOKY_MAX_CPU_COUNT"))  # should print "1"
   ```

---

### Error: `ValueError: Column 'is_deceased' has sdtype 'boolean' but contains non-boolean values`

**Cause:** SDV 1.37.2 strict boolean validation. The column contains 0/1 integers.

**Fix:** The `_coerce_booleans()` function in `ctgan_trainer.py` handles this. If you
are seeing this error, check that:
1. `ctgan_trainer.py` has `_coerce_booleans()` defined
2. `df_prepared = _coerce_booleans(df, table_meta_dict)` is called before `synth.fit(df_prepared)`

---

### Error: `TypeError: unexpected keyword argument 'sequence_key'` (PAR training)

**Cause:** SDV API changed in 1.37.2. `sequence_key` is not a `PARSynthesizer` constructor
argument.

**Fix:** Sequence keys must be embedded in the `SingleTableMetadata` object. Use
`build_par_metadata()` in `par_trainer.py`:
```python
meta = build_par_metadata(table_meta_dict)
synth = PARSynthesizer(meta, context_columns=[], epochs=300)
# NOT: PARSynthesizer(meta, sequence_key="patient_id")
```

---

### Error: `FileNotFoundError: data/ready/patients_ready.csv not found`

**Cause:** Phase 3 has not been run yet.

**Fix:** Run Phase 3 first:
```bash
python run_phase3.py
```

If you have already run it, check `data/ready/`:
```bash
ls data/ready/
```
The directory should contain all five `*_ready.csv` files and `metadata.json`.

---

### Error: `ModuleNotFoundError: No module named 'sdv'`

**Cause:** Dependencies not installed.

**Fix:**
```bash
pip install -r requirements.txt
```

If on Kaggle, add `sdv` to the notebook:
```python
!pip install sdv>=1.37.0,<2.0.0
```

---

### Error: `torch.cuda.is_available()` returns `False` when GPU should be available

**Cause:** CUDA not available or wrong PyTorch version installed.

**Fix for Kaggle:** In the notebook settings, enable GPU (Settings → Accelerator → GPU T4 x1).
Then restart the kernel.

**Fix for local:**
```bash
# Check CUDA version
nvcc --version

# Install PyTorch with matching CUDA version
pip install torch --index-url https://download.pytorch.org/whl/cu121  # for CUDA 12.1
```

---

### Warning: `psutil not found — memory monitoring disabled`

**Cause:** `psutil` is not installed.

**Fix:**
```bash
pip install psutil>=5.9.0
```

This is a non-critical dependency. Training will proceed without memory monitoring.

---

### manifest.json shows `"status": "file_missing"` for a table

**Cause:** The manifest recorded a table as complete, but the model or CSV file was
deleted or is missing.

**Fix:**
- If the ZIP checkpoint is available, extract it to recover the files
- Otherwise, delete the table's entry from `manifest.json` and retrain that table:
  ```bash
  python run_phase4b.py --table patients --force
  ```

---

## 3. Glossary: Healthcare Terms

**Abatement:** The end or resolution of a condition. A condition's `abatement_datetime`
is when it resolved. An active chronic condition has no abatement.

**Ambulatory (AMB):** An encounter class where the patient visits the clinic and leaves
the same day (outpatient). Contrast with Inpatient (IMP) where the patient stays overnight.

**Chronic condition:** A health condition with no expected resolution — it persists
indefinitely. Hypertension, type 2 diabetes, and asthma are examples.

**Clinical pathway:** The sequence of healthcare events (visits, tests, diagnoses,
treatments) that a patient experiences over time.

**Condition:** A clinical diagnosis — a disease or health problem.

**DALY (Disability-Adjusted Life Years):** A measure of disease burden. One DALY = one
year of healthy life lost due to disability, illness, or premature death. Higher = worse health.

**Discharge disposition:** For inpatient encounters, how the patient left the hospital:
discharged home, transferred to another facility, died, etc.

**EHR (Electronic Health Record):** A digital system for recording patient health
information in a healthcare setting.

**Encounter:** A single interaction between a patient and the healthcare system. An
encounter has a specific class (outpatient, emergency, inpatient), type (annual physical,
emergency visit, follow-up), location, and duration.

**FHIR (Fast Healthcare Interoperability Resources):** The HL7 standard for exchanging
healthcare data using RESTful APIs and JSON.

**HL7 (Health Level 7):** The organisation that creates healthcare data exchange standards.

**HbA1c (Haemoglobin A1c):** A blood test measuring average blood glucose over the past
2–3 months. The key monitoring test for diabetes management. Normal: <5.7%. Diabetic: ≥6.5%.

**Inpatient (IMP):** An encounter class where the patient is admitted to hospital and
stays overnight or longer.

**LOINC (Logical Observation Identifiers Names and Codes):** A universal coding system
for laboratory tests and clinical measurements. LOINC 8480-6 = Systolic Blood Pressure.

**Observation:** A clinical measurement taken during an encounter. Examples: blood
pressure, body weight, blood glucose, haemoglobin A1c.

**Onset:** When a condition first appeared or was first diagnosed.

**QALY (Quality-Adjusted Life Year):** A measure of quality + quantity of life. One year
in perfect health = 1 QALY. One year severely disabled = 0.2 QALY.

**RxNorm:** The US standard drug coding system. Identifies specific drug formulations
by name, dose form, and strength.

**SNOMED CT (Systematized Nomenclature of Medicine — Clinical Terms):** The comprehensive
clinical terminology for conditions, procedures, and findings.

**Synthea:** An open-source synthetic patient generator from The MITRE Corporation.
Produces FHIR JSON bundles.

---

## 4. Glossary: Machine Learning Terms

**Autoregressive model:** A model that predicts the next element in a sequence from the
previous elements. PAR (Probabilistic AutoRegressive) is the autoregressive model used
for encounter sequences.

**Batch size:** The number of training examples used in one gradient update step.

**Class imbalance:** When training data contains far more examples of one class than
another (e.g., 95% no-diabetes, 5% diabetes). Models trained on imbalanced data tend
to ignore the minority class.

**Copula:** A statistical function that captures the dependency structure between random
variables independently of their individual distributions.

**CTGAN (Conditional Tabular GAN):** A GAN architecture for tabular data that uses a
conditional generator to handle categorical imbalance.

**Differential privacy:** A mathematical framework for training models that provides
provable privacy guarantees by adding calibrated noise to the training process.

**Epoch:** One complete pass through the training data.

**GAN (Generative Adversarial Network):** A generative model architecture consisting
of a Generator (creates fake data) and a Discriminator (distinguishes real from fake).

**Gaussian Copula:** A copula that assumes the dependency structure is multivariate
Gaussian.

**Gradient:** The direction of steepest increase for a loss function. Gradient descent
moves in the opposite direction to minimise the loss.

**IQR (Interquartile Range):** The difference between the 75th percentile (Q3) and
25th percentile (Q1) of a distribution. Used for outlier detection (3×IQR method).

**KL Divergence (Kullback-Leibler Divergence):** A measure of how one probability
distribution differs from a reference distribution. Used in VAE loss functions.

**KS Test (Kolmogorov-Smirnov test):** A statistical test comparing two sample
distributions. KS statistic = 0 means identical distributions; 1 means completely different.

**LSTM (Long Short-Term Memory):** A type of recurrent neural network that can learn
long-range dependencies in sequential data. Used inside PAR.

**Mode collapse:** A failure mode of GAN training where the Generator learns to produce
only a few output types, ignoring most of the training distribution.

**Overfitting:** When a model memorises training data so well that it performs poorly
on new data. For synthetic data, overfitting means memorising real patient records.

**PAR (Probabilistic AutoRegressive):** SDV's sequential synthesis model based on LSTM.

**SDV (Synthetic Data Vault):** The Python library providing CTGAN, TVAE, PAR, and
GaussianCopula implementations.

**SIGKILL:** Linux signal number 9. Sent by the OS to unconditionally terminate a process.
Cannot be caught, blocked, or handled.

**TSTR (Train on Synthetic, Test on Real):** An evaluation paradigm: train a classifier
on synthetic data and test it on real data. If performance matches a real-trained
classifier, the synthetic data is statistically faithful.

**TVD (Total Variation Distance):** A measure of how different two categorical
distributions are. TVD = 0.5 × Σ|P(x) - Q(x)| over all categories.

**TVAE (Tabular Variational Autoencoder):** A VAE-based tabular generative model.
Uses the same RDT preprocessing as CTGAN.

**VAE (Variational Autoencoder):** A generative model that learns a compressed latent
representation of data and generates new samples by decoding from the latent space.

**VGM (Variational Gaussian Mixture):** A method for normalising skewed numerical
distributions by identifying multiple modes and normalising relative to each mode.

---

## 5. Glossary: Software and Infrastructure Terms

**Atomic write:** A file write operation where the file appears at the target path
completely or not at all — no partial state is ever visible.

**Colab (Google Colaboratory):** A free cloud notebook environment from Google with
optional GPU access.

**CUDA:** NVIDIA's parallel computing platform for GPU programming. Required for GPU-
accelerated PyTorch training.

**Foreign key:** A column in one table that references the primary key of another table.
Defines relationships between tables.

**joblib:** A Python library for lightweight parallel computing. Used internally by
SDV and scikit-learn.

**Kaggle:** A data science platform with free GPU-accelerated notebook kernels.

**loky:** The default backend for joblib, based on the loky process pool executor.
`LOKY_MAX_CPU_COUNT` controls how many workers it spawns.

**Manifest:** A file listing the status, location, and integrity of all completed
training outputs. The `outputs/logs/manifest.json` file in this project.

**OOM Killer:** The Linux Out-Of-Memory Killer. A kernel mechanism that terminates
processes when RAM is critically low. Sends SIGKILL.

**Pickle (.pkl):** A Python binary serialisation format. SDV models are saved as `.pkl` files.

**Primary key:** A column whose values are unique within a table. Identifies each row.

**psutil:** A Python library for system monitoring — CPU, RAM, disk, processes.

**PyTorch:** Facebook's deep learning library. Used by CTGAN and PAR for neural
network operations.

**RDT (Reversible Data Transformers):** SDV's preprocessing library that converts
tabular data into numerical form suitable for neural network training.

**SHA-256:** A cryptographic hash function producing a 256-bit hash. Used to verify
file integrity — if a file changes, its SHA-256 hash changes.

**UUID (Universally Unique Identifier):** A 128-bit random number used as a unique
identifier. Format: `xxxxxxxx-xxxx-xxxx-xxxx-xxxxxxxxxxxx`.

---

## 6. Useful Commands Reference

```bash
# Install all dependencies
pip install -r requirements.txt

# Run Phase 2: Parse FHIR bundles
python run_phase2.py

# Run Phase 3: Preprocess
python run_phase3.py

# Run smoke test (5 epochs, fast)
python run_smoke_4b.py

# Run full training (all tables)
python run_phase4b.py

# Run full training with GPU
python run_phase4b.py --gpu

# Train one table only
python run_phase4b.py --table patients
python run_phase4b.py --table encounters
python run_phase4b.py --table observations
python run_phase4b.py --table conditions
python run_phase4b.py --table medications

# Force retrain even if checkpoint exists
python run_phase4b.py --table patients --force

# Specify Google Drive backup directory (Colab)
python run_phase4b.py --gdrive-dir /content/drive/MyDrive/SynthFHIR

# Check manifest status
python -c "import json; print(json.dumps(json.load(open('outputs/logs/manifest.json'))['tables'], indent=2))"

# Count rows in all ready CSVs
python -c "
import pandas as pd; from pathlib import Path
for f in sorted(Path('data/ready').glob('*_ready.csv')):
    df = pd.read_csv(f, usecols=[0])
    print(f'{f.name}: {len(df):,} rows')
"
```

---

## 7. Repository Structure Reference

```
synthetic_health/
│
├── README.md                      Root overview and quick start
├── requirements.txt               Python dependencies
│
├── config/
│   └── settings.yaml              All hyperparameters, paths, model selection
│
├── data/
│   ├── raw/                       1000 Synthea FHIR JSON bundles (input, not committed)
│   ├── processed/                 Phase 2 output: raw extracted CSVs
│   └── ready/                     Phase 3 output: preprocessed training tables
│       ├── patients_ready.csv     (998 rows × 18 cols)
│       ├── encounters_ready.csv   (57,667 rows × 12 cols)
│       ├── observations_ready.csv (303,696 rows × 9 cols)
│       ├── conditions_ready.csv   (37,835 rows × 10 cols)
│       ├── medications_ready.csv  (46,734 rows × 10 cols)
│       └── metadata.json          SDV MultiTableMetadata
│
├── src/
│   ├── __init__.py
│   ├── config_loader.py           Loads settings.yaml into Config object
│   ├── feature_engineering.py     Phase 3 transformations (pure functions)
│   ├── metadata_generator.py      SDV metadata builder
│   ├── preprocessor.py            Phase 3 orchestrator
│   ├── validator.py               FK validation utilities
│   └── parsers/                   Phase 2 FHIR parsers
│       ├── patient.py
│       ├── encounter.py
│       ├── observation.py
│       ├── condition.py
│       └── medication.py
│   └── synthesis/                 Phase 4 synthesis engine
│       ├── pipeline.py            Training orchestrator
│       ├── ctgan_trainer.py       CTGAN/TVAE/GaussianCopula wrapper
│       ├── par_trainer.py         PAR wrapper
│       ├── sampler.py             Synthetic data sampling
│       ├── checkpoint.py          manifest.json management
│       ├── backup.py              ZIP creation and distribution
│       ├── memory.py              RAM/GPU monitoring
│       ├── config.py              SynthesisConfig dataclass
│       └── progress.py            ProgressTracker
│
├── outputs/
│   ├── models/                    Trained .pkl model files
│   ├── synthetic/                 Synthetic CSV outputs
│   ├── logs/                      manifest.json, training.log, statistics
│   ├── reports/                   Preprocessing reports
│   └── figures/                   EDA charts
│
├── run_phase2.py                  CLI: FHIR parsing
├── run_phase3.py                  CLI: preprocessing
├── run_phase4a.py                 CLI: Phase 4A (legacy)
├── run_phase4b.py                 CLI: fault-tolerant full training
├── run_smoke_4b.py                CLI: smoke test
├── run_readiness.py               CLI: data readiness check
│
├── tests/                         Test suite
│
└── docs/                          Engineering documentation
    ├── README.md
    ├── 00_Project_Overview.md
    ├── 01_Background.md
    ├── 02_Dataset.md
    ├── 03_Preprocessing.md
    ├── 04_Model_Architecture.md
    ├── 05_Training_Pipeline.md
    ├── 06_Evaluation.md
    ├── 07_FHIR_Reconstruction.md
    ├── 08_Dashboard.md
    ├── 09_Engineering_Log.md
    ├── 10_FAQ.md
    ├── diagrams/
    └── images/
```

---

## 8. References and Further Reading

### Papers

1. **CTGAN:** Xu, L., Skoularidou, M., Cuesta-Infante, A., & Veeramachaneni, K. (2019).
   *Modeling Tabular Data using Conditional GAN.*
   NeurIPS 2019. https://arxiv.org/abs/1907.00503

2. **TVAE:** Same paper as CTGAN (TVAE was introduced in the same paper as a comparison
   model).

3. **Goodfellow GAN:** Goodfellow, I., et al. (2014). *Generative Adversarial Networks.*
   NeurIPS 2014. https://arxiv.org/abs/1406.2661

4. **VAE:** Kingma, D., & Welling, M. (2013). *Auto-Encoding Variational Bayes.*
   ICLR 2014. https://arxiv.org/abs/1312.6114

5. **PAR:** Jarrett, D., Cebere, B., Liu, T., Curth, A., & van der Schaar, M. (2021).
   *HyperImpute: Generalised Iterative Imputation with Automatic Model Selection.*
   (PAR is documented in the SDV library, not in a standalone paper.)

6. **Synthetic Health Data Review:** Walonoski, J., et al. (2018). *Synthea: An Approach,
   Method, and Software Mechanism for Generating Synthetic Patients and the Synthetic
   Electronic Health Care Record.* Journal of the American Medical Informatics Association.

7. **Re-identification Risk:** Sweeney, L. (2000). *Simple Demographics Often Identify
   People Uniquely.* Carnegie Mellon University, Data Privacy Working Paper 3.

### Documentation

- **SDV Documentation:** https://docs.sdv.dev/sdv
- **FHIR R4 Specification:** https://hl7.org/fhir/R4/
- **Synthea Documentation:** https://synthetichealth.github.io/synthea/
- **LOINC:** https://loinc.org
- **SNOMED CT:** https://www.snomed.org
- **RxNorm:** https://www.nlm.nih.gov/research/umls/rxnorm/

### Tools Used

- **SDV (Synthetic Data Vault):** https://github.com/sdv-dev/SDV
- **PyTorch:** https://pytorch.org
- **Pandas:** https://pandas.pydata.org
- **Streamlit:** https://streamlit.io
- **psutil:** https://github.com/giampaolo/psutil
- **anonymeter:** https://github.com/statice/anonymeter
