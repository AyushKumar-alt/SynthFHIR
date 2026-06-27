# 04 — Model Architecture

**Version:** 1.0  
**Last Updated:** 2026-06-28  
**Phase Coverage:** Phase 4A, 4B  
**Source Files:** [src/synthesis/ctgan_trainer.py](../src/synthesis/ctgan_trainer.py), [src/synthesis/par_trainer.py](../src/synthesis/par_trainer.py)

---

## Table of Contents

1. [The Core Challenge: Synthesising Tabular Healthcare Data](#1-the-core-challenge-synthesising-tabular-healthcare-data)
2. [Background: Generative Models](#2-background-generative-models)
3. [Generative Adversarial Networks (GANs)](#3-generative-adversarial-networks-gans)
4. [Variational Autoencoders (VAEs)](#4-variational-autoencoders-vaes)
5. [Copula Models](#5-copula-models)
6. [CTGAN — Conditional Tabular GAN](#6-ctgan--conditional-tabular-gan)
7. [TVAE — Tabular Variational Autoencoder](#7-tvae--tabular-variational-autoencoder)
8. [GaussianCopulaSynthesizer](#8-gaussiancopulasynthesizer)
9. [PAR — Probabilistic AutoRegressive Model](#9-par--probabilistic-autoregressive-model)
10. [HMA — Hierarchical Modelling Algorithm](#10-hma--hierarchical-modelling-algorithm)
11. [Model Selection Decisions](#11-model-selection-decisions)
12. [Per-Table Model Assignments](#12-per-table-model-assignments)
13. [Hyperparameter Decisions](#13-hyperparameter-decisions)
14. [GPU Support and Requirements](#14-gpu-support-and-requirements)
15. [SDV Library Architecture](#15-sdv-library-architecture)

---

## 1. The Core Challenge: Synthesising Tabular Healthcare Data

Synthesising tabular data is harder than it appears. Healthcare tables have specific
properties that naive approaches cannot handle:

**Property 1: Mixed data types.** A single table contains numerical columns (age: 48.7),
categorical columns (gender: "female"), boolean columns (is_deceased: 0), and columns with
high cardinality (loinc_display: 30 unique values).

**Property 2: Skewed distributions.** Healthcare values are rarely normally distributed.
Body mass index clusters around 25–30 but has a long right tail. Age at death has a
bimodal distribution (infant mortality + old age). Standard statistical models assuming
Gaussian distributions fail badly here.

**Property 3: Class imbalance.** Only ~30% of patients may have diabetes. A model that
does not account for this imbalance might generate a synthetic dataset where only 5% have
diabetes (the model learned "mostly no diabetes") or 80% have it (the model failed to
distinguish modes).

**Property 4: Inter-column correlations.** BMI correlates with age, gender, and race.
Hemoglobin A1c correlates with age and the presence of diabetes. A model that ignores
these correlations would generate records where a young, thin patient has diabetic-range
HbA1c values — clinically unrealistic.

**Property 5: Sequential dependencies.** Encounter #5 depends on the history of encounters
#1–4. A 75-year-old patient at encounter #50 looks very different from a 25-year-old at
encounter #5. Standard models treat rows independently.

The models selected for this project were chosen specifically because they address these
properties.

---

## 2. Background: Generative Models

A **generative model** is a statistical model that learns the probability distribution
of training data and can sample new examples from that distribution.

The simplest generative model is a normal distribution fitted to a dataset: estimate the
mean and standard deviation, then sample from `Normal(mean, std)` to generate new values.
This works for simple numerical data but fails for:

- Mixed types (you cannot sample "female" from a normal distribution)
- Skewed distributions (income data is log-normal, not normal)
- Multi-modal distributions (bimodal HbA1c: one peak for diabetics, one for non-diabetics)
- Correlated multi-dimensional data (the joint distribution of blood pressure + age + BMI)

Deep generative models — GANs, VAEs, and autoregressive models — were developed precisely
to handle these complex, high-dimensional distributions.

---

## 3. Generative Adversarial Networks (GANs)

A **GAN (Generative Adversarial Network)** was introduced by Goodfellow et al. in 2014.
It consists of two neural networks trained simultaneously in competition:

```
Real data ──→┐
              ├──→ Discriminator ──→ "Real" or "Fake" signal
Generator ───→┘
   ▲                                         │
   └─────────────────────────────────────────┘
              (adversarial feedback)
```

**The Generator:** Takes a vector of random noise as input and outputs synthetic data.
Initially, it produces garbage. Over time, it learns to produce data that the Discriminator
cannot distinguish from real data.

**The Discriminator:** Takes a sample (either real or generated) and outputs a probability
that it is real. Initially, it is easily fooled. Over time, it learns to detect fake data.

**The Training Process:**  
At each training step:
1. Sample a batch of real data
2. Sample a batch of noise → pass through Generator → fake data
3. Train Discriminator to classify real as "real" and fake as "fake"
4. Train Generator to produce data that Discriminator classifies as "real"
5. Repeat for N epochs

**Why GANs work for tabular data:**  
The Discriminator acts as a learned "realism detector" — it captures complex distributions,
correlations, and patterns without requiring the programmer to specify them mathematically.
The Generator is forced to learn all of these to fool the Discriminator.

**Problems with vanilla GANs for tabular data:**

1. **Mode collapse:** The Generator finds a small number of outputs that fool the
   Discriminator and stops exploring the full data distribution. Synthetic data clusters
   into a few unrealistic "modes."

2. **Categorical variables:** Neural networks output continuous numbers. Generating
   `gender = "female"` requires converting the string to a number and back. Naive
   one-hot encoding fails when the Generator produces soft assignments
   (0.3 male, 0.7 female).

3. **Class imbalance:** If only 5% of training samples have a rare condition, the
   Generator may never learn to produce records with that condition.

CTGAN solves these problems — see section 6.

---

## 4. Variational Autoencoders (VAEs)

A **VAE (Variational Autoencoder)** is a different type of generative model introduced
by Kingma and Welling in 2013.

```
Input data ──→ Encoder ──→ μ (mean), σ (std) of latent space
                              │
                              └──→ z = μ + σ × ε (sampling with reparameterisation)
                                         │
                              Decoder ←──┘
                                 │
                              Reconstructed / Generated data
```

**The Encoder:** Compresses input data into a lower-dimensional "latent space."
Instead of outputting a single point in latent space, it outputs a probability distribution
(mean μ and standard deviation σ) over possible encodings.

**The Latent Space:** A random sample is drawn from the distribution `N(μ, σ²)`.
This is the "code" for that data point. Nearby points in latent space correspond to
similar data.

**The Decoder:** Takes a point in latent space and reconstructs the original data.
To generate new data, sample a random point from the prior distribution `N(0, I)` in
latent space and pass it through the Decoder.

**Training objective:** Minimise reconstruction error (how well the Decoder recreates the
input) plus KL divergence (how much the learned latent distribution deviates from the
prior `N(0, I)`).

**TVAE** applies this approach to tabular data — see section 7.

---

## 5. Copula Models

A **copula** is a statistical function that captures the dependency structure between
random variables, independent of their marginal distributions.

**Intuition:** Two variables X and Y might each be non-normally distributed individually,
but their joint distribution has a specific correlation structure. A copula separates:
1. The marginal distribution of X (e.g., log-normal)
2. The marginal distribution of Y (e.g., skewed normal)
3. The dependency structure between X and Y (correlation)

**Gaussian Copula** assumes the dependency structure is Gaussian (even though the
individual marginals may not be). It:
1. Transforms each column to a uniform distribution [0,1] using the empirical CDF
2. Transforms uniform to Gaussian using the inverse normal CDF
3. Fits a multivariate Gaussian to capture correlations
4. Generates new data by sampling from the Gaussian, then reversing the transforms

See section 8 for why Gaussian Copula is used only as a fallback in this project.

---

## 6. CTGAN — Conditional Tabular GAN

**CTGAN (Conditional Tabular GAN)** was introduced by Xu et al. (2019) in the paper
"Modeling Tabular Data using Conditional GAN." It is the primary synthesis model for
most tables in this project.

CTGAN addresses the three problems with vanilla GANs for tabular data:

### Solution 1: Mode-Specific Normalisation for Skewed Data

Instead of normalising numerical columns globally (subtract mean, divide by std),
CTGAN uses **Variational Gaussian Mixture (VGM)** normalisation:

1. Fit a Gaussian mixture model to each numerical column
2. Identify the number of modes (peaks in the distribution)
3. Normalise values relative to the nearest mode's mean and standard deviation
4. Encode "which mode does this value belong to?" as an additional one-hot feature

For a column like `age` with modes at "young adults (25–35)" and "elderly (65–80)",
this encoding tells the generator "this is a mode-2 value" so it knows to produce
values in the elderly range rather than the adult range.

### Solution 2: Conditional Generator for Categorical Imbalance

CTGAN's **Conditional Generator (CTGAN)** addresses class imbalance:

1. At each training step, randomly select one categorical column
2. Randomly select a value from that column (with uniform probability across categories)
3. Train the Generator to produce data conditioned on this (column, value) pair
4. Use this "cond" vector as additional input to both Generator and Discriminator

This forces the Generator to learn ALL categories of ALL columns, even rare ones.
Without conditioning, a rare category like "stroke (disorder)" that appears in only 2%
of patients might never appear in training batches — the Generator never learns to produce it.

### Solution 3: Structured Architecture for Tabular Data

CTGAN uses Batch Normalisation in the Generator and Spectral Normalisation in the
Discriminator to stabilise training. The Discriminator uses multiple Fully Connected
layers with LeakyReLU activation.

### RDT: Reversible Data Transformer

Before training, CTGAN (via SDV's library) applies **RDT (Reversible Data Transformers)**
to convert all column types into numerical form:

| Column Type | RDT Transformation |
|---|---|
| Categorical | Label encoding → one-hot |
| Boolean | Map True→1.0, False→0.0 |
| Numerical | VGM normalisation (as described above) |
| UUID/ID | Skipped (not modelled) |

The `DataTransformer.transform()` method in RDT is where the Kaggle SIGKILL crash
occurred — see [09_Engineering_Log.md](09_Engineering_Log.md) for details.

### CTGAN Training Hyperparameters

As configured in `config/settings.yaml`:

| Parameter | Value | Justification |
|---|---|---|
| `epochs` | 300 | Standard for medical tabular data; enough epochs for convergence without overfitting |
| `batch_size` | 500 | Large enough for stable gradient estimates; small enough to fit in GPU VRAM |
| `generator_dim` | (256, 256) | Default SDV architecture |
| `discriminator_dim` | (256, 256) | Default SDV architecture |
| `generator_lr` | 2e-4 | Adam optimiser default |
| `discriminator_lr` | 2e-4 | Adam optimiser default |
| `discriminator_steps` | 1 | Number of Discriminator updates per Generator update |

---

## 7. TVAE — Tabular Variational Autoencoder

**TVAE** was introduced alongside CTGAN in the same Xu et al. (2019) paper as a
VAE-based alternative for tabular data.

TVAE uses the same RDT preprocessing as CTGAN (including VGM normalisation) but replaces
the adversarial training with a VAE:

- **Encoder:** 3-layer fully connected network → μ, σ of latent space
- **Decoder:** 3-layer fully connected network → reconstructed tabular features
- **Loss:** Reconstruction loss + KL divergence

### Why TVAE Was Rejected for This Project

**Critical reason: Same memory crash as CTGAN.**

TVAE uses the **same** `DataTransformer.transform()` method from RDT for preprocessing.
On Kaggle with the 303,696-row observations table:

1. `DataTransformer.transform()` uses `joblib.Parallel(n_jobs=-1)` to parallelise
   column transformations
2. On Kaggle's 4-core CPU, this spawns 4 worker processes
3. Each process copies the full 140 MB DataFrame into its own memory space
4. Total memory: 4 × 140 MB = 560 MB, instantly — before any computation begins
5. The Linux OOM killer sends SIGKILL to the entire Python process
6. Result: all training data lost, kernel dead

**This crash is identical for CTGAN and TVAE** because the crash happens in the shared
preprocessing step, not in the model itself.

**Utility reason: No meaningful improvement for this data type.**

CTGAN and TVAE produce similar quality results for mixed numerical/categorical tabular
data. The CTGAN paper showed TVAE slightly outperforming CTGAN on some benchmarks and
vice versa on others — neither is clearly superior. Given that TVAE does not fix the
memory crash and does not offer meaningful quality advantages for this specific dataset
(LOINC-coded observations with mixed numerical/categorical structure), TVAE was not selected.

---

## 8. GaussianCopulaSynthesizer

The Gaussian Copula fits a multivariate Gaussian to model correlations between columns,
after transforming each column's marginal distribution to Gaussian shape.

### Advantages

- **Extremely fast:** No neural network training. Fitting takes seconds, not hours.
- **Deterministic:** Same training data always produces the same model.
- **Memory efficient:** No 303K × batch_size tensor operations.
- **No SIGKILL risk:** Does not use the joblib-parallelised DataTransformer.

### Why GaussianCopula Was Not Selected as Primary Model

**It cannot model conditional distributions within a categorical column.**

Consider the observations table. The column `loinc_display` has 30 values. The column
`value_quantity` contains measurements. The critical relationship is:

```
value_quantity | loinc_display = "Body Height"     → Normal(170, 12)  [cm]
value_quantity | loinc_display = "Heart rate"      → Normal(75, 15)   [bpm]
value_quantity | loinc_display = "Hemoglobin A1c"  → Normal(5.5, 1.2) [%]
```

The Gaussian Copula models the **joint** distribution of `(loinc_display, value_quantity)`.
It transforms `loinc_display` to a numerical code and fits a multivariate Gaussian to
`(code, value_quantity)`. The problem is that the resulting Gaussian blends all the
per-LOINC distributions into one:

```
Synthesised value_quantity → blended distribution peaking around the global mean
→ Blood glucose values that look like heart rates
→ Heart rates that look like heights
→ Clinical nonsense
```

CTGAN's conditional training mechanism explicitly conditions on the value of `loinc_display`
when generating `value_quantity`, preserving the per-LOINC distributions correctly.

### GaussianCopula as Memory Fallback

GaussianCopula is implemented as a fallback when CTGAN raises a `MemoryError`:

```python
try:
    synth, model_type = CTGANTrainer(cfg).train(...), "ctgan"
except MemoryError:
    logger.warning("[OOM] %s: falling back to GaussianCopulaSynthesizer", table_name)
    mem.cleanup(f"{table_name}_oom_fallback")
    synth = GaussianCopulaSynthesizer(metadata).fit(df)
    model_type = "gaussian_copula"
```

Note: `MemoryError` (catchable Python exception) is different from `SIGKILL` (uncatchable
OS signal). The joblib fix ensures OOM manifests as `MemoryError` rather than SIGKILL.
See [09_Engineering_Log.md](09_Engineering_Log.md).

---

## 9. PAR — Probabilistic AutoRegressive Model

**PAR (Probabilistic AutoRegressive)** is SDV's model for sequential data. It was
designed specifically for the healthcare use case of generating patient event sequences.

### How PAR Works

PAR uses a recurrent neural network (specifically, an LSTM — Long Short-Term Memory)
to model the conditional probability of the next event given the history of previous events:

```
P(x_t | x_1, x_2, ..., x_{t-1}, patient_context)
```

Where:
- `x_t` is the t-th event in the patient's sequence (an encounter, condition, or medication)
- `x_1, ..., x_{t-1}` are all previous events in the patient's history
- `patient_context` are static patient features (age, gender, etc.) — optional

**Architecture:**

```
Patient Sequence: [enc_1, enc_2, enc_3, ..., enc_N]
                     │       │       │
                     ▼       ▼       ▼
                   LSTM ──→ LSTM ──→ LSTM ──→ ... Output Distribution
                     │       │       │
                  Hidden   Hidden   Hidden
                  State    State    State
```

The LSTM reads each event in the sequence and updates its hidden state (a compressed
representation of "what this patient looks like at this point in time"). After processing
the first N events, the LSTM predicts the distribution of the (N+1)-th event.

### PAR in SDV

SDV's PAR implementation requires:
- `sequence_key`: the column that groups rows by sequence (here: `patient_id`)
- `sequence_index`: the column that defines order within a sequence (here: `sequence_index`)
- `context_columns`: optional patient-level features (not used here — patient context
  lives in the separate patients table)

**Critical SDV 1.37.2 API note:** `sequence_key` and `sequence_index` must be set in
the `SingleTableMetadata` object *before* constructing the `PARSynthesizer`. They are
NOT constructor arguments. The `build_par_metadata()` function in `par_trainer.py` handles
this:

```python
st_dict = {
    **table_meta_dict,
    "METADATA_SPEC_VERSION": "V1",
    "sequence_key": "patient_id",
    "sequence_index": "sequence_index",
}
meta = SingleTableMetadata.load_from_dict(st_dict)
synth = PARSynthesizer(meta, context_columns=[], epochs=300, ...)
```

### Why PAR for Encounters, Conditions, and Medications

These three tables have a property that makes them fundamentally different from patients
and observations: **each table is a sequence of events ordered in time for each patient.**

The order and timing of events matter:
- A patient cannot have a condition diagnosed before they were born
- A medication cannot be prescribed before the condition that requires it
- The gap between consecutive encounters depends on how sick the patient has been recently

PAR models these temporal dependencies by learning from the sequence directly. CTGAN
would treat each row as independent — it could generate encounter #50 before encounter #1,
or a gap of -100 days between consecutive visits.

---

## 10. HMA — Hierarchical Modelling Algorithm

**HMA (Hierarchical Modelling Algorithm)** is SDV's approach for multi-table synthesis.
Instead of training separate models for each table and then linking them, HMA:

1. Trains a model on the parent table (patients)
2. Computes summary statistics of each child table per parent
3. Models the parent + child summaries jointly
4. At generation time, synthesises parents first, then generates child tables conditioned
   on each parent's profile

### Why HMA Was Not Selected

HMA was evaluated but rejected for several reasons:

**Reason 1: API instability in SDV 1.37.2.**
HMA underwent major refactoring in SDV 1.0–1.37.2. Several methods and parameter names
changed, and some documented features were missing in the installed version.

**Reason 2: Loss of PAR-specific sequential modelling.**
HMA uses a single model type for all child tables. Using HMA would mean giving up
PAR's sequential modelling for encounters, conditions, and medications — replacing it with
a non-sequential model that cannot capture visit patterns.

**Reason 3: Black-box FK handling.**
HMA generates FK values automatically during sampling. When something goes wrong with
FK integrity in the output, it is difficult to debug because the FK handling is internal.
Our approach (generate UUIDs, then remap FK columns in a post-processing step) is
explicit and debuggable.

**Reason 4: Memory issues at scale.**
HMA loads all tables into memory simultaneously. With 446,930 rows across five tables,
this creates memory pressure comparable to the observations crash.

The selected approach — training each table independently with the most appropriate model,
then remapping FK columns — provides more control, clearer debugging, and avoids the
API issues.

---

## 11. Model Selection Decisions

### Decision 1: CTGAN vs TVAE for patients, observations

| Criterion | CTGAN | TVAE |
|---|---|---|
| Memory usage (DataTransformer) | Same | Same (same code path) |
| Training stability | Generally stable | Occasionally unstable |
| Categorical handling | Explicit conditional training | Implicit via reconstruction |
| Utility (CTGAN benchmark) | Baseline | Similar, no clear winner |
| Failure mode | Mode collapse | Posterior collapse |
| Selected? | **YES** | No |

TVAE offers no meaningful advantage and shares the exact same memory risk (same DataTransformer
code path). CTGAN is the established standard for tabular synthesis and was selected.

### Decision 2: CTGAN vs PAR for encounters

| Criterion | CTGAN | PAR |
|---|---|---|
| Models row independence | Yes (wrong) | No |
| Models sequential dependencies | No | **Yes** |
| Preserves visit ordering | Not guaranteed | **Yes** |
| Models inter-visit gaps | No | **Yes** |
| Training time | Faster | Slower |
| Selected? | No | **YES** |

### Decision 3: CTGAN vs PAR for observations

| Criterion | CTGAN | PAR |
|---|---|---|
| Models LOINC-conditional distributions | **Yes** (conditional generator) | Yes (via sequence) |
| Handles 303,696 rows | **Yes** (batch training) | Not practical (requires 304-element sequences per patient) |
| Training time on GPU | ~2–3 hours | Would exceed Kaggle time limits |
| Selected? | **YES** | No |

### Decision 4: CTGAN vs GaussianCopula for observations

| Criterion | CTGAN | GaussianCopula |
|---|---|---|
| Models per-LOINC value distributions | **Yes** | No (blends distributions) |
| Memory usage | Moderate | Very low |
| Training time | Hours | Minutes |
| Output quality | **High** | Poor for this use case |
| Selected? | **YES** (primary) | Fallback only |

---

## 12. Per-Table Model Assignments

| Table | Model | Reason |
|---|---|---|
| patients | CTGAN | Non-sequential cross-sectional demographics; many categorical columns with class imbalance |
| encounters | PAR | Sequential events; inter-visit gap is the key feature to preserve |
| observations | CTGAN | 303K rows prevents PAR; CTGAN conditional generator handles 30-category LOINC split |
| conditions | PAR | Sequential diagnoses with onset/abatement temporal structure |
| medications | PAR | Sequential prescriptions tied to condition progression |

This is configured in `config/settings.yaml`:

```yaml
synthesis:
  table_models:
    patients:     "ctgan"
    encounters:   "par"
    observations: "ctgan"
    conditions:   "par"
    medications:  "par"
```

Changing any of these strings requires no code changes — the factory function
`create_single_table_synthesizer()` in `ctgan_trainer.py` dispatches on the string value.

---

## 13. Hyperparameter Decisions

### 300 Epochs

**What it means:** The training loop runs through the entire dataset 300 times.

**Why 300?**  
This is the standard recommendation from the CTGAN and SDV papers for medical tabular data.
It is a balance between:
- Underfitting (too few epochs → model has not learned the distribution)
- Overfitting (too many epochs → model memorises training data → privacy risk)
- Training time (300 epochs on the observations table takes ~2 hours on a T4 GPU)

In practice, we observed that loss plateaus around epoch 200–250, so 300 provides a
comfortable margin for convergence.

### Batch Size 500

**What it means:** At each training step, 500 rows are sampled from the training data
to compute the gradient.

**Why 500?**  
500 is large enough for stable gradient estimates (smaller batches produce noisy gradients
that slow convergence) while fitting in the GPU's VRAM (Video RAM). The Kaggle T4 GPU
has 16 GB VRAM; a batch of 500 rows with the model in memory uses approximately 1–2 GB.

### Seed 42

The random seed ensures that training runs are reproducible. Given the same data and seed,
CTGAN will produce the same model weights. This is important for debugging and for
comparing different versions of the pipeline.

---

## 14. GPU Support and Requirements

### Why GPU Matters

Both CTGAN and PAR perform matrix multiplications at each training step. For a batch of
500 rows transformed into a ~1,000-dimension feature vector:

- One training step requires: 500 × 1,000 matrix × 1,000 × 256 weight matrix → many
  floating point operations
- 300 epochs × (50,000 rows / 500 batch) = 30,000 training steps for observations
- On a CPU: each step takes ~0.5 seconds → 30,000 steps = 4.2 hours
- On a T4 GPU: each step takes ~0.05 seconds → 30,000 steps = 25 minutes

GPU reduces training time by approximately 10×.

### CUDA Detection

```python
import torch
use_cuda = torch.cuda.is_available()
device_name = torch.cuda.get_device_name(0) if use_cuda else "CPU"
```

The pipeline detects GPU availability at startup and passes `cuda=True` to all synthesisers
if a GPU is available. This works automatically on:

- **Kaggle:** T4 GPU available in GPU-accelerated notebooks
- **Google Colab:** T4 GPU available in GPU runtime
- **Local Windows:** Only if CUDA-compatible GPU and CUDA drivers are installed
- **Local CPU:** Falls back to CPU automatically

---

## 15. SDV Library Architecture

**SDV (Synthetic Data Vault)** is the open-source library from DataCebo that provides
all generative model implementations used in this project.

Version: `sdv>=1.37.0,<2.0.0`

**Why pin below 2.0.0?**  
SDV 2.0 introduced breaking API changes to the Metadata class. Our `metadata.json`
format and all model training code was validated against 1.37.2. Upgrading to 2.0 would
require significant refactoring of the metadata builder and trainer files.

### SDV Component Map

```
sdv.single_table
├─ CTGANSynthesizer      ← patients, observations (+ GaussianCopula fallback)
├─ TVAESynthesizer       ← not selected for primary use
└─ GaussianCopulaSynthesizer ← MemoryError fallback only

sdv.sequential
└─ PARSynthesizer        ← encounters, conditions, medications

sdv.metadata
├─ Metadata              ← multi-table metadata (used in checkpoint.py)
└─ SingleTableMetadata   ← single-table metadata (used in par_trainer.py directly)

rdt (Reversible Data Transformers)
└─ DataTransformer       ← shared preprocessing for CTGAN and TVAE
   └─ joblib.Parallel    ← source of Kaggle SIGKILL (see Engineering Log)
```

### How Our Code Wraps SDV

We do not call SDV classes directly from the training pipeline. Instead, two trainer
classes provide a stable API layer:

- `CTGANTrainer` in `ctgan_trainer.py`: wraps `CTGANSynthesizer`, `TVAESynthesizer`,
  and `GaussianCopulaSynthesizer` behind a `create_single_table_synthesizer()` factory
- `PARTrainer` in `par_trainer.py`: wraps `PARSynthesizer` and handles the
  SDV 1.37.2-specific metadata injection requirement

This isolation means that when we upgrade SDV or change models, we only need to modify
the trainer files — not the pipeline, checkpoint, or backup code.
