# 09 — Engineering Log

**Version:** 1.0  
**Last Updated:** 2026-06-28  
**Policy:** This log is append-only. Entries are never deleted. Newest entries at the bottom.

---

This is the chronological engineering diary of the SynthFHIR project. It records every
significant incident, design decision, discovery, and lesson. The purpose is to prevent
solving the same problem twice and to provide context for future engineers who inherit
this codebase.

---

## Entry 001 — Phase 2: FHIR Parsing API Mismatch

**Date:** (early phase 2)  
**Severity:** Medium — blocked development for ~1 session

### Problem

The `fhir.resources` Python library (version ≥7.0.0) changed its API between versions.
In the FHIR R4 model, `Patient.name` is a list of `HumanName` objects. The original code
accessed it as `patient.name[0].family` — which works in some versions but raises
`AttributeError` or returns `None` in others, depending on whether the Pydantic model
serialises the name field differently.

Similarly, `Observation.value_quantity` vs `Observation.valueQuantity` (camelCase vs
snake_case) was inconsistent across library versions.

### Root Cause

`fhir.resources` uses Pydantic for data validation. Pydantic v1 and v2 have different
attribute access patterns. The version installed on Kaggle differed from the version
on the development machine.

### Fix

Added explicit version pinning in `requirements.txt`:
```
fhir.resources>=7.0.0
```

Rewrote all parsers to use `.dict()` and dictionary key access instead of attribute
access, making the code robust to both Pydantic v1 and v2 serialisation:

```python
# Fragile:
patient.name[0].family

# Robust:
patient_dict = patient.dict()
family = patient_dict.get("name", [{}])[0].get("family", "")
```

### Lesson

When using a third-party library that depends on Pydantic, always test on the target
environment (Kaggle kernel) before assuming development environment behaviour is correct.
Add integration tests that run the parsers on 1–2 real Synthea files.

---

## Entry 002 — Phase 2: Blood Pressure Panel Parsing

**Date:** (Phase 2)  
**Severity:** Medium — silent data loss without this fix

### Problem

Blood pressure in FHIR is stored as a single Observation resource with a `component`
array, rather than as two separate Observation resources with individual values. The
initial parser only handled `valueQuantity` observations and silently dropped all
`component` observations — discarding all blood pressure data.

Without blood pressure, the observations table was missing one of the most clinically
significant vital signs.

### Root Cause

The initial parser assumed all observations would have a `valueQuantity`. Blood pressure
is a special case in FHIR that uses the `component` structure.

### Fix

Added `_extract_component_observations()` in `src/parsers/observation.py` to detect
`valueType == "component"` rows and store the raw `component_json` for later processing.

Then added `_expand_bp_components()` in `src/feature_engineering.py` (Phase 3) to
convert the stored `component_json` into individual systolic and diastolic rows.

### Lesson

Always validate that the output tables contain the expected observation types. A simple
check:
```python
assert "Systolic Blood Pressure" in observations["loinc_display"].values
```
should be part of the Phase 2 test suite.

---

## Entry 003 — Phase 3: SDV Metadata Boolean Encoding Issue

**Date:** (Phase 3)  
**Severity:** Low — caused assertion errors in smoke test, not in production

### Problem

Phase 3 preprocessing stores `is_deceased`, `is_chronic`, and `is_active` as integer
columns (0 or 1) in the CSV files. The SDV metadata generator marked these as `sdtype: boolean`.

When SDV's `CTGANSynthesizer` received these columns:
- Expected: Python `True` / `False` (boolean)
- Received: numpy `int64` values `0` and `1`

SDV 1.37.2 raises a validation error:
```
ValueError: Column 'is_deceased' has sdtype 'boolean' but contains non-boolean values
```

### Root Cause

SDV 1.37.2 tightened boolean validation. Earlier versions accepted 0/1 integers as
booleans. The SDV team decided to enforce strict typing.

### Fix

Added `_coerce_booleans()` in `ctgan_trainer.py`:

```python
def _coerce_booleans(df: pd.DataFrame, table_meta_dict: dict) -> pd.DataFrame:
    bool_cols = [
        col for col, info in table_meta_dict.get("columns", {}).items()
        if info.get("sdtype") == "boolean" and col in df.columns
    ]
    df = df.copy()
    for col in bool_cols:
        df[col] = df[col].map(lambda x: None if pd.isna(x) else bool(int(x)))
    return df
```

This converts 0 → `False`, 1 → `True`, NaN → `None` before passing to SDV.

### Why Store as Integer in CSV?

`True`/`False` in CSV files is ambiguous — different parsers treat the string "True"
differently. Integer 0/1 is universally understood and never causes encoding issues on
read. The conversion to Python boolean happens only at the last moment, just before
passing to SDV.

### Lesson

Always test with the exact SDV version that will be used in production. Add a validation
step in the smoke test that verifies SDV accepts the prepared training data.

---

## Entry 004 — Phase 4A: PAR API Change (sequence_key Not a Constructor Argument)

**Date:** (Phase 4A)  
**Severity:** High — blocked all PAR training

### Problem

The SDV documentation (for versions prior to 1.37.x) shows:

```python
PARSynthesizer(
    metadata,
    sequence_key="patient_id",     # WRONG in SDV 1.37.2
    sequence_index="sequence_index"  # WRONG in SDV 1.37.2
)
```

In SDV 1.37.2, `sequence_key` and `sequence_index` are NOT constructor arguments.
Passing them causes a `TypeError: unexpected keyword argument 'sequence_key'`.

### Root Cause

SDV 1.37.2 changed PARSynthesizer to read `sequence_key` and `sequence_index` from the
`SingleTableMetadata` object rather than accepting them as constructor arguments.

### Fix

The `build_par_metadata()` function in `par_trainer.py` injects these fields directly
into the metadata dictionary before loading:

```python
st_dict = {
    **table_meta_dict,
    "METADATA_SPEC_VERSION": "V1",
    "sequence_key":   "patient_id",
    "sequence_index": "sequence_index",
}
meta = SingleTableMetadata.load_from_dict(st_dict)
# Now: PARSynthesizer(meta)  ← no sequence_key in constructor
```

The PAR model reads `sequence_key` and `sequence_index` from `meta.sequence_key` and
`meta.sequence_index` (internal attributes of `SingleTableMetadata`).

**Note:** The newer `Metadata.get_table_metadata()` approach (returning a new-style
metadata object) also does NOT work — that class exposes `set_sequence_key()` but NOT
`.sequence_key` as a plain attribute, so PAR's internal read fails. Only the
`SingleTableMetadata.load_from_dict()` approach works in 1.37.2.

### Lesson

For SDV, always read the source code of the specific version you are using, not just
the documentation. The documentation lags the codebase and often covers multiple
incompatible versions simultaneously. Pinning `sdv>=1.37.0,<2.0.0` prevents accidental
upgrade to a breaking version.

---

## Entry 005 — Phase 4A: Kaggle SIGKILL — The Critical Incident

**Date:** (Phase 4A — first full training run on Kaggle)  
**Severity:** CRITICAL — total data loss; all completed work wiped

### Problem

The first full training run on Kaggle crashed with:

```
joblib.externals.loky.process_executor.TerminatedWorkerError:
A worker process managed by the executor was unexpectedly terminated.
The process quit with exit code -9 [SIGKILL]

During handling of the above exception, another exception occurred:
RuntimeError: DataTransformer.transform failed
```

This crash occurred inside `DataTransformer.transform()` during the preprocessing of the
`observations` table (303,696 rows). The crash happened at the very beginning of the
first training step — before a single epoch completed.

**After the crash, Kaggle wiped `/kaggle/working` completely.** All output files were lost:
- `patients_model.pkl` (trained in ~4 minutes)
- `encounters_model.pkl` (trained)
- All synthetic CSVs
- Training logs

Nothing was recoverable.

### Root Cause (Full Technical Analysis)

`DataTransformer.transform()` is the RDT preprocessing step that converts categorical
and numerical columns into numerical tensors for the neural network.

Inside `DataTransformer.transform()`, the implementation is:

```python
# From SDV/RDT source (approximately):
results = joblib.Parallel(n_jobs=-1)(
    joblib.delayed(self._transform_column)(col, data)
    for col, data in self._get_columns(X)
)
```

`n_jobs=-1` means "use all available CPU cores." On Kaggle's 4-core CPU:
- 4 worker processes are spawned
- Each worker process starts as a fresh Python interpreter
- The `data` (the full 303,696-row DataFrame, ~140 MB) is passed to each worker via
  pickle serialisation over a pipe
- Each worker must deserialise its copy: 4 × 140 MB = 560 MB of RAM allocated
  instantaneously
- This is on top of the parent process's 140 MB and all other Python/numpy/SDV overhead
- The Linux OOM (Out-Of-Memory) Killer detects available RAM below threshold
- OOM Killer sends `SIGKILL` to the Python process
- `SIGKILL` cannot be intercepted by Python — `try/except` cannot catch it
- The process dies instantly, mid-write

### Why This Is NOT Catchable by Python

`SIGKILL` (signal number 9) is sent by the kernel directly to the process. It is the
"kill with extreme prejudice" signal. Unlike `SIGTERM` (signal 15), which processes can
handle, `SIGKILL` cannot be:
- Caught (signal handlers cannot intercept it)
- Blocked (signal masks cannot mask it)
- Ignored

Python's `try/except` operates at the Python interpreter level. When the kernel sends
SIGKILL, the Python interpreter itself is killed — it never gets a chance to run the
`except` clause.

`MemoryError` (the Python exception for out-of-memory) IS catchable. But the OOM Killer
kills the process before Python ever sees a `MemoryError`.

### Fix

**Layer 1: Prevent the worker processes from being spawned**

```python
# MUST be before all imports
os.environ.setdefault("LOKY_MAX_CPU_COUNT", "1")
```

`LOKY_MAX_CPU_COUNT=1` forces joblib's loky backend to use at most 1 CPU core.
With 1 core, `Parallel(n_jobs=-1)` runs in the main process (no workers spawned,
no DataFrame copies created).

**Layer 2: Belt-and-suspenders joblib context**

```python
from joblib import parallel_backend

with parallel_backend("sequential"):
    synth.fit(df)
```

`parallel_backend("sequential")` is a context manager that overrides any backend
requests made inside `synth.fit()`. Any `joblib.Parallel` call inside the context
runs sequentially in the main process.

**Layer 3: Thread pool suppression**

```python
os.environ.setdefault("OMP_NUM_THREADS",       "1")
os.environ.setdefault("MKL_NUM_THREADS",       "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS",   "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS",  "1")
```

These suppress thread pools in OpenBLAS (BLAS matrix operations), MKL (Intel Math Kernel
Library), OpenMP (parallel loops in C extensions), and NumExpr. Without these, numpy
matrix operations in the DataTransformer may still spin up multiple threads.

**Result after fix:**

Memory usage during DataTransformer.transform() drops from:
```
Before: 140 MB × 5 (parent + 4 workers) = 700 MB instantaneous
After:  140 MB × 1 (parent only)         = 140 MB
```

If memory is still insufficient, Python raises `MemoryError` (catchable) instead of
the OS sending SIGKILL (not catchable). The pipeline catches MemoryError and falls
back to GaussianCopulaSynthesizer.

### Why the ENV Vars Must Be Set Before ALL Imports

joblib reads `LOKY_MAX_CPU_COUNT` when it is first imported. NumPy reads `OMP_NUM_THREADS`
when OpenBLAS is loaded (which happens when numpy is imported). If these variables are
set after the imports, they have no effect.

This is why `run_phase4b.py` begins with:
```python
from __future__ import annotations   # Python grammar requires this first

import os
os.environ.setdefault("LOKY_MAX_CPU_COUNT", "1")
...
# Then and only then:
import pandas as pd
import torch
from sdv import ...
```

### The Phase 4B Redesign (Consequence of This Incident)

This crash motivated a complete redesign of the training pipeline. The 18 requirements
of Phase 4B are all direct responses to the failure modes exposed by this incident:

1. Immediate backup after every table (prevent total loss)
2. Checkpointing with atomic manifest writes (prevent manifest corruption)
3. Per-table ZIPs (backup that survives session wipe)
4. Multi-destination copy (/kaggle/output, /kaggle/working, Google Drive)
5. Resume from next pending table (not from scratch)
6. manifest.json with SHA-256 hashes (detect corrupted files)
7. joblib sequential backend (prevent SIGKILL)
8. Safe memory cleanup after each table (prevent accumulation)
9. os.sync() before ZIP (prevent partial writes)
10. Per-table metrics (visibility into training progress)
...

### Lesson

**Design for failure from the beginning.** A training pipeline that runs for 3+ hours
on cloud infrastructure WILL be interrupted. The question is not "if" but "when." Every
output file must be immediately backed up upon completion, and the pipeline must be able
to resume from exactly where it left off.

Do not test pipeline stability only on fast smoke tests (5 epochs, 20 rows). The crash
in this incident did not occur with small data — it occurred at production scale.

---

## Entry 006 — Phase 4B: from __future__ SyntaxError

**Date:** (Phase 4B redesign implementation)  
**Severity:** Low — caught immediately during testing

### Problem

The first implementation of the Phase 4B `run_phase4b.py` placed the `os.environ.setdefault`
calls before `from __future__ import annotations`:

```python
# WRONG — causes SyntaxError
import os
os.environ.setdefault("LOKY_MAX_CPU_COUNT", "1")

from __future__ import annotations  # SyntaxError: future import must be first
```

### Root Cause

Python grammar requires `from __future__ import X` to be the very first statement in a
module after the module docstring. Any executable statement before it (including `import os`
and `os.environ.setdefault()`) causes `SyntaxError: from __future__ imports must occur at
the beginning of the file`.

### Fix

Correct ordering:

```python
from __future__ import annotations   # First: Python grammar requirement

import os                             # Second: standard library import
os.environ.setdefault("LOKY_MAX_CPU_COUNT", "1")  # Before all other imports

# Then everything else
import sys
import logging
import pandas as pd
...
```

### Lesson

`from __future__ import annotations` is a compile-time directive, not a runtime statement.
Python checks for it during parsing (before execution). The environment variable setup
must come immediately after it — but `os` and `os.environ.setdefault()` can legitimately
come after `from __future__`. The constraint is only that `from __future__` precedes
all other import statements.

---

## Entry 007 — Phase 4B: Windows cp1252 UnicodeDecodeError in CSV Inspection

**Date:** 2026-06-27  
**Severity:** Low — only affected debugging scripts, not production code

### Problem

A debugging script used `open(csv_path)` without specifying `encoding`. On Windows,
the default encoding is `cp1252` (Windows code page 1252). The `observations_ready.csv`
file contained byte 0x90 (a Windows-specific extended character), which is not valid
in cp1252.

```
UnicodeDecodeError: 'charmap' codec can't decode byte 0x90 in position 4214
```

### Root Cause

Files written by pandas on Windows with `df.to_csv(path, index=False)` use the system's
default encoding if `encoding` is not specified — also cp1252 on Windows. But the file
contained content (from Synthea JSON, which is UTF-8) that was not valid cp1252.

### Fix

Always specify `encoding='utf-8'` when opening CSV files:

```python
pd.read_csv(path, encoding='utf-8', encoding_errors='replace')
open(path, encoding='utf-8')
```

In the production code (`preprocessor.py`, `pipeline.py`), pandas was already using
UTF-8 implicitly on Linux/Kaggle. The issue only appeared in Windows debugging scripts.

### Lesson

**Always specify encoding explicitly.** The default encoding is platform-dependent.
A script that works on Linux may fail on Windows (cp1252) and vice versa. Adding
`encoding='utf-8'` to every file open and `pd.read_csv()` is zero cost and prevents
a class of cryptic errors. The production pipeline code uses UTF-8 throughout;
debugging scripts should follow the same standard.

---

## Entry 008 — Phase 3: TVAE Analysis — Why TVAE Was Rejected

**Date:** (Phase 4B design)  
**Severity:** N/A — design decision documented for completeness

### Context

During the Phase 4B design, there was a question of whether TVAE (Tabular Variational
Autoencoder) might be better suited for the observations table than CTGAN. TVAE was
investigated as a potential replacement.

### Analysis

**Hypothesis 1: TVAE might not trigger the SIGKILL crash.**  
WRONG. TVAE uses the EXACT same `DataTransformer.transform()` code path as CTGAN (both
are part of the SDV single-table synthesiser family and share the RDT preprocessing
layer). A TVAE training run on 303,696 rows would crash identically to CTGAN before
the fix.

**Hypothesis 2: TVAE produces higher-quality output for numerical LOINC data.**  
UNPROVEN. The Xu et al. 2019 benchmark shows TVAE slightly outperforming CTGAN on
some numerical datasets and CTGAN outperforming TVAE on others. For mixed datasets
with both numerical and categorical columns (which describes observations: `value_quantity`
is numerical, `loinc_display` and `value_unit` are categorical), neither model clearly
dominates.

**Hypothesis 3: TVAE avoids mode collapse that CTGAN might suffer.**  
PARTIALLY TRUE. TVAE's KL divergence regularisation prevents degenerate outputs somewhat
more reliably than adversarial training. But CTGAN's conditional training mechanism
specifically addresses the mode collapse issue for categorical imbalance (which is the
main risk for observations with rare LOINC codes).

### Decision

TVAE was rejected. No advantage over CTGAN was identified that would justify the
additional complexity of supporting a second model type. CTGAN was already selected,
tested, and the fallback path (GaussianCopula on MemoryError) was already designed
for CTGAN's failure mode. Adding TVAE would add configuration complexity with no clear
benefit.

### Lesson

When evaluating alternative models, always start by asking "does this alternative fix
the actual root cause of the problem I'm trying to solve?" In this case, the root cause
was the joblib fork bomb — and TVAE shares the exact same code path. Alternative model
selection is irrelevant until the root cause is fixed.

---

## Entry 009 — Phase 4B: GaussianCopula Cannot Model Conditional LOINC Distributions

**Date:** (Phase 4B design)  
**Severity:** N/A — design decision documented

### Context

GaussianCopula was proposed as an alternative to CTGAN for the observations table,
since it is much less memory-intensive.

### Analysis

The observations table has a fundamental conditional structure:

```
P(value_quantity | loinc_display)
```

For Body Height: values cluster around 160–185 cm (adults)
For Heart Rate: values cluster around 60–100 bpm
For Hemoglobin A1c: values cluster around 5.0–8.0 %

These are completely different distributions conditioned on `loinc_display`.

The Gaussian Copula models the joint distribution `P(loinc_display, value_quantity)` by:
1. Transforming `loinc_display` to a numerical code (e.g., ordinal encoding: "Body Height"→0, "Heart rate"→1, ...)
2. Transforming `value_quantity` to Gaussian via marginal CDF
3. Fitting a 2D Gaussian to the (code, normalized_value) pairs

The problem: the 2D Gaussian fits ONE joint distribution. It cannot represent that code=0
(Body Height) is associated with values in [100, 250] while code=1 (Heart Rate) is associated
with values in [40, 200]. The joint Gaussian blends these distributions, creating synthetic
records where:
- Body Height = 72 bpm (clinically nonsensical)
- Heart Rate = 172 cm (clinically nonsensical)

CTGAN's conditional generator explicitly conditions on `loinc_display` when generating
`value_quantity`, preserving the per-LOINC distributions correctly.

### Decision

GaussianCopula rejected as primary model for observations. Retained as `MemoryError`
fallback — if CTGAN fails due to memory pressure on a smaller dataset, GaussianCopula
is better than nothing, but its output quality for observations is poor.

### Lesson

Model selection must consider the structural properties of the data, not just computational
efficiency. A fast model that produces clinically invalid output is worse than a slow model
that produces clinically valid output.

---

## Entry 010 — Phase 4B: Manifest Atomic Write Design

**Date:** (Phase 4B implementation)  
**Severity:** N/A — proactive design decision

### Problem Anticipated

If `manifest.json` is written using a standard `file.write()` call and the process is
killed (SIGKILL, power loss) during the write, the resulting file can be:
- Truncated (partial JSON)
- Empty (OS wrote zero bytes before kill)
- Garbled (OS write cache partially flushed)

A corrupted `manifest.json` would cause the next run to crash on load, with no way to
recover which tables completed. This would negate the entire fault-tolerance design.

### Solution

Atomic write pattern:

```python
def save(self) -> None:
    tmp = self._path.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(self._data, indent=2, default=str), encoding="utf-8")
    try:
        tmp.replace(self._path)    # POSIX: atomic rename
    except Exception:
        shutil.copy2(tmp, self._path)  # Windows fallback
        tmp.unlink(missing_ok=True)
```

On Linux/macOS, `Path.replace()` calls `os.rename()`, which on the same filesystem is
guaranteed by POSIX to be atomic. The file appears at the destination instantaneously
from the perspective of any other process — there is no window where a partially-written
file exists at the target path.

On Windows, `os.rename()` with different drives is not atomic, but same-drive operations
are effectively atomic in modern NTFS. The `try/except` with `shutil.copy2` handles
cross-drive scenarios.

### Lesson

Persist critical state with atomic writes. The cost is trivial (one extra file, one
rename call). The benefit is correctness guarantees that no partial state can corrupt
the manifest.

---

## Entry 011 — Phase 4B: ZIP Records Real Patient Data Exclusion

**Date:** (Phase 4B implementation)  
**Severity:** N/A — privacy design decision

### Decision

The per-table ZIP checkpoints intentionally exclude `data/ready/*.csv` (the real patient
training data). This was deliberate.

### Reasoning

The checkpoints are designed to be:
1. Downloaded from Kaggle Output
2. Re-uploaded to a new Kaggle session
3. Shared with collaborators (for multi-machine training)

If the real patient training data (even Synthea synthetic data) were in the ZIP, anyone
who obtained the ZIP would have the training data. While Synthea data is not truly
private, establishing this exclusion now ensures that when the pipeline is used with
real EHR data (the intended future use case), real patient records are never included
in ZIPs.

The `metadata.json` (which describes the schema but contains no records) IS included
because it is needed to reload models in a fresh session without re-running Phase 3.

### Lesson

Design privacy properties for the intended future use case, not just the current use case.
A habit of excluding patient data from checkpoints now prevents accidental exposure when
the pipeline is used with real data later.
