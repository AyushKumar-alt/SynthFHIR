# 03 — Preprocessing

**Version:** 1.0  
**Last Updated:** 2026-06-28  
**Phase Coverage:** Phase 3  
**Source Files:** [src/feature_engineering.py](../src/feature_engineering.py), [src/preprocessor.py](../src/preprocessor.py)

---

## Table of Contents

1. [Overview: What Preprocessing Does and Why It Exists](#1-overview-what-preprocessing-does-and-why-it-exists)
2. [Pipeline Stages](#2-pipeline-stages)
3. [Temporal Strategy: Days Since Birth](#3-temporal-strategy-days-since-birth)
4. [Patients Table Preprocessing](#4-patients-table-preprocessing)
5. [Encounters Table Preprocessing](#5-encounters-table-preprocessing)
6. [Observations Table Preprocessing](#6-observations-table-preprocessing)
7. [Conditions Table Preprocessing](#7-conditions-table-preprocessing)
8. [Medications Table Preprocessing](#8-medications-table-preprocessing)
9. [Cross-Table: Patient Aggregate Features](#9-cross-table-patient-aggregate-features)
10. [Foreign Key Validation](#10-foreign-key-validation)
11. [SDV Metadata Generation](#11-sdv-metadata-generation)
12. [Summary of All Preprocessing Decisions](#12-summary-of-all-preprocessing-decisions)
13. [What Would Change for Real EHR Data](#13-what-would-change-for-real-ehr-data)

---

## 1. Overview: What Preprocessing Does and Why It Exists

Machine learning models cannot learn from raw FHIR data. The raw data has several
properties that make it unsuitable for direct use:

**Problem 1: Timestamps are absolute calendar dates.**  
A date like "2018-04-22" means nothing to a model unless the model knows when the
patient was born. "Day 13,960 of the patient's life" is meaningful because it tells
us the patient was 38 years old. Absolute dates also expose calendar-year information
which can be used to re-identify patients (the only person born on 1953-07-16 in a given
zip code).

**Problem 2: PII columns exist in the raw data.**  
Names, Social Security Numbers, driver's licenses, and GPS coordinates are in the raw
FHIR output. These must be removed before any model sees the data.

**Problem 3: Some columns carry no information.**  
`country = "US"` for every single patient. This column has zero variance — no model
could learn anything from it. Including it wastes model capacity.

**Problem 4: High-cardinality categorical columns exist.**  
`organization_display` might have 200 unique hospital names in a real dataset. A
generative model trying to reproduce 200 rare categories is wasting capacity on
things that appear once or twice. Bucketing to "top 50 + other" focuses capacity
on common values.

**Problem 5: Outliers distort model training.**  
A single erroneous measurement — a blood glucose reading of 5,000 mg/dL instead of
the plausible 500 mg/dL — forces the model to allocate capacity to a region of the
distribution that should never appear in real data.

**Problem 6: Sequential structure must be explicit.**  
PAR needs to know which column defines "patient sequence" and which column defines
"position in sequence." These must be added before training.

**Problem 7: Rare LOINC codes are unsynthesisable.**  
An observation type that appears 3 times in 1,000 patients cannot have its distribution
reliably learned by any model.

Preprocessing solves all seven problems. Every decision below is justified against one
or more of these motivations.

---

## 2. Pipeline Stages

The preprocessing pipeline (`src/preprocessor.py`) has six steps:

```
Step 1: Load Phase 2 CSVs (data/processed/*.csv)
   │
   ▼
Step 2: Feature engineering — per table transformations
   ├─ clean_patients()
   ├─ engineer_encounters()
   ├─ engineer_observations()
   ├─ engineer_conditions()
   └─ engineer_medications()
   │
   ▼
Step 3: Patient aggregate enrichment
   └─ add_patient_aggregates()  — joins encounter/condition/medication counts to patients
   │
   ▼
Step 4: Foreign key validation
   └─ Verify all encounter.patient_id values exist in patients.patient_id, etc.
   │
   ▼
Step 5: Write ready CSVs (data/ready/*_ready.csv)
   │
   ▼
Step 6: Generate SDV metadata (data/ready/metadata.json)
```

Total run time on this dataset: approximately 45–90 seconds on a standard laptop.

---

## 3. Temporal Strategy: Days Since Birth

Every date-type column across all five tables is converted to a floating-point number
representing the number of days elapsed since the patient's date of birth.

```
days_since_birth = (event_datetime - patient_birth_datetime) / 1 day

Example:
  Patient born:      1975-06-20
  Encounter date:    2023-03-15
  Difference:        17,435 days
  days_since_birth = 17,435.0
```

**Why days since birth, not days since some fixed date (e.g., 2000-01-01)?**

A fixed reference date would preserve absolute calendar years. A patient whose first
encounter is at `days_since_reference = 14,610` (exactly 40 years after 2000-01-01) was
born in 1960 — that information is sensitive. Using birth date as the reference makes the
number relative to the patient's own life, not to the calendar.

**Why days, not years or months?**

Days provide the most granular temporal information without the complexity of a datetime
object. Years would lose information about seasonality in disease patterns. Months are
arbitrary. Days is a natural unit for medical time (length of hospital stay, time between
visits).

**Why floating point?**

A patient born at 10:00 AM who has an encounter at midnight will have
`days_since_birth = 0.58` (14 hours / 24). This sub-day precision matters for newborn
care records where events within hours of birth are clinically significant.

**Why clip to lower=0?**

A FHIR parsing error or a data entry error could produce a negative delta (event before
birth). Clipping to ≥0 prevents nonsensical negative temporal values from entering the
model.

**Timezone handling:**

Synthea emits timestamps with mixed UTC offsets (e.g., `+01:00` for winter records,
`+02:00` for summer records due to daylight saving). Converting all timestamps to UTC
before computing the delta prevents a 1-hour spurious "jump" in the timeline at each
daylight saving transition. The implementation uses:

```python
pd.to_datetime(series, utc=True, errors="coerce").dt.tz_localize(None)
```

`utc=True` normalises everything to UTC. `.dt.tz_localize(None)` removes the timezone
object so subsequent arithmetic works without pandas raising timezone warnings.

---

## 4. Patients Table Preprocessing

**Source function:** `clean_patients()` in `src/feature_engineering.py`  
**Input:** `data/processed/patients.csv` (1,000 rows × ~25 columns)  
**Output:** `data/ready/patients_ready.csv` (998 rows × 18 columns)

### Step 4.1: PII Column Removal

```python
drop_cols = [
    "family_name", "given_name",          # Real names
    "ssn", "drivers_license", "passport",  # Government identifiers
    "lat", "lon",                           # Precise GPS location
    "postal_code",                          # High-granularity geography
    "country",                              # 100% "US" — zero variance
    "state",                               # 100% "Massachusetts" — zero variance
    "birth_place_city",                    # Very sparse
    "mothers_maiden_name",                 # PII
]
```

**Why names are removed:**  
First and last names directly identify a real person. Even in synthetic data, training
on names would cause the model to generate plausible-sounding names — and synthetic data
with realistic names is more likely to be confused with real data by downstream users.

**Why SSN, DL, passport are removed:**  
Government identifiers have no clinical value. They exist in FHIR for administrative
(billing, insurance) purposes. The synthesis target is clinical data, not administrative
records.

**Why lat/lon are removed but city is retained:**  
GPS coordinates are precise enough to identify a specific building. A city name is
a population-level aggregate. "Arlington, MA" contains ~46,000 people; latitude
42.4154, longitude -71.1565 identifies a specific city block. City is retained because
it carries information about the local disease environment (rural vs urban, access to
specialised care) that is clinically meaningful for synthesis.

**Why country and state are removed:**  
All 1,000 Synthea patients are generated in Massachusetts, USA. Every row has
`country = "US"` and `state = "Massachusetts"`. A column with a single unique value
(zero variance) conveys zero information to a model. Including it wastes one input
dimension for no benefit.

**Why postal_code is removed:**  
In a 1,000-patient dataset, many postal codes appear only once or twice. A generative
model would have to learn the distribution of ~20 different zip codes from ~50 samples
each — insufficient to model reliably. More importantly, specific postal codes combined
with demographics (age + gender + race) create high re-identification risk even in
synthetic data.

### Step 4.2: Age Computation from Birth Date

```python
today = pd.Timestamp.today().normalize()
birth_dt = pd.to_datetime(df["birth_date"], errors="coerce")
df["age"] = ((today - birth_dt).dt.days / 365.25).round(1)
```

**Why convert birth_date → age?**  
Exact dates of birth are PII. "Born 1975-06-20" combined with city and gender is
potentially identifying. "Age 48.7" is less identifying — many thousands of people are
48 years old.

**Why 365.25 and not 365?**  
365.25 accounts for leap years. A person's age calculated over 80 years differs by 20 days
between using 365 and 365.25. Rounding to 1 decimal place further reduces precision while
retaining meaningful clinical information (a model can distinguish 48.7 from 49.3).

### Step 4.3: Mortality Encoding

```python
df["is_deceased"] = deceased_dt.notna().astype(int)
df["age_at_death"] = np.where(
    deceased_dt.notna(),
    ((deceased_dt - birth_dt).dt.days / 365.25).round(1),
    np.nan,
)
```

**Why is_deceased as 0/1 integer, not boolean?**  
SDV handles integer 0/1 columns differently from Python booleans. Integers are stored
as numerical; booleans require the SDV boolean encoder. We store as 0/1 integer for
CSV compatibility and convert to Python bool only when passing to SDV (in `ctgan_trainer.py`).

**Why age_at_death (not datetime of death)?**  
Same reasoning as age: the exact death date is PII. Age at death is a continuous variable
that the model can learn from — older patients are more likely to die than younger patients,
and the model can learn the age-at-death distribution from the training data.

### Step 4.4: Categorical Value Standardisation

```python
cat_cols = ["gender", "race", "ethnicity", "marital_status", "language", ...]
for col in cat_cols:
    df[col] = df[col].astype(str).str.strip().str.lower().replace("nan", pd.NA)
```

**Why lowercase?**  
Synthea sometimes produces mixed-case strings: "Female", "FEMALE", "female" all appear.
A generative model treating these as three distinct categories would learn an incorrect
distribution (33% each instead of ~50% female). Lowercasing collapses them into one.

**Why `.replace("nan", pd.NA)`?**  
When a pandas DataFrame is read from CSV, null values become the string `"nan"` (four
characters) if not handled carefully. Without this replacement, `"nan"` becomes a valid
categorical value that the model treats as a category. We replace it with `pd.NA` (the
proper missing value marker) so SDV handles it correctly.

### Step 4.5: Deduplication

```python
df = df.drop_duplicates(subset=["patient_id"], keep="first")
```

Two patients (of the 1,000 generated) had duplicate `patient_id` values in the raw
Synthea output — a known edge case in batch Synthea runs. Deduplication removes them,
leaving 998 unique patients.

---

## 5. Encounters Table Preprocessing

**Source function:** `engineer_encounters()` in `src/feature_engineering.py`  
**Input:** `data/processed/encounters.csv`  
**Output:** `data/ready/encounters_ready.csv` (57,667 rows × 12 columns)

### Step 5.1: Temporal Feature Engineering

```python
df["days_since_birth"] = _days_delta(df["start_datetime"], df["patient_id"], birth_map)
df["encounter_duration_hours"] = (
    (end - start).dt.total_seconds() / 3600
).clip(lower=0).round(2)
```

`days_since_birth` anchors the encounter to the patient's life timeline. Without this
column, a model seeing two encounters at "2010-06-15" and "2018-04-22" cannot know that
the first patient was 35 years old and the second was 55 years old at the time of their
respective visits.

`encounter_duration_hours` captures visit length. A 15-minute outpatient visit
(0.25 hours) is very different clinically from a 3-day inpatient admission (72 hours).
Clipping to `lower=0` handles the rare case where the end datetime precedes the start
(a data entry error).

### Step 5.2: Sequence Index and Inter-Visit Gap

```python
df = df.sort_values(["patient_id", "days_since_birth"])
df["sequence_index"] = df.groupby("patient_id").cumcount()
df["days_since_prev_encounter"] = df.groupby("patient_id")["days_since_birth"].diff()
```

**`sequence_index`:** The PAR model requires a column that defines the position of each
event within the patient's history. `sequence_index = 0` is the first ever encounter,
`sequence_index = 1` is the second, etc. Without this, PAR cannot order the generated
sequence correctly.

**`days_since_prev_encounter`:** This feature directly encodes the inter-visit gap — the
time between consecutive visits. It is NaN for the first encounter (no previous encounter
to compute a gap from). This feature allows PAR to learn that a patient with heart failure
has gaps of 14–30 days while a healthy patient has gaps of 365 days.

### Step 5.3: High-Cardinality Capping

```python
for col in ("organization_display", "location_display"):
    top50 = df[col].value_counts().nlargest(50).index
    df[col] = df[col].where(df[col].isin(top50), other="other")
```

**Why cap at 50?**  
A Synthea run with 1,000 patients generates encounters at many different healthcare
organisations (hospitals, clinics, specialist offices). Many of these appear only 1–3
times. A categorical generative model allocated to represent 200 organisations from
~2 training samples each will produce garbage. Keeping the top 50 (by frequency) captures
the organisations where most visits occur, while rare ones are collapsed into the catchall
category `"other"`.

**Why not drop these columns entirely?**  
The type of organisation matters clinically. Encounters at emergency departments have
different distributions of duration, diagnoses, and observations than ambulatory clinic
visits. The column carries real information — just too many unique values to synthesise
reliably.

### Step 5.4: Constant Column Removal

```python
df = df.drop(columns=["type_code", "practitioner_npi", "reason_code", "status"])
```

| Removed Column | Reason |
|---|---|
| `type_code` | Redundant with `type_display` (keeps text, drops the numeric code) |
| `practitioner_npi` | Provider identifier — high cardinality, PII adjacent |
| `reason_code` | Redundant with `reason_display` |
| `status` | Always "finished" in Synthea — zero variance |

---

## 6. Observations Table Preprocessing

**Source function:** `engineer_observations()` in `src/feature_engineering.py`  
**Input:** `data/processed/observations.csv`  
**Output:** `data/ready/observations_ready.csv` (303,696 rows × 9 columns)

### Step 6.1: Value Type Filtering

FHIR Observation resources can contain values in different forms:

| FHIR value type | Example | Retained? |
|---|---|---|
| `valueQuantity` | Blood glucose: 5.2 mmol/L | Yes (`value_type = "quantity"`) |
| `valueCodeableConcept` | Smoking status: "Never smoked" | No |
| `valueString` | Free text note | No |
| `component` | Blood pressure panel | Yes (expanded separately) |

Only `valueQuantity` (numerical) observations are retained. Categorical observations
(`valueCodeableConcept`) and free-text strings are excluded because:

1. There are dozens of unique string values per LOINC code — too high cardinality
2. Their distributions are not well-represented in a 1,000-patient dataset
3. The clinical value of synthesising "smoking status" categories is lower than
   synthesising blood pressure values

**Exception: Blood pressure components.** Blood pressure uses the `component` structure
rather than `valueQuantity`. A special expansion step handles this (see step 6.2).

### Step 6.2: Blood Pressure Component Expansion

Blood pressure is stored as:

```json
"component": [
  {"code": "8480-6", "display": "Systolic BP", "valueQuantity": {"value": 128}},
  {"code": "8462-4", "display": "Diastolic BP", "valueQuantity": {"value": 82}}
]
```

The expansion function (`_expand_bp_components()`) converts this into two rows:

```
Row 1: loinc_display="Systolic Blood Pressure",  value_quantity=128, value_unit="mm[Hg]"
Row 2: loinc_display="Diastolic Blood Pressure", value_quantity=82,  value_unit="mm[Hg]"
```

IDs are suffixed: `obs-uuid_c0` (systolic) and `obs-uuid_c1` (diastolic) to ensure
uniqueness.

**Why expand rather than keep as a single row?**  
The synthesiser handles one numerical value per row. A single blood pressure row would
need to encode two correlated values (systolic and diastolic) in a non-standard structure
that CTGAN cannot process. After expansion, each value is a standard numerical column
in its own right, and CTGAN can learn the joint distribution of systolic and diastolic
values across all patients.

### Step 6.3: LOINC Frequency Filter

```python
loinc_counts = df_all["loinc_display"].value_counts()
top_loinc = loinc_counts[loinc_counts >= min_frequency].head(top_n_loinc).index
df_all = df_all[df_all["loinc_display"].isin(top_loinc)]
```

**Parameters (from `config/settings.yaml`):**
- `min_loinc_frequency = 100` — a LOINC type must appear at least 100 times
- `top_n_loinc = 30` — keep the top 30 by frequency

**Why 100 as the minimum?**  
With 1,000 patients, 100 occurrences means the measurement was taken in approximately
10% of patients (adjusted for frequency of measurement). 100 samples is a practical
minimum for a generative model to learn a distribution shape.

**Why 30 as the maximum?**  
30 LOINC types captures the clinically most important routine measurements (vital signs,
core lab panels) without including rare tests. 30 types × ~10,000 observations each =
~300,000 rows — computationally manageable for CTGAN training.

### Step 6.4: Outlier Capping (3×IQR)

```python
def _cap_outliers(df, value_col, group_col, factor=3.0):
    for name, grp in df.groupby(group_col):
        q1 = grp[value_col].quantile(0.25)
        q3 = grp[value_col].quantile(0.75)
        iqr = q3 - q1
        lo = q1 - factor * iqr
        hi = q3 + factor * iqr
        mask = df[group_col] == name
        df.loc[mask, value_col] = df.loc[mask, value_col].clip(lo, hi)
```

**Why per-LOINC outlier capping?**  
Outliers must be evaluated in context. A `value_quantity` of 180 is:
- Extremely high for `Hemoglobin A1c` (normal range: 4–6%)
- Perfectly normal for `Systolic Blood Pressure` (normal range: 90–140 mm[Hg])
- Low for `Body Height` (normal adult range: 150–200 cm)

The IQR (Interquartile Range) method computes outlier bounds separately for each LOINC
type, making the capping clinically meaningful.

**Why 3×IQR rather than 2×IQR or z-score?**  
`3×IQR` is a conservative threshold. For a normally distributed variable, `3×IQR` caps
approximately the top and bottom 0.35% of values. Most erroneous measurements will be
far outside this range. Using `2×IQR` would also cap legitimate extreme values (e.g., a
genuinely hypertensive patient with systolic of 220 mm[Hg]).

**What "capping" means vs "removal":**  
Instead of deleting rows with outlying values, we replace extreme values with the boundary
value. A systolic of 380 mm[Hg] becomes the upper bound of the normal range. This:
1. Preserves the row (no information loss for other columns)
2. Prevents the model from learning that extreme values are common
3. Maintains a plausible clinical value in the output

---

## 7. Conditions Table Preprocessing

**Source function:** `engineer_conditions()` in `src/feature_engineering.py`  
**Input:** `data/processed/conditions.csv`  
**Output:** `data/ready/conditions_ready.csv` (37,835 rows × 10 columns)

### Step 7.1: Temporal Encoding

```python
df["onset_days_since_birth"]      = _days_delta(df["onset_datetime"], ...)
df["abatement_days_since_birth"]  = _days_delta(df["abatement_datetime"], ...)
df["duration_days"]               = (abatement - onset).clip(lower=0)
df["is_chronic"]                  = df["abatement_datetime"].isna().astype(int)
```

**Why separate onset and abatement columns?**  
These are clinically distinct events. A condition's onset tells you when it was diagnosed.
Its abatement tells you when it resolved. Many conditions (hypertension, type 2 diabetes)
never resolve — they are chronic. For these, `abatement_days_since_birth` is NaN and
`is_chronic = 1`.

**Why `duration_days`?**  
Duration is derived from onset and abatement, but it is a more direct feature for synthesis.
A model can more naturally learn "most resolved upper respiratory infections last 7–14 days"
from `duration_days` than by computing the difference itself.

### Step 7.2: Constant Column Removal

| Removed Column | Reason |
|---|---|
| `snomed_code` | Redundant with `snomed_display` |
| `verification_status` | Always "confirmed" in Synthea — zero variance |
| `category` | Always "encounter" in Synthea — zero variance |

---

## 8. Medications Table Preprocessing

**Source function:** `engineer_medications()` in `src/feature_engineering.py`  
**Input:** `data/processed/medications.csv`  
**Output:** `data/ready/medications_ready.csv` (46,734 rows × 10 columns)

### Temporal Encoding

```python
df["days_since_birth"] = _days_delta(df["authored_on"], df["patient_id"], birth_map)
```

`authored_on` is when the prescription was written — converted to the patient's life
timeline offset using the same approach as other tables.

### Clinical Flag

```python
df["is_active"] = (df["status"].str.lower() == "active").astype(int)
```

`is_active` is a derived boolean: 1 if the patient is currently taking this medication,
0 if the course is completed or the drug was stopped. This is more useful for synthesis
than the raw `status` string because it gives the model a binary signal for "current
medication" that correlates with conditions in the patient's condition table.

### Removed Columns

| Removed Column | Reason |
|---|---|
| `rxnorm_code` | Redundant with `rxnorm_display` |
| `requester_display` | Provider name — high cardinality, PII adjacent |
| `dosage_text` | Free text ("Take 1 tablet daily with food") — not synthesisable |
| `intent` | Always "order" in Synthea — zero variance |
| `authored_on` | Converted to `days_since_birth`; raw datetime no longer needed |

**Why is `dosage_text` not synthesisable?**  
Free text fields like "Take 2 tablets by mouth twice daily with meals" are natural language.
CTGAN is a tabular model — it operates on categorical or numerical values, not text strings.
To synthesise dosage instructions, you would need a language model (GPT-class). This is
outside the scope of this project.

---

## 9. Cross-Table: Patient Aggregate Features

**Source function:** `add_patient_aggregates()` in `src/feature_engineering.py`

After all five child tables are processed, three aggregate features are computed and added
to the patients table:

```python
enc_counts  = encounters.groupby("patient_id").size().rename("encounter_count")
cond_counts = conditions.groupby("patient_id").size().rename("condition_count")
med_counts  = medications.groupby("patient_id").size().rename("medication_count")

patients = patients.merge(enc_counts,  on="patient_id", how="left")
patients = patients.merge(cond_counts, on="patient_id", how="left")
patients = patients.merge(med_counts,  on="patient_id", how="left")
```

**Why are these added to the patients table?**  
When PAR generates synthetic encounter sequences, it can optionally condition on patient
context columns. A patient with `encounter_count = 500` should generate a longer, denser
sequence than a patient with `encounter_count = 10`. These aggregate features provide
the PAR model with a compressed summary of the patient's clinical complexity.

**Why left join?**  
A patient who has zero encounters (unlikely but possible if a newborn's record is
incomplete) would produce a NaN from `groupby().size()`. The left join preserves all
patients; NaN values are then filled with 0.

---

## 10. Foreign Key Validation

After all transformations, the pipeline validates referential integrity:

```python
valid_pids = set(patients["patient_id"])
valid_eids = set(encounters["encounter_id"])

# Check all encounters reference a real patient
orphans = (~encounters["patient_id"].isin(valid_pids)).sum()
# Check all observations reference a real encounter
orphans = (~observations["encounter_id"].isin(valid_eids)).sum()
# etc.
```

In this dataset, the validation passes with zero orphan records. The validation is
included because:

1. A bug in a future Phase 2 parser could produce orphan records
2. The deduplication steps could theoretically remove a parent while a child record
   still references it
3. Providing explicit validation is good engineering practice for data pipelines

Orphan records (if found) are logged as warnings, not errors. The pipeline continues
because downstream SDV synthesis can handle a small number of FK violations, and
preventing training from running entirely would be too disruptive.

---

## 11. SDV Metadata Generation

**Source file:** `src/metadata_generator.py`

After all ready CSVs are written, the pipeline generates a `metadata.json` file in
SDV's MultiTableMetadata format. This file tells SDV:

- The schema of each table (column names, data types)
- Which columns are primary keys
- Which columns are foreign keys
- Which tables are related (parent-child relationships)
- Which columns should be treated as sequential (for PAR)

```json
{
  "METADATA_SPEC_VERSION": "MULTI_TABLE_V1",
  "tables": {
    "patients": {
      "primary_key": "patient_id",
      "columns": {
        "patient_id":  {"sdtype": "id", "regex_format": "[0-9a-f-]{36}"},
        "gender":      {"sdtype": "categorical"},
        "age":         {"sdtype": "numerical"},
        "is_deceased": {"sdtype": "boolean"},
        ...
      }
    },
    "encounters": {
      "primary_key": "encounter_id",
      "sequence_key": "patient_id",
      "sequence_index": "sequence_index",
      "columns": {...}
    }
  },
  "relationships": [
    {
      "parent_table_name": "patients",
      "parent_primary_key": "patient_id",
      "child_table_name": "encounters",
      "child_foreign_key": "patient_id"
    }
  ]
}
```

**Why is this file necessary?**  
SDV cannot infer the schema of tabular data automatically with sufficient accuracy.
For example, a UUID column (`patient_id`) looks like a string to a naive parser — but
SDV needs to know it is an `id` (primary key) so it does not try to learn its distribution.
Similarly, SDV needs to know `is_deceased` is a boolean, not a numerical variable ranging
from 0.0 to 1.0.

**Why `sdtype: "id"` for UUID columns?**  
SDV's `id` type tells the synthesiser: "do not model this column's distribution; instead,
generate new unique IDs for synthetic records." Without this, SDV might try to learn that
`patient_id` values contain sequences of hexadecimal digits — computationally wasteful
and producing incorrect output (real UUIDs re-appearing in synthetic data).

---

## 12. Summary of All Preprocessing Decisions

| Technique | Applied To | Why |
|---|---|---|
| PII column removal | patients | Privacy — names/SSN/DL cannot be in training data |
| Zero-variance removal | all tables | Columns with one unique value carry no information |
| Temporal encoding (days_since_birth) | all child tables | Removes calendar dates while preserving temporal patterns |
| Date → age conversion | patients | Removes exact birthdate while preserving age information |
| Deceased → is_deceased + age_at_death | patients | Removes exact death datetime while preserving mortality signal |
| Categorical standardisation (lowercase) | all tables | Collapses mixed-case duplicates into single categories |
| High-cardinality capping → "other" | encounters | Prevents rare categories from consuming model capacity |
| LOINC frequency filter | observations | Removes observation types too rare to model reliably |
| Blood pressure expansion | observations | Converts BP panel into two standard numerical rows |
| 3×IQR outlier capping per LOINC | observations | Removes extreme values caused by data entry errors |
| Sequence index | all child tables | Required for PAR to order events within a patient |
| Inter-visit gap | encounters | Encodes temporal spacing between visits |
| Duration calculation | conditions | Derives condition length from onset/abatement dates |
| Clinical boolean flags | conditions, medications | Creates binary features from categorical status columns |
| Redundant code removal | all tables | Removes numeric codes when human-readable text retained |
| Patient aggregate enrichment | patients | Adds summary features that inform sequential generation |
| Deduplication on primary key | all tables | Prevents duplicate records from corrupting FK relationships |
| Foreign key validation | all child tables | Confirms referential integrity after all transformations |
| SDV metadata generation | all tables | Provides SDV with schema required for correct modelling |

---

## 13. What Would Change for Real EHR Data

This project uses Synthea-generated data, where there are no real people and re-identification
risk is zero. For real EHR data, the following additional steps would be required:

### Patient ID Hashing

```python
import hashlib, secrets

SALT = secrets.token_bytes(32)  # Per-run secret; never store with data

def hash_patient_id(real_mrn: str) -> str:
    return hashlib.sha256(SALT + real_mrn.encode()).hexdigest()[:32]
```

Real Medical Record Numbers (MRNs) must be replaced with deterministic random hashes
so they remain consistent across tables (FK relationships preserved) but cannot be
reverse-mapped to real patients.

### Quasi-Identifier Risk Assessment

The combination of `(gender, race, ethnicity, age_band, city)` in `patients_ready.csv`
creates quasi-identifiers. For real data, a k-anonymity audit is required:

```
For every combination of (gender, race, age_band, city) that appears in the data,
count the number of patients. If any combination has fewer than k=5 patients,
suppress or generalise that combination.
```

The `anonymeter` library (in `requirements.txt`, commented out) implements this check.

### Age Generalisation

Exact ages should be rounded to 5-year bands:

```python
df["age_band"] = pd.cut(df["age"], bins=range(0, 120, 5), 
                         labels=[f"{i}-{i+4}" for i in range(0, 115, 5)])
```

### Geography Generalisation

For real data, `city` should be replaced with a higher-level aggregate (region, state),
or suppressed entirely if city + demographics creates a small cell.

### Minimum k-Anonymity for All Groups

Any value in any column that appears fewer than k=5 times is a re-identification risk.
These rare values should be grouped into an "other" category before training.
