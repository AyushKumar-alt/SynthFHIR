# 07 — FHIR Reconstruction

**Version:** 1.0  
**Last Updated:** 2026-06-28  
**Phase Coverage:** Phase 6 (Planned)  
**Status:** Design complete — implementation pending

---

## Table of Contents

1. [Why Reconstruct FHIR Bundles](#1-why-reconstruct-fhir-bundles)
2. [The Reconstruction Problem](#2-the-reconstruction-problem)
3. [Reconstruction Strategy](#3-reconstruction-strategy)
4. [Mapping Synthetic CSVs Back to FHIR Resources](#4-mapping-synthetic-csvs-back-to-fhir-resources)
5. [Challenges and Solutions](#5-challenges-and-solutions)
6. [Implementation Plan](#6-implementation-plan)
7. [Expected Outputs](#7-expected-outputs)

---

## 1. Why Reconstruct FHIR Bundles

The five synthetic CSVs produced in Phase 4B are useful for researchers and data
scientists who work with tabular data. However, many healthcare applications speak FHIR:

- **FHIR-native analytics platforms** (Microsoft Azure Health Data Services, AWS
  HealthLake, Google Cloud Healthcare API) ingest FHIR JSON bundles, not CSVs
- **EHR test environments** need FHIR resources to populate test databases
- **FHIR validators** test applications against realistic patient records
- **Interoperability testing** between hospital systems uses FHIR as the exchange format

By converting the synthetic CSVs back into valid FHIR R4 JSON bundles, SynthFHIR
produces synthetic data that is immediately usable by any FHIR-compliant application —
not just data science workflows.

---

## 2. The Reconstruction Problem

The original FHIR bundles contained information that was intentionally dropped during
preprocessing:

| Preprocessing step | FHIR information lost |
|---|---|
| PII removal | Patient names, SSN, exact birth dates |
| Date conversion to days_since_birth | All absolute timestamps |
| Code column removal (loinc_code, snomed_code, rxnorm_code) | Numeric coding system codes |
| Cardinality capping ("other") | Specific organisation names for rare providers |

Reconstruction must handle each of these:

1. **Names:** Generate plausible synthetic names (not real names from training data)
2. **Dates:** Convert `days_since_birth` back to calendar dates using a synthetic birth date
3. **Codes:** Look up LOINC/SNOMED/RxNorm codes for the display names
4. **Organisations:** Generate plausible provider names for `"other"` category records

---

## 3. Reconstruction Strategy

```
Synthetic CSVs (tabular)
     │
     ▼
Step 1: Generate synthetic birth dates
  └─ Convert synthetic age → plausible birth date
     (e.g., age 45.3 → born approximately 45 years ago)
     Use synthetic reference date (not today) to avoid de-anonymisation

     ▼
Step 2: Reverse temporal encoding
  └─ Convert days_since_birth → absolute datetime
     absolute_date = synthetic_birth_date + timedelta(days=days_since_birth)

     ▼
Step 3: Generate synthetic PII (names, addresses)
  └─ Generate plausible names and addresses that match demographics
     (gender-appropriate first names, common surnames, US addresses)
     These are entirely synthetic — not from any real person

     ▼
Step 4: Look up coding system codes
  └─ Reverse map display names to LOINC/SNOMED/RxNorm codes
     Using pre-built lookup tables derived from the coding systems
     e.g., "Body Height" → LOINC code "8302-2"

     ▼
Step 5: Construct FHIR resources
  └─ Map CSV columns to FHIR resource fields
     Generate valid JSON for Patient, Encounter, Observation, Condition, MedicationRequest

     ▼
Step 6: Bundle resources per patient
  └─ Create one FHIR Bundle per synthetic patient
     containing all their Encounters, Observations, Conditions, Medications

     ▼
Step 7: Validate FHIR compliance
  └─ Validate each bundle against the FHIR R4 specification
     using the fhir.resources Python library
```

---

## 4. Mapping Synthetic CSVs Back to FHIR Resources

### Patient → FHIR Patient

| CSV Column | FHIR Field | Notes |
|---|---|---|
| patient_id | id | Use as-is (already a UUID) |
| gender | gender | Direct mapping |
| race | extension[us-core-race] | HL7 US Core race extension |
| ethnicity | extension[us-core-ethnicity] | HL7 US Core ethnicity extension |
| birth_sex | extension[us-core-birthsex] | |
| language | communication[0].language | |
| city | address[0].city | |
| is_deceased + age_at_death | deceasedDateTime | Reconstruct from synthetic birth date |
| (generated) | name[0].family + name[0].given | Synthetic name generation |
| (generated) | birthDate | Derived from synthetic age |

### Encounter → FHIR Encounter

| CSV Column | FHIR Field | Notes |
|---|---|---|
| encounter_id | id | |
| patient_id | subject.reference | "Patient/{patient_id}" |
| class_code | class.code | Direct (AMB, EMER, IMP, etc.) |
| type_display | type[0].coding[0].display | |
| organization_display | serviceProvider.display | |
| location_display | location[0].location.display | |
| reason_display | reasonCode[0].coding[0].display | |
| days_since_birth → datetime | period.start | synthetic_birth + days |
| days_since_birth + duration_hours | period.end | start + duration |
| discharge_disposition | hospitalization.dischargeDisposition | |

### Observation → FHIR Observation

| CSV Column | FHIR Field | Notes |
|---|---|---|
| observation_id | id | |
| patient_id | subject.reference | |
| encounter_id | encounter.reference | |
| category | category[0].coding[0].code | |
| loinc_display → code | code.coding[0] | Look up LOINC code from display |
| value_quantity | valueQuantity.value | |
| value_unit | valueQuantity.unit | |
| days_since_birth → datetime | effectiveDateTime | |

### Condition → FHIR Condition

| CSV Column | FHIR Field | Notes |
|---|---|---|
| condition_id | id | |
| patient_id | subject.reference | |
| encounter_id | encounter.reference | |
| snomed_display → code | code.coding[0] | Look up SNOMED code from display |
| clinical_status | clinicalStatus.coding[0].code | |
| onset_days_since_birth → datetime | onsetDateTime | |
| abatement_days_since_birth → datetime | abatementDateTime | NaN → null |
| is_chronic | (derived from abatementDateTime) | |

### MedicationRequest → FHIR MedicationRequest

| CSV Column | FHIR Field | Notes |
|---|---|---|
| medication_id | id | |
| patient_id | subject.reference | |
| encounter_id | encounter.reference | |
| rxnorm_display → code | medicationCodeableConcept.coding[0] | Look up RxNorm code |
| status | status | |
| category | category[0].coding[0].code | |
| reason_display | reasonCode[0].coding[0].display | |
| days_since_birth → datetime | authoredOn | |
| is_active | (derived from status) | |

---

## 5. Challenges and Solutions

### Challenge 1: Synthetic Birth Date Generation

The synthetic patients table has `age` (e.g., 45.3 years) but no birth date. We need
a birth date to reconstruct absolute timestamps.

**Solution:** Generate a synthetic reference date for each patient:

```python
import random
from datetime import date, timedelta

def synthetic_birth_date(age_years: float, reference_date: date) -> date:
    days_lived = int(age_years * 365.25)
    birth = reference_date - timedelta(days=days_lived)
    # Add small random jitter (±30 days) to prevent deterministic mapping
    birth += timedelta(days=random.randint(-30, 30))
    return birth
```

**Why add jitter?** Without jitter, every patient of the same synthetic age would have
the same synthetic birth date, which would look unrealistic in the FHIR bundle.

**Reference date:** Use the Phase 3 preprocessing date (2026-06-28) so that
`age = 45.3 → birth ≈ 1981-03-01`.

### Challenge 2: Coding System Reverse Lookup

During preprocessing, `loinc_code` was dropped, keeping only `loinc_display`. To
reconstruct valid FHIR, we need the code back.

**Solution:** Build lookup dictionaries from LOINC, SNOMED, and RxNorm public release
files (all freely available) for the specific codes used in this dataset:

```python
LOINC_LOOKUP = {
    "Body Height": "8302-2",
    "Systolic Blood Pressure": "8480-6",
    "Diastolic Blood Pressure": "8462-4",
    "Heart rate": "8867-4",
    ...
}

SNOMED_LOOKUP = {
    "Diabetes mellitus type 2": "44054006",
    "Essential hypertension": "59621000",
    ...
}
```

Since we filtered to the top-30 LOINC codes, the lookup table has exactly 30 entries.
The SNOMED and RxNorm tables are larger but bounded by what appeared in the training data.

### Challenge 3: FHIR Validation

Not every generated FHIR resource will be perfectly valid. Common issues:

- Required fields missing (some FHIR elements are mandatory)
- Wrong value types (FHIR expects specific code systems)
- Logical inconsistencies (abatement before onset)

**Solution:** Use the `fhir.resources` Python library to validate each resource
after construction. Invalid resources are logged and corrected where possible (e.g.,
swap onset/abatement if reversed), or excluded with a warning.

---

## 6. Implementation Plan

Phase 6 will add:

```
src/fhir_reconstruction/
├── date_generator.py     — synthetic birth date generation
├── code_lookup.py        — LOINC/SNOMED/RxNorm lookup tables
├── name_generator.py     — gender/ethnicity-appropriate name generation
├── resource_builders.py  — CSV row → FHIR resource constructors
├── bundle_assembler.py   — assemble per-patient FHIR bundles
└── validator.py          — FHIR R4 validation

run_phase6.py             — CLI entry point
```

---

## 7. Expected Outputs

After Phase 6 completes:

```
outputs/fhir_synthetic/
├── bundle_patient_0001.json    ← One FHIR Bundle per synthetic patient
├── bundle_patient_0002.json
├── ...
├── bundle_patient_1000.json
├── validation_report.json      ← Count of valid vs invalid resources per type
└── summary.json                ← Total resource counts, validation rate
```

Each bundle will contain:
- 1 Patient resource
- ~58 Encounter resources (average)
- ~304 Observation resources (average)
- ~38 Condition resources (average)
- ~47 MedicationRequest resources (average)
