# 01 — Background Knowledge

**Version:** 1.0  
**Last Updated:** 2026-06-28  
**Audience:** Readers with no prior knowledge of healthcare IT or FHIR

---

## Table of Contents

1. [Electronic Health Records (EHR)](#1-electronic-health-records-ehr)
2. [HL7 — The Healthcare Data Standard Organisation](#2-hl7--the-healthcare-data-standard-organisation)
3. [FHIR — Fast Healthcare Interoperability Resources](#3-fhir--fast-healthcare-interoperability-resources)
4. [FHIR Resources — The Building Blocks](#4-fhir-resources--the-building-blocks)
5. [FHIR Bundles — Packaging Multiple Resources](#5-fhir-bundles--packaging-multiple-resources)
6. [FHIR References — How Resources Link to Each Other](#6-fhir-references--how-resources-link-to-each-other)
7. [The Five Clinical Tables in This Project](#7-the-five-clinical-tables-in-this-project)
8. [Foreign Keys — Relational Integrity in Healthcare Data](#8-foreign-keys--relational-integrity-in-healthcare-data)
9. [The Clinical Pathway — A Patient's Journey Through the System](#9-the-clinical-pathway--a-patients-journey-through-the-system)
10. [Healthcare Interoperability — Why Standards Matter](#10-healthcare-interoperability--why-standards-matter)
11. [Synthea — The Synthetic Patient Generator](#11-synthea--the-synthetic-patient-generator)

---

## 1. Electronic Health Records (EHR)

An **Electronic Health Record (EHR)** is a digital version of a patient's complete
medical history. Before EHRs, hospitals kept paper charts — physical folders containing
handwritten notes, lab results, and medication lists. EHR systems replaced these with
structured digital databases.

A modern EHR contains:

- **Demographics:** Name, date of birth, gender, address, insurance information
- **Encounters:** Every visit to a doctor, hospital, or clinic
- **Observations:** Every measurement taken — temperature, blood pressure, blood glucose,
  cholesterol, body weight
- **Conditions:** Every diagnosis — hypertension, diabetes, asthma, cancer
- **Medications:** Every drug prescribed — name, dose, frequency, duration
- **Allergies:** Known drug and food allergies
- **Procedures:** Surgeries, imaging, diagnostic procedures
- **Immunisations:** Vaccination history
- **Documents:** Discharge summaries, clinical notes, referral letters

The largest EHR vendors include Epic, Cerner (now Oracle Health), Allscripts, and
Meditech. The problem is that each vendor stores data differently. A patient record in
Epic looks structurally different from the same patient record in Cerner. This makes it
nearly impossible to share data between hospitals that use different systems.

This is the **interoperability problem** that FHIR was designed to solve.

---

## 2. HL7 — The Healthcare Data Standard Organisation

**HL7** stands for **Health Level 7**. It is a non-profit organisation founded in 1987
that creates and maintains standards for exchanging healthcare data. The "Level 7" refers
to the seventh layer of the OSI network model — the application layer where data is
formatted and interpreted.

HL7 has published multiple data exchange standards over the decades:

| Standard | Year | Description |
|---|---|---|
| HL7 v2 | 1989 | Pipe-delimited text messages (still widely used today) |
| HL7 v3 | 2005 | XML-based, but complex and poorly adopted |
| CDA (Clinical Document Architecture) | 2005 | XML documents for clinical notes |
| FHIR | 2014 | RESTful API standard using JSON or XML (current standard) |

HL7 v2 messages look like this:

```
MSH|^~\&|ADT|HOSPITAL|LAB|REFERENCE|20240101||ADT^A01|MSG001|P|2.3
PID|1||12345^^^HOSPITAL^MR||Doe^John^A||19800101|M|||123 Main St
```

This is unreadable to humans, difficult to parse programmatically, and has hundreds of
optional fields with inconsistent usage across vendors. FHIR was designed to replace this
with something modern, human-readable, and machine-friendly.

---

## 3. FHIR — Fast Healthcare Interoperability Resources

**FHIR** (pronounced "fire") stands for **Fast Healthcare Interoperability Resources**.
It is the current HL7 standard for exchanging healthcare data. Released in 2014 and now
at version R4 (the fourth major release), FHIR is built on web technologies that software
engineers already know: REST APIs, JSON, and HTTP.

### Why FHIR Was Created

Before FHIR, a patient moving from one hospital to another had to physically carry their
records (or hope a fax machine would work). Researchers who needed data from multiple
hospitals had to negotiate separate data sharing agreements with each institution and write
custom software to convert each institution's data format.

FHIR solves this by defining a common vocabulary: a specific JSON structure for what a
"Patient" looks like, a specific structure for what a "Medication" looks like, a specific
structure for what a "Laboratory Result" looks like — and these structures are the same
everywhere that FHIR is implemented.

### What FHIR Looks Like

A FHIR Patient resource in JSON format:

```json
{
  "resourceType": "Patient",
  "id": "617a71a9-1df5-4a2d-277b-1f89906ee9e0",
  "name": [
    {
      "family": "Smith",
      "given": ["John", "Edward"]
    }
  ],
  "gender": "male",
  "birthDate": "1980-03-15",
  "address": [
    {
      "line": ["123 Main Street"],
      "city": "Arlington",
      "state": "MA",
      "postalCode": "02476"
    }
  ]
}
```

This is standard JSON — any programming language can parse it without a specialised library.
The field names (`gender`, `birthDate`, `address`) are defined by the FHIR specification
and mean the same thing in every FHIR-compliant system.

---

## 4. FHIR Resources — The Building Blocks

In FHIR, every piece of clinical information is represented as a **Resource** — a JSON
object with a specific structure. FHIR defines over 150 resource types. This project uses
five of them:

### Patient

Represents a single person receiving healthcare. Contains demographic information.

```json
{
  "resourceType": "Patient",
  "id": "uuid-here",
  "gender": "female",
  "birthDate": "1975-06-20",
  "extension": [
    {
      "url": "http://hl7.org/fhir/us/core/StructureDefinition/us-core-race",
      "valueCodeableConcept": {
        "coding": [{"display": "White"}]
      }
    }
  ]
}
```

A patient has no clinical data — they are just the person. All clinical events are
stored in other resources that reference this patient.

### Encounter

Represents a single interaction between a patient and the healthcare system. An encounter
could be a routine check-up, an emergency room visit, an inpatient hospital stay, or a
telemedicine call.

```json
{
  "resourceType": "Encounter",
  "id": "encounter-uuid",
  "status": "finished",
  "class": {
    "code": "AMB",
    "display": "ambulatory"
  },
  "subject": {
    "reference": "Patient/617a71a9-1df5-4a2d-277b-1f89906ee9e0"
  },
  "period": {
    "start": "2023-03-15T09:00:00+00:00",
    "end": "2023-03-15T09:30:00+00:00"
  }
}
```

The `subject.reference` field links this encounter back to its patient. This is how FHIR
maintains relationships between resources.

### Observation

Represents a single clinical measurement. Observations are the most numerous resource
in any EHR — a single visit might generate 10–20 observations (blood pressure, heart rate,
body weight, blood glucose, haemoglobin, etc.).

```json
{
  "resourceType": "Observation",
  "id": "obs-uuid",
  "status": "final",
  "code": {
    "coding": [
      {
        "system": "http://loinc.org",
        "code": "8480-6",
        "display": "Systolic Blood Pressure"
      }
    ]
  },
  "subject": {
    "reference": "Patient/617a71a9-1df5-4a2d-277b-1f89906ee9e0"
  },
  "encounter": {
    "reference": "Encounter/encounter-uuid"
  },
  "valueQuantity": {
    "value": 128.0,
    "unit": "mm[Hg]"
  }
}
```

The `code` field uses **LOINC** (Logical Observation Identifiers Names and Codes) — a
universal coding system for lab tests and clinical measurements. LOINC code `8480-6`
always means "Systolic Blood Pressure" regardless of which EHR system generated the record.

### Condition

Represents a clinical diagnosis — a disease or health problem that the patient has or
has had.

```json
{
  "resourceType": "Condition",
  "id": "cond-uuid",
  "clinicalStatus": {
    "coding": [{"code": "active"}]
  },
  "code": {
    "coding": [
      {
        "system": "http://snomed.info/sct",
        "code": "44054006",
        "display": "Diabetes mellitus type 2"
      }
    ]
  },
  "subject": {
    "reference": "Patient/617a71a9-1df5-4a2d-277b-1f89906ee9e0"
  },
  "onsetDateTime": "2018-04-22T00:00:00+00:00",
  "abatementDateTime": null
}
```

The `code` field uses **SNOMED CT** (Systematized Nomenclature of Medicine — Clinical Terms)
— a comprehensive coding system for clinical conditions. Code `44054006` always means
"Type 2 Diabetes" regardless of how the hospital documents it in their own notes.

When `abatementDateTime` is null, the condition is ongoing (chronic). When it has a value,
the condition resolved (e.g., a broken bone that healed, an infection that cleared).

### MedicationRequest

Represents a prescription — a request for a patient to receive a medication.

```json
{
  "resourceType": "MedicationRequest",
  "id": "med-uuid",
  "status": "active",
  "intent": "order",
  "medicationCodeableConcept": {
    "coding": [
      {
        "system": "http://www.nlm.nih.gov/research/umls/rxnorm",
        "code": "860975",
        "display": "Metformin 500 MG Oral Tablet"
      }
    ]
  },
  "subject": {
    "reference": "Patient/617a71a9-1df5-4a2d-277b-1f89906ee9e0"
  },
  "encounter": {
    "reference": "Encounter/encounter-uuid"
  },
  "authoredOn": "2018-04-22T00:00:00+00:00"
}
```

The `medicationCodeableConcept` uses **RxNorm** — a standard drug coding system maintained
by the US National Library of Medicine. RxNorm code `860975` always means exactly this
formulation of Metformin.

---

## 5. FHIR Bundles — Packaging Multiple Resources

A **FHIR Bundle** is a container that holds multiple FHIR resources together. When Synthea
generates a patient record, it produces a single JSON file — a Bundle — that contains
all the resources for that patient: the Patient resource plus all their Encounters,
Observations, Conditions, and MedicationRequests.

```json
{
  "resourceType": "Bundle",
  "type": "collection",
  "entry": [
    {
      "resource": {
        "resourceType": "Patient",
        "id": "617a71a9-...",
        ...
      }
    },
    {
      "resource": {
        "resourceType": "Encounter",
        "id": "enc-001",
        "subject": {"reference": "Patient/617a71a9-..."},
        ...
      }
    },
    {
      "resource": {
        "resourceType": "Observation",
        "id": "obs-001",
        "subject": {"reference": "Patient/617a71a9-..."},
        "encounter": {"reference": "Encounter/enc-001"},
        ...
      }
    }
  ]
}
```

In this project, 1,000 Synthea patients each produce one JSON file. Each file is a Bundle
containing all resources for that patient. Phase 2 parses these 1,000 files and extracts
the resources into five relational tables.

---

## 6. FHIR References — How Resources Link to Each Other

FHIR resources link to each other using **References** — a field containing the resource
type and ID of the linked resource.

```json
"subject": {
  "reference": "Patient/617a71a9-1df5-4a2d-277b-1f89906ee9e0"
}
```

This reference says: "the subject of this observation is the Patient whose ID is
`617a71a9-1df5-4a2d-277b-1f89906ee9e0`." To find the patient's details, you look up
the Patient resource with that ID.

References are the FHIR equivalent of **foreign keys** in a relational database. When
Phase 2 parses the FHIR bundles, these references are extracted as foreign key columns
in the CSV tables:

| FHIR Reference | Becomes |
|---|---|
| `encounter.subject.reference` | `encounters.patient_id` |
| `observation.subject.reference` | `observations.patient_id` |
| `observation.encounter.reference` | `observations.encounter_id` |
| `condition.subject.reference` | `conditions.patient_id` |
| `medicationRequest.subject.reference` | `medications.patient_id` |

---

## 7. The Five Clinical Tables in This Project

After parsing, the FHIR bundles become five related tables. Here is a conceptual diagram:

```
patients (998 rows)
│
│  patient_id (UUID) ← primary key
│  gender, race, ethnicity, age, ...
│
├──────────────────────────────────────────────────┐
│                                                  │
encounters (57,667 rows)                          (other child tables)
│  encounter_id ← primary key                      │
│  patient_id ──────────────────────────────────→ patients.patient_id
│  type_display, class_code, days_since_birth, ...
│                                                  │
├─────────────────────┐                            │
│                     │                            │
observations           conditions                  medications
(303,696 rows)         (37,835 rows)               (46,734 rows)
patient_id ──────→ patients    patient_id ──→ patients   patient_id ──→ patients
encounter_id ────→ encounters  encounter_id → encounters  encounter_id → encounters
```

### Why Five Tables?

Healthcare data is inherently relational. A patient is not defined by a single row of
data — they are defined by their demographics PLUS their clinical history across multiple
types of events. Flattening all this into one table would:

1. Create an enormous number of columns (hundreds)
2. Create massive redundancy (patient demographics repeated for every observation)
3. Make it impossible to capture the sequential nature of clinical events
4. Destroy temporal relationships between events

Keeping the data in five linked tables preserves the relational structure that mirrors
how healthcare data actually works.

---

## 8. Foreign Keys — Relational Integrity in Healthcare Data

A **foreign key** is a column in one table that contains values that match the primary key
of another table. Foreign keys define relationships between tables and enforce referential
integrity — the guarantee that every child record has a valid parent record.

In this project:

| Child Column | Parent Table | Parent Column | Meaning |
|---|---|---|---|
| `encounters.patient_id` | patients | `patient_id` | Every encounter belongs to exactly one patient |
| `observations.patient_id` | patients | `patient_id` | Every observation belongs to exactly one patient |
| `observations.encounter_id` | encounters | `encounter_id` | Every observation occurred during exactly one encounter |
| `conditions.patient_id` | patients | `patient_id` | Every condition belongs to exactly one patient |
| `conditions.encounter_id` | encounters | `encounter_id` | Every condition was diagnosed during exactly one encounter |
| `medications.patient_id` | patients | `patient_id` | Every prescription belongs to exactly one patient |
| `medications.encounter_id` | encounters | `encounter_id` | Every prescription was written during exactly one encounter |

**Why this matters for synthetic data generation:**

When we generate synthetic data, we cannot simply generate each table independently.
If we generate 1,000 synthetic patients with UUIDs A, B, C... and then generate
encounters with random patient IDs, those IDs will not match any patient.

The SynthFHIR pipeline solves this by generating tables in a specific order (patients first,
then their child tables) and remapping synthetic UUIDs to maintain referential integrity.

---

## 9. The Clinical Pathway — A Patient's Journey Through the System

To understand why the data looks the way it does, it helps to follow one patient through
the healthcare system from birth to death.

```
BIRTH
  │
  ├─ Patient record created: demographics, insurance, language
  │
  ▼
CHILDHOOD
  ├─ Annual well-child visits (Encounters: AMB / ambulatory)
  │     ├─ Height, weight, head circumference (Observations)
  │     └─ Vaccinations (Immunisations — not in scope for this project)
  │
  ▼
ADULTHOOD
  ├─ Annual physical examinations (Encounters)
  │     ├─ Blood pressure, heart rate, BMI (Observations)
  │     ├─ Cholesterol panel (Observations: LDL, HDL, triglycerides)
  │     └─ Blood glucose (Observation: fasting glucose, HbA1c)
  │
  ├─ New diagnosis: Type 2 Diabetes (Condition: onset 2018)
  │     └─ Medication started: Metformin (MedicationRequest: 2018)
  │
  ├─ Emergency visit: acute chest pain (Encounter: EMER)
  │     ├─ ECG, troponin, BNP (Observations)
  │     └─ Diagnosis: Unstable Angina (Condition: onset 2022)
  │
  ├─ Hospitalisation: cardiac catheterisation (Encounter: IMP / inpatient)
  │     ├─ Pre-procedure bloods (Observations)
  │     └─ Post-procedure vitals (Observations)
  │
  ▼
DEATH (or end of observation period)
  └─ Patient deceased_datetime recorded
```

This pathway explains why:

- **Observations are the most numerous table:** Every visit generates multiple measurements.
  A patient with 500 encounters over their lifetime easily generates 3,000–5,000 observations.
- **Encounters come before all other clinical events:** You cannot have an observation or
  a new diagnosis without a visit where it was recorded.
- **Conditions can span multiple encounters:** A chronic condition like diabetes has one
  onset date but is referenced in dozens of subsequent encounters.
- **Sequential modelling is important for encounters:** The time between visit #1 and
  visit #2 is correlated with the patient's health status. A patient with severe chronic
  disease visits more frequently than a healthy patient. This temporal dependency is why
  encounters use the PAR (Probabilistic AutoRegressive) model rather than CTGAN.

---

## 10. Healthcare Interoperability — Why Standards Matter

**Interoperability** means the ability of different computer systems to exchange and use
data. Healthcare interoperability means a patient's record can flow from their GP, to a
specialist, to a hospital, to a pharmacist, and back — without manual re-entry.

Without interoperability:

- A patient arriving at an emergency room unconscious cannot have their allergies or
  current medications looked up from their GP's record
- A researcher studying a drug's long-term effects cannot link pharmacy dispensing records
  with hospital admission records without months of manual data linkage
- An AI model trained on data from one hospital cannot be validated on data from another
  without a custom data transformation for each institution

FHIR provides the common language. When every institution uses FHIR, data flows freely.

For this project, FHIR is important for two reasons:

1. **Input:** The Synthea generator produces FHIR bundles — the same format a real hospital
   would produce if it were a FHIR-compliant EHR.
2. **Output:** Phase 6 of this project will reconstruct the synthetic data back into FHIR
   bundles — making the synthetic records directly usable by any FHIR-compliant application.

---

## 11. Synthea — The Synthetic Patient Generator

**Synthea** is an open-source synthetic patient generator developed by The MITRE Corporation.
It uses rule-based clinical models to simulate the health histories of synthetic patients
from birth to death.

Synthea is not a machine learning model. It generates patients according to pre-programmed
disease progression rules derived from clinical epidemiology literature:

- Age-specific incidence rates for diabetes, hypertension, cancer, etc.
- Disease progression pathways (pre-diabetes → type 2 diabetes → complications)
- Clinical guidelines for when to order lab tests, prescribe medications, and refer to specialists
- US-specific demographics from Census data

### Why Synthea Is Used as Input

Synthea is the gold standard for synthetic health data in research. It is:

- **Free and open-source** — no licensing restrictions
- **FHIR-native** — outputs valid FHIR R4 JSON bundles directly
- **Medically realistic** — grounded in published clinical epidemiology
- **Widely used** — the default test dataset for dozens of healthcare IT tools

### Synthea's Limitations (Why SynthFHIR Adds Value)

Synthea's rule-based approach has inherent limitations:

| Limitation | Consequence |
|---|---|
| Fixed disease pathways | Rare or unusual clinical presentations are not modelled |
| Rule-derived, not data-derived | Does not reflect a specific real population's demographics |
| US-centric | Designed for US epidemiology; poorly reflects other countries |
| Independence assumption | Conditions and medications are generated somewhat independently; complex comorbidity patterns are not fully captured |
| No learning | Cannot improve by exposure to real data |

SynthFHIR addresses these by learning from Synthea data using generative models. If
replaced with real hospital data (respecting all privacy requirements), the same pipeline
would produce synthetic data reflecting that specific population.

### Synthea in This Project

- **1,000 patients** were generated using the default Synthea Massachusetts state module
- **FHIR R4 JSON** output format
- Each patient produces one `.json` file (a FHIR Bundle)
- The 1,000 bundles contain:
  - 998 patients (2 were filtered during preprocessing)
  - 57,667 encounters
  - 303,696 observations
  - 37,835 conditions
  - 46,734 medications
  - Total: 446,930 clinical records across all tables
