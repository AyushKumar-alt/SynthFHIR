# Data Dictionary

All tables are written to `data/processed/` after running `run_phase2.py`.

---

## Table Relationships

```
patients.csv (1)
    │
    ├──< encounters.csv   (patient_id FK)
    │        │
    │        ├──< observations.csv   (patient_id FK, encounter_id FK)
    │        ├──< conditions.csv     (patient_id FK, encounter_id FK)
    │        └──< medications.csv    (patient_id FK, encounter_id FK)
```

---

## patients.csv

One row per patient (998 rows expected).

| Column | Type | Source | Description |
|--------|------|--------|-------------|
| patient_id | str | Patient.id | Primary key — bare UUID |
| family_name | str | Patient.name[use=official].family | Surname |
| given_name | str | Patient.name[use=official].given | Given names joined with space |
| gender | str | Patient.gender | `male` or `female` |
| birth_date | date | Patient.birthDate | YYYY-MM-DD |
| deceased_datetime | datetime | Patient.deceasedDateTime | ISO datetime; empty if living |
| race | str | Patient.extension[us-core-race].text | e.g. White, Asian |
| ethnicity | str | Patient.extension[us-core-ethnicity].text | e.g. Not Hispanic or Latino |
| birth_sex | str | Patient.extension[us-core-birthsex].valueCode | M or F |
| marital_code | str | Patient.maritalStatus.coding.code | M, S, D, W |
| marital_status | str | Patient.maritalStatus.text | Display text |
| city | str | Patient.address[0].city | Current city |
| state | str | Patient.address[0].state | e.g. MA |
| postal_code | str | Patient.address[0].postalCode | US ZIP code |
| country | str | Patient.address[0].country | US |
| lat | float | Patient.address[0].extension[geolocation].latitude | GPS latitude |
| lon | float | Patient.address[0].extension[geolocation].longitude | GPS longitude |
| birth_place_city | str | Patient.extension[patient-birthPlace].city | City of birth |
| birth_place_state | str | Patient.extension[patient-birthPlace].state | State of birth |
| language | str | Patient.communication[0].language.coding.display | Primary language |
| ssn | str | Patient.identifier[type=SS] | Synthetic SSN — NOT real |
| drivers_license | str | Patient.identifier[type=DL] | Synthetic DL number |
| passport | str | Patient.identifier[type=PPN] | Synthetic passport number |
| daly | float | Patient.extension[disability-adjusted-life-years] | Synthea DALY score |
| qaly | float | Patient.extension[quality-adjusted-life-years] | Synthea QALY score |

---

## encounters.csv

One row per clinical encounter (~57,667 rows expected).

| Column | Type | Source | Description |
|--------|------|--------|-------------|
| encounter_id | str | Encounter.id | Primary key |
| patient_id | str | Encounter.subject.reference | FK → patients.patient_id |
| status | str | Encounter.status | `finished` |
| class_code | str | Encounter.class.code | AMB, EMER, IMP, HH, VR |
| type_code | str | Encounter.type[0].coding.code | SNOMED procedure code |
| type_display | str | Encounter.type[0].coding.display | Human-readable type |
| start_datetime | datetime | Encounter.period.start | ISO datetime with TZ |
| end_datetime | datetime | Encounter.period.end | ISO datetime with TZ |
| practitioner_npi | str | Encounter.participant.individual.reference | NPI number |
| organization_display | str | Encounter.serviceProvider.display | Hospital / clinic name |
| location_display | str | Encounter.location[0].location.display | Facility name |
| reason_code | str | Encounter.reasonCode[0].coding.code | Reason SNOMED code |
| reason_display | str | Encounter.reasonCode[0].coding.display | Reason text |
| discharge_disposition | str | Encounter.hospitalization.dischargeDisposition | Inpatient only |

---

## observations.csv

One row per observation (~531,367 rows expected). The `value_type` column
indicates which value field is populated.

| Column | Type | Source | Description |
|--------|------|--------|-------------|
| observation_id | str | Observation.id | Primary key |
| patient_id | str | Observation.subject.reference | FK → patients.patient_id |
| encounter_id | str | Observation.encounter.reference | FK → encounters.encounter_id |
| status | str | Observation.status | `final` |
| category | str | Observation.category[0].coding.display | laboratory, vital-signs, survey |
| loinc_code | str | Observation.code.coding[system=loinc].code | LOINC code |
| loinc_display | str | Observation.code.coding[system=loinc].display | Observation name |
| effective_datetime | datetime | Observation.effectiveDateTime | When observation was taken |
| value_type | str | Derived | quantity, codeable_concept, string, component |
| value_quantity | float | Observation.valueQuantity.value | Numeric result |
| value_unit | str | Observation.valueQuantity.unit | Unit of measure (e.g. %) |
| value_code | str | Observation.valueCodeableConcept.coding.code | Coded result |
| value_display | str | Observation.valueCodeableConcept.coding.display | Coded result text |
| value_string | str | Observation.valueString | Free-text result (rare) |
| component_json | str | Observation.component[] | JSON string for BP panels |

---

## conditions.csv

One row per condition record (~37,835 rows expected).

| Column | Type | Source | Description |
|--------|------|--------|-------------|
| condition_id | str | Condition.id | Primary key |
| patient_id | str | Condition.subject.reference | FK → patients.patient_id |
| encounter_id | str | Condition.encounter.reference | FK → encounters.encounter_id |
| snomed_code | str | Condition.code.coding[system=snomed].code | SNOMED-CT code |
| snomed_display | str | Condition.code.coding[system=snomed].display | Diagnosis name |
| clinical_status | str | Condition.clinicalStatus.coding.code | active, resolved, inactive |
| verification_status | str | Condition.verificationStatus.coding.code | confirmed, unconfirmed |
| category | str | Condition.category[0].coding.display | encounter-diagnosis |
| onset_datetime | datetime | Condition.onsetDateTime | When condition started |
| abatement_datetime | datetime | Condition.abatementDateTime | When condition ended (empty = ongoing) |
| recorded_date | date | Condition.recordedDate | When recorded in chart |

---

## medications.csv

One row per MedicationRequest (~46,734 rows expected).

| Column | Type | Source | Description |
|--------|------|--------|-------------|
| medication_id | str | MedicationRequest.id | Primary key |
| patient_id | str | MedicationRequest.subject.reference | FK → patients.patient_id |
| encounter_id | str | MedicationRequest.encounter.reference | FK → encounters.encounter_id |
| status | str | MedicationRequest.status | active, stopped, completed |
| intent | str | MedicationRequest.intent | order |
| rxnorm_code | str | MedicationRequest.medicationCodeableConcept.coding.code | RxNorm code |
| rxnorm_display | str | MedicationRequest.medicationCodeableConcept.coding.display | Drug name + strength |
| category | str | MedicationRequest.category[0].text | community, inpatient |
| authored_on | datetime | MedicationRequest.authoredOn | Prescription date |
| requester_display | str | MedicationRequest.requester.display | Prescribing clinician |
| reason_display | str | MedicationRequest.reasonReference[0].display | Clinical indication |
| dosage_text | str | MedicationRequest.dosageInstruction[0].text | Dosage instructions |
