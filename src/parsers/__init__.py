"""FHIR resource parsers.

Each parser module exposes a single extract_*_row(resource: dict) function
that accepts a raw FHIR resource dict and returns a flat row dict (or None
on failure). The pipeline dispatches to the correct extractor via the
RESOURCE_EXTRACTORS registry below.
"""

from __future__ import annotations

from .patient import extract_patient_row
from .encounter import extract_encounter_row
from .observation import extract_observation_row
from .condition import extract_condition_row
from .medication import extract_medication_row

# Maps FHIR resourceType → extractor function
RESOURCE_EXTRACTORS: dict[str, callable] = {
    "Patient": extract_patient_row,
    "Encounter": extract_encounter_row,
    "Observation": extract_observation_row,
    "Condition": extract_condition_row,
    "MedicationRequest": extract_medication_row,
}

# Maps FHIR resourceType → output CSV filename
RESOURCE_CSV_NAMES: dict[str, str] = {
    "Patient": "patients.csv",
    "Encounter": "encounters.csv",
    "Observation": "observations.csv",
    "Condition": "conditions.csv",
    "MedicationRequest": "medications.csv",
}

__all__ = [
    "RESOURCE_EXTRACTORS",
    "RESOURCE_CSV_NAMES",
    "extract_patient_row",
    "extract_encounter_row",
    "extract_observation_row",
    "extract_condition_row",
    "extract_medication_row",
]
