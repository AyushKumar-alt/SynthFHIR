"""Observation resource extractor.

Handles three value shapes found in Synthea output:
  - valueQuantity      → numeric lab/vital result  (e.g. HbA1c = 5.9 %)
  - valueCodeableConcept → categorical result       (e.g. tobacco status)
  - valueString        → free-text result           (rare)
  - component[]        → multi-part panel           (e.g. blood pressure)
"""

from __future__ import annotations

import json
import logging
from typing import Any

from .base import get_coding, get_quantity, get_reference_id

logger = logging.getLogger(__name__)


def extract_observation_row(resource: dict) -> dict[str, Any] | None:
    """Extract a flat row dict from a FHIR Observation resource.

    Blood pressure panels use the 'component' field; these are stored as a
    JSON string so no information is lost, and they can be expanded in the
    preprocessing phase.

    Args:
        resource: Raw FHIR Observation resource dict from a bundle entry.

    Returns:
        Flat dict suitable for a DataFrame row, or None if extraction fails.
    """
    try:
        # ── Category (laboratory, vital-signs, survey, etc.) ───────────────
        categories = resource.get("category", [])
        _, _, category_display = (
            get_coding(categories[0].get("coding", [])) if categories else ("", "", "")
        )

        # ── LOINC observation code ─────────────────────────────────────────
        _, loinc_code, loinc_display = get_coding(
            resource.get("code", {}).get("coding", [])
        )

        # ── References ────────────────────────────────────────────────────
        patient_id = get_reference_id(
            resource.get("subject", {}).get("reference", "")
        )
        encounter_id = get_reference_id(
            resource.get("encounter", {}).get("reference", "")
        )

        # ── Value — detect type and extract accordingly ────────────────────
        value_type = ""
        value_quantity: float | None = None
        value_unit = ""
        value_code = ""
        value_display = ""
        value_string = ""
        component_json = ""

        if "valueQuantity" in resource:
            value_type = "quantity"
            value_quantity, value_unit = get_quantity(resource["valueQuantity"])

        elif "valueCodeableConcept" in resource:
            value_type = "codeable_concept"
            vcc = resource["valueCodeableConcept"]
            _, value_code, value_display = get_coding(vcc.get("coding", []))
            if not value_display:
                value_display = vcc.get("text", "")

        elif "valueString" in resource:
            value_type = "string"
            value_string = str(resource["valueString"])

        elif "component" in resource:
            # Blood pressure panels: store raw JSON to preserve systolic/diastolic
            value_type = "component"
            component_json = json.dumps(resource["component"])

        return {
            "observation_id": resource.get("id", ""),
            "patient_id": patient_id,
            "encounter_id": encounter_id,
            "status": resource.get("status", ""),
            "category": category_display,
            "loinc_code": loinc_code,
            "loinc_display": loinc_display,
            "effective_datetime": resource.get("effectiveDateTime", ""),
            "value_type": value_type,
            "value_quantity": value_quantity,
            "value_unit": value_unit,
            "value_code": value_code,
            "value_display": value_display,
            "value_string": value_string,
            "component_json": component_json,
        }

    except Exception as exc:
        logger.warning("Observation extraction failed (id=%s): %s", resource.get("id"), exc)
        return None
