"""MedicationRequest resource extractor.

Synthea produces two variants of the medication field:
  - medicationCodeableConcept  → RxNorm code + display inline
  - medicationReference        → reference to a separate Medication resource

Both are handled; the referenced variant stores the display text when
available (the UUID reference is not useful for synthesis).
"""

from __future__ import annotations

import logging
from typing import Any

from .base import get_coding, get_reference_id

logger = logging.getLogger(__name__)


def extract_medication_row(resource: dict) -> dict[str, Any] | None:
    """Extract a flat row dict from a FHIR MedicationRequest resource.

    Args:
        resource: Raw FHIR MedicationRequest resource dict from a bundle entry.

    Returns:
        Flat dict suitable for a DataFrame row, or None if extraction fails.
    """
    try:
        # ── Medication code — two possible locations ───────────────────────
        rxnorm_code = ""
        rxnorm_display = ""

        if "medicationCodeableConcept" in resource:
            mcc = resource["medicationCodeableConcept"]
            _, rxnorm_code, rxnorm_display = get_coding(mcc.get("coding", []))
            if not rxnorm_display:
                rxnorm_display = mcc.get("text", "")

        elif "medicationReference" in resource:
            med_ref = resource["medicationReference"]
            rxnorm_display = med_ref.get("display", "")
            # Store the UUID so it can be joined to medications.csv if needed
            rxnorm_code = get_reference_id(med_ref.get("reference", ""))

        # ── Category (community, inpatient, etc.) ─────────────────────────
        categories = resource.get("category", [])
        category_display = ""
        if categories:
            _, _, category_display = get_coding(categories[0].get("coding", []))
            if not category_display:
                category_display = categories[0].get("text", "")

        # ── Reason — prefer reasonReference.display, fall back to reasonCode ─
        reason_display = ""
        reason_refs = resource.get("reasonReference", [])
        reason_codes = resource.get("reasonCode", [])
        if reason_refs:
            reason_display = reason_refs[0].get("display", "")
        elif reason_codes:
            _, _, reason_display = get_coding(reason_codes[0].get("coding", []))

        # ── Dosage (free text) ────────────────────────────────────────────
        dosage_instructions = resource.get("dosageInstruction", [])
        dosage_text = dosage_instructions[0].get("text", "") if dosage_instructions else ""

        # ── Requester ─────────────────────────────────────────────────────
        requester_display = resource.get("requester", {}).get("display", "")

        # ── References ────────────────────────────────────────────────────
        patient_id = get_reference_id(
            resource.get("subject", {}).get("reference", "")
        )
        encounter_id = get_reference_id(
            resource.get("encounter", {}).get("reference", "")
        )

        return {
            "medication_id": resource.get("id", ""),
            "patient_id": patient_id,
            "encounter_id": encounter_id,
            "status": resource.get("status", ""),
            "intent": resource.get("intent", ""),
            "rxnorm_code": rxnorm_code,
            "rxnorm_display": rxnorm_display,
            "category": category_display,
            "authored_on": resource.get("authoredOn", ""),
            "requester_display": requester_display,
            "reason_display": reason_display,
            "dosage_text": dosage_text,
        }

    except Exception as exc:
        logger.warning(
            "MedicationRequest extraction failed (id=%s): %s", resource.get("id"), exc
        )
        return None
