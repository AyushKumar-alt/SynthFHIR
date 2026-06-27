"""Condition resource extractor."""

from __future__ import annotations

import logging
from typing import Any

from .base import get_coding, get_reference_id

logger = logging.getLogger(__name__)


def extract_condition_row(resource: dict) -> dict[str, Any] | None:
    """Extract a flat row dict from a FHIR Condition resource.

    Args:
        resource: Raw FHIR Condition resource dict from a bundle entry.

    Returns:
        Flat dict suitable for a DataFrame row, or None if extraction fails.
    """
    try:
        # ── SNOMED diagnosis code ──────────────────────────────────────────
        _, snomed_code, snomed_display = get_coding(
            resource.get("code", {}).get("coding", [])
        )

        # ── Clinical + verification status ────────────────────────────────
        clinical_codings = resource.get("clinicalStatus", {}).get("coding", [])
        clinical_status = clinical_codings[0].get("code", "") if clinical_codings else ""

        verification_codings = resource.get("verificationStatus", {}).get("coding", [])
        verification_status = (
            verification_codings[0].get("code", "") if verification_codings else ""
        )

        # ── Category (encounter-diagnosis vs problem-list-item) ────────────
        categories = resource.get("category", [])
        _, _, category_display = (
            get_coding(categories[0].get("coding", [])) if categories else ("", "", "")
        )

        # ── References ────────────────────────────────────────────────────
        patient_id = get_reference_id(
            resource.get("subject", {}).get("reference", "")
        )
        encounter_id = get_reference_id(
            resource.get("encounter", {}).get("reference", "")
        )

        return {
            "condition_id": resource.get("id", ""),
            "patient_id": patient_id,
            "encounter_id": encounter_id,
            "snomed_code": snomed_code,
            "snomed_display": snomed_display,
            "clinical_status": clinical_status,
            "verification_status": verification_status,
            "category": category_display,
            "onset_datetime": resource.get("onsetDateTime", ""),
            "abatement_datetime": resource.get("abatementDateTime", ""),
            "recorded_date": resource.get("recordedDate", ""),
        }

    except Exception as exc:
        logger.warning("Condition extraction failed (id=%s): %s", resource.get("id"), exc)
        return None
