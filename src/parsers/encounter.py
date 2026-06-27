"""Encounter resource extractor."""

from __future__ import annotations

import logging
from typing import Any

from .base import get_coding, get_period_dates, get_reference_id

logger = logging.getLogger(__name__)


def extract_encounter_row(resource: dict) -> dict[str, Any] | None:
    """Extract a flat row dict from a FHIR Encounter resource.

    Args:
        resource: Raw FHIR Encounter resource dict from a bundle entry.

    Returns:
        Flat dict suitable for a DataFrame row, or None if extraction fails.
    """
    try:
        # ── Encounter class (AMB, EMER, IMP, HH, VR) ──────────────────────
        class_code = resource.get("class", {}).get("code", "")

        # ── Encounter type (SNOMED) ────────────────────────────────────────
        enc_types = resource.get("type", [])
        _, type_code, type_display = (
            get_coding(enc_types[0].get("coding", [])) if enc_types else ("", "", "")
        )

        # ── Visit period ───────────────────────────────────────────────────
        start_dt, end_dt = get_period_dates(resource.get("period"))

        # ── Practitioner (primary performer) ──────────────────────────────
        practitioner_npi = ""
        for participant in resource.get("participant", []):
            ref = participant.get("individual", {}).get("reference", "")
            if "Practitioner" in ref:
                # Reference format: "Practitioner?identifier=.../us-npi|<NPI>"
                practitioner_npi = ref.split("|")[-1] if "|" in ref else ref
                break

        # ── Service provider (organisation) ───────────────────────────────
        service_provider = resource.get("serviceProvider", {})
        organization_display = service_provider.get("display", "")

        # ── Location ──────────────────────────────────────────────────────
        locations = resource.get("location", [])
        location_display = (
            locations[0].get("location", {}).get("display", "") if locations else ""
        )

        # ── Reason ────────────────────────────────────────────────────────
        reason_codes = resource.get("reasonCode", [])
        _, reason_code, reason_display = (
            get_coding(reason_codes[0].get("coding", [])) if reason_codes else ("", "", "")
        )

        # ── Discharge disposition (inpatient only) ────────────────────────
        hospitalization = resource.get("hospitalization", {})
        discharge_display = ""
        if hospitalization:
            _, _, discharge_display = get_coding(
                hospitalization.get("dischargeDisposition", {}).get("coding", [])
            )

        # ── Patient reference ──────────────────────────────────────────────
        patient_id = get_reference_id(
            resource.get("subject", {}).get("reference", "")
        )

        return {
            "encounter_id": resource.get("id", ""),
            "patient_id": patient_id,
            "status": resource.get("status", ""),
            "class_code": class_code,
            "type_code": type_code,
            "type_display": type_display,
            "start_datetime": start_dt,
            "end_datetime": end_dt,
            "practitioner_npi": practitioner_npi,
            "organization_display": organization_display,
            "location_display": location_display,
            "reason_code": reason_code,
            "reason_display": reason_display,
            "discharge_disposition": discharge_display,
        }

    except Exception as exc:
        logger.warning("Encounter extraction failed (id=%s): %s", resource.get("id"), exc)
        return None
