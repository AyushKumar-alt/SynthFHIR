"""Patient resource extractor."""

from __future__ import annotations

import logging
from typing import Any

from .base import (
    get_coding,
    get_extension,
    get_extension_text,
    get_first_identifier_by_type,
    get_reference_id,
)

logger = logging.getLogger(__name__)


def extract_patient_row(resource: dict) -> dict[str, Any] | None:
    """Extract a flat row dict from a FHIR Patient resource.

    Args:
        resource: Raw FHIR Patient resource dict from a bundle entry.

    Returns:
        Flat dict suitable for a DataFrame row, or None if extraction fails.
    """
    try:
        patient_id = resource.get("id", "")
        extensions = resource.get("extension", [])

        # ── Demographics from US Core extensions ──────────────────────────
        race = get_extension_text(extensions, "us-core-race")
        ethnicity = get_extension_text(extensions, "us-core-ethnicity")

        birth_sex_ext = get_extension(extensions, "us-core-birthsex")
        birth_sex = birth_sex_ext.get("valueCode", "") if birth_sex_ext else ""

        birth_place_ext = get_extension(extensions, "patient-birthPlace")
        birth_place_city, birth_place_state = "", ""
        if birth_place_ext:
            addr = birth_place_ext.get("valueAddress", {})
            birth_place_city = addr.get("city", "")
            birth_place_state = addr.get("state", "")

        # ── Synthea-specific quality metrics ──────────────────────────────
        daly_ext = get_extension(extensions, "disability-adjusted-life-years")
        qaly_ext = get_extension(extensions, "quality-adjusted-life-years")
        daly = daly_ext.get("valueDecimal") if daly_ext else None
        qaly = qaly_ext.get("valueDecimal") if qaly_ext else None

        # ── Name ──────────────────────────────────────────────────────────
        names = resource.get("name", [])
        official = next(
            (n for n in names if n.get("use") == "official"),
            names[0] if names else {},
        )
        family_name = official.get("family", "")
        given_name = " ".join(official.get("given", []))

        # ── Address + geo-coordinates ─────────────────────────────────────
        addresses = resource.get("address", [])
        addr = addresses[0] if addresses else {}
        city = addr.get("city", "")
        state = addr.get("state", "")
        postal_code = addr.get("postalCode", "")
        country = addr.get("country", "")

        lat: float | None = None
        lon: float | None = None
        for addr_ext in addr.get("extension", []):
            if "geolocation" in addr_ext.get("url", ""):
                for geo in addr_ext.get("extension", []):
                    if geo.get("url") == "latitude":
                        lat = geo.get("valueDecimal")
                    elif geo.get("url") == "longitude":
                        lon = geo.get("valueDecimal")

        # ── Marital status ────────────────────────────────────────────────
        marital = resource.get("maritalStatus", {})
        marital_codings = marital.get("coding", [])
        marital_code = marital_codings[0].get("code", "") if marital_codings else ""
        marital_text = marital.get("text", "")

        # ── Identifiers ───────────────────────────────────────────────────
        identifiers = resource.get("identifier", [])
        ssn = get_first_identifier_by_type(identifiers, "SS")
        drivers_license = get_first_identifier_by_type(identifiers, "DL")
        passport = get_first_identifier_by_type(identifiers, "PPN")

        # ── Language ──────────────────────────────────────────────────────
        communications = resource.get("communication", [])
        language = ""
        if communications:
            lang_codings = communications[0].get("language", {}).get("coding", [])
            if lang_codings:
                language = lang_codings[0].get("display", "")

        return {
            "patient_id": patient_id,
            "family_name": family_name,
            "given_name": given_name,
            "gender": resource.get("gender", ""),
            "birth_date": resource.get("birthDate", ""),
            "deceased_datetime": resource.get("deceasedDateTime", ""),
            "race": race,
            "ethnicity": ethnicity,
            "birth_sex": birth_sex,
            "marital_code": marital_code,
            "marital_status": marital_text,
            "city": city,
            "state": state,
            "postal_code": postal_code,
            "country": country,
            "lat": lat,
            "lon": lon,
            "birth_place_city": birth_place_city,
            "birth_place_state": birth_place_state,
            "language": language,
            "ssn": ssn,
            "drivers_license": drivers_license,
            "passport": passport,
            "daly": daly,
            "qaly": qaly,
        }

    except Exception as exc:
        logger.warning("Patient extraction failed (id=%s): %s", resource.get("id"), exc)
        return None
