"""Shared FHIR extraction utilities used by all parser modules.

All functions are pure (no side-effects, no state) so each parser
can import only what it needs and remain independently testable.
"""

from __future__ import annotations

from typing import Any


def get_coding(coding_list: list[dict]) -> tuple[str, str, str]:
    """Return (system, code, display) from the first entry in a FHIR coding array.

    Returns three empty strings when the list is absent or empty.
    """
    if not coding_list:
        return ("", "", "")
    first = coding_list[0]
    return (
        first.get("system", ""),
        first.get("code", ""),
        first.get("display", ""),
    )


def get_reference_id(reference_str: str | None) -> str:
    """Strip the urn:uuid: prefix from a FHIR reference and return the bare UUID.

    Also handles relative references like "Patient/abc-123" by returning the
    last path segment.
    """
    if not reference_str:
        return ""
    ref = reference_str.replace("urn:uuid:", "")
    return ref.split("/")[-1]


def get_extension(extensions: list[dict], url_fragment: str) -> dict | None:
    """Return the first extension whose URL contains url_fragment, or None."""
    for ext in extensions or []:
        if url_fragment in ext.get("url", ""):
            return ext
    return None


def get_extension_text(extensions: list[dict], url_fragment: str) -> str:
    """Return the 'text' sub-extension valueString for a US Core-style extension.

    Used for race, ethnicity, and similar coded extensions that carry a
    human-readable 'text' child alongside the machine-readable ombCategory.
    """
    ext = get_extension(extensions, url_fragment)
    if ext is None:
        return ""
    for sub in ext.get("extension", []):
        if sub.get("url") == "text":
            return sub.get("valueString", "")
    # Fallback: try direct valueString on the extension itself
    return ext.get("valueString", "")


def get_period_dates(period: dict | None) -> tuple[str, str]:
    """Return (start, end) ISO datetime strings from a FHIR period object.

    Both values default to empty string when absent.
    """
    if not period:
        return ("", "")
    return (period.get("start", ""), period.get("end", ""))


def get_quantity(quantity: dict | None) -> tuple[float | None, str]:
    """Return (numeric_value, unit_string) from a FHIR valueQuantity object."""
    if not quantity:
        return (None, "")
    raw = quantity.get("value")
    value = float(raw) if raw is not None else None
    unit = quantity.get("unit") or quantity.get("code", "")
    return (value, unit)


def get_first_identifier_by_type(
    identifiers: list[dict], type_code: str
) -> str:
    """Find the first identifier whose type.coding contains type_code and return its value."""
    for ident in identifiers or []:
        for coding in ident.get("type", {}).get("coding", []):
            if coding.get("code") == type_code:
                return ident.get("value", "")
    return ""
