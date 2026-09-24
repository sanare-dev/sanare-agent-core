"""Validate owner intake proposals before any Brain tool can write them.

This module is deliberately model- and storage-free. A model response is
untrusted data; a validated proposal is still pending owner confirmation.
"""

import hashlib
import re


MAX_INPUT_BYTES = 20_000
MAX_ITEMS = 30
MAX_QUOTE_CHARS = 4_000
KINDS = frozenset({"company", "account", "note", "task"})
SENSITIVE = re.compile(
    r"(?:-----BEGIN [A-Z ]+PRIVATE KEY-----|"
    r"\b(?:api[_ -]?key|password|passwd|secret|token)\s*[:=]\s*\S+|"
    r"\bBearer\s+[A-Za-z0-9._~+/-]{12,}|\bsk-[A-Za-z0-9_-]{16,}|"
    r"\b[A-Z0-9._%+-]+@[A-Z0-9.-]+\.[A-Z]{2,}\b|"
    r"\b(?:\+?\d[\d ()-]{8,}\d)\b)",
    re.IGNORECASE,
)


class IntakeError(ValueError):
    """Error codes only: source text and model output never enter logs."""


def _text(value: object, *, max_chars: int, code: str) -> str:
    if not isinstance(value, str) or not value.strip() or len(value) > max_chars:
        raise IntakeError(code)
    value = value.strip()
    try:
        value.encode("utf-8")
    except UnicodeEncodeError:
        raise IntakeError(code) from None
    return value


def preflight_source(source_text: str, source_ref: str) -> tuple[str, str]:
    """Reject unsafe input before a caller sends any text to a model.

    Returns the normalized source reference and source SHA-256. The heuristic
    check is conservative and cannot replace a full privacy review.
    """
    if not isinstance(source_text, str) or not source_text.strip():
        raise IntakeError("source_empty")
    try:
        encoded = source_text.encode("utf-8")
    except UnicodeEncodeError:
        raise IntakeError("source_invalid_encoding") from None
    if len(encoded) > MAX_INPUT_BYTES:
        raise IntakeError("source_too_large")
    if SENSITIVE.search(source_text):
        raise IntakeError("source_sensitive")
    reference = _text(source_ref, max_chars=500, code="source_ref_invalid")
    if (any(ord(char) < 32 for char in reference) or "?" in reference or
            SENSITIVE.search(reference)):
        raise IntakeError("source_ref_invalid")
    return reference, hashlib.sha256(encoded).hexdigest()


def validate(source_text: str, source_ref: str, model_output: object) -> dict:
    """Return stable pending candidates grounded in exact source spans.

    The caller must run preflight_source before a model call. This validator
    rechecks it after the call and rejects model-added credentials. No
    candidate is permission to mutate the owner's systems.
    """
    reference, source_hash = preflight_source(source_text, source_ref)
    if not isinstance(model_output, dict) or set(model_output) != {"items"}:
        raise IntakeError("output_invalid")
    items = model_output["items"]
    if not isinstance(items, list) or len(items) > MAX_ITEMS:
        raise IntakeError("items_invalid")

    proposals = []
    seen: set[tuple[str, int, int]] = set()
    for item in items:
        if not isinstance(item, dict) or set(item) != {"kind", "title", "quote", "start", "end", "fields"}:
            raise IntakeError("item_invalid")
        kind = item["kind"]
        if not isinstance(kind, str) or kind not in KINDS:
            raise IntakeError("kind_invalid")
        quote = _text(item["quote"], max_chars=MAX_QUOTE_CHARS, code="quote_invalid")
        start, end = item["start"], item["end"]
        if (type(start) is not int or type(end) is not int or start < 0 or
                end <= start or end > len(source_text) or source_text[start:end] != quote):
            raise IntakeError("quote_not_in_source")
        if SENSITIVE.search(quote):
            raise IntakeError("quote_sensitive")
        title = _text(item["title"], max_chars=160, code="title_invalid")
        if title not in quote or SENSITIVE.search(title):
            raise IntakeError("title_not_grounded")
        fields = item["fields"]
        if not isinstance(fields, list) or len(fields) > 12:
            raise IntakeError("fields_invalid")
        clean_fields = []
        labels: set[str] = set()
        for field in fields:
            if not isinstance(field, dict) or set(field) != {"label", "value"}:
                raise IntakeError("field_invalid")
            label = _text(field["label"], max_chars=60, code="field_invalid")
            value = _text(field["value"], max_chars=500, code="field_invalid")
            if label.casefold() in labels or value not in quote or SENSITIVE.search(label + " " + value):
                raise IntakeError("field_not_grounded")
            labels.add(label.casefold())
            clean_fields.append({"label": label, "value": value})
        key = (kind, start, end)
        if key in seen:
            raise IntakeError("duplicate_item")
        seen.add(key)
        stable_id = hashlib.sha256(f"{source_hash}\0{kind}\0{start}\0{end}".encode()).hexdigest()
        proposals.append({
            "id": stable_id,
            "status": "pending_review",
            "kind": kind,
            "title": title,
            "body": quote,
            "fields": clean_fields,
            "source": {"ref": reference, "sha256": source_hash,
                       "start": start, "end": end},
        })
    return {"schema": 1, "source_sha256": source_hash, "proposals": proposals}
