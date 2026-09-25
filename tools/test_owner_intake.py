"""Offline checks for the owner intake validation boundary."""

from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from deep_agent.owner_intake import IntakeError, preflight_source, validate  # noqa: E402


SOURCE = "Компания Example Ltd работает в Лиссабоне. Документы хранятся локально."


def item(quote: str, kind: str = "company") -> dict:
    start = SOURCE.index(quote)
    return {"kind": kind, "title": "Example Ltd" if kind == "company" else "Документы",
            "quote": quote, "start": start, "end": start + len(quote), "fields": []}


class OwnerIntakeTests(unittest.TestCase):
    def test_exact_citations_stable_ids_and_pending_state(self):
        first = item("Компания Example Ltd работает в Лиссабоне.")
        second = item("Документы хранятся локально.", "note")
        result = validate(SOURCE, "braindesk://thread/example", {"items": [first, second]})
        self.assertEqual(len(result["proposals"]), 2)
        self.assertEqual(result["proposals"][0]["status"], "pending_review")
        self.assertEqual(result["proposals"][0]["body"], first["quote"])
        self.assertEqual(result["proposals"][0]["source"]["start"], 0)
        self.assertEqual(result, validate(SOURCE, "braindesk://thread/example", {"items": [first, second]}))

    def test_hallucinated_quote_and_wrong_offset_rejected(self):
        proposal = item("Компания Example Ltd работает в Лиссабоне.")
        proposal["quote"] = "Компания Example Ltd работает в Париже."
        with self.assertRaisesRegex(IntakeError, "quote_not_in_source"):
            validate(SOURCE, "braindesk://thread/example", {"items": [proposal]})
        proposal["quote"] = "Компания Example Ltd работает в Лиссабоне."
        proposal["start"] = 1
        with self.assertRaisesRegex(IntakeError, "quote_not_in_source"):
            validate(SOURCE, "braindesk://thread/example", {"items": [proposal]})

    def test_field_requires_literal_source_and_unique_label(self):
        proposal = item("Компания Example Ltd работает в Лиссабоне.")
        proposal["fields"] = [{"label": "Город", "value": "Порту"}]
        with self.assertRaisesRegex(IntakeError, "field_not_grounded"):
            validate(SOURCE, "braindesk://thread/example", {"items": [proposal]})
        proposal["fields"] = [{"label": "Город", "value": "Лиссабоне"},
                              {"label": "город", "value": "Лиссабоне"}]
        with self.assertRaisesRegex(IntakeError, "field_not_grounded"):
            validate(SOURCE, "braindesk://thread/example", {"items": [proposal]})

    def test_sensitive_input_and_reference_rejected(self):
        with self.assertRaisesRegex(IntakeError, "source_sensitive"):
            preflight_source("api_key = hidden-value", "braindesk://thread/example")
        with self.assertRaisesRegex(IntakeError, "source_ref_invalid"):
            preflight_source(SOURCE, "https://example.com/?token=hidden")
        with self.assertRaisesRegex(IntakeError, "source_sensitive"):
            validate("api_key = hidden-value", "braindesk://thread/example", {"items": []})

    def test_preflight_is_stable_and_rejects_invalid_unicode(self):
        reference, digest = preflight_source(SOURCE, " braindesk://thread/example ")
        self.assertEqual(reference, "braindesk://thread/example")
        self.assertEqual(digest, validate(SOURCE, reference, {"items": []})["source_sha256"])
        with self.assertRaisesRegex(IntakeError, "source_invalid_encoding"):
            preflight_source("broken\ud800", reference)

    def test_caps_shape_and_duplicate_rejected(self):
        proposal = item("Компания Example Ltd работает в Лиссабоне.")
        with self.assertRaisesRegex(IntakeError, "source_too_large"):
            validate("а" * 10001, "braindesk://thread/example", {"items": []})
        with self.assertRaisesRegex(IntakeError, "items_invalid"):
            validate(SOURCE, "braindesk://thread/example", {"items": [proposal] * 31})
        with self.assertRaisesRegex(IntakeError, "duplicate_item"):
            validate(SOURCE, "braindesk://thread/example", {"items": [proposal, proposal]})
        with self.assertRaisesRegex(IntakeError, "item_invalid"):
            validate(SOURCE, "braindesk://thread/example", {"items": [{**proposal, "action": "write"}]})
        with self.assertRaisesRegex(IntakeError, "title_invalid"):
            validate(SOURCE, "braindesk://thread/example", {"items": [{**proposal, "title": "\ud800"}]})


if __name__ == "__main__":
    unittest.main()
