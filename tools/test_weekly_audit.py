"""Offline checks for the read-only audit baseline."""

from datetime import datetime, timezone
import json
from pathlib import Path
import sqlite3

from tools.weekly_audit import build_report


def _registry(path: Path) -> None:
    with sqlite3.connect(path) as conn:
        conn.executescript(
            """
            CREATE TABLE locations (id TEXT, kind TEXT, host TEXT, path TEXT, verified_at TEXT);
            CREATE TABLE services (label TEXT, type TEXT, location TEXT,
                                   purpose TEXT, status TEXT, verified_at TEXT);
            CREATE TABLE databases (id TEXT, path TEXT, host TEXT,
                                    purpose TEXT, verified_at TEXT);
            INSERT INTO locations VALUES ('private-location', 'mac', 'host',
                                          '/private/secret-path', '2026-09-01T00:00:00Z');
            INSERT INTO services VALUES ('service', 'local', 'host', 'purpose',
                                         'active', NULL);
            INSERT INTO databases VALUES ('database', '/private/db', 'host',
                                          '', 'bad-date');
            """
        )


def test_report_counts_without_leaking_source_values(tmp_path: Path) -> None:
    registry = tmp_path / "registry.sqlite"
    documents = tmp_path / "documents.json"
    _registry(registry)
    documents.write_text(
        json.dumps(
            {
                "documents": [
                    {"path": "private-name", "status": "current", "sha256": "a" * 64},
                    {"path": "private-name-2", "status": "existing_unreviewed"},
                ]
            }
        ),
        encoding="utf-8",
    )
    before = (registry.read_bytes(), documents.read_bytes())
    report = build_report(
        registry,
        documents,
        as_of=datetime(2026, 9, 24, tzinfo=timezone.utc),
        max_age_days=14,
    )
    groups = report["registry"]["groups"]
    assert groups["locations"]["stale_verification"] == 1
    assert groups["services"]["unverified"] == 1
    assert groups["databases"]["missing_required"] == 1
    assert groups["databases"]["invalid_verification_date"] == 1
    assert report["documents"]["counts"]["inventory_or_unreviewed"] == 1
    assert report["documents"]["counts"]["missing_hash"] == 1
    assert (registry.read_bytes(), documents.read_bytes()) == before
    rendered = json.dumps(report)
    assert "private-location" not in rendered
    assert "private-name" not in rendered
    assert "/private/" not in rendered


def test_missing_sources_are_unknown_not_zero(tmp_path: Path) -> None:
    report = build_report(
        tmp_path / "missing.sqlite",
        tmp_path / "missing.json",
        as_of=datetime(2026, 9, 24, tzinfo=timezone.utc),
        max_age_days=30,
    )
    assert report["registry"] == {"state": "unavailable", "reason": "source_missing"}
    assert report["documents"] == {"state": "unavailable", "reason": "source_missing"}
    assert "backup_restore" in report["not_measured"]


def test_schema_mismatch_is_unknown(tmp_path: Path) -> None:
    registry = tmp_path / "registry.sqlite"
    with sqlite3.connect(registry) as conn:
        conn.execute("CREATE TABLE locations (id TEXT)")
    documents = tmp_path / "documents.json"
    documents.write_text("{}", encoding="utf-8")
    report = build_report(
        registry,
        documents,
        as_of=datetime(2026, 9, 24, tzinfo=timezone.utc),
        max_age_days=30,
    )
    assert report["registry"]["reason"] == "schema_mismatch"
    assert report["documents"]["reason"] == "schema_mismatch"
