"""Read-only baseline for the owner's weekly evidence audit.

This CLI deliberately reports aggregates only. Registry rows and document paths
can contain private information, so neither is copied into its output.
"""

from __future__ import annotations

import argparse
from contextlib import closing
from datetime import datetime, timedelta, timezone
import json
from pathlib import Path
import sqlite3


REGISTRY_FIELDS = {
    "locations": ("id", "kind", "host", "path", "verified_at"),
    "services": ("label", "type", "location", "purpose", "status", "verified_at"),
    "databases": ("id", "path", "host", "purpose", "verified_at"),
}


def _present(value: object) -> bool:
    return isinstance(value, str) and bool(value.strip())


def _date(value: object) -> datetime | None:
    if not isinstance(value, str) or not value.strip():
        return None
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        return None
    if parsed.tzinfo is None:
        return None
    return parsed.astimezone(timezone.utc)


def audit_registry(path: Path, cutoff: datetime) -> dict[str, object]:
    """Inspect the existing SQLite catalog without creating or changing it."""
    if not path.is_file():
        return {"state": "unavailable", "reason": "source_missing"}
    try:
        uri = f"{path.resolve().as_uri()}?mode=ro"
        with closing(sqlite3.connect(uri, uri=True)) as conn:
            conn.row_factory = sqlite3.Row
            tables = {
                row[0]
                for row in conn.execute("SELECT name FROM sqlite_master WHERE type='table'")
            }
            if not set(REGISTRY_FIELDS).issubset(tables):
                return {"state": "unavailable", "reason": "schema_mismatch"}
            groups: dict[str, object] = {}
            for table, required in REGISTRY_FIELDS.items():
                columns = {row[1] for row in conn.execute(f"PRAGMA table_info({table})")}
                if not set(required).issubset(columns):
                    return {"state": "unavailable", "reason": "schema_mismatch"}
                counts = {
                    "total": 0,
                    "missing_required": 0,
                    "unverified": 0,
                    "stale_verification": 0,
                    "invalid_verification_date": 0,
                }
                for row in conn.execute(f"SELECT {', '.join(required)} FROM {table}"):
                    counts["total"] += 1
                    if any(not _present(row[field]) for field in required):
                        counts["missing_required"] += 1
                    raw_date = row["verified_at"]
                    if not _present(raw_date):
                        counts["unverified"] += 1
                    elif (verified := _date(raw_date)) is None:
                        counts["invalid_verification_date"] += 1
                    elif verified < cutoff:
                        counts["stale_verification"] += 1
                groups[table] = counts
            return {"state": "observed", "groups": groups}
    except (OSError, sqlite3.Error):
        return {"state": "unavailable", "reason": "source_unreadable"}


def audit_documents(path: Path) -> dict[str, object]:
    """Count catalog metadata, without reading any registered document."""
    if not path.is_file():
        return {"state": "unavailable", "reason": "source_missing"}
    try:
        catalog = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, ValueError):
        return {"state": "unavailable", "reason": "source_unreadable"}
    if not isinstance(catalog, dict) or not isinstance(catalog.get("documents"), list):
        return {"state": "unavailable", "reason": "schema_mismatch"}
    documents = catalog["documents"]
    counts = {
        "total": len(documents),
        "missing_path": 0,
        "missing_status": 0,
        "missing_hash": 0,
        "inventory_or_unreviewed": 0,
    }
    for item in documents:
        if not isinstance(item, dict):
            counts["missing_path"] += 1
            counts["missing_status"] += 1
            counts["missing_hash"] += 1
            continue
        for key, output_key in (
            ("path", "missing_path"),
            ("status", "missing_status"),
            ("sha256", "missing_hash"),
        ):
            if not _present(item.get(key)):
                counts[output_key] += 1
        status = item.get("status")
        if isinstance(status, str) and status in {"inventory_only", "existing_unreviewed"}:
            counts["inventory_or_unreviewed"] += 1
    return {"state": "observed", "counts": counts}


def build_report(
    registry: Path, documents: Path, *, as_of: datetime, max_age_days: int
) -> dict[str, object]:
    as_of = as_of.astimezone(timezone.utc)
    cutoff = as_of - timedelta(days=max_age_days)
    return {
        "schema": 1,
        "as_of": as_of.isoformat(),
        "verification_cutoff": cutoff.isoformat(),
        "registry": audit_registry(registry, cutoff),
        "documents": audit_documents(documents),
        "not_measured": [
            "contradictory_knowledge",
            "unused_skills",
            "repeated_errors",
            "backup_restore",
            "memory_usage_in_answers",
            "chat_summary_coverage",
        ],
        "scope": "Metadata completeness and verification age only; no content or live health check",
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--registry", required=True, type=Path)
    parser.add_argument("--documents", required=True, type=Path)
    parser.add_argument("--as-of", help="UTC date YYYY-MM-DD (default: current UTC day)")
    parser.add_argument("--max-age-days", type=int, default=30)
    args = parser.parse_args()
    if args.max_age_days < 1:
        parser.error("--max-age-days must be positive")
    try:
        as_of = (
            datetime.strptime(args.as_of, "%Y-%m-%d").replace(tzinfo=timezone.utc)
            if args.as_of
            else datetime.now(timezone.utc)
        )
    except ValueError:
        parser.error("--as-of must be an ISO date")
    report = build_report(
        args.registry, args.documents, as_of=as_of, max_age_days=args.max_age_days
    )
    print(json.dumps(report, ensure_ascii=False, sort_keys=True))
    return 0 if all(report[key]["state"] == "observed" for key in ("registry", "documents")) else 2


if __name__ == "__main__":
    raise SystemExit(main())
