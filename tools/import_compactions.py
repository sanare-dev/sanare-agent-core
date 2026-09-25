"""Stage existing Brain Desk compaction summaries for the owner's inbox.

This local Mac tool reads chat files without changing them. It does not call a
model or send data to the bridge. Candidates stay in a private pending folder
until a separate reviewed Brain ingestion path is available.
"""

import argparse
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import re
import sqlite3
import stat


DEFAULT_THREADS = Path.home() / "Library/Application Support/BrainDesk/threads"
DEFAULT_KIMI = (Path.home() / "Library/Application Support/kimi-desktop/daimon-share/daimon/"
                "runtime/kimi-code/home/sessions")
DEFAULT_CLAUDE = Path.home() / ".claude/projects"
DEFAULT_PENDING = Path.home() / "Library/Application Support/SanareOrchestrator/owner-inbox/pending"
UUID = re.compile(r"^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$")
OPENWEBUI_MESSAGE_ID = re.compile(r"^[A-Za-z0-9_-]{1,128}$")
SENSITIVE = re.compile(
    r"(?:-----BEGIN [A-Z ]+PRIVATE KEY-----|\b(?:api[_ -]?key|password|passwd|secret|token)\s*[:=]\s*\S+|"
    r"\bBearer\s+[A-Za-z0-9._~+/-]{12,}|\bsk-[A-Za-z0-9_-]{16,}|"
    r"\b[A-Z0-9._%+-]+@[A-Z0-9.-]+\.[A-Z]{2,}\b|"
    r"\b(?:\+?\d[\d ()-]{8,}\d)\b)",
    re.IGNORECASE,
)
MAX_THREAD_BYTES = 64 * 1024 * 1024
MAX_SUMMARY_CHARS = 12_000


@dataclass(frozen=True)
class Candidate:
    schema: int
    status: str
    source_kind: str
    source: str
    thread_id: str
    compaction_id: str
    source_sha256: str
    observed_at: str
    summary: str


def _candidate(thread: dict, message: dict) -> Candidate | None:
    thread_id = thread.get("thread_id")
    if not isinstance(thread_id, str) or not UUID.fullmatch(thread_id):
        return None
    extra = message.get("additional_kwargs")
    compaction = extra.get("brain_desk_compaction") if isinstance(extra, dict) else None
    if not isinstance(compaction, dict):
        return None
    summary = compaction.get("summary")
    compaction_id = compaction.get("id")
    if (not isinstance(summary, str) or not summary.strip() or
            len(summary) > MAX_SUMMARY_CHARS or
            not isinstance(compaction_id, str) or not compaction_id.strip()):
        return None
    # Heuristic protection only: every candidate still requires review before
    # being sent to Brain or promoted to shared memory.
    if SENSITIVE.search(summary):
        return None
    observed_at = thread.get("updated_at")
    if not isinstance(observed_at, str):
        return None
    try:
        datetime.fromisoformat(observed_at.replace("Z", "+00:00"))
    except ValueError:
        return None
    digest = hashlib.sha256(summary.encode("utf-8")).hexdigest()
    return Candidate(
        schema=1,
        status="pending_review",
        source_kind="braindesk",
        source=f"braindesk://thread/{thread_id}#compaction={compaction_id}",
        thread_id=thread_id,
        compaction_id=compaction_id,
        source_sha256=digest,
        observed_at=observed_at,
        summary=summary,
    )


def read_thread(path: Path) -> dict | None:
    if not UUID.fullmatch(path.stem) or path.suffix != ".json":
        return None
    try:
        # Open the file without following a final symlink, then validate the
        # opened descriptor. This also closes the stat/read replacement gap.
        fd = os.open(path, os.O_RDONLY | os.O_NOFOLLOW)
        with os.fdopen(fd, "rb") as input_file:
            metadata = os.fstat(input_file.fileno())
            if not stat.S_ISREG(metadata.st_mode) or metadata.st_size > MAX_THREAD_BYTES:
                return None
            raw = input_file.read(MAX_THREAD_BYTES + 1)
        if len(raw) > MAX_THREAD_BYTES:
            return None
        data = json.loads(raw.decode("utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError):
        return None
    if not isinstance(data, dict) or data.get("thread_id") != path.stem:
        return None
    return data


def extract_candidates(path: Path) -> list[Candidate]:
    thread = read_thread(path)
    if thread is None or not isinstance(thread.get("messages"), list):
        return []
    found: dict[tuple[str, str], Candidate] = {}
    for message in thread["messages"]:
        if not isinstance(message, dict) or message.get("type") != "ai":
            continue
        candidate = _candidate(thread, message)
        if candidate is not None:
            found[(candidate.compaction_id, candidate.source_sha256)] = candidate
    return list(found.values())


def candidate_name(candidate: Candidate) -> str:
    key = f"{candidate.source_kind}\0{candidate.thread_id}\0{candidate.compaction_id}\0{candidate.source_sha256}"
    return hashlib.sha256(key.encode("utf-8")).hexdigest() + ".json"


def _prepare_pending(pending: Path, dry_run: bool) -> None:
    if dry_run:
        return
    pending.mkdir(mode=0o700, parents=True, exist_ok=True)
    if pending.is_symlink():
        raise OSError("pending directory is a symlink")
    pending.chmod(0o700)


def _already_seen(pending: Path, name: str) -> bool:
    """A candidate already staged, or already reviewed and moved out (#238):
    either way this run must not recreate it in the pending queue."""
    if (pending / name).exists():
        return True
    reviewed = pending.parent / "reviewed"
    return any((reviewed / decision / name).exists() for decision in DECISIONS)


def _stage_candidate(candidate: Candidate, pending: Path, counts: dict[str, int], dry_run: bool) -> None:
    counts["candidates"] += 1
    name = candidate_name(candidate)
    if _already_seen(pending, name):
        counts["existing"] += 1
        return
    target = pending / name
    if dry_run:
        counts["new"] += 1
        return
    payload = json.dumps(asdict(candidate), ensure_ascii=False, indent=2) + "\n"
    try:
        fd = os.open(target, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    except FileExistsError:
        counts["existing"] += 1
        return
    with os.fdopen(fd, "w", encoding="utf-8") as output:
        output.write(payload)
    counts["new"] += 1


CANDIDATE_ID = re.compile(r"^[0-9a-f]{64}$")
DECISIONS = ("written", "rejected")


def list_pending(pending: Path) -> list[dict]:
    """Read every staged candidate for an external reader (Brain Desk #238).

    Read-only: parses each `pending/<id>.json`, adds its filename stem as
    `id`, and drops anything that fails to parse as one of our own records
    instead of raising, so one corrupt file cannot break the whole listing.
    Sorted oldest first, matching the order candidates were observed.
    """
    if not pending.is_dir() or pending.is_symlink():
        return []
    found: list[dict] = []
    for path in sorted(pending.glob("*.json")):
        if not CANDIDATE_ID.fullmatch(path.stem) or path.is_symlink():
            continue
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, UnicodeError, json.JSONDecodeError):
            continue
        if not isinstance(data, dict) or data.get("status") != "pending_review":
            continue
        found.append({"id": path.stem, **data})
    found.sort(key=lambda c: str(c.get("observed_at") or ""))
    return found


def resolve_pending(pending: Path, candidate_id: str, decision: str) -> bool:
    """Move a reviewed candidate out of the pending queue (Brain Desk #238).

    Called once the owner's tick has gone through the existing memory adapter
    (`decision="written"`) or been declined (`decision="rejected"`); either
    way the file leaves `pending/` so it is never re-shown or re-imported,
    while the record itself is kept under `reviewed/<decision>/` for audit.
    Returns False (no exception) when the id is invalid or already resolved,
    since a repeated resolve of the same id is a normal race, not an error.
    """
    if decision not in DECISIONS or not CANDIDATE_ID.fullmatch(candidate_id):
        return False
    source = pending / f"{candidate_id}.json"
    if not source.is_file() or source.is_symlink():
        return False
    destination_dir = pending.parent / "reviewed" / decision
    destination_dir.mkdir(mode=0o700, parents=True, exist_ok=True)
    if destination_dir.is_symlink():
        raise OSError("reviewed directory is a symlink")
    destination = destination_dir / source.name
    try:
        os.replace(source, destination)
    except FileNotFoundError:
        return False
    return True


def stage(threads: Path, pending: Path, *, dry_run: bool = False) -> dict[str, int]:
    counts = {"files": 0, "candidates": 0, "new": 0, "existing": 0}
    if not threads.is_dir():
        return counts
    _prepare_pending(pending, dry_run)
    for path in threads.iterdir():
        counts["files"] += 1
        for candidate in extract_candidates(path):
            _stage_candidate(candidate, pending, counts, dry_run)
    return counts


def _read_jsonl_lines(path: Path) -> list[bytes]:
    try:
        fd = os.open(path, os.O_RDONLY | os.O_NOFOLLOW)
        with os.fdopen(fd, "rb") as input_file:
            metadata = os.fstat(input_file.fileno())
            if not stat.S_ISREG(metadata.st_mode) or metadata.st_size > MAX_THREAD_BYTES:
                return []
            raw = input_file.read(MAX_THREAD_BYTES + 1)
        if len(raw) > MAX_THREAD_BYTES:
            return []
        return raw.splitlines()
    except OSError:
        return []


def _has_symlink_parent(path: Path, root: Path) -> bool:
    parent = path.parent
    while parent != root:
        if parent.is_symlink() or parent == parent.parent:
            return True
        parent = parent.parent
    return False


def extract_kimi_candidates(path: Path, root: Path) -> list[Candidate]:
    if path.name != "wire.jsonl":
        return []
    try:
        relative = path.relative_to(root)
    except ValueError:
        return []
    if _has_symlink_parent(path, root):
        return []
    session_id = hashlib.sha256(str(relative.parent).encode("utf-8")).hexdigest()
    found: list[Candidate] = []
    for ordinal, raw_line in enumerate(_read_jsonl_lines(path), start=1):
        if len(raw_line) > 1_000_000 or b'"context.apply_compaction"' not in raw_line:
            continue
        try:
            event = json.loads(raw_line)
        except (UnicodeError, json.JSONDecodeError):
            continue
        if not isinstance(event, dict) or event.get("type") != "context.apply_compaction":
            continue
        summary = event.get("summary")
        timestamp = event.get("time")
        if (not isinstance(summary, str) or not summary.strip() or
                len(summary) > MAX_SUMMARY_CHARS or SENSITIVE.search(summary) or
                not isinstance(timestamp, int) or timestamp < 1_500_000_000_000):
            continue
        try:
            observed_at = datetime.fromtimestamp(timestamp / 1000, timezone.utc).isoformat()
        except (ValueError, OverflowError, OSError):
            continue
        found.append(Candidate(
            schema=1,
            status="pending_review",
            source_kind="kimi",
            source=f"kimi://{relative.as_posix()}#compaction={ordinal}",
            thread_id=session_id,
            compaction_id=str(ordinal),
            source_sha256=hashlib.sha256(summary.encode("utf-8")).hexdigest(),
            observed_at=observed_at,
            summary=summary,
        ))
    return found


def stage_kimi(root: Path, pending: Path, *, dry_run: bool = False) -> dict[str, int]:
    counts = {"files": 0, "candidates": 0, "new": 0, "existing": 0}
    if not root.is_dir() or root.is_symlink():
        return counts
    _prepare_pending(pending, dry_run)
    for path in root.rglob("wire.jsonl"):
        counts["files"] += 1
        for candidate in extract_kimi_candidates(path, root):
            _stage_candidate(candidate, pending, counts, dry_run)
    return counts


def extract_claude_candidates(path: Path, root: Path) -> list[Candidate]:
    if path.suffix != ".jsonl" or path.parent.parent != root:
        return []
    try:
        relative = path.relative_to(root)
    except ValueError:
        return []
    if _has_symlink_parent(path, root):
        return []
    session_id = hashlib.sha256(str(relative).encode("utf-8")).hexdigest()
    found: list[Candidate] = []
    for ordinal, raw_line in enumerate(_read_jsonl_lines(path), start=1):
        if len(raw_line) > 1_000_000 or b'"isCompactSummary"' not in raw_line:
            continue
        try:
            event = json.loads(raw_line)
        except (UnicodeError, json.JSONDecodeError):
            continue
        if (not isinstance(event, dict) or event.get("type") != "user" or
                event.get("isCompactSummary") is not True):
            continue
        message = event.get("message")
        summary = message.get("content") if isinstance(message, dict) else None
        observed_at = event.get("timestamp")
        if (not isinstance(summary, str) or not summary.strip() or
                len(summary) > 30_000 or SENSITIVE.search(summary) or
                not isinstance(observed_at, str)):
            continue
        try:
            datetime.fromisoformat(observed_at.replace("Z", "+00:00"))
        except ValueError:
            continue
        found.append(Candidate(
            schema=1,
            status="pending_review",
            source_kind="claude",
            source=f"claude://{relative.as_posix()}#compaction={ordinal}",
            thread_id=session_id,
            compaction_id=str(ordinal),
            source_sha256=hashlib.sha256(summary.encode("utf-8")).hexdigest(),
            observed_at=observed_at,
            summary=summary,
        ))
    return found


def stage_claude(root: Path, pending: Path, *, dry_run: bool = False) -> dict[str, int]:
    counts = {"files": 0, "candidates": 0, "new": 0, "existing": 0}
    if not root.is_dir() or root.is_symlink():
        return counts
    _prepare_pending(pending, dry_run)
    for path in root.glob("*/*.jsonl"):
        counts["files"] += 1
        for candidate in extract_claude_candidates(path, root):
            _stage_candidate(candidate, pending, counts, dry_run)
    return counts


def stage_openwebui(database: Path, pending: Path, *, dry_run: bool = False) -> dict[str, int]:
    """Stage only saved context summaries from an Open WebUI SQLite snapshot."""
    counts = {"files": 0, "candidates": 0, "new": 0, "existing": 0}
    if database.is_symlink() or not database.is_file():
        return counts
    _prepare_pending(pending, dry_run)
    try:
        # Immutable mode reads a consistent, closed snapshot and never writes
        # SQLite sidecar files into the source directory.
        connection = sqlite3.connect(database.as_uri() + "?mode=ro&immutable=1", uri=True)
        with connection:
            rows = connection.execute(
                "SELECT id, chat_id, updated_at, context_summary "
                "FROM chat_message WHERE context_summary IS NOT NULL"
            )
            counts["files"] = 1
            for message_id, chat_id, timestamp, summary in rows:
                if (not isinstance(message_id, str) or not OPENWEBUI_MESSAGE_ID.fullmatch(message_id) or
                        not isinstance(chat_id, str) or not UUID.fullmatch(chat_id) or
                        not isinstance(summary, str) or not summary.strip() or
                        len(summary) > MAX_SUMMARY_CHARS or SENSITIVE.search(summary) or
                        not isinstance(timestamp, int) or timestamp < 1_500_000_000):
                    continue
                try:
                    observed_at = datetime.fromtimestamp(timestamp, timezone.utc).isoformat()
                except (ValueError, OverflowError, OSError):
                    continue
                candidate = Candidate(
                    schema=1,
                    status="pending_review",
                    source_kind="openwebui",
                    source=f"openwebui://chat/{chat_id}#message={message_id}",
                    thread_id=chat_id,
                    compaction_id=message_id,
                    source_sha256=hashlib.sha256(summary.encode("utf-8")).hexdigest(),
                    observed_at=observed_at,
                    summary=summary,
                )
                _stage_candidate(candidate, pending, counts, dry_run)
    except sqlite3.Error:
        # The source may be an older Open WebUI schema or a broken snapshot.
        # Never fall back to raw chat text.
        return counts
    finally:
        if "connection" in locals():
            connection.close()
    return counts


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--threads", type=Path, default=DEFAULT_THREADS)
    parser.add_argument("--kimi-root", type=Path, default=DEFAULT_KIMI)
    parser.add_argument("--claude-root", type=Path, default=DEFAULT_CLAUDE)
    parser.add_argument("--openwebui-db", type=Path,
                        help="closed Open WebUI SQLite snapshot with chat_message.context_summary")
    parser.add_argument("--pending", type=Path, default=DEFAULT_PENDING)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--source", choices=("all", "braindesk", "kimi", "claude", "openwebui"), default="all")
    parser.add_argument("--list-pending", action="store_true",
                        help="print the pending queue as JSON for an external reader (Brain Desk #238)")
    parser.add_argument("--resolve-pending", metavar="ID",
                        help="move one candidate out of the pending queue once it has been reviewed")
    parser.add_argument("--decision", choices=DECISIONS,
                        help="outcome for --resolve-pending: written or rejected")
    options = parser.parse_args()
    if options.list_pending:
        print(json.dumps(list_pending(options.pending), ensure_ascii=False, sort_keys=True))
        return 0
    if options.resolve_pending:
        if not options.decision:
            parser.error("--resolve-pending requires --decision")
        moved = resolve_pending(options.pending, options.resolve_pending, options.decision)
        print(json.dumps({"moved": moved}, sort_keys=True))
        return 0 if moved else 1
    if options.source == "openwebui" and options.openwebui_db is None:
        parser.error("--source openwebui requires --openwebui-db")
    result = {"files": 0, "candidates": 0, "new": 0, "existing": 0}
    for source, run in (("braindesk", lambda: stage(options.threads, options.pending, dry_run=options.dry_run)),
                        ("kimi", lambda: stage_kimi(options.kimi_root, options.pending, dry_run=options.dry_run)),
                        ("claude", lambda: stage_claude(options.claude_root, options.pending, dry_run=options.dry_run))):
        if options.source in ("all", source):
            counts = run()
            for key, value in counts.items():
                result[key] += value
    if options.openwebui_db is not None and options.source in ("all", "openwebui"):
        counts = stage_openwebui(options.openwebui_db, options.pending, dry_run=options.dry_run)
        for key, value in counts.items():
            result[key] += value
    result["at"] = datetime.now(timezone.utc).isoformat()
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
