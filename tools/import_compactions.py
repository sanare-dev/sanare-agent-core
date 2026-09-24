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
import stat


DEFAULT_THREADS = Path.home() / "Library/Application Support/BrainDesk/threads"
DEFAULT_PENDING = Path.home() / "Library/Application Support/SanareOrchestrator/owner-inbox/pending"
UUID = re.compile(r"^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$")
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
    key = f"{candidate.thread_id}\0{candidate.compaction_id}\0{candidate.source_sha256}"
    return hashlib.sha256(key.encode("utf-8")).hexdigest() + ".json"


def stage(threads: Path, pending: Path, *, dry_run: bool = False) -> dict[str, int]:
    counts = {"files": 0, "candidates": 0, "new": 0, "existing": 0}
    if not threads.is_dir():
        return counts
    if not dry_run:
        pending.mkdir(mode=0o700, parents=True, exist_ok=True)
        if pending.is_symlink():
            raise OSError("pending directory is a symlink")
        pending.chmod(0o700)
    for path in threads.iterdir():
        counts["files"] += 1
        for candidate in extract_candidates(path):
            counts["candidates"] += 1
            target = pending / candidate_name(candidate)
            if target.exists():
                counts["existing"] += 1
                continue
            if dry_run:
                counts["new"] += 1
                continue
            payload = json.dumps(asdict(candidate), ensure_ascii=False, indent=2) + "\n"
            try:
                fd = os.open(target, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
            except FileExistsError:
                counts["existing"] += 1
                continue
            with os.fdopen(fd, "w", encoding="utf-8") as output:
                output.write(payload)
            counts["new"] += 1
    return counts


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--threads", type=Path, default=DEFAULT_THREADS)
    parser.add_argument("--pending", type=Path, default=DEFAULT_PENDING)
    parser.add_argument("--dry-run", action="store_true")
    options = parser.parse_args()
    result = stage(options.threads, options.pending, dry_run=options.dry_run)
    result["at"] = datetime.now(timezone.utc).isoformat()
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
