"""Read-only weekly discovery of public SKILL.md candidates for Brain.

This tool never installs, executes, or approves a discovered skill. GitHub
responses are untrusted metadata; the owner-facing output is a review queue.
"""

from __future__ import annotations

import argparse
import base64
import json
import os
import re
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.parse import quote
from urllib.request import Request, urlopen


API_BASE = "https://api.github.com"
SOURCES = (
    ("anthropics/skills", "skills/"),
    ("openai/skills", "skills/"),
    ("openclaw/agent-skills", "skills/"),
    ("NousResearch/hermes-agent", "skills/"),
)
TASK_TERMS = {
    "сайты и релизы": ("site", "website", "deploy", "vercel", "wordpress"),
    "магазины": ("amazon", "shopify", "commerce", "store", "retail"),
    "деньги и налоги": ("tax", "finance", "payment", "invoice", "accounting"),
    "агенты и процессы": ("agent", "workflow", "automation", "ops", "mcp"),
    "память и документы": ("memory", "document", "pdf", "knowledge", "research"),
    "безопасность": ("security", "audit", "secret", "backup", "privacy"),
}
MAX_RESPONSE_BYTES = 5_000_000
MAX_SKILL_BYTES = 64_000


class ScoutError(RuntimeError):
    """A public source could not be inspected safely or completely."""


class GitHubClient:
    def __init__(self, token: str | None = None):
        self.token = token

    def get(self, path: str) -> dict[str, Any]:
        if not path.startswith("/repos/") or ".." in path:
            raise ScoutError("invalid GitHub API path")
        headers = {"Accept": "application/vnd.github+json", "User-Agent": "brain-skill-scout"}
        if self.token:
            headers["Authorization"] = f"Bearer {self.token}"
        request = Request(API_BASE + path, headers=headers)
        try:
            with urlopen(request, timeout=15) as response:
                payload = response.read(MAX_RESPONSE_BYTES + 1)
        except (HTTPError, URLError, TimeoutError) as exc:
            # Never include response bodies or request headers in reports.
            status = getattr(exc, "code", "network")
            raise ScoutError(f"GitHub API unavailable ({status})") from None
        if len(payload) > MAX_RESPONSE_BYTES:
            raise ScoutError("GitHub API response exceeds size limit")
        try:
            result = json.loads(payload)
        except (ValueError, UnicodeDecodeError):
            raise ScoutError("GitHub API returned invalid JSON") from None
        if not isinstance(result, dict):
            raise ScoutError("GitHub API returned an unexpected shape")
        return result


@dataclass(frozen=True)
class SourceSnapshot:
    repo: str
    commit: str
    pushed_at: str
    repo_license: str
    paths: tuple[str, ...]


def _safe_text(value: object, limit: int = 180) -> str:
    text = value if isinstance(value, str) else ""
    return re.sub(r"[\x00-\x1f\x7f]", " ", text).strip()[:limit]


def _frontmatter(markdown: str) -> tuple[str, str]:
    lines = markdown.splitlines()
    if not lines or lines[0].strip() != "---":
        return "", ""
    fields: dict[str, str] = {}
    for line in lines[1:81]:
        if line.strip() == "---":
            break
        match = re.match(r"^(name|description):\s*(.*)$", line)
        if match:
            fields[match.group(1)] = match.group(2).strip().strip("\"'")
    return _safe_text(fields.get("name"), 80), _safe_text(fields.get("description"))


def _contents(client: GitHubClient, repo: str, path: str, commit: str) -> str:
    encoded_path = "/".join(quote(part, safe="") for part in path.split("/"))
    response = client.get(f"/repos/{repo}/contents/{encoded_path}?ref={commit}")
    if response.get("encoding") != "base64" or not isinstance(response.get("content"), str):
        raise ScoutError("SKILL.md content unavailable")
    try:
        raw = base64.b64decode(response["content"], validate=False)
        if len(raw) > MAX_SKILL_BYTES:
            raise ScoutError("SKILL.md exceeds size limit")
        return raw.decode("utf-8")
    except (ValueError, UnicodeDecodeError):
        raise ScoutError("SKILL.md content is invalid") from None


def _source(client: GitHubClient, repo: str, root: str) -> SourceSnapshot:
    metadata = client.get(f"/repos/{repo}")
    if metadata.get("archived"):
        raise ScoutError("repository archived")
    branch = metadata.get("default_branch")
    if not isinstance(branch, str) or not branch:
        raise ScoutError("default branch unavailable")
    branch_data = client.get(f"/repos/{repo}/branches/{quote(branch, safe='')}")
    commit = (branch_data.get("commit") or {}).get("sha")
    if not isinstance(commit, str) or not re.fullmatch(r"[0-9a-f]{40}", commit):
        raise ScoutError("commit SHA unavailable")
    tree = client.get(f"/repos/{repo}/git/trees/{commit}?recursive=1")
    if tree.get("truncated"):
        raise ScoutError("repository tree truncated")
    entries = tree.get("tree")
    if not isinstance(entries, list):
        raise ScoutError("repository tree unavailable")
    paths = tuple(
        entry["path"] for entry in entries
        if isinstance(entry, dict) and entry.get("type") == "blob"
        and isinstance(entry.get("path"), str) and entry["path"].startswith(root)
    )
    license_data = metadata.get("license") or {}
    license_id = license_data.get("spdx_id") if isinstance(license_data, dict) else None
    return SourceSnapshot(repo, commit, _safe_text(metadata.get("pushed_at"), 40),
                          _safe_text(license_id, 40) or "UNKNOWN", paths)


def _matches(text: str) -> list[str]:
    normalized = text.lower()
    return [task for task, terms in TASK_TERMS.items() if any(term in normalized for term in terms)]


def scout(client: GitHubClient, *, per_source: int = 5, max_cards: int = 20) -> dict[str, Any]:
    cards: list[dict[str, Any]] = []
    errors: list[dict[str, str]] = []
    sources: list[dict[str, str | int]] = []
    for repo, root in SOURCES:
        try:
            snapshot = _source(client, repo, root)
            paths = [path for path in snapshot.paths if path.endswith("/SKILL.md")
                     and not any(part.startswith(".") for part in path.split("/"))]
            ranked = sorted(paths, key=lambda path: (-len(_matches(path)), path))
            sources.append({"repo": repo, "commit": snapshot.commit,
                            "pushed_at": snapshot.pushed_at, "skills_found": len(paths)})
            for path in ranked[:per_source]:
                try:
                    name, description = _frontmatter(_contents(client, repo, path, snapshot.commit))
                except ScoutError as exc:
                    errors.append({"source": f"{repo}/{path}", "error": str(exc)})
                    continue
                task_matches = _matches(f"{path} {name} {description}")
                if not task_matches:
                    continue
                directory = path.rsplit("/", 1)[0] + "/"
                local_license = next((item for item in snapshot.paths if item.startswith(directory)
                                      and item[len(directory):].lower() in
                                      {"license", "license.md", "license.txt"}), None)
                has_scripts = any(item.startswith(directory + "scripts/") for item in snapshot.paths)
                cards.append({
                    "task": task_matches[0], "name": name or path.split("/")[-2],
                    "description": description or "Описание в frontmatter недоступно",
                    "source": repo, "path": path, "commit": snapshot.commit,
                    "url": f"https://github.com/{repo}/blob/{snapshot.commit}/{path}",
                    "license": "needs_per_skill_review" if local_license else snapshot.repo_license,
                    "license_scope": "skill_file" if local_license else "repository_only",
                    "risk": "Есть scripts; нужен просмотр кода" if has_scripts else
                            "Внешние инструкции; нужен ручной аудит",
                    "test": "not_run", "state": "candidate_unreviewed",
                })
        except ScoutError as exc:
            errors.append({"source": repo, "error": str(exc)})
    cards.sort(key=lambda card: (card["task"], card["source"], card["name"]))
    return {"schema": 1, "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "mode": "public_metadata_only", "sources": sources,
            "candidates": cards[:max_cards], "errors": errors,
            "note": "Ни один навык не установлен и не запущен. Лицензия и безопасность каждой карточки требуют ручной проверки."}


def markdown_report(result: dict[str, Any]) -> str:
    rows = ["# Кандидаты навыков Brain", "",
            "Только публичные метаданные; навыки не установлены и не запускались.", "",
            "| Задача владельца | Навык | Источник | Лицензия | Риск и тест |",
            "| --- | --- | --- | --- | --- |"]
    for card in result["candidates"]:
        safe = lambda value: str(value).replace("|", "\\|").replace("\n", " ")
        rows.append(f"| {safe(card['task'])} | [{safe(card['name'])}]({card['url']}) | "
                    f"{safe(card['source'])}@{card['commit'][:8]} | {safe(card['license'])} "
                    f"({safe(card['license_scope'])}) | {safe(card['risk'])}; тест не запускался |")
    if result["errors"]:
        rows.extend(["", "## Неполные источники", ""])
        for error in result["errors"]:
            rows.append(f"- {error['source']}: {error['error']}")
    return "\n".join(rows) + "\n"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--format", choices=("json", "markdown"), default="json")
    parser.add_argument("--per-source", type=int, default=5)
    args = parser.parse_args(argv)
    if not 1 <= args.per_source <= 10:
        parser.error("--per-source must be 1..10")
    result = scout(GitHubClient(os.environ.get("GITHUB_TOKEN")), per_source=args.per_source)
    print(markdown_report(result) if args.format == "markdown" else
          json.dumps(result, ensure_ascii=False, indent=2))
    return 2 if result["errors"] else 0


if __name__ == "__main__":
    sys.exit(main())
