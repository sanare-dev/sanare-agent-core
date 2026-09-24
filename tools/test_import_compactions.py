"""Offline tests for the local Brain Desk compaction inbox staging tool."""

import json
from pathlib import Path
import tempfile
import unittest

from import_compactions import extract_candidates, stage, stage_claude, stage_kimi


THREAD_ID = "11111111-2222-4333-8444-555555555555"


def thread(summary: str) -> dict:
    return {
        "thread_id": THREAD_ID,
        "updated_at": "2026-09-24T00:00:00Z",
        "messages": [
            {"type": "human", "content": "Текст полного чата не копировать"},
            {"type": "ai", "content": "Ответ", "additional_kwargs": {
                "brain_desk_compaction": {
                    "id": "compact-1", "summary": summary, "messages": 9,
                },
            }},
        ],
    }


class ImportCompactionsTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.threads = self.root / "threads"
        self.pending = self.root / "pending"
        self.threads.mkdir()

    def tearDown(self):
        self.temp.cleanup()

    def write_thread(self, content: dict | str):
        path = self.threads / f"{THREAD_ID}.json"
        path.write_text(content if isinstance(content, str) else json.dumps(content))
        return path

    def test_stages_only_summary_and_is_idempotent(self):
        self.write_thread(thread("Решение: использовать локальное хранилище."))
        self.assertEqual(stage(self.threads, self.pending)["new"], 1)
        files = list(self.pending.glob("*.json"))
        self.assertEqual(len(files), 1)
        payload = json.loads(files[0].read_text())
        self.assertEqual(payload["status"], "pending_review")
        self.assertEqual(payload["summary"], "Решение: использовать локальное хранилище.")
        self.assertNotIn("Текст полного чата", files[0].read_text())
        self.assertEqual(files[0].stat().st_mode & 0o777, 0o600)
        self.assertEqual(stage(self.threads, self.pending)["existing"], 1)

    def test_changed_summary_creates_new_candidate(self):
        self.write_thread(thread("Первое решение."))
        stage(self.threads, self.pending)
        self.write_thread(thread("Уточнённое решение."))
        self.assertEqual(stage(self.threads, self.pending)["new"], 1)
        self.assertEqual(len(list(self.pending.glob("*.json"))), 2)

    def test_sensitive_summary_is_not_staged(self):
        for content in (
            "api_key = hidden-value",
            "Пишите на owner@example.com",
            "-----BEGIN RSA PRIVATE KEY-----",
            "Bearer abcdefghijklmnop1234",
            "sk-abcdefghijklmnopqrstuvwxyz",
        ):
            self.write_thread(thread(content))
            self.assertEqual(stage(self.threads, self.pending)["candidates"], 0)
        self.assertEqual(list(self.pending.iterdir()), [])

    def test_invalid_json_mismatched_id_and_symlink_are_skipped(self):
        path = self.write_thread("{broken")
        self.assertEqual(extract_candidates(path), [])
        self.write_thread({**thread("Безопасная выжимка."), "thread_id": "wrong"})
        self.assertEqual(extract_candidates(path), [])
        path.unlink()
        path.symlink_to(self.root / "outside.json")
        self.assertEqual(extract_candidates(path), [])

    def test_dry_run_does_not_write(self):
        self.write_thread(thread("Безопасная выжимка."))
        result = stage(self.threads, self.pending, dry_run=True)
        self.assertEqual(result["new"], 1)
        self.assertFalse(self.pending.exists())

    def test_oversized_file_is_skipped_before_parsing(self):
        path = self.write_thread(thread("Безопасная выжимка."))
        with path.open("ab") as output:
            output.truncate(64 * 1024 * 1024 + 1)
        self.assertEqual(extract_candidates(path), [])

    def test_kimi_stages_only_existing_compaction_summary(self):
        root = self.root / "kimi" / "sessions"
        wire = root / "conversation-1" / "agents" / "main" / "wire.jsonl"
        wire.parent.mkdir(parents=True)
        events = [
            {"type": "turn.prompt", "input": "Полная переписка не копируется"},
            {"type": "context.apply_compaction", "summary": "Вывод: проект принят.",
             "contextSummary": "Другой текст", "time": 1790200000000},
        ]
        wire.write_text("\n".join(json.dumps(event) for event in events) + "\n")
        self.assertEqual(stage_kimi(root, self.pending)["new"], 1)
        payload = json.loads(next(self.pending.glob("*.json")).read_text())
        self.assertEqual(payload["source_kind"], "kimi")
        self.assertEqual(payload["summary"], "Вывод: проект принят.")
        self.assertIn("conversation-1/agents/main/wire.jsonl", payload["source"])
        self.assertEqual(stage_kimi(root, self.pending)["existing"], 1)
        self.assertNotIn("Полная переписка", json.dumps(payload))

    def test_kimi_skips_sensitive_and_symlinked_wire(self):
        root = self.root / "kimi" / "sessions"
        wire = root / "conversation-1" / "agents" / "main" / "wire.jsonl"
        wire.parent.mkdir(parents=True)
        wire.write_text(json.dumps({"type": "context.apply_compaction",
                                   "summary": "password = hidden-value", "time": 1790200000000}) + "\n")
        self.assertEqual(stage_kimi(root, self.pending)["candidates"], 0)
        wire.unlink()
        wire.symlink_to(self.write_thread(thread("Чужой файл.")))
        self.assertEqual(stage_kimi(root, self.pending)["candidates"], 0)

    def test_claude_stages_only_marked_compaction(self):
        root = self.root / "claude" / "projects"
        session = root / "project-1" / "session.jsonl"
        session.parent.mkdir(parents=True)
        events = [
            {"type": "user", "message": {"content": "Сырая реплика"}},
            {"type": "system", "subtype": "compact_boundary"},
            {"type": "user", "isCompactSummary": True,
             "message": {"content": "Выжимка: принято решение."},
             "timestamp": "2026-09-24T01:00:00Z"},
        ]
        session.write_text("\n".join(json.dumps(event) for event in events) + "\n")
        self.assertEqual(stage_claude(root, self.pending)["new"], 1)
        payload = json.loads(next(self.pending.glob("*.json")).read_text())
        self.assertEqual(payload["source_kind"], "claude")
        self.assertEqual(payload["summary"], "Выжимка: принято решение.")
        self.assertNotIn("Сырая реплика", json.dumps(payload))
        self.assertEqual(stage_claude(root, self.pending)["existing"], 1)

    def test_claude_skips_unmarked_and_sensitive_compaction(self):
        root = self.root / "claude" / "projects"
        session = root / "project-1" / "session.jsonl"
        session.parent.mkdir(parents=True)
        session.write_text(json.dumps({"type": "user", "isCompactSummary": True,
                                       "message": {"content": "owner@example.com"},
                                       "timestamp": "2026-09-24T01:00:00Z"}) + "\n")
        self.assertEqual(stage_claude(root, self.pending)["candidates"], 0)

    def test_claude_skips_symlinked_project_directory(self):
        root = self.root / "claude" / "projects"
        root.mkdir(parents=True)
        outside = self.root / "outside"
        outside.mkdir()
        (outside / "session.jsonl").write_text(json.dumps({
            "type": "user", "isCompactSummary": True,
            "message": {"content": "Текст выжимки."},
            "timestamp": "2026-09-24T01:00:00Z",
        }) + "\n")
        (root / "linked").symlink_to(outside, target_is_directory=True)
        self.assertEqual(stage_claude(root, self.pending)["candidates"], 0)


if __name__ == "__main__":
    unittest.main()
