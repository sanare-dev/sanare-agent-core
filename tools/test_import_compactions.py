"""Offline tests for the local Brain Desk compaction inbox staging tool."""

import json
from pathlib import Path
import sqlite3
import tempfile
import unittest

from import_compactions import (
    extract_candidates,
    list_pending,
    resolve_pending,
    stage,
    stage_claude,
    stage_kimi,
    stage_openwebui,
)


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

    def test_openwebui_stages_only_saved_context_summary(self):
        database = self.root / "webui.db"
        message_id = "msg_" + "a" * 69
        with sqlite3.connect(database) as connection:
            connection.execute(
                "CREATE TABLE chat_message (id TEXT, chat_id TEXT, updated_at INTEGER, "
                "context_summary TEXT, content TEXT)"
            )
            connection.execute(
                "INSERT INTO chat_message VALUES (?, ?, ?, ?, ?)",
                (message_id, THREAD_ID, 1790200000, "Выжимка контекста.", "Сырая переписка"),
            )
            connection.execute(
                "INSERT INTO chat_message VALUES (?, ?, ?, ?, ?)",
                ("33333333-4444-4555-8666-777777777777", THREAD_ID,
                 1790200000, None, "Другое сообщение"),
            )
        self.assertEqual(stage_openwebui(database, self.pending)["new"], 1)
        payload = json.loads(next(self.pending.glob("*.json")).read_text())
        self.assertEqual(payload["source_kind"], "openwebui")
        self.assertEqual(payload["summary"], "Выжимка контекста.")
        self.assertNotIn("Сырая переписка", json.dumps(payload))
        self.assertEqual(stage_openwebui(database, self.pending)["existing"], 1)

    def test_openwebui_skips_sensitive_and_old_schema(self):
        database = self.root / "webui.db"
        with sqlite3.connect(database) as connection:
            connection.execute(
                "CREATE TABLE chat_message (id TEXT, chat_id TEXT, updated_at INTEGER, "
                "context_summary TEXT)"
            )
            connection.execute(
                "INSERT INTO chat_message VALUES (?, ?, ?, ?)",
                ("22222222-3333-4444-8555-666666666666", THREAD_ID,
                 1790200000, "password = hidden-value"),
            )
        self.assertEqual(stage_openwebui(database, self.pending)["candidates"], 0)
        database.unlink()
        with sqlite3.connect(database) as connection:
            connection.execute("CREATE TABLE chat_message (id TEXT, content TEXT)")
        self.assertEqual(stage_openwebui(database, self.pending)["candidates"], 0)
        self.assertEqual(list(self.pending.glob("*.json")), [])

    def test_list_pending_reads_staged_candidates_with_id(self):
        self.write_thread(thread("Решение: использовать локальное хранилище."))
        stage(self.threads, self.pending)
        [row] = list_pending(self.pending)
        self.assertRegex(row["id"], r"^[0-9a-f]{64}$")
        self.assertEqual(row["summary"], "Решение: использовать локальное хранилище.")
        self.assertEqual(row["status"], "pending_review")

    def test_list_pending_ignores_missing_dir_and_junk_files(self):
        self.assertEqual(list_pending(self.pending), [])
        self.pending.mkdir()
        (self.pending / "not-a-candidate.json").write_text("{}")
        (self.pending / ("a" * 64 + ".json")).write_text("not json")
        self.assertEqual(list_pending(self.pending), [])

    def test_resolve_pending_moves_candidate_out_of_queue(self):
        self.write_thread(thread("Решение: использовать локальное хранилище."))
        stage(self.threads, self.pending)
        [row] = list_pending(self.pending)
        self.assertTrue(resolve_pending(self.pending, row["id"], "written"))
        self.assertEqual(list_pending(self.pending), [])
        moved = self.pending.parent / "reviewed" / "written" / f"{row['id']}.json"
        self.assertTrue(moved.is_file())
        self.assertEqual(json.loads(moved.read_text())["summary"], row["summary"])
        # Re-running import for the same source does not resurrect it: it is
        # counted as already seen (in "reviewed/"), never staged again.
        self.assertEqual(stage(self.threads, self.pending)["existing"], 1)
        self.assertEqual(stage(self.threads, self.pending)["new"], 0)
        self.assertEqual(list(self.pending.glob("*.json")), [])

    def test_resolve_pending_rejects_bad_id_and_repeat(self):
        self.write_thread(thread("Решение: использовать локальное хранилище."))
        stage(self.threads, self.pending)
        [row] = list_pending(self.pending)
        self.assertFalse(resolve_pending(self.pending, "../etc/passwd", "written"))
        self.assertFalse(resolve_pending(self.pending, row["id"], "bogus"))
        self.assertTrue(resolve_pending(self.pending, row["id"], "written"))
        self.assertFalse(resolve_pending(self.pending, row["id"], "written"))


if __name__ == "__main__":
    unittest.main()
