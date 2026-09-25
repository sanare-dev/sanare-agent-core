"""An incognito Brain Desk thread must never enter candidate memory."""

import asyncio
from datetime import datetime, timezone
from types import SimpleNamespace

import pytest

from deep_agent import consolidator


@pytest.mark.parametrize(
    "kind, content",
    [
        ("system", "[BRAIN_DESK_INCOGNITO_V1] Не сохраняй в память"),
        ("human", "[BRAIN_DESK_INCOGNITO_V1] Не сохраняй в память"),
        ("ai", "[BRAIN_DESK_INCOGNITO_V1] Не сохраняй в память"),
        ("tool", [{"type": "text", "text": "[BRAIN_DESK_INCOGNITO_V1]"}]),
    ],
)
def test_incognito_marker_in_any_turn_excludes_whole_thread(kind, content):
    since = datetime(2026, 9, 23, tzinfo=timezone.utc)
    marked = [{"type": kind, "content": content}]
    # The marker can fall outside the last 12 displayed messages.
    marked += [{"type": "human", "content": f"private-{i}"} for i in range(13)]
    threads = [
        {
            "thread_id": "incognito",
            "updated_at": "2026-09-23T06:00:00Z",
            "values": {"messages": marked},
        },
        {
            "thread_id": "ordinary",
            "updated_at": "2026-09-23T06:00:00Z",
            "values": {"messages": [{"type": "human", "content": "public fact"}]},
        },
    ]

    digest = consolidator.format_threads(threads, since)

    assert "thread incognito" not in digest
    assert "private-" not in digest
    assert "thread ordinary" in digest
    assert "public fact" in digest


def test_no_incognito_digest_has_neutral_empty_result():
    since = datetime(2026, 9, 23, tzinfo=timezone.utc)
    threads = [
        {
            "thread_id": "incognito",
            "updated_at": "2026-09-23T06:00:00Z",
            "values": {
                "messages": [
                    {"type": "system", "content": consolidator.INCOGNITO_MARKER},
                    {"type": "human", "content": "private fact"},
                ]
            },
        }
    ]

    assert (
        consolidator.format_threads(threads, since) == "Нет обновлённых тредов за окно."
    )


def test_recent_conversations_tool_does_not_return_incognito_content(monkeypatch):
    class Threads:
        async def search(self, **_kwargs):
            return [
                {
                    "thread_id": "incognito",
                    "updated_at": "2999-01-01T00:00:00Z",
                    "values": {
                        "messages": [
                            {"type": "system", "content": "[BRAIN_DESK_INCOGNITO_V1]"},
                            {"type": "human", "content": "private fact"},
                        ]
                    },
                }
            ]

    monkeypatch.setattr(
        consolidator, "get_client", lambda: SimpleNamespace(threads=Threads())
    )

    result = asyncio.run(consolidator.recent_conversations.ainvoke({}))
    assert result == "Нет обновлённых тредов за окно."
