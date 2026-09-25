"""Candidate-memory consolidation run by Brain itself through the local bridge.

Why not an Agent Server cron: the owner's emergency stop is a local SQLite flag
(right-hand.sqlite, read by brain_stop.require_running) that the cloud cannot
read, and cloud-side calls bypass the bridge ledger (reserve before send,
measured settle, unknown != 0). Instead a local launchd job
(tools/consolidate_memory.py) sends CONSOLIDATION_PROMPT to `team.brain` over
the existing bridge, which checks the stop before every paid stage and meters
every stage in the ledger. Brain gets one read-only, model-free tool that
returns a bounded digest of recently updated Brain threads; cards are written
with the stock native file tools into /memories/ only.

Pattern: https://docs.langchain.com/oss/python/deepagents/memory (background
consolidation reads recent threads through the SDK client).
"""
from datetime import datetime, timedelta, timezone

from langchain.tools import tool
from langgraph_sdk import get_client

RECENT_CONVERSATIONS_TOOL = 'native_recent_conversations'
CONSOLIDATION_MARKER = 'MSTY_MEMORY_CONSOLIDATION_V1'
INCOGNITO_MARKER = 'BRAIN_DESK_INCOGNITO_V1'
SOURCE_GRAPH = 'msty_native'
WINDOW_HOURS = 8  # launchd interval is 6 h; 2 h overlap, cards are updated, not duplicated
MAX_THREADS = 10
MAX_MESSAGES_PER_THREAD = 12
MAX_CHARS_PER_MESSAGE = 1200
MAX_TOTAL_CHARS = 40000

CONSOLIDATION_PROMPT = f"""{CONSOLIDATION_MARKER}
Плановая консолидация памяти-кандидата.
1. Вызови {RECENT_CONVERSATIONS_TOOL} один раз.
2. Посмотри существующие карточки: native_ls / native_grep в /memories/. Обновляй
   подходящую карточку (native_edit_file), новую создавай только для новой темы.
3. Карточка — /memories/<проект>/<тема>.md, кратко: что сделано; где (пути, URL,
   проект); как; источники и проверки (thread_id); остаток; дата.
4. Только проверяемые факты из переписки; догадки не записывай. Секреты, ключи,
   пароли и персональные данные не записывай. Карточка — reference_only кандидат,
   не approved-память и не полномочие. Подагентов и внешние инструменты не используй.
Если существенной работы нет — ничего не пиши и ответь «нет новых фактов».
В конце перечисли изменённые карточки."""


def _text(content) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return ' '.join(part.get('text', '') if isinstance(part, dict) else str(part)
                        for part in content)
    return ''


def format_threads(threads: list[dict], since: datetime) -> str:
    """Bounded plain-text digest of human/AI turns; tool payloads are skipped,
    and consolidation and incognito threads are excluded by their markers."""
    chunks, total = [], 0
    for thread in threads:
        updated = str(thread.get('updated_at') or '')
        try:
            if datetime.fromisoformat(updated.replace('Z', '+00:00')) < since:
                continue
        except ValueError:
            continue
        messages = [m for m in ((thread.get('values') or {}).get('messages') or [])
                    if isinstance(m, dict)]
        if any(m.get('type') == 'human' and CONSOLIDATION_MARKER in _text(m.get('content'))
               for m in messages):
            continue
        # Brain Desk puts this note in a system turn. Check every message before
        # selecting the last N visible turns, including system and tool turns.
        if any(INCOGNITO_MARKER in _text(m.get('content')) for m in messages):
            continue
        lines = [f"{m.get('type')}: {_text(m.get('content'))[:MAX_CHARS_PER_MESSAGE]}"
                 for m in messages[-MAX_MESSAGES_PER_THREAD:] if m.get('type') in ('human', 'ai')
                 and _text(m.get('content')).strip()]
        if not lines:
            continue
        chunk = f"## thread {thread.get('thread_id')} updated {updated}\n" + '\n'.join(lines)
        if total + len(chunk) > MAX_TOTAL_CHARS:
            break
        chunks.append(chunk)
        total += len(chunk)
    return '\n\n'.join(chunks) or 'Нет обновлённых тредов за окно.'


@tool(RECENT_CONVERSATIONS_TOOL)
async def recent_conversations() -> str:
    """Read-only bounded digest of Brain conversations updated in the last hours
    (human/assistant text only). For candidate-memory consolidation; not live status."""
    since = datetime.now(timezone.utc) - timedelta(hours=WINDOW_HOURS)
    client = get_client()  # in-process Agent Server connection, no model call
    threads = await client.threads.search(
        metadata={'graph_id': SOURCE_GRAPH}, sort_by='updated_at', sort_order='desc',
        limit=MAX_THREADS, select=['thread_id', 'updated_at', 'values'])
    return format_threads(threads, since)
