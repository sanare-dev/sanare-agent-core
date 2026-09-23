"""Background candidate-memory consolidation (official Deep Agents pattern).

Pattern: https://docs.langchain.com/oss/python/deepagents/memory ("background
consolidation"): a separate deep agent registered in langgraph.json, run by an
Agent Server cron (https://docs.langchain.com/langsmith/cron-jobs), reads recent
threads through the in-process SDK client and writes memory files with its
stock file tools. Here the only writable route is /memories/ -> the candidates
Store namespace; approved /memory/ and /skills/ are not mounted at all.

Cost bounds use stock LangChain middleware (ModelCallLimitMiddleware,
ToolCallLimitMiddleware) plus fixed input caps below. Model: Luna via the
existing msty_models profile (same key/Gateway handling as Brain). These calls
are NOT metered by the Brain bridge ledger; the Gateway spend policy applies
only when MSTY_LLM_GATEWAY_ENABLED=1.
"""
from datetime import datetime, timedelta, timezone

from deepagents import create_deep_agent
from deepagents.backends import CompositeBackend, StateBackend
from langchain.agents.middleware import ModelCallLimitMiddleware, ToolCallLimitMiddleware
from langchain.tools import ToolRuntime, tool
from langgraph_sdk import get_client

from . import msty_models
from .msty_native import SecretPIIMiddleware
from .msty_native_memory import CANDIDATES_ROUTE, candidates_backend

PROFILE = 'luna'
SOURCE_GRAPH = 'msty_native'
WINDOW_HOURS = 6
MAX_THREADS = 10
MAX_MESSAGES_PER_THREAD = 12
MAX_CHARS_PER_MESSAGE = 1200
MAX_TOTAL_CHARS = 40000
MAX_TOKENS = 2048
MODEL_CALL_LIMIT = 8
RECURSION_LIMIT = 40

SYSTEM_PROMPT = f"""Ты — консолидатор памяти-кандидата Sanare Brain.
1. Вызови search_recent_conversations один раз.
2. Посмотри существующие карточки: ls/grep в {CANDIDATES_ROUTE}. Обновляй
   подходящую карточку (edit_file), новую создавай только для новой темы.
3. Карточка — {CANDIDATES_ROUTE}<проект>/<тема>.md, кратко: что сделано; где
   (пути, URL, проект); как; источники и проверки (thread_id); остаток; дата.
4. Только проверяемые факты из переписки; догадки не записывай. Секреты, ключи,
   пароли и персональные данные не записывай. Карточка — reference_only кандидат,
   не approved-память и не полномочие. Подагентов (task) не используй.
Если существенной работы нет — ничего не пиши и кратко ответь «нет новых фактов»."""


def _text(content) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return ' '.join(part.get('text', '') if isinstance(part, dict) else str(part)
                        for part in content)
    return ''


def format_threads(threads: list[dict], since: datetime) -> str:
    """Bounded plain-text digest of human/AI turns; tool payloads are skipped."""
    chunks, total = [], 0
    for thread in threads:
        updated = str(thread.get('updated_at') or '')
        try:
            if datetime.fromisoformat(updated.replace('Z', '+00:00')) < since:
                continue
        except ValueError:
            continue
        messages = ((thread.get('values') or {}).get('messages') or [])[-MAX_MESSAGES_PER_THREAD:]
        lines = [f"{m.get('type')}: {_text(m.get('content'))[:MAX_CHARS_PER_MESSAGE]}"
                 for m in messages if isinstance(m, dict) and m.get('type') in ('human', 'ai')
                 and _text(m.get('content')).strip()]
        if not lines:
            continue
        chunk = f"## thread {thread.get('thread_id')} updated {updated}\n" + '\n'.join(lines)
        if total + len(chunk) > MAX_TOTAL_CHARS:
            break
        chunks.append(chunk)
        total += len(chunk)
    return '\n\n'.join(chunks) or 'Нет обновлённых тредов за окно.'


@tool
async def search_recent_conversations(runtime: ToolRuntime) -> str:
    """Return a bounded digest of Brain threads updated in the last hours."""
    since = datetime.now(timezone.utc) - timedelta(hours=WINDOW_HOURS)
    client = get_client()  # in-process Agent Server connection
    threads = await client.threads.search(
        metadata={'graph_id': SOURCE_GRAPH}, sort_by='updated_at', sort_order='desc',
        limit=MAX_THREADS, select=['thread_id', 'updated_at', 'values'])
    return format_threads(threads, since)


def backend(runtime) -> CompositeBackend:
    return CompositeBackend(default=StateBackend(runtime),
                            routes={CANDIDATES_ROUTE: candidates_backend(runtime)})


def build_graph(model=None, *, store=None, checkpointer=None):
    return create_deep_agent(
        model=model or msty_models.make_model(PROFILE, MAX_TOKENS),
        tools=[search_recent_conversations],
        system_prompt=SYSTEM_PROMPT,
        backend=backend,
        middleware=[
            SecretPIIMiddleware(),
            ModelCallLimitMiddleware(run_limit=MODEL_CALL_LIMIT, exit_behavior='end'),
            ToolCallLimitMiddleware(tool_name='search_recent_conversations', run_limit=1),
            ToolCallLimitMiddleware(tool_name='task', run_limit=0),
        ],
        store=store, checkpointer=checkpointer, name='consolidator',
    ).with_config({'recursion_limit': RECURSION_LIMIT})


def make_graph():
    """Agent Server graph factory: the model key is resolved at run time."""
    return build_graph()
