"""Per-task reasoning effort: one deterministic rule, no extra model call.

Owner order 2026-09-26: the reasoning level of every model is chosen by the
task instead of being fixed (Luna had effort='max' for every turn since #35, so
a bare «привет» reasoned for minutes). ``choose_effort`` reads only signals the
graph already has: the latest owner message, the Brain role and the Brain Desk
agent persona of the first system message. That turn level is kept for the
plan and the synthesis; ``step_effort`` lowers the steps that only continue a
simple tool chain (owner order 2026-09-28, see below).

Levels are abstract (low < medium < high < max); ``msty_models.make_model``
maps them to each provider's own parameter, and providers without a safe
per-call parameter ignore them (recorded as ``provider_value=None``).
"""
from __future__ import annotations

import json
import re
from typing import Any

LEVELS = ('low', 'medium', 'high', 'max')
DEFAULT_LEVEL = 'medium'
METADATA_KEY = 'msty_reasoning_effort'

# Explicit owner force wins over every other signal: «!low» … «!max».
_FORCE_TOKEN = re.compile(r'(?<![\w!])!(low|medium|high|max)\b', re.IGNORECASE)
_FORCE_MAX = re.compile(
    r'максимально\s+(?:глубоко\s+)?(?:подумай|продумай|разбери|рассуди)'
    r'|(?:подумай|продумай|разбери)\s+(?:максимально|по\s+максимуму)'
    r'|максимальн\w*\s+(?:уровень\s+)?(?:рассуждени|глубин)', re.IGNORECASE)
_GREETING_WORD = (
    r'(?:привет\w*|здравствуй\w*|добр\w+\s+(?:утро|день|вечер|ночи)|хай|хелло|салют'
    r'|hi|hello|hey|спасибо|благодарю|ок|окей|ok|понял\w*|ясно|ага|угу|да|нет|отлично|супер'
    r'|как\s+дела|как\s+ты|ты\s+тут|ты\s+здесь|brain|брейн|бро|дружище)')
_GREETING = re.compile(r'^(?:' + _GREETING_WORD + r'[\s,!.?)(:\-]*)+$', re.IGNORECASE)
_HIGH = re.compile(
    r'подумай|продумай|глубок|разбер|разбор|проанализ|анализ|исследу|сравни'
    r'|план|спланир|архитектур|стратеги|декомпоз|многошаг|пошагов'
    r'|аудит|ревью|review|перепровер|провер\w*\s+(?:код|pr|пул|вс[её])'
    r'|код\b|кода\b|коде\b|скрипт|патч|pull\s*request|\bpr\b|рефактор|исправ|почини|отлад'
    r'|\b(?:fix|fixes|fixed|debug|refactor|plan|planning|review|audit|analy[sz]e|analysis'
    r'|compare|investigate|implement|deploy|architecture|step\s+by\s+step|think\s+hard)\b'
    r'|реализу|внедри|разработ|миграци|деплой|deploy'
    r'|поручи|делегир|отдел(?:у|ам|ы|а|ом|ов)?\b|архитектор|разработчик'
    r'|почему\s+(?:не\s+работает|падает|сломал)|причин\w*\s+(?:сбоя|ошибки|отказа)'
    r'|```', re.IGNORECASE)
_ARCHITECT_PERSONA = re.compile(r'как его агент «[^»]*архитект[^»]*»', re.IGNORECASE)
LONG_TEXT = 1200
SHORT_TEXT = 80


def _text(content: Any) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return '\n'.join(item if isinstance(item, str) else item.get('text', '')
                         for item in content
                         if isinstance(item, str) or (isinstance(item, dict)
                                                      and isinstance(item.get('text'), str)))
    return ''


def _role(message: Any) -> Any:
    if isinstance(message, dict):
        return message.get('role', message.get('type'))
    return getattr(message, 'type', None)


def _content(message: Any) -> Any:
    return message.get('content') if isinstance(message, dict) else getattr(message, 'content', None)


def latest_owner_text(messages: list[Any]) -> str:
    for message in reversed(messages or []):
        if _role(message) in ('user', 'human'):
            return _text(_content(message)).strip()
    return ''


def _architect(messages: list[Any]) -> bool:
    for message in (messages or [])[:3]:
        if _role(message) == 'system' and _ARCHITECT_PERSONA.search(_text(_content(message))):
            return True
    return False


def classify(text: str, *, architect: bool = False) -> tuple[str, str]:
    """(level, reason) for one owner message; pure and deterministic."""
    forced = _FORCE_TOKEN.search(text)
    if forced:
        return forced.group(1).lower(), 'owner_force'
    if _FORCE_MAX.search(text):
        return 'max', 'owner_force'
    if not text:
        return DEFAULT_LEVEL, 'no_owner_text'
    if len(text) <= SHORT_TEXT and _GREETING.match(text):
        return 'low', 'greeting'
    if _HIGH.search(text) or len(text) > LONG_TEXT:
        return 'high', 'deep_work' if len(text) <= LONG_TEXT else 'long_request'
    if architect:
        return 'high', 'architect'
    if len(text) <= SHORT_TEXT and '\n' not in text:
        return 'low', 'short_question'
    return 'medium', 'ordinary'


def choose_effort(state: dict) -> dict:
    """The turn's level from state: {'version', 'level', 'reason'}."""
    messages = state.get('messages') or []
    level, reason = classify(latest_owner_text(messages), architect=_architect(messages))
    return {'version': 1, 'level': level, 'reason': reason}



# ---------------------------------------------------------------------------
# Per-step level (owner order 2026-09-28, «тупит нереально»): the turn level of
# choose_effort is right for the plan and the final answer, not for every step
# of a browser/tool chain. Live 28.09 (Brain Desk, owner_browser_* chain, 8 min
# for 7 steps): every step re-reasoned at the turn level (medium there); the
# model call was 3.5-7 s of each ~55 s step, the rest is outside this graph.
# A step that only continues a chain after
# a simple tool result runs at STEP_LEVEL; the turn level is kept for the plan
# (no tool result yet), after a non-simple tool, for the synthesis (msty.py
# re-runs a low step that turned out to be the final answer, see
# _synthesis_step) and an explicit owner force. Two failed tool batches in a
# row ask for rethinking at FAILURE_LEVEL. gpt-6-luna documents none/low/
# medium/high/xhigh/max: no 'minimal' tier, and 'none' drops reasoning
# entirely, so 'low' is the floor here.
STEP_LEVEL = 'low'
FAILURE_LEVEL = 'medium'
FAILURE_STREAK = 2
_SIMPLE_TOOL = re.compile(
    r'(?:^|[_.])(?:owner_browser_[a-z0-9_]+|browser_[a-z0-9_]+|web_read|web_search'
    r'|brain_desk_read_tool_result)$')
_READ_METHODS = frozenset(('GET', 'HEAD'))


def _call_parts(call: Any) -> tuple[Any, Any, Any]:
    """(id, name, args) of one assistant tool call in dict or LangChain form."""
    if not isinstance(call, dict):
        return None, None, None
    function = call.get('function') if isinstance(call.get('function'), dict) else {}
    name = call.get('name', function.get('name'))
    args = call.get('args', function.get('arguments'))
    if isinstance(args, str):
        try:
            args = json.loads(args)
        except ValueError:
            args = None
    return call.get('id'), name, args


def simple_tool(name: Any, args: Any = None) -> bool:
    """A tool whose result needs no deep re-planning to take the next step."""
    if not isinstance(name, str):
        return False
    if _SIMPLE_TOOL.search(name):
        return True
    if name == 'service_call' or name.endswith('_service_call'):
        method = args.get('method') if isinstance(args, dict) else None
        return isinstance(method, str) and method.upper() in _READ_METHODS
    return False


def _field(message: Any, key: str) -> Any:
    return message.get(key) if isinstance(message, dict) else getattr(message, key, None)


def _failed(message: Any) -> bool:
    if _field(message, 'status') == 'error':
        return True
    from . import msty_taxonomy  # lazy: taxonomy imports the registry
    return msty_taxonomy.classify_tool_text(_content(message)) is not None


def tool_batches(messages: list[Any]) -> list[dict]:
    """Tool batches after the latest owner message, oldest first."""
    start = 0
    for index, message in enumerate(messages or []):
        if _role(message) in ('user', 'human'):
            start = index + 1
    batches, current = [], None
    for message in (messages or [])[start:]:
        role = _role(message)
        if role in ('assistant', 'ai'):
            calls = [_call_parts(c) for c in (_field(message, 'tool_calls') or [])]
            current = {'calls': {c[0]: (c[1], c[2]) for c in calls}, 'results': []} if calls else None
            if current is not None:
                batches.append(current)
        elif role == 'tool' and current is not None:
            current['results'].append(message)
    return [b for b in batches if b['results']]


def step_effort(state: dict, task: dict) -> dict:
    """This step's level from the turn level ``task`` and the tool chain so far.

    Returns the choice dict of choose_effort plus 'step' (plan | tool_chain |
    failure_rethink | task) and, when lowered or raised, 'task_level' and
    'task_reason' so the synthesis can restore the turn level.
    """
    messages = state.get('messages') or []
    batches = tool_batches(messages)
    last = messages[-1] if messages else None
    if not batches or _role(last) != 'tool':
        return {**task, 'step': 'plan' if not batches else 'task'}
    choice = state.get('tool_choice')
    if (task.get('reason') == 'owner_force' or choice == 'none' or
            (isinstance(choice, dict) and choice.get('type') == 'none')):
        return {**task, 'step': 'task'}
    streak = 0
    for batch in reversed(batches):
        if not any(_failed(m) for m in batch['results']):
            break
        streak += 1
    base = {'version': 1, 'task_level': task['level'], 'task_reason': task['reason'],
            'chain_steps': len(batches)}
    if streak >= FAILURE_STREAK:
        return {**base, 'level': FAILURE_LEVEL, 'reason': 'tool_failures', 'step': 'failure_rethink'}
    batch = batches[-1]
    names = [batch['calls'].get(_field(m, 'tool_call_id'), (None, None)) for m in batch['results']]
    if all(simple_tool(name, args) for name, args in names) and \
            LEVELS.index(task['level']) > LEVELS.index(STEP_LEVEL):
        return {**base, 'level': STEP_LEVEL, 'reason': 'tool_chain', 'step': 'tool_chain'}
    return {**task, 'step': 'task'}


#: Reasoning tokens are billed inside the output limit (Responses API
#: max_output_tokens). Live 2026-09-26 (#41): Luna at 'max' spent the whole 4096
#: on reasoning and returned no text. A level runs only when the step's output
#: limit reaches its floor; otherwise it steps down. Limits are not raised here.
MIN_OUTPUT_TOKENS = {'max': 8192, 'high': 4096, 'medium': 1024}


def fit(level: str, output_limit: int) -> str:
    """Highest level <= ``level`` whose output floor fits ``output_limit``."""
    index = LEVELS.index(level)
    while index > 0 and output_limit < MIN_OUTPUT_TOKENS.get(LEVELS[index], 0):
        index -= 1
    return LEVELS[index]


#: Out-of-limit recovery (owner order 2026-09-28): a retry must leave most of
#: its output for the answer text, so its level needs RETRY_HEADROOM_FACTOR x
#: that level's reasoning floor.
RETRY_HEADROOM_FACTOR = 4


def retry_level(applied: str, headroom: int) -> str:
    """One level below ``applied`` (high→medium→low), lower still if the
    retry's output ``headroom`` cannot hold that level's reasoning and a reply."""
    index = max(LEVELS.index(applied) - 1, 0)
    while index > 0 and headroom < RETRY_HEADROOM_FACTOR * MIN_OUTPUT_TOKENS.get(LEVELS[index], 0):
        index -= 1
    return LEVELS[index]


# Server sub-agent roles: short atomic operator work, long read-only research,
# and cross-checking of critical claims.
SUBAGENT_LEVELS = {'operator': 'low', 'researcher': 'medium', 'auditor': 'high'}
COMPACTION_LEVEL = 'low'
