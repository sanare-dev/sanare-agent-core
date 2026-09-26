"""Per-task reasoning effort: one deterministic rule, no extra model call.

Owner order 2026-09-26: the reasoning level of every model is chosen by the
task instead of being fixed (Luna had effort='max' for every turn since #35, so
a bare «привет» reasoned for minutes). ``choose_effort`` reads only signals the
graph already has: the latest owner message, the Brain role and the Brain Desk
agent persona of the first system message. Tool-loop follow-up steps of the
same turn see the same latest owner message and therefore reuse its level.

Levels are abstract (low < medium < high < max); ``msty_models.make_model``
maps them to each provider's own parameter, and providers without a safe
per-call parameter ignore them (recorded as ``provider_value=None``).
"""
from __future__ import annotations

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


# Server sub-agent roles: short atomic operator work, long read-only research,
# and cross-checking of critical claims.
SUBAGENT_LEVELS = {'operator': 'low', 'researcher': 'medium', 'auditor': 'high'}
COMPACTION_LEVEL = 'low'
