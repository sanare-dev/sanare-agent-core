"""TAU, слой L5: Evidence Gate на финальном ответе.

Негативное утверждение («не настроен», «не работает», «отсутствует», «сломан» и
английские аналоги) допускается в финальном ответе только после УСПЕШНОГО чтения
профильного статуса в этой сессии — инструмента с evidence_class='status_read'
из манифеста (§4.3 эталонной архитектуры). Иначе ответ переписывается в честное
«не могу подтвердить» + перечень проверенного из evidence log; исходный текст
сохраняется, но помечается как неподтверждённый.

Уверенная ложь хуже честного «не знаю»: ложный диагноз «cron не настроен» без
успешного статус-чтения — исходный дефект 2026-09-22, этот слой делает его
невозможным на уровне конвейера, а не prompt-правила.

Маркеры конфигурируемы: переменная окружения MSTY_NEGATIVE_MARKERS (через запятую)
заменяет список по умолчанию; пустое значение отключает Gate.
"""
from __future__ import annotations

import os

from . import msty_registry, msty_taxonomy

DEFAULT_NEGATIVE_MARKERS = (
    'не настроен', 'не работает', 'отсутствует', 'сломан', 'не установлен',
    'not configured', 'not working', 'is missing', 'broken', 'unavailable',
)


def negative_markers() -> tuple[str, ...]:
    override = os.getenv('MSTY_NEGATIVE_MARKERS')
    if override is None:
        return DEFAULT_NEGATIVE_MARKERS
    return tuple(marker.strip().lower() for marker in override.split(',') if marker.strip())


def _text(content) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return '\n'.join(part if isinstance(part, str) else str(part.get('text', ''))
                         for part in content
                         if isinstance(part, str) or
                         isinstance(part, dict) and isinstance(part.get('text'), str))
    return ''


def has_negative_claim(text: str) -> bool:
    lowered = text.lower()
    return any(marker in lowered for marker in negative_markers())


def _call_names(messages) -> dict[str, str]:
    """tool_call_id → имя вызова (OpenAI-dict и LangChain формы ассистента)."""
    names = {}
    for message in messages or ():
        calls = (message.get('tool_calls') if isinstance(message, dict)
                 else getattr(message, 'tool_calls', None)) or ()
        for call in calls:
            if not isinstance(call, dict):
                continue
            name = call.get('name') or (call.get('function') or {}).get('name')
            if isinstance(call.get('id'), str) and isinstance(name, str):
                names[call['id']] = name
    return names


def _tool_messages(messages):
    for message in messages or ():
        is_tool = (message.get('role') == 'tool' if isinstance(message, dict)
                   else getattr(message, 'type', None) == 'tool')
        if is_tool:
            yield message


def successful_status_reads(state) -> list[str]:
    """Имена evidence-инструментов (status_read) с успешным результатом в сессии.

    Источники: evidence log состояния графа (tau_evidence, пишет контур L4) и
    скан истории сообщений — результаты инструментов без признаков отказа.
    """
    found = []
    for item in state.get('tau_evidence') or ():
        if (isinstance(item, dict) and item.get('evidence_class') == 'status_read'
                and isinstance(item.get('tool'), str) and item['tool'] not in found):
            found.append(item['tool'])
    messages = state.get('messages') or ()
    names = _call_names(messages)
    for message in _tool_messages(messages):
        if isinstance(message, dict):
            name = message.get('name') or names.get(message.get('tool_call_id'))
            content, status = message.get('content'), message.get('status')
        else:
            name = getattr(message, 'name', None) or names.get(getattr(message, 'tool_call_id', None))
            content, status = message.content, getattr(message, 'status', None)
        if status == 'error' or not isinstance(name, str):
            continue
        entry = msty_registry.find(name)
        if entry is None or entry.evidence_class != 'status_read':
            continue
        text = _text(content)
        if text.startswith('error=') or msty_taxonomy.classify_tool_text(text) is not None:
            continue  # неуспешное чтение — не доказательство
        if name not in found:
            found.append(name)
    return found


def evidence_summary(state) -> str:
    """Перечень проверенного: успешные чтения и классифицированные отказы."""
    checked = []
    for item in state.get('tau_evidence') or ():
        if isinstance(item, dict) and isinstance(item.get('tool'), str):
            checked.append(f"{item['tool']}: ok")
    for item in state.get('tau_errors') or ():
        if isinstance(item, dict) and isinstance(item.get('tool'), str):
            checked.append(f"{item['tool']}: {item.get('class', 'отказ')}")
    return '; '.join(checked) if checked else 'ничего'


def rewrite_unconfirmed(content: str, state) -> str:
    status_tools = ', '.join(sorted(
        entry.name for entry in msty_registry.TOOLS
        if entry.evidence_class == 'status_read'))
    return ('Не могу подтвердить негативный вывод: в этой сессии не было успешного '
            f'профильного статус-чтения ({status_tools}). Проверено: {evidence_summary(state)}. '
            'Ниже — исходная оценка модели, она НЕ подтверждена и не является диагнозом.'
            f'\n\n{content}')


def gate_final_answer(state: dict, result):
    """Пропускает или переписывает финальный ответ; вызовы инструментов не трогает.

    Применяется только к настоящему финалу модели: без tool_calls, без уже
    сработавших блокировок/гейтов конвейера (их текст — системный, не оценка).
    """
    if result.tool_calls or result.invalid_tool_calls:
        return result, None
    meta = result.response_metadata or {}
    if meta.get('msty_blocked') or meta.get('msty_completion_gate') or meta.get('msty_generation'):
        return result, None
    if not negative_markers():
        return result, None
    text = _text(result.content)
    if not text.strip() or not has_negative_claim(text):
        return result, None
    reads = successful_status_reads(state)
    if reads:
        return result, None  # негатив подтверждён успешным статус-чтением
    return (result.model_copy(update={
        'content': rewrite_unconfirmed(text, state),
        'response_metadata': {**meta, 'tau_evidence_gate': 'unconfirmed'},
    }), 'unconfirmed')
