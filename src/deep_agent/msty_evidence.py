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

Точность (аудит 2026-09-23): подстрочный поиск переписывал ~13 из 20 обычных
ответов («дубликат отсутствует», «если кнопка не работает, обновите»,
«Nothing is missing»). Теперь негативом считается только предложение, где
маркер стоит рядом с объектом инфраструктуры (cron, синхронизация, сервис,
интеграция, ключ, база, …) и которое не является условием, советом,
отрицанием отрицания или историей исправления. Доказательство принимается
только из ТЕКУЩЕГО хода владельца (после последнего его сообщения): одно
статус-чтение в прошлом ходе больше не разрешает любой негатив навсегда.

Область (24.09.2026, brain-desk: «Архитектор отморозился»): Gate применяется
только к вопросам о состоянии системы и сервисов — когда начало последнего
сообщения владельца спрашивает о статусе/сбое («работает ли», «проверь
синхронизацию», «почему не обновляются») или в этом ходе модель вызывала
профильное статус-чтение. Планирование, брифы, разборы очереди и тексты
Gate не трогает: там упоминание «отсутствует»/«нет данных» — не диагноз
инфраструктуры, а оговорка-приставка сбивала ответ («Не могу подтвердить
негативный вывод…» над утренним брифом и планом Архитектора).
"""
from __future__ import annotations

import json
import os
import re

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


# Объекты инфраструктуры, о состоянии которых делается диагноз.
_SUBJECT = re.compile(
    r'(?is)(?:cron|крон|синхрониз|\bsync|сервис|service|интеграц|integration|'
    r'подключен|коннектор|connector|webhook|вебхук|\bключ|\bkey|токен|token|'
    r'баз[аеуы]\b|database|\bdb\b|сервер|server|магазин|store|деплой|deploy|'
    r'очеред|queue|\bjob|джоб|расписан|schedule|\bmcp\b|\bapi\b|vercel|supabase|'
    r'pressable|\bbrain\b|контур|мониторинг|monitoring|бэкап|backup|воркер|worker|'
    r'заказ|order|оплат|payment|почт|email|домен|domain|dns|ssl|сертификат)')
# Предложения, где негатив — не диагноз: условное придаточное В НАЧАЛЕ
# предложения («Если кнопка не работает, …»), «ничего не отсутствует»,
# прошлое исправленное состояние, желаемое поведение. Совет в конце диагноза
# («Интеграция не работает — проверьте токен») диагноз не отменяет.
_CONDITIONAL_START = re.compile(
    r'(?is)^\W*(?:(?:если|в\s+случае|if|in\s+case)\b'
    r'(?!\s+(?:you\s+(?:ask|look|check)|коротко|честно|кратко|по\s+сути))|'
    r'(?:убедитесь|проверьте|make\s+sure|check|ensure)\b[^.!?—–]{0,40}\b(?:что|that)\b(?![^.!?]*[—–]))')
_NOT_A_DIAGNOSIS = re.compile(
    r'(?is)(?:^|\W)(?:nothing\s+is\s+(?:missing|broken)|ничего\s+не\s+(?:отсутствует|сломано)|'
    r'не\s+отсутству\w*|исправлен\w*|fixed|был\w*\s+сломан|was\s+broken|'
    r'как\s+вы\s+(?:и\s+)?просили|as\s+(?:you\s+)?requested|по\s+задумке|by\s+design)(?:\W|$)')
_SENTENCE = re.compile(r'(?<=[.!?;])\s+|\n+')


def _marker_pattern(marker: str) -> re.Pattern[str]:
    # Границы слова: «broken» не должно совпадать внутри «unbroken».
    return re.compile(r'(?<![\wа-яё])' + re.escape(marker), re.IGNORECASE)


def has_negative_claim(text: str) -> bool:
    """Есть ли в тексте диагноз-негатив о состоянии инфраструктуры.

    Явно заданные владельцем маркеры (MSTY_NEGATIVE_MARKERS) ищутся как есть:
    владелец сам выбрал точные формулировки; фильтры контекста применяются
    только к списку по умолчанию.
    """
    markers = negative_markers()
    patterns = [_marker_pattern(marker) for marker in markers]
    custom = os.getenv('MSTY_NEGATIVE_MARKERS') is not None
    for sentence in _SENTENCE.split(text or ''):
        if not any(pattern.search(sentence) for pattern in patterns):
            continue
        if not custom and (_CONDITIONAL_START.search(sentence) or
                           _NOT_A_DIAGNOSIS.search(sentence) or not _SUBJECT.search(sentence)):
            continue
        return True
    return False


# Вопрос о состоянии системы: статус, сбой, «работает ли», проверка сервиса.
# Ищется только в начале сообщения владельца (STATUS_HEAD символов): длинные
# поручения и брифы несут данные ниже («статус», «ошибка» в выгрузке), это не
# их вопрос.
STATUS_HEAD = 400
_STATUS_INTENT = re.compile(
    r'(?is)(?:статус|состояни|как\s+там|работа\w*\s+ли|не\s+работа|почему\s+не\b|'
    r'сломал|сломан|упал|падает|лежит|сбо[йия]|ошибк|здоров|жив\w*\s+ли|'
    r'доступ\w*\s+ли|настроен|подключ[её]н|не\s+обновля|обновля\w*\s+ли|'
    r'синхрониз|\bsync|провер(?:ь|ьте|ить|им)\b|диагност|почин|'
    r'\bhealth|\bstatus\b|is\s+\w+\s+(?:up|down|running|working)|not\s+working|'
    r'\bbroken\b|\bdown\b|failing|\bcheck\b)')


def _owner_text(messages) -> str:
    """Текст последнего сообщения владельца (user/human)."""
    for message in reversed(list(messages or ())):
        role = (message.get('role') if isinstance(message, dict)
                else getattr(message, 'type', None))
        if role in ('user', 'human'):
            return _text(message.get('content') if isinstance(message, dict)
                         else message.content)
    return ''


def is_status_question(text: str) -> bool:
    """Начало сообщения спрашивает о состоянии системы/сервиса."""
    return bool(_STATUS_INTENT.search((text or '')[:STATUS_HEAD]))


def _status_tool_called(state) -> bool:
    """В текущем ходе модель вызывала профильное статус-чтение (любой исход)."""
    messages = state.get('messages') or ()
    names = _call_names(messages)
    for message in _tool_messages(_current_turn(messages)):
        name = (message.get('name') if isinstance(message, dict) else getattr(message, 'name', None))
        call_id = (message.get('tool_call_id') if isinstance(message, dict)
                   else getattr(message, 'tool_call_id', None))
        entry = msty_registry.find(name or names.get(call_id) or '')
        if entry is not None and entry.evidence_class == 'status_read':
            return True
    return False


def in_scope(state) -> bool:
    """Gate касается только диагноза состояния системы, не планов и текстов."""
    return (is_status_question(_owner_text(state.get('messages') or ()))
            or _status_tool_called(state))


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


def _current_turn(messages) -> list:
    """Сообщения после последнего сообщения владельца (текущий ход)."""
    messages = list(messages or ())
    for index in range(len(messages) - 1, -1, -1):
        message = messages[index]
        role = (message.get('role') if isinstance(message, dict)
                else getattr(message, 'type', None))
        if role in ('user', 'human'):
            return messages[index + 1:]
    return messages


# Только «не смог прочитать». state=failed/error — данные о системе (сбой
# синхронизации), это как раз доказательство негативного диагноза.
_UNREAD_STATES = frozenset(('unavailable', 'rejected', 'denied'))


def _unread_state(text: str) -> bool:
    """Статус-инструмент ответил, что сам статус прочитать не удалось.

    Для таксономии это данные (вызов прошёл), но доказательством состояния
    системы такой ответ не является: {"state":"unavailable","code":...}.
    """
    stripped = text.strip()
    if not stripped.startswith('{'):
        return False
    try:
        data = json.loads(stripped)
    except ValueError:
        return False
    value = data.get('state') if isinstance(data, dict) else None
    return isinstance(value, str) and value.lower() in _UNREAD_STATES


def successful_status_reads(state) -> list[str]:
    """Имена evidence-инструментов (status_read) с успешным результатом в ТЕКУЩЕМ ходе.

    Источник — только история сообщений этого хода: результаты статус-чтений
    без признаков отказа. Журнал tau_evidence — наблюдаемость, не доказательство:
    он переживает ходы и не должен разрешать негатив на новую тему.
    """
    found = []
    messages = _current_turn(state.get('messages') or ())
    names = _call_names(state.get('messages') or ())
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
        if _unread_state(text):
            continue  # инструмент не смог прочитать статус (state=unavailable/rejected)
        if name not in found:
            found.append(name)
    return found


def evidence_summary(state) -> str:
    """Перечень проверенного в текущем ходе: имена вызванных инструментов и исход."""
    messages = state.get('messages') or ()
    names = _call_names(messages)
    checked = []
    for message in _tool_messages(_current_turn(messages)):
        if isinstance(message, dict):
            name = message.get('name') or names.get(message.get('tool_call_id'))
            status = message.get('status')
        else:
            name = getattr(message, 'name', None) or names.get(getattr(message, 'tool_call_id', None))
            status = getattr(message, 'status', None)
        if isinstance(name, str):
            item = f"{name}: {'отказ' if status == 'error' else 'ok'}"
            if item not in checked:
                checked.append(item)
    return '; '.join(checked[:12]) if checked else 'ничего'


def rewrite_unconfirmed(content: str, state) -> str:
    status_tools = ', '.join(sorted(
        entry.name for entry in msty_registry.TOOLS
        if entry.evidence_class == 'status_read'))
    return ('Не могу подтвердить негативный вывод: в этом ходе не было успешного '
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
    if not in_scope(state):
        return result, None  # план, бриф, текст — не диагноз системы
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
