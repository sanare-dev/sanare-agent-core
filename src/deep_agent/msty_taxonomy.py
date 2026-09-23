"""TAU, слой L4: таксономия отказов и политики восстановления.

Кодовая классификация вместо prompt-правила (таблица §4.2 эталонной архитектуры):

| класс            | сигнатуры                              | политика |
|------------------|----------------------------------------|----------|
| unknown_tool     | имя вне реестра / «Unknown tool»       | корректирующее сообщение + fuzzy; 1 повтор |
| invalid_args     | нарушение схемы                        | ошибки валидации модели; 1 повтор |
| transient        | timeout, 429, 5xx, обрыв соединения    | retry ×2 с backoff+jitter, ТОЛЬКО идемпотентные |
| deterministic    | 400/401/403/404/422, отказ бизнес-логики | не повторять; смена инструмента |
| unknown_state    | ответ потерян, эффект неизвестен       | freeze + сверка, не слепой retry |

Исполнение внешних MCP-инструментов живёт за коннектором клиента Msty: в графе
нет их HTTP-статусов, поэтому класс внешнего отказа определяется по сигнатуре
текста результата. Это эвристика аннотации и учёта — протокольные проверки
(допуск, проверенные наблюдения) она не подменяет.

Бюджет восстановления: суммарно 2 попытки на один и тот же вызов (имя + каноничный
дайджест аргументов); дальше — честный дегрейд с перечнем проверенного, а не
зацикливание корректирующих сообщений. Учёт ведётся в состоянии графа
(`tau_errors`); это же задел наблюдаемости и Evidence Gate.
"""
from __future__ import annotations

import re

from langchain_core.messages import ToolMessage

from . import msty_execution, msty_registry

UNKNOWN_TOOL = 'unknown_tool'
INVALID_ARGS = 'invalid_args'
TRANSIENT = 'transient'
DETERMINISTIC = 'deterministic'
UNKNOWN_STATE = 'unknown_state'
CLASSES = frozenset((UNKNOWN_TOOL, INVALID_ARGS, TRANSIENT, DETERMINISTIC, UNKNOWN_STATE))

# Суммарный бюджет попыток на один вызов (исходная + один повтор).
ATTEMPT_BUDGET = 2
# Inline-повторы transient-отказа идемпотентного вызова внутри одной границы
# исполнения; поверх действует общий бюджет ATTEMPT_BUDGET.
TRANSIENT_RETRIES = 2

#: Подсказка модели на каждый класс отказа (детерминированная политика, не prompt).
POLICY_HINT = {
    UNKNOWN_TOOL: ('имя относится к другому коннектору или не существует; вызови '
                   'инструмент напрямую по его собственной схеме из реестра, не через execute_tool'),
    INVALID_ARGS: 'исправь аргументы по схеме и повтори вызов один раз',
    TRANSIENT: 'временный сбой; допустим один повтор того же вызова без изменений',
    DETERMINISTIC: 'повтор без изменения условий бесполезен; смени инструмент или зафиксируй блокер',
    UNKNOWN_STATE: ('исход неизвестен; сначала сверь состояние read-only вызовом, '
                    'не повторяй побочный эффект вслепую'),
}

_UNKNOWN_TOOL_TEXT = re.compile(
    r'(?is)\bunknown tool\b|инструмент\s+\S+\s+не\s+(?:существует|найден)|no such tool')
_INVALID_ARGS_TEXT = re.compile(
    r'(?is)invalid (?:argument|parameter|args)|validation error|schema violation|'
    r'не\s+прош[её]л\s+валидац|наруша\w+\s+схем')
_TRANSIENT_TEXT = re.compile(
    r'(?is)time[ds]? ?out|тайм-?аут|\b429\b|\b50[234]\b|rate\s*limit|temporarily unavailable|'
    r'connection (?:refused|reset|aborted)|временно недоступ|превышено время ожидания')
_DETERMINISTIC_TEXT = re.compile(
    r'(?is)\b40[0134]\b|\b422\b|permission denied|\bforbidden\b|\bbad request\b|'
    r'отказано в доступе|недопустим\w+\s+запрос')


def classify_tool_text(content) -> str | None:
    """Класс отказа по тексту результата инструмента; None — признаков отказа нет."""
    if not isinstance(content, str) or not content.strip():
        return None
    head = content[:2000]  # сигнатуры ищутся в заголовке отказа, не в данных
    if _UNKNOWN_TOOL_TEXT.search(head):
        return UNKNOWN_TOOL
    if _INVALID_ARGS_TEXT.search(head):
        return INVALID_ARGS
    if _TRANSIENT_TEXT.search(head):
        return TRANSIENT
    if _DETERMINISTIC_TEXT.search(head):
        return DETERMINISTIC
    return None


# Имена типов исключений, которые заведомо являются временными сбоями транспорта.
# Перечисление по именам, а не импорт SDK провайдеров: адаптер не должен
# затаскивать их иерархии в слой восстановления.
_TRANSIENT_EXC_NAMES = frozenset((
    'TimeoutError', 'ConnectError', 'ConnectTimeout', 'ReadTimeout', 'WriteTimeout',
    'PoolTimeout', 'ConnectionError', 'APIConnectionError', 'APITimeoutError',
    'RateLimitError', 'InternalServerError', 'ServiceUnavailableError',
    'RemoteProtocolError', 'ReadError', 'ConnectError',
))
_TRANSIENT_STATUS = frozenset((408, 409, 425, 429, 500, 502, 503, 504))


def is_transient_exception(error: BaseException) -> bool:
    """Transient-сбой по типу исключения или HTTP-статусу; неизвестное = False.

    Консервативно: нераспознанное исключение НЕ transient, повторять его нельзя.
    """
    if isinstance(error, (TimeoutError, ConnectionError)):
        return True
    if type(error).__name__ in _TRANSIENT_EXC_NAMES:
        return True
    status = getattr(error, 'status_code', None)
    if status is None:
        response = getattr(error, 'response', None)
        status = getattr(response, 'status_code', None)
    return status in _TRANSIENT_STATUS


def fingerprint(call: dict) -> str:
    """Стабильный ключ вызова: имя + каноничный дайджест аргументов."""
    name = call.get('name') if isinstance(call.get('name'), str) else ''
    try:
        digest = msty_execution.canonical_digest(call.get('args'))[:16]
    except msty_execution.ExecutionProtocolError:
        digest = 'unhashable'
    return name + ':' + digest


def is_idempotent(name: str) -> bool:
    """Консервативная идемпотентность: read по манифесту; неизвестное = False."""
    entry = msty_registry.find(name)
    return entry is not None and entry.access == 'read'


def prior_attempts(errors, call: dict) -> int:
    """Сколько раз этот точный вызов уже отказал в текущем состоянии графа."""
    mark = fingerprint(call)
    return sum(1 for entry in errors or ()
               if isinstance(entry, dict) and entry.get('fingerprint') == mark)


def error_entry(call: dict, failure_class: str, source: str, attempt: int) -> dict:
    """Запись errors[] в состоянии графа: класс, инструмент, попытка, источник."""
    return {'version': 1, 'tool': call.get('name'), 'fingerprint': fingerprint(call),
            'class': failure_class, 'source': source, 'attempt': attempt,
            'tool_call_id': call.get('id')}


def budget_exhausted_message(call: dict, errors) -> ToolMessage:
    """Честный дегрейд: бюджет исчерпан, перечислено проверенное. Не сдача молча."""
    mark = fingerprint(call)
    history = [entry for entry in errors or ()
               if isinstance(entry, dict) and entry.get('fingerprint') == mark]
    classes = []
    for entry in history:
        if entry.get('class') not in classes:
            classes.append(entry.get('class'))
    checked = (', '.join(f"{entry.get('class')} (попытка {entry.get('attempt')})"
                         for entry in history) or 'нет записей')
    return ToolMessage(
        content=(f"error=budget_exhausted. Вызов '{call.get('name')}' не удался: бюджет "
                 f'{ATTEMPT_BUDGET} попыток исчерпан (классы: {", ".join(classes) or "неизвестно"}). '
                 f'Проверено: {checked}. Не повторяй этот вызов без изменения условий; '
                 'продолжай доступными инструментами или честно доложи блокер владельцу.'),
        name=call.get('name'), tool_call_id=call.get('id'), status='error')


def annotate_failure(content: str, failure_class: str) -> str:
    """Класс и детерминированная политика поверх текста отказа внешнего вызова."""
    hint = POLICY_HINT.get(failure_class)
    if not hint:
        return content
    return f'tau_class={failure_class}. Политика: {hint}.\n---\n{content}'
