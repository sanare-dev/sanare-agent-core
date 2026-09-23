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

import json
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
    r'(?is)time[ds]? ?out|тайм-?аут|\b429\b|\b5\d\d\b|rate\s*limit|temporarily unavailable|'
    r'connection (?:refused|reset|aborted)|временно недоступ|превышено время ожидания')
_DETERMINISTIC_TEXT = re.compile(
    r'(?is)\b40[0134]\b|\b422\b|permission denied|\bforbidden\b|\bbad request\b|'
    r'отказано в доступе|недопустим\w+\s+запрос')


# Конверт отказа: результат ЯВНО объявляет себя ошибкой — в начале текста
# («Error: …», «MCP error -32000: …», «Ошибка: …», собственные префиксы TAU
# error=/tau_class=) или фразой «Tool 'x' failed» в первой строке. Сигнатуры
# классов ищутся только внутри конверта: успешный ответ с полями вроде
# "timeout": 300, историей статусов [200, 502] или job «check-404-pages»
# раньше помечался сбоем. «Error count: 0», «Сбой не обнаружен», «Не удалось
# найти ошибок» — данные, не конверт.
_ERROR_ENVELOPE = re.compile(
    r'(?is)^\W{0,3}(?:error\s*[:=\-—]|error\s+(?:executing|calling|while|in|during)\b|'
    r'tau_class=|mcp\s+error|tool\s+(?:execution\s+)?(?:failed|error)\b|failed\s+to\b|'
    r'exception\s*[:\-]|traceback\b|unknown\s+tool\b|no\s+such\s+tool\b|request\s+failed|'
    r'http\s+(?:error\s+)?[45]\d\d\b|ошибка\s*[:\-—]|ошибка\s+(?:выполнения|вызова|при)\b|'
    r'сбой\s*[:\-—]|не\s+удалось(?!\s+(?:найти|обнаружить)\s+(?:ни\s+)?(?:ошиб|сбо|проблем))\b|'
    r'инструмент\s+\S+\s+не\s+(?:существует|найден))')
_FIRST_LINE_FAILURE = re.compile(
    r'(?is)\btool\s+\S+\s+(?:failed|errored)\b|\bfailed\s+with\s+(?:status|error)\b|'
    r'(?:произошла|возникла)\s+ошибка\b')
_ENVELOPE_HEAD = 600
# Результат без конверта тоже отказ, если его ПЕРВАЯ СТРОКА (или весь короткий
# ответ) несёт строгую сигнатуру: HTTP-код с фразой причины, а не голое число.
_FIRST_LINE = 300
_STRICT_SIGNATURE = re.compile(
    r'(?is)\b(?:[45]\d\d\s+(?:not\s+found|unauthori[sz]ed|forbidden|bad\s+request|'
    r'service\s+unavailable|bad\s+gateway|gateway\s+time-?out|internal\s+server\s+error|'
    r'too\s+many\s+requests|unprocessable|conflict|request\s+timeout)|'
    r'(?:http|status(?:\s+code)?)\s*[:=]?\s*[45]\d\d\b|timed\s+out|time-?out\s+(?:after|exceeded|error)|'
    r'rate\s*limit\w*\s+(?:exceeded|reached|hit)|connection\s+(?:refused|reset|aborted)|'
    r'permission\s+denied|access\s+denied|invalid\s+(?:argument|parameter|args)|'
    r'validation\s+error|unknown\s+tool|no\s+such\s+tool|temporarily\s+unavailable|'
    r'тайм-?аут|превышено\s+время|отказано\s+в\s+доступе|временно\s+недоступ)')


def _json_error_text(data) -> str | None:
    """Текст ошибки разобранного JSON-результата; '' — ошибки нет."""
    if not isinstance(data, dict):
        return ''
    dump = json.dumps(data, ensure_ascii=False)[:_ENVELOPE_HEAD]
    if data.get('isError') is True or data.get('is_error') is True:
        return 'error: ' + dump
    error = data.get('error')
    if error and data.get('ok') is not True:
        return ('error: ' + (error if isinstance(error, str)
                             else json.dumps(error, ensure_ascii=False)))[:_ENVELOPE_HEAD]
    # success:false описывает сам вызов. ok:false / status:"failed" НЕ считаются
    # отказом: у статус-чтений (health, job status) это данные о состоянии
    # системы — ровно то доказательство, которое нужно Evidence Gate.
    if data.get('success') is False:
        return 'error: ' + dump
    return ''


def _text_parts(content) -> str | None:
    """Текст из списка MCP-частей [{"type":"text","text":...}]; None — не такой список."""
    if not isinstance(content, list):
        return None
    parts = [part.get('text') for part in content
             if isinstance(part, dict) and isinstance(part.get('text'), str)]
    return '\n'.join(parts) if parts else None


def classify_tool_text(content) -> str | None:
    """Класс отказа результата инструмента; None — результат не объявлен ошибкой.

    Консервативно в сторону «успех» только для данных: без конверта ошибки
    текст — данные, даже если в нём встречаются числа 404/502 или слово timeout.
    """
    joined = _text_parts(content)
    if joined is not None:
        content = joined
    if not isinstance(content, str) or not content.strip():
        return None
    stripped = content.strip()
    head = None
    if stripped[:1] in ('{', '['):
        try:
            data = json.loads(stripped)
        except ValueError:
            data = None
        if data is not None:
            nested = _text_parts(data)
            if nested is not None:
                return classify_tool_text(nested)
            head = _json_error_text(data)
            if not head:
                return None
    if head is None:
        head = stripped[:_ENVELOPE_HEAD]
        first = stripped.split('\n', 1)[0][:_FIRST_LINE]
        if not (_ERROR_ENVELOPE.search(head) or _FIRST_LINE_FAILURE.search(first) or
                _STRICT_SIGNATURE.search(first)):
            return None
    if _UNKNOWN_TOOL_TEXT.search(head):
        return UNKNOWN_TOOL
    if _INVALID_ARGS_TEXT.search(head):
        return INVALID_ARGS
    if _TRANSIENT_TEXT.search(head):
        return TRANSIENT
    if _DETERMINISTIC_TEXT.search(head):
        return DETERMINISTIC
    # Явная ошибка без распознанной сигнатуры: исход известен (отказ), повтор
    # без изменения условий бесполезен.
    return DETERMINISTIC


# Имена типов исключений, которые заведомо являются временными сбоями транспорта.
# Перечисление по именам, а не импорт SDK провайдеров: адаптер не должен
# затаскивать их иерархии в слой восстановления.
_TRANSIENT_EXC_NAMES = frozenset((
    'TimeoutError', 'ConnectError', 'ConnectTimeout', 'ReadTimeout', 'WriteTimeout',
    'PoolTimeout', 'ConnectionError', 'APIConnectionError', 'APITimeoutError',
    'RateLimitError', 'InternalServerError', 'ServiceUnavailableError',
    'RemoteProtocolError', 'ReadError', 'ConnectError',
))
_TRANSIENT_STATUS = frozenset((408, 425, 429, 500, 502, 503, 504))


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


#: Предел журналов tau_errors/tau_evidence в checkpoint (последние записи).
LOG_LIMIT = 50


def turn_of(messages) -> int:
    """Номер хода владельца: число человеческих сообщений в истории.

    Бюджет попыток живёт в пределах одного хода: после двух сбоев чтение не
    должно блокироваться навсегда во всех следующих ходах треда.
    """
    count = 0
    for message in messages or ():
        role = (message.get('role') if isinstance(message, dict)
                else getattr(message, 'type', None))
        count += role in ('user', 'human')
    return count


def prior_attempts(errors, call: dict, turn: int | None = None) -> int:
    """Сколько раз этот точный вызов уже отказал в текущем ходе (turn=None — всего)."""
    mark = fingerprint(call)
    return sum(1 for entry in errors or ()
               if isinstance(entry, dict) and entry.get('fingerprint') == mark
               and (turn is None or entry.get('turn') == turn))


def error_entry(call: dict, failure_class: str, source: str, attempt: int,
                turn: int | None = None) -> dict:
    """Запись errors[] в состоянии графа: класс, инструмент, попытка, источник, ход."""
    return {'version': 1, 'tool': call.get('name'), 'fingerprint': fingerprint(call),
            'class': failure_class, 'source': source, 'attempt': attempt,
            'tool_call_id': call.get('id'), 'turn': turn}


def budget_exhausted_message(call: dict, errors, turn: int | None = None) -> ToolMessage:
    """Честный дегрейд: бюджет исчерпан, перечислено проверенное. Не сдача молча."""
    mark = fingerprint(call)
    history = [entry for entry in errors or ()
               if isinstance(entry, dict) and entry.get('fingerprint') == mark
               and (turn is None or entry.get('turn') == turn)]
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
        name=call.get('name'), tool_call_id=call.get('id') or 'unknown', status='error')


def annotate_failure(content: str, failure_class: str, name: str | None = None) -> str:
    """Класс и детерминированная политика поверх текста отказа внешнего вызова.

    Для не-идемпотентного вызова (write или неизвестный) временный сбой — это
    неизвестный исход: подсказка «повтори» могла бы продублировать эффект.
    """
    if failure_class == TRANSIENT and name is not None and not is_idempotent(name):
        failure_class = UNKNOWN_STATE
    hint = POLICY_HINT.get(failure_class)
    if not hint:
        return content
    return f'tau_class={failure_class}. Политика: {hint}.\n---\n{content}'
