"""TAU, слой L4: таксономия отказов и политики восстановления.

Кодовая классификация вместо prompt-правила (таблица §4.2 эталонной архитектуры):

| класс            | сигнатуры                              | политика |
|------------------|----------------------------------------|----------|
| unknown_tool     | имя вне реестра / «Unknown tool»       | корректирующее сообщение + fuzzy; 1 повтор |
| invalid_args     | нарушение схемы                        | ошибки валидации модели; 1 повтор |
| transient        | timeout, 429, 5xx, обрыв соединения    | retry ×2 с backoff+jitter, ТОЛЬКО идемпотентные |
| deterministic    | 400/401/403/404/422, отказ бизнес-логики | не повторять; смена инструмента |
| unknown_state    | ответ потерян, эффект неизвестен       | freeze + сверка, не слепой retry |
| needs_owner      | нет входа / сервер не подключён (Brain Desk auth, not_connected) | не повторять; карточка «Переподключить», пробел в ответе |
| policy_refusal   | отказ прав/политики владельца или Brain Desk | не повторять и не обходить; спросить владельца |

Исполнение внешних MCP-инструментов живёт за коннектором клиента Msty: в графе
нет их HTTP-статусов, поэтому класс внешнего отказа определяется по сигнатуре
текста результата. Это эвристика аннотации и учёта — протокольные проверки
(допуск, проверенные наблюдения) она не подменяет.

Окно Brain Desk исполняет MCP на клиенте и сообщает отказ текстом с
префиксами «Ошибка инструмента:», «Инструмент отказал:», «Результат
неизвестен (…)», «Отклонено/Отказано Brain Desk…», а с brain-desk #309 —
завершающим блоком «[Brain Desk · самовосстановление] Класс: <класс>». Этот
блок — самый надёжный сигнал: класс окна переводится в класс TAU напрямую.

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
# Отдельные классы, а не deterministic: подсказка deterministic «смени
# инструмент» для них вредна — отказ прав нельзя обходить другим путём, а
# потерянный вход чинит только владелец (карточка «Переподключить»).
NEEDS_OWNER = 'needs_owner'
POLICY_REFUSAL = 'policy_refusal'
CLASSES = frozenset((UNKNOWN_TOOL, INVALID_ARGS, TRANSIENT, DETERMINISTIC, UNKNOWN_STATE,
                     NEEDS_OWNER, POLICY_REFUSAL))

# Суммарный бюджет попыток на один вызов (исходная + один повтор).
ATTEMPT_BUDGET = 2
# Inline-повторы transient-отказа идемпотентного вызова внутри одной границы
# исполнения; поверх действует общий бюджет ATTEMPT_BUDGET.
TRANSIENT_RETRIES = 2

#: Подсказка модели на каждый класс отказа (детерминированная политика, не prompt).
POLICY_HINT = {
    UNKNOWN_TOOL: ('имя относится к другому коннектору или не существует; вызови '
                   'инструмент напрямую по его собственной схеме из реестра, не через execute_tool'),
    INVALID_ARGS: ('исправь аргументы по схеме и повтори вызов один раз до ответа владельцу; '
                   'идентификаторы не угадывай — возьми точное значение из инструмента списка'),
    TRANSIENT: 'временный сбой; допустим один повтор того же вызова без изменений',
    DETERMINISTIC: 'повтор без изменения условий бесполезен; смени инструмент или зафиксируй блокер',
    UNKNOWN_STATE: ('исход неизвестен; сначала сверь состояние read-only вызовом, '
                    'не повторяй побочный эффект вслепую'),
    NEEDS_OWNER: ('нет входа или подключения к источнику; не повторяй вызов и не подменяй '
                  'источник; скажи владельцу о карточке «Переподключить» и назови пробел в ответе'),
    POLICY_REFUSAL: ('отказ прав или политики (владелец, Brain Desk, только чтение); не повторяй '
                     'и не обходи другим инструментом; если действие нужно — попроси владельца'),
}

_UNKNOWN_TOOL_TEXT = re.compile(
    r'(?is)\bunknown tool\b|инструмент\s+\S+\s+не\s+(?:существует|найден)|no such tool')
_INVALID_ARGS_TEXT = re.compile(
    r'(?is)invalid (?:argument|parameter|args)|validation error|schema violation|'
    r'не\s+прош[её]л\s+валидац|наруша\w+\s+схем|'
    # ZodError MCP-серверов (Supabase: «ref must be exactly 20 characters long»),
    # JSON-RPC -32602 Invalid params.
    r'\bZodError\b|\\?"code\\?"\s*:\s*\\?"(?:too_small|too_big|invalid_type|invalid_string|invalid_enum_value)|'
    r'must\s+be\s+exactly\b|\binvalid\s+params\b|-32602\b')
_TRANSIENT_TEXT = re.compile(
    r'(?is)time[ds]? ?out|тайм-?аут|\b429\b|\b5\d\d\b|rate\s*limit|temporarily unavailable|'
    r'connection (?:refused|reset|aborted)|временно недоступ|превышено время ожидания|'
    r'upstream\s+connect\s+error|no\s+healthy\s+upstream|ECONN(?:REFUSED|RESET)')
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
    r'http\s+(?:error\s+)?[45]\d\d\b(?!\s+count)|ошибка\s*[:\-—]|ошибка\s+(?:выполнения|вызова|при)\b|'
    r'сбой\s*[:\-—]|не\s+удалось(?!\s+(?:найти|обнаружить)\s+(?:ни\s+)?(?:ошиб|сбо|проблем))\b|'
    r'инструмент\s+\S+\s+не\s+(?:существует|найден))')
# Отказ, объявленный фразой В НАЧАЛЕ ответа (не в середине строки лога).
_FIRST_LINE_FAILURE = re.compile(
    r'(?is)^\W{0,3}(?:tool\s+\S+\s+(?:failed|errored)\b|'
    r'(?:the\s+)?service\s+is\s+temporarily\s+unavailable|'
    r'(?:при\s+\S+\s+)?(?:произошла|возникла)\s+ошибка\b)')
# Ответ прокси/шлюза вместо результата: статус-строка HTTP или HTML-страница
# ошибки в самом начале, типовые сообщения Envoy/Node/Cloudflare.
_PROXY_FAILURE = re.compile(
    r'(?is)^\W{0,3}(?:(?:HTTP/\d(?:\.\d)?\s+|status:\s*)[45]\d\d\b|'
    r'<(?:!doctype\s+html|html)[^>]*>.{0,300}?\b[45]\d\d\s+(?:bad\s+gateway|service\s+unavailable|'
    r'gateway\s+time-?out|internal\s+server\s+error|not\s+found|forbidden)|'
    r'upstream\s+connect\s+error|no\s+healthy\s+upstream|connect\s+ECONNREFUSED|'
    r'error\s+code:\s*5\d\d)')
# Конверты окна Brain Desk (src/lib/server/mcp/pool.ts): isError результата,
# JSON-RPC отказ, потерянный исход, отказ до вызова. Только в начале текста.
_CLIENT_ENVELOPE = re.compile(
    r'(?is)^\W{0,3}(?:ошибка\s+инструмента\s*:|инструмент\s+отказал\s*:|'
    r'результат\s+неизвестен\b|отклонено\s+brain\s+desk\b|отказано\s+brain\s+desk\b|'
    r'инструмент\s+\S+\s+запрещ[её]н\s+владельцем)')
# Исход неизвестен (обрыв, таймаут окна): действие могло выполниться.
_CLIENT_UNKNOWN_STATE = re.compile(r'(?is)^\W{0,3}результат\s+неизвестен\b')
# Отказ прав/политики окна: «Отказано Brain Desk», запрет владельца.
_CLIENT_REFUSAL = re.compile(
    r'(?is)^\W{0,3}(?:отказано\s+brain\s+desk\b|инструмент\s+\S+\s+запрещ[её]н\s+владельцем)|'
    r'требует\s+подтверждения\s+владельца')
# Предпроверка идентификатора окном (brain-desk #309): id не из инструмента списка.
_CLIENT_PRECHECK = re.compile(r'(?is)^\W{0,3}отклонено\s+brain\s+desk\s+до\s+вызова\b')
# Блок самовосстановления окна (brain-desk #309) — последний абзац результата:
# «\n\n[Brain Desk · самовосстановление] Класс: validation (…). подсказка[\nУрок: …]».
_RECOVERY_TAG = re.compile(r'(?:^|\n\n)\[Brain Desk · самовосстановление\] Класс: ([a-z_]+)\b')
#: Класс окна → класс TAU. not_found уточняется ниже (инструмент или объект).
WINDOW_CLASS = {
    'validation': INVALID_ARGS,
    'not_found': INVALID_ARGS,
    'auth': NEEDS_OWNER,
    'not_connected': NEEDS_OWNER,
    'transient': TRANSIENT,
    'permission': POLICY_REFUSAL,
}
# Счётчики вида «Jobs timed out: 0», «HTTP 503 count: 0» — данные.
_ZERO_COUNTER = re.compile(r'(?is)(?:count|errors?|failures?|timed\s+out|out)\s*[:=]\s*0\b')
_ENVELOPE_HEAD = 600
# Результат без конверта тоже отказ, если его ПЕРВАЯ СТРОКА (или весь короткий
# ответ) несёт строгую сигнатуру: HTTP-код с фразой причины, а не голое число.
_SHORT_RESULT = 300
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
    if isinstance(error, str) and error.strip().lower() in ('', 'none', 'null', 'ok', 'false'):
        error = None
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
    """Текст списка MCP-частей, где ВСЕ элементы {"type":"text","text":...};
    None — не такой список (строки из базы с колонкой text — данные)."""
    if not isinstance(content, list) or not content:
        return None
    if not all(isinstance(part, dict) and part.get('type') == 'text'
               and isinstance(part.get('text'), str) for part in content):
        return None
    return '\n'.join(part['text'] for part in content)


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
    window = _window_class(stripped)
    if window is not None:
        return window
    if _CLIENT_ENVELOPE.search(stripped[:_ENVELOPE_HEAD]):
        return _client_class(stripped)
    head = None
    if stripped[:1] == '{' or stripped[:2] in ('[{', '["', '[]'):
        try:
            data = json.loads(stripped)
        except ValueError:
            return None  # усечённый/битый JSON — данные, конверта ошибки нет
        if data is not None:
            nested = _text_parts(data)
            if nested is not None:
                return classify_tool_text(nested)
            head = _json_error_text(data)
            if not head:
                return None
    if head is None:
        head = stripped[:_ENVELOPE_HEAD]
        short = len(stripped) <= _SHORT_RESULT and '\n' not in stripped
        # Нулевой счётчик («HTTP 503 count: 0») гасит только строгую сигнатуру
        # короткого ответа; явный конверт ошибки («Failed to fetch …: 503 …
        # (retry count: 0)») остаётся отказом.
        if not (_ERROR_ENVELOPE.search(head) or _FIRST_LINE_FAILURE.search(head) or
                _PROXY_FAILURE.search(head) or
                short and _STRICT_SIGNATURE.search(stripped)
                and not _ZERO_COUNTER.search(stripped)):
            return None
    return _signature_class(head)


def _signature_class(head: str) -> str:
    """Класс отказа по сигнатуре текста, уже признанного ошибкой."""
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


def _client_class(text: str) -> str:
    """Класс отказа, объявленного конвертом окна Brain Desk (без блока класса)."""
    if _CLIENT_UNKNOWN_STATE.search(text):
        return UNKNOWN_STATE
    if _CLIENT_REFUSAL.search(text[:_ENVELOPE_HEAD]):
        return POLICY_REFUSAL
    if _CLIENT_PRECHECK.search(text):
        return INVALID_ARGS
    return _signature_class(text[:_ENVELOPE_HEAD])


def _window_class(text: str) -> str | None:
    """Класс из завершающего блока самовосстановления окна; None — блока нет.

    Блок принимается только последним абзацем результата: строка с тем же
    тегом в середине прочитанного файла или журнала — данные, не объявление.
    """
    match = None
    for match in _RECOVERY_TAG.finditer(text):
        pass
    if match is None or '\n\n' in text[match.end():]:
        return None
    window = match.group(1)
    # Потерянный исход важнее класса транспорта: запись могла выполниться.
    if _CLIENT_UNKNOWN_STATE.search(text) and window != 'permission':
        return UNKNOWN_STATE
    if window == 'not_found' and _UNKNOWN_TOOL_TEXT.search(text[:match.start()]):
        return UNKNOWN_TOOL
    if window in WINDOW_CLASS:
        return WINDOW_CLASS[window]
    # «unknown» и будущие классы окна: ошибка объявлена, класс — по сигнатуре.
    return _signature_class(text[:min(match.start(), _ENVELOPE_HEAD)])


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


def effective_class(failure_class: str, name: str | None = None) -> str:
    """Для не-идемпотентного вызова (write или неизвестный) временный сбой — это
    неизвестный исход: подсказка «повтори» могла бы продублировать эффект."""
    if failure_class == TRANSIENT and name is not None and not is_idempotent(name):
        return UNKNOWN_STATE
    return failure_class


#: Предел строк корректирующей заметки (один шаг — не больше нескольких вызовов).
NOTE_LIMIT = 5


def recovery_note(failures) -> str:
    """Короткая system-заметка следующему шагу модели об отказах прошлого шага.

    По образцу LangGraph ToolNode (handle_tool_errors: «Error: … Please fix your
    mistakes.») и вербальной рефлексии Reflexion: модель видит класс и политику
    отказа до того, как напишет ответ владельцу. '' — отказов не было.
    """
    lines = []
    for entry in failures or ():
        if not isinstance(entry, dict):
            continue
        tool, failure_class = entry.get('tool'), entry.get('class')
        if not isinstance(tool, str) or failure_class not in POLICY_HINT:
            continue
        failure_class = effective_class(failure_class, tool)
        attempt = entry.get('attempt')
        policy = ('бюджет попыток исчерпан: не повторяй, доложи владельцу проверенное и точный блокер'
                  if type(attempt) is int and attempt >= ATTEMPT_BUDGET else POLICY_HINT[failure_class])
        lines.append(f'- {tool[:80]}: tau_class={failure_class}; {policy}.')
    if not lines:
        return ''
    return ('TOOL_ERROR_RECOVERY_NOTE. Прошлый вызов инструмента отказал:\n'
            + '\n'.join(lines[:NOTE_LIMIT])
            + '\nСледуй политике класса до ответа владельцу; одинаковый вызов без изменений не повторяй.')


def annotate_failure(content: str, failure_class: str, name: str | None = None) -> str:
    """Класс и детерминированная политика поверх текста отказа внешнего вызова.

    Для не-идемпотентного вызова (write или неизвестный) временный сбой — это
    неизвестный исход: подсказка «повтори» могла бы продублировать эффект.
    """
    failure_class = effective_class(failure_class, name)
    hint = POLICY_HINT.get(failure_class)
    if not hint:
        return content
    return f'tau_class={failure_class}. Политика: {hint}.\n---\n{content}'
