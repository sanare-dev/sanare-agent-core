"""TAU, слой L4: circuit breaker на удалённых вызовах native-контура.

MCP-соединения инструментов живут за коннектором клиента Msty и в графе
отсутствуют, поэтому breaker стоит на тех удалённых вызовах, которые граф
делает сам: генерация модели по серверным профилям (msty_models / LLM Gateway).

Семантика: N подряд transient-отказов одного соединения → контур открывается на
cooldown; пока контур открыт, вызовы получают детерминированный отказ «контур
недоступен, cooldown до T» без обращения к провайдеру (без расхода и без
нагрузки на лежащий сервис). Успешный вызов сбрасывает счётчик. После cooldown —
полуоткрытое состояние: ровно один пробный вызов; его провал сразу открывает
контур снова, остальные запросы во время пробы получают отказ.

Ключ — профиль модели: провайдер и его ключ общие для всех пользователей
процесса, поэтому лежащий провайдер закрывается для всех. 409 (конфликт) в
transient не входит — это ответ бизнес-логики, не сбой транспорта.

Состояние — в памяти процесса (Agent Server — один процесс-рантайм); файловая
система не используется. Breaker никогда не падает сам: его сбой не должен
становиться сбоем контура.
"""
from __future__ import annotations

import threading
import time

#: Подряд идущих transient-отказов до открытия контура.
FAILURE_THRESHOLD = 3
#: Секунды охлаждения открытого контура.
COOLDOWN_SECONDS = 60.0
#: Предел ожидания исхода пробного вызова полуоткрытого контура.
PROBE_SECONDS = 150.0

_lock = threading.Lock()
_token_seq = 0
_connections: dict[str, dict] = {}


def open_remaining(connection: str) -> float | None:
    """Чистое чтение: секунды до закрытия контура или None. Пробу не занимает;
    для допуска вызова использовать admit()."""
    try:
        with _lock:
            state = _connections.get(connection)
            if not state:
                return None
            now = time.monotonic()
            remaining = state.get('open_until', 0.0) - now
            if remaining > 0:
                return remaining
            if state.get('probe_until', 0.0) > now:
                return state['probe_until'] - now
            return None
    except Exception:
        return None


def admit(connection: str) -> tuple[float | None, int]:
    """(секунды до закрытия | None, токен пробы). Токен != 0 — этот вызов
    владеет пробой полуоткрытого контура и только он снимает её в release_probe."""
    try:
        with _lock:
            state = _connections.get(connection)
            if not state:
                return None, 0
            now = time.monotonic()
            remaining = state.get('open_until', 0.0) - now
            if remaining > 0:
                return remaining, 0
            if state.get('probe_until', 0.0) > now:
                return state['probe_until'] - now, 0  # проба уже идёт
            if state.get('open_until'):
                # Cooldown истёк: полуоткрытое состояние, один пробный вызов.
                # Провал пробы (failures = порог-1 → +1) сразу открывает контур.
                global _token_seq
                _token_seq += 1
                state['open_until'] = 0.0
                state['failures'] = FAILURE_THRESHOLD - 1
                state['probe_until'] = now + PROBE_SECONDS
                state['probe_token'] = _token_seq
                return None, _token_seq
            return None, 0
    except Exception:
        return None, 0


def record_success(connection: str) -> None:
    try:
        with _lock:
            _connections.pop(connection, None)
    except Exception:
        pass


def record_transient_failure(connection: str) -> bool:
    """Учесть transient-отказ; True — контур только что открылся."""
    try:
        with _lock:
            state = _connections.setdefault(connection, {'failures': 0, 'open_until': 0.0})
            state['failures'] += 1
            state['probe_until'] = 0.0
            if state['failures'] >= FAILURE_THRESHOLD and not state.get('open_until'):
                state['open_until'] = time.monotonic() + COOLDOWN_SECONDS
                return True
            return False
    except Exception:
        return False


def release_probe(connection: str, token: int = 0) -> None:
    """Снять отметку пробы без изменения счётчика; вызывать в finally.

    Проба, прерванная отменой (CancelledError, таймаут под-прогона) или
    нетранспортной ошибкой, иначе держала бы профиль закрытым для всех до
    PROBE_SECONDS, хотя отказов не было.
    """
    try:
        with _lock:
            state = _connections.get(connection)
            # Снимает только владелец: отмена постороннего вызова, начатого до
            # открытия контура, не должна пропускать всех во время пробы.
            if state and token and state.get('probe_token') == token:
                state['probe_until'] = 0.0
                state['probe_token'] = 0
    except Exception:
        pass


def reset() -> None:
    """Полный сброс; для офлайн-тестов, не для рантайм-логики."""
    with _lock:
        _connections.clear()
