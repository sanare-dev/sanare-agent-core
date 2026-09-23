"""TAU, слой L4: circuit breaker на удалённых вызовах native-контура.

MCP-соединения инструментов живут за коннектором клиента Msty и в графе
отсутствуют, поэтому breaker стоит на тех удалённых вызовах, которые граф
делает сам: генерация модели по серверным профилям (msty_models / LLM Gateway).

Семантика: N подряд transient-отказов одного соединения → контур открывается на
cooldown; пока контур открыт, вызовы получают детерминированный отказ «контур
недоступен, cooldown до T» без обращения к провайдеру (без расхода и без
нагрузки на лежащий сервис). Успешный вызов сбрасывает счётчик.

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

_lock = threading.Lock()
_connections: dict[str, dict] = {}


def open_remaining(connection: str) -> float | None:
    """Секунды до закрытия контура, если он открыт; None — вызовы допустимы."""
    try:
        with _lock:
            state = _connections.get(connection)
            if not state:
                return None
            remaining = state.get('open_until', 0.0) - time.monotonic()
            if remaining > 0:
                return remaining
            if state.get('open_until'):
                # Cooldown истёк: полуоткрытое состояние, один пробный вызов.
                state['open_until'] = 0.0
                state['failures'] = 0
            return None
    except Exception:
        return None


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
            if state['failures'] >= FAILURE_THRESHOLD and not state.get('open_until'):
                state['open_until'] = time.monotonic() + COOLDOWN_SECONDS
                return True
            return False
    except Exception:
        return False


def reset() -> None:
    """Полный сброс; для офлайн-тестов, не для рантайм-логики."""
    with _lock:
        _connections.clear()
