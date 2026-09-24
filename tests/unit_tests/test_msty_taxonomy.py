"""TAU L4: таксономия отказов, идемпотентность, бюджет, circuit breaker."""
import time

import pytest

from deep_agent import msty_breaker, msty_taxonomy


@pytest.mark.parametrize('content,expected', [
    ('Unknown tool: msty_store_sync_status', 'unknown_tool'),
    ('Error: Unknown tool', 'unknown_tool'),
    ('Invalid argument: path must be a string', 'invalid_args'),
    ('validation error: field required', 'invalid_args'),
    ('Request timed out after 30s', 'transient'),
    ('HTTP 503 Service Unavailable', 'transient'),
    ('rate limit exceeded, retry later', 'transient'),
    ('connection refused by upstream', 'transient'),
    ('404 Not Found', 'deterministic'),
    ('permission denied for role', 'deterministic'),
])
def test_classify_tool_text_failure_signatures(content, expected):
    assert msty_taxonomy.classify_tool_text(content) == expected


@pytest.mark.parametrize('content', [
    '{"status": "ok", "sha256": "41fd2b12"}',
    '/tmp/test_forbidden_local_write_cor0/external-verified.txt',  # путь с "forbidden"
    'Rows: 42. Last sync 3 minutes ago.',
    '',
    None,
])
def test_classify_tool_text_ignores_data_and_paths(content):
    assert msty_taxonomy.classify_tool_text(content) is None


class _HttpError(Exception):
    def __init__(self, status_code):
        self.status_code = status_code


@pytest.mark.parametrize('error,expected', [
    (TimeoutError('t'), True),
    (ConnectionError('reset'), True),
    (_HttpError(503), True),
    (_HttpError(429), True),
    (_HttpError(400), False),
    (ValueError('bad'), False),  # неизвестное — не transient, повторять нельзя
])
def test_is_transient_exception(error, expected):
    assert msty_taxonomy.is_transient_exception(error) is expected


def test_fingerprint_is_stable_and_distinguishes_args():
    one = msty_taxonomy.fingerprint({'name': 'execute_sql', 'args': {'query': 'select 1'}})
    same = msty_taxonomy.fingerprint({'name': 'execute_sql', 'args': {'query': 'select 1'}})
    other = msty_taxonomy.fingerprint({'name': 'execute_sql', 'args': {'query': 'select 2'}})
    assert one == same and one != other


def test_idempotency_comes_from_manifest_access():
    assert msty_taxonomy.is_idempotent('read_file') is True
    assert msty_taxonomy.is_idempotent('write_file') is False
    assert msty_taxonomy.is_idempotent('no_such_tool') is False  # консервативно


def test_prior_attempts_and_budget_message():
    call = {'name': 'execute_sql', 'args': {'query': 'x'}, 'id': 'c1'}
    errors = [msty_taxonomy.error_entry(call, 'transient', 'external', 1),
              msty_taxonomy.error_entry(call, 'transient', 'external', 2),
              msty_taxonomy.error_entry({'name': 'other', 'args': {}}, 'deterministic', 'guard', 1)]
    assert msty_taxonomy.prior_attempts(errors, call) == 2
    message = msty_taxonomy.budget_exhausted_message(call, errors)
    assert message.status == 'error'
    assert 'budget_exhausted' in message.content
    assert 'transient' in message.content and 'попытка 2' in message.content
    assert 'execute_sql' in message.content


def test_annotate_failure_adds_class_and_policy():
    annotated = msty_taxonomy.annotate_failure('HTTP 503', 'transient')
    assert annotated.startswith('tau_class=transient.')
    assert 'один повтор' in annotated and 'HTTP 503' in annotated


# --- Circuit breaker -----------------------------------------------------------

@pytest.fixture(autouse=True)
def reset_breaker():
    msty_breaker.reset()
    yield
    msty_breaker.reset()


def test_breaker_opens_after_threshold_and_blocks_during_cooldown():
    assert msty_breaker.open_remaining('model:luna') is None
    assert msty_breaker.record_transient_failure('model:luna') is False
    assert msty_breaker.record_transient_failure('model:luna') is False
    assert msty_breaker.record_transient_failure('model:luna') is True  # открылся
    remaining = msty_breaker.open_remaining('model:luna')
    assert remaining is not None and 0 < remaining <= msty_breaker.COOLDOWN_SECONDS
    # Другое соединение не затронуто.
    assert msty_breaker.open_remaining('model:deepseek') is None


def test_breaker_success_resets_and_expired_cooldown_half_opens():
    for _ in range(msty_breaker.FAILURE_THRESHOLD):
        msty_breaker.record_transient_failure('model:luna')
    msty_breaker.record_success('model:luna')
    assert msty_breaker.open_remaining('model:luna') is None
    for _ in range(msty_breaker.FAILURE_THRESHOLD):
        msty_breaker.record_transient_failure('model:luna')
    msty_breaker._connections['model:luna']['open_until'] = time.monotonic() - 1
    assert msty_breaker.open_remaining('model:luna') is None  # cooldown истёк


@pytest.mark.parametrize('content,expected', [
    ("Tool 'x' failed: upstream returned 503", 'transient'),
    ('Tool X failed with status 503 Service Unavailable. ' + 'x' * 400, 'transient'),
    ('[{"type":"text","text":"Error: timeout"}]', 'transient'),
    ('The service is temporarily unavailable. ' + 'y' * 400, 'transient'),
    ('При вызове произошла ошибка: 502 Bad Gateway. ' + 'z' * 400, 'transient'),
    ('{"success": false, "message": "Database connection failed"}', 'deterministic'),
    ('{"error": {"code": 500}}', 'transient'),
    ('tau_class=transient. Политика: x\n---\nError: 503 ' + 'w' * 400, 'transient'),
    # Данные, а не отказ.
    ('Error count: 0. All 12 jobs healthy.', None),
    ('Сбой не обнаружен, все сервисы работают.', None),
    ('Не удалось найти ошибок: всё ок', None),
    ('{"jobs":[{"timeout":300,"status":"ok"}]}', None),
    ('Job check-404-pages scheduled', None),
    ('{"history":[200,502]}', None),
    ('{"state":"healthy","evidence":[{"message":"fetch timeout"}]}', None),
    # ok:false / status:failed у статус-чтения — данные о системе (доказательство).
    ('{"ok": false, "checks": [{"name": "cron", "ok": false}]}', None),
    ('{"status": "failed", "job_id": "worker-1"}', None),
])
def test_classify_envelopes_and_data_after_review(content, expected):
    assert msty_taxonomy.classify_tool_text(content) == expected


@pytest.mark.parametrize('content', [
    '2026-09-23 10:00 Request timed out after 30s\n' + 'ok line\n' * 50,
    '[cron] job sync failed with status 1\nnext',
    'Журнал: 22.09 произошла ошибка, 23.09 исправлено',
    'Last 3 runs: tool sync failed once, then recovered',
    'Jobs timed out: 0',
    'HTTP 503 count: 0',
    '[{"id": 1, "text": "Request timed out"}]',
    '{"rows": ["a", "b timed out' + 'q' * 500,
    '{"error": "none", "state": "ok"}',
])
def test_successful_data_is_not_a_failure_round2(content):
    assert msty_taxonomy.classify_tool_text(content) is None


@pytest.mark.parametrize('content,expected', [
    ('Failed to fetch https://api.x/v1: 503 Service Unavailable (retry count: 0)', 'transient'),
    ('Request failed: timeout (retry_count=0)', 'transient'),
    ('HTTP 502 Bad Gateway (errors: 0 retried)', 'transient'),
    ('HTTP/1.1 503 Service Unavailable\nContent-Type: text/html', 'transient'),
    ('<html><body><h1>503 Service Unavailable</h1>\nNo server is available', 'transient'),
    ('<html>\n<head><title>502 Bad Gateway</title></head>\n<body>nginx</body></html>', 'transient'),
    ('upstream connect error or disconnect/reset before headers', 'transient'),
    ('connect ECONNREFUSED 127.0.0.1:5432', 'transient'),
    ('HTTP 503 count: 0', None),
    ('2026-09-23 HTTP 503 in logs\nok', None),
])
def test_proxy_failures_and_counters_round3(content, expected):
    assert msty_taxonomy.classify_tool_text(content) == expected


# --- Окно Brain Desk (brain-desk #309) ------------------------------------------

# Живой инцидент: Supabase list_tables с угаданным project_id из 19 символов.
ZOD_INCIDENT = ('Ошибка инструмента: {"error":{"name":"ZodError","message":"[\\n  {\\n    '
                '\\"origin\\": \\"string\\",\\n    \\"code\\": \\"too_small\\",\\n    '
                '\\"minimum\\": 20,\\n    \\"inclusive\\": true,\\n    \\"exact\\": true,\\n    '
                '\\"path\\": [],\\n    \\"message\\": \\"ref must be exactly 20 characters long\\"\\n  '
                '}\\n]"}}')
TAG = '\n\n[Brain Desk · самовосстановление] Класс: '


def test_live_zod_incident_is_invalid_args_not_success():
    # Раньше «Ошибка инструмента:» не узнавалась — отказ засчитывался успехом.
    assert msty_taxonomy.classify_tool_text(ZOD_INCIDENT) == 'invalid_args'
    tagged = (ZOD_INCIDENT + TAG + 'validation (неверный аргумент). Аргументы отклонены '
              'инструментом. Исправь их по схеме и повтори вызов один раз, до ответа владельцу. '
              'Вызови list_projects и возьми project_id оттуда.'
              '\nУрок: Supabase: не угадывать project_id — сначала list_projects')
    assert msty_taxonomy.classify_tool_text(tagged) == 'invalid_args'
    # Тот же текст частями MCP.
    assert msty_taxonomy.classify_tool_text([{'type': 'text', 'text': tagged}]) == 'invalid_args'
    annotated = msty_taxonomy.annotate_failure(tagged, 'invalid_args', 'list_tables')
    assert annotated.startswith('tau_class=invalid_args.')
    assert 'инструмента списка' in annotated and 'ref must be exactly' in annotated


@pytest.mark.parametrize('content', [
    '[{"schema":"public","name":"inbox_events","rows":12}]',
    '{"projects":[{"id":"abcdefghijklmnopqrst","name":"sanare-tax"}]}',
    'Tables: inbox_events, tax_obligations',
    # Тег в середине прочитанного журнала — данные, не объявление окна.
    'Журнал окна:' + TAG + 'validation (неверный аргумент). x\n\nследующая запись: ok',
])
def test_successful_client_results_stay_success(content):
    assert msty_taxonomy.classify_tool_text(content) is None


@pytest.mark.parametrize('window,body,expected', [
    ('validation', 'Ошибка инструмента: expected string, received number', 'invalid_args'),
    ('not_found', 'Инструмент отказал: 404 Not Found', 'invalid_args'),
    ('not_found', 'Ошибка инструмента: relation "public.x" does not exist', 'invalid_args'),
    ('not_found', 'Инструмент supabase_x не найден среди включённых серверов Brain Desk; '
                  'ничего не выполнено.', 'unknown_tool'),
    ('auth', 'Ошибка инструмента: 401 Unauthorized', 'needs_owner'),
    ('not_connected', 'Инструмент отказал: MCP-сервер: не подключён', 'needs_owner'),
    ('transient', 'Ошибка инструмента: 503 Service Unavailable', 'transient'),
    ('permission', 'Отказано Brain Desk: приватная папка. Ничего не выполнено; '
                   'не пытайся обойти другим путём.', 'policy_refusal'),
    ('validation', 'Отклонено Brain Desk до вызова: project_id не из list_projects.', 'invalid_args'),
    # Потерянный исход важнее класса транспорта: запись могла выполниться.
    ('not_connected', 'Результат неизвестен (MCP-сервер: процесс завершился (1)); действие '
                      'могло выполниться — не повторяй его без проверки.', 'unknown_state'),
    # Нераспознанный окном класс — по сигнатуре текста.
    ('unknown', 'Ошибка инструмента: something odd', 'deterministic'),
])
def test_window_recovery_class_is_authoritative(window, body, expected):
    content = body + TAG + window + ' (подпись). Подсказка окна.'
    assert msty_taxonomy.classify_tool_text(content) == expected


@pytest.mark.parametrize('content,expected', [
    # Окно без блока самовосстановления (до brain-desk #313).
    (ZOD_INCIDENT, 'invalid_args'),
    ('Инструмент отказал: MCP error -32602: Invalid params', 'invalid_args'),
    ('Ошибка инструмента: 503 Service Unavailable', 'transient'),
    ('Ошибка инструмента: relation does not exist', 'deterministic'),
    ('Результат неизвестен (timeout); действие могло выполниться — не повторяй его без проверки.',
     'unknown_state'),
    ('Отказано Brain Desk: только чтение. Ничего не выполнено; не пытайся обойти другим путём.',
     'policy_refusal'),
    ('Инструмент supabase_apply_migration запрещён владельцем (уровень риска «запрет»); '
     'ничего не выполнено. Не пытайся обойти другим путём.', 'policy_refusal'),
    ('Отклонено Brain Desk до вызова: project_id не из list_projects.', 'invalid_args'),
])
def test_client_envelopes_without_recovery_block(content, expected):
    assert msty_taxonomy.classify_tool_text(content) == expected


def test_policy_refusal_and_needs_owner_hints_forbid_bypass_and_retry():
    refusal = msty_taxonomy.POLICY_HINT['policy_refusal']
    assert 'не обходи' in refusal and 'не повторяй' in refusal
    owner = msty_taxonomy.POLICY_HINT['needs_owner']
    assert 'Переподключить' in owner and 'пробел' in owner and 'не повторяй' in owner
    assert {'needs_owner', 'policy_refusal'} <= msty_taxonomy.CLASSES


def test_recovery_note_names_tool_class_policy_and_budget():
    call = {'name': 'list_tables', 'args': {'project_id': 'x' * 19}, 'id': 'c1'}
    first = msty_taxonomy.error_entry(call, 'invalid_args', 'external', 1, 1)
    note = msty_taxonomy.recovery_note([first])
    assert note.startswith('TOOL_ERROR_RECOVERY_NOTE.')
    assert 'list_tables: tau_class=invalid_args' in note and 'до ответа владельцу' in note
    assert 'не повторяй' in note
    second = msty_taxonomy.error_entry(call, 'invalid_args', 'external', 2, 1)
    assert 'бюджет попыток исчерпан' in msty_taxonomy.recovery_note([second])
    # Временный сбой вызова вне манифеста — неизвестный исход, как в annotate_failure.
    lost = msty_taxonomy.error_entry({'name': 'no_such_write', 'args': {}}, 'transient', 'external', 1)
    assert 'tau_class=unknown_state' in msty_taxonomy.recovery_note([lost])
    assert msty_taxonomy.recovery_note([]) == ''
    assert msty_taxonomy.recovery_note([{'tool': 'x', 'class': 'bogus'}, 'junk']) == ''
