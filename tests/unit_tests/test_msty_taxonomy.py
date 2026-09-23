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
