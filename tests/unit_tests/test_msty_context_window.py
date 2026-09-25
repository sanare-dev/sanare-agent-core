"""Window-sized admission (brain-desk #145): offline, no paid model or network."""
import asyncio

import pytest

from deep_agent import msty, msty_execution as execution
from tests.unit_tests.test_msty_compaction import initial, install


def bound(profile='luna', limit=936000, **changes):
    state = initial(**changes)
    state['task_budget_binding'].update(profile=profile, input_limit=limit)
    return state


def test_window_limits_leave_output_and_count_reserve():
    assert execution.window_input_limit('luna') == 986000
    assert execution.window_input_limit('deepseek') == 936000
    # 200K models keep the historical admission (window minus 10%).
    assert execution.window_input_limit('opus') == 180000
    assert execution.window_input_limit('fable') == 180000
    assert set(execution.BINDING_PROFILES) <= set(execution.CONTEXT_WINDOWS)


def test_input_limit_reads_only_a_sane_binding():
    assert execution.input_limit(None) == execution.LEGACY_INPUT_LIMIT
    assert execution.input_limit({}) == execution.LEGACY_INPUT_LIMIT
    assert execution.input_limit(bound()) == 936000
    assert execution.input_limit(bound(limit=True)) == execution.LEGACY_INPUT_LIMIT
    assert execution.input_limit(bound(limit=10**7)) == execution.LEGACY_INPUT_LIMIT
    assert execution.input_limit(bound(profile='opus')) == execution.LEGACY_INPUT_LIMIT


def test_bound_window_admits_large_input_and_attests_that_limit(monkeypatch):
    seen = install(monkeypatch, ['OK'], [500000])
    result = asyncio.run(msty.graph.ainvoke(bound()))
    assert len(seen['requests']) == 1
    check = result['context_budget_check']
    assert check['status'] == 'accepted'
    assert check['limit'] == 936000 and check['input_tokens'] == 500000


def test_legacy_bridge_binding_keeps_180000(monkeypatch):
    seen = install(monkeypatch, ['OK'], [100])
    result = asyncio.run(msty.graph.ainvoke(initial()))
    assert len(seen['requests']) == 1
    assert result['context_budget_check']['limit'] == 180000


def test_overflow_of_bound_window_never_generates(monkeypatch):
    seen = install(monkeypatch, [], [936001])
    result = asyncio.run(msty.graph.ainvoke(bound()))
    assert not seen['requests']
    check = result['context_budget_check']
    assert check == {'version': 1, 'status': 'rejected', 'input_tokens': 936001, 'limit': 936000,
                     'window_admission': True}
    assert '936000' in result['result']['content']
    assert result['result']['usage_metadata']['total_tokens'] == 0


@pytest.mark.parametrize('profile,limit', [
    ('luna', 986001),       # above Luna's window minus reserve
    ('luna', 179999),       # below the legacy floor
    ('luna', True),         # bool is not an int limit
])
def test_binding_outside_profile_window_is_rejected_before_generation(monkeypatch, profile, limit):
    seen = install(monkeypatch, [], [])
    state = bound(profile=profile, limit=limit)
    result = asyncio.run(msty.graph.ainvoke(state))
    assert not seen['requests'] and not seen['caps']
    assert result['result']['usage_metadata']['total_tokens'] == 0
    assert result['context_budget_check']['status'] == 'rejected'
    assert result['context_budget_check']['limit'] == 180000


def test_200k_profile_cannot_be_bound_to_a_1m_limit():
    with pytest.raises(execution.ExecutionProtocolError):
        execution.validate_binding(bound(profile='opus'), 'opus', 100)
    execution.validate_binding(bound(profile='opus', limit=180000), 'opus', 100)


def test_binding_with_extra_field_is_rejected(monkeypatch):
    seen = install(monkeypatch, [], [])
    state = bound()
    state['task_budget_binding']['extra'] = 1
    result = asyncio.run(msty.graph.ainvoke(state))
    assert not seen['requests']
    assert result['execution']['status'] == 'blocked'


def test_every_attestation_declares_window_admission(monkeypatch):
    # The bridge sends a window binding only after this attestation (incident
    # 2026-09-24): accepted and rejected checks both carry it.
    seen = install(monkeypatch, ['OK'], [100])
    accepted = asyncio.run(msty.graph.ainvoke(initial()))['context_budget_check']
    assert accepted['window_admission'] is True and accepted['limit'] == 180000
    seen = install(monkeypatch, [], [936001])
    rejected = asyncio.run(msty.graph.ainvoke(bound()))['context_budget_check']
    assert not seen['requests']
    assert rejected['window_admission'] is True and rejected['status'] == 'rejected'
