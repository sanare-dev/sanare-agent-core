"""Additive local learning receipts never substitute for artifact evidence.

Offline only: synthetic receipts and a mocked model on the compiled native graph.
No local artifact reads, NAS projection, credentials or model APIs are used.
"""
import asyncio
from copy import deepcopy
import json

import pytest
from langchain_core.messages import AIMessage
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.types import Command

from deep_agent import msty, msty_task as task
from tests.unit_tests.test_msty_compaction import initial, install, USAGE
from tests.unit_tests.test_msty_task import PLAN, VERIFY, TOOLS, plan_call, plan_receipt, verify_receipt, payload


def context(count=1):
    return {'schema': 'msty.learning.context.v1', 'state': 'loaded', 'authority': 'reference_only',
        'retrieved_lesson_count': count, 'used_lesson_count': None,
        'lessons': [{'id': 'lesson-' + str(index), 'lesson': 'Проверить критерии текущего артефакта.',
            'source': 'synthetic-' + str(index) + '.json', 'source_sha256': '1' * 64,
            'lesson_sha256': '2' * 64, 'state': 'candidate', 'authority': 'reference_only',
            'published': False, 'semantic_verification': 'not_performed'} for index in range(count)],
        'quarantined_count': 0, 'scan_truncated': False,
        'reuse_receipt_path': task.JOURNAL + 'msty-learning-reuse-' + 'a' * 32 + '.json',
        'reuse_receipt_sha256': '3' * 64, 'code': None}


def outcome():
    return {'schema': 'msty.learning.outcome.v1', 'state': 'recorded',
        'lesson_id': 'msty-brain-lesson-auto-' + 'd' * 64,
        'lesson_receipt_path': task.JOURNAL + 'msty-brain-lesson-auto-' + 'd' * 64 + '.json',
        'lesson_receipt_sha256': '4' * 64, 'source_sha256': 'd' * 64,
        'metrics': {'outcome': 'passed', 'passed_checks': 1, 'failed_checks': 0, 'error_checks': 0,
            'baseline_outcome': 'failed', 'baseline_source_sha256': '5' * 64,
            'recovery_observed': True, 'retrieved_lesson_count': 1, 'used_lesson_count': None},
        'engram': {'state': 'recorded', 'code': None, 'event_id': 'msty-auto-' + 'd' * 64}, 'code': None}


def updated(receipt, contract=None):
    if contract is None:
        contract = task._plan(plan_call(), {**plan_receipt(), 'learning': context()})
    issued = {'native': {'id': 'verify', 'name': VERIFY, 'args': {'plan_id': contract['plan_id']}}}
    return task.observe({'task_contract': contract}, issued,
                        [{'tool_call_id': 'native', 'content': json.dumps(receipt, ensure_ascii=False)}])


def test_old_receipts_remain_compatible_without_synthetic_learning_success():
    contract = task._plan(plan_call(), plan_receipt())
    result = updated(verify_receipt(), contract)
    assert result['status'] == 'verified_against_observations'
    assert result['whole_task_completion_verified'] is False
    assert 'learning_context' not in contract and 'learning_outcome' not in result


@pytest.mark.parametrize('count', [0, 1, 3])
def test_context_preserves_reference_provenance_not_candidate_text(count):
    receipt = {**plan_receipt(), 'learning': context(count)}
    contract = task._plan(plan_call(), receipt)
    saved = contract['learning_context']
    assert saved['authority'] == 'reference_only' and saved['used_lesson_count'] is None
    assert saved['retrieved_lesson_count'] == len(saved['lesson_refs']) == count
    assert 'lessons' not in saved
    assert all(set(ref) == {'id', 'source', 'source_sha256', 'lesson_sha256'} for ref in saved['lesson_refs'])
    assert contract['status'] == 'planned' and contract['criteria_origin'] == 'model_proposed'
    assert contract['whole_task_completion_verified'] is False
    receipt['learning']['retrieved_lesson_count'] = 99
    assert saved['retrieved_lesson_count'] == count


@pytest.mark.parametrize('code,has_receipt', [('learning_context_unavailable', True),
                                           ('learning_context_unavailable', False),
                                           ('learning_reuse_not_persisted', False)])
def test_unavailable_context_remains_explicit_without_preventing_real_plan(code, has_receipt):
    value = context(0)
    value.update(state='unavailable', code=code)
    if not has_receipt:
        value.update(reuse_receipt_path=None, reuse_receipt_sha256=None)
    contract = task._plan(plan_call(), {**plan_receipt(), 'learning': value})
    assert contract['status'] == 'planned'
    assert contract['learning_context']['state'] == 'unavailable'


@pytest.mark.parametrize('mutate', [
    lambda c: c.update(unexpected=True), lambda c: c.pop('code'),
    lambda c: c.update(schema='msty.learning.context.v2'),
    lambda c: c.update(state='approved'), lambda c: c.update(authority='system'),
    lambda c: c.update(retrieved_lesson_count=True), lambda c: c.update(retrieved_lesson_count=4),
    lambda c: c.update(retrieved_lesson_count=0), lambda c: c.update(used_lesson_count=1),
    lambda c: c.update(quarantined_count=True), lambda c: c.update(quarantined_count=257),
    lambda c: c.update(scan_truncated='false'), lambda c: c.update(code='unknown'),
    lambda c: c.update(reuse_receipt_path=None, reuse_receipt_sha256=None),
    lambda c: c.update(reuse_receipt_path=task.JOURNAL + 'other.json'),
    lambda c: c.update(reuse_receipt_sha256='not-a-hash'),
    lambda c: c['lessons'][0].update(published='false'),
    lambda c: c['lessons'][0].update(published=True),
    lambda c: c['lessons'][0].update(authority='owner_approved'),
    lambda c: c['lessons'][0].update(state='promoted'),
    lambda c: c['lessons'][0].update(semantic_verification='verified'),
    lambda c: c['lessons'][0].update(id='bad/id'),
    lambda c: c['lessons'][0].update(lesson='short'),
    lambda c: c['lessons'][0].update(lesson='x' * 2001),
    lambda c: c['lessons'][0].update(source='../private.json'),
    lambda c: c['lessons'][0].update(source_sha256='0' * 63),
    lambda c: c['lessons'][0].update(lesson_sha256='F' * 64),
    lambda c: c['lessons'][0].update(extra='instruction'),
])
def test_invalid_context_is_rejected_not_silently_dropped(mutate):
    value = context()
    mutate(value)
    assert task._plan(plan_call(), {**plan_receipt(), 'learning': value}) is None


def test_context_aggregate_utf8_limit_and_duplicate_ids():
    value = context(3)
    for lesson in value['lessons']:
        lesson['lesson'] = 'я' * 1500
    assert task._plan(plan_call(), {**plan_receipt(), 'learning': value}) is None
    value = context(2)
    value['lessons'][1]['id'] = value['lessons'][0]['id']
    assert task._plan(plan_call(), {**plan_receipt(), 'learning': value}) is None


@pytest.mark.parametrize('state,code', [('recorded', None),
    ('unknown', 'engram_timeout_no_retry'), ('unknown', 'engram_failed_outcome_unknown'),
    ('unknown', 'engram_adapter_error'), ('unknown', 'engram_prior_attempt_unconfirmed'),
    ('unknown', 'engram_delivery_record_unavailable'),
    ('not_attempted', 'engram_adapter_unavailable'), ('not_attempted', 'engram_adapter_error')])
def test_valid_outcome_keeps_exact_nas_status_and_observed_not_causal_metrics(state, code):
    value = outcome()
    value['engram'].update(state=state, code=code)
    result = updated({**verify_receipt(), 'learning': value})
    assert result['status'] == 'verified_against_observations'
    assert result['learning_outcome']['engram']['state'] == state
    assert result['learning_outcome']['engram']['code'] == code
    assert result['learning_outcome']['metrics']['used_lesson_count'] is None
    assert result['whole_task_completion_verified'] is False
    value['engram']['state'] = 'modified-after-observation'
    assert result['learning_outcome']['engram']['state'] == state


def test_local_learning_error_is_explicit_and_does_not_rewrite_actual_file_check():
    value = {'schema': 'msty.learning.outcome.v1', 'state': 'error', 'lesson_id': None,
        'lesson_receipt_path': None, 'lesson_receipt_sha256': None, 'source_sha256': None,
        'metrics': None, 'engram': {'state': 'not_attempted', 'code': 'local_lesson_not_persisted', 'event_id': None},
        'code': 'local_learning_not_persisted'}
    result = updated({**verify_receipt(), 'learning': value})
    assert result['status'] == 'verified_against_observations'
    assert result['learning_outcome']['state'] == 'error'
    assert result['learning_outcome']['engram']['state'] == 'not_attempted'
    value['engram'].update(state='recorded')
    assert updated({**verify_receipt(), 'learning': value})['status'] == 'blocked'


@pytest.mark.parametrize('mutate', [
    lambda o: o.update(extra=True), lambda o: o.pop('code'),
    lambda o: o.update(schema='msty.learning.outcome.v2'),
    lambda o: o.update(state='verified'), lambda o: o.update(code='fake'),
    lambda o: o.update(source_sha256='0' * 64),
    lambda o: o.update(lesson_id='other'),
    lambda o: o.update(lesson_receipt_path=task.JOURNAL + 'other.json'),
    lambda o: o.update(lesson_receipt_sha256=True),
    lambda o: o['metrics'].update(outcome='failed'),
    lambda o: o['metrics'].update(passed_checks=True),
    lambda o: o['metrics'].update(passed_checks=0),
    lambda o: o['metrics'].update(failed_checks=1),
    lambda o: o['metrics'].update(error_checks=-1),
    lambda o: o['metrics'].update(baseline_outcome='unknown'),
    lambda o: o['metrics'].update(baseline_source_sha256='not-a-hash'),
    lambda o: o['metrics'].update(recovery_observed='true'),
    lambda o: o['metrics'].update(recovery_observed=False),
    lambda o: o['metrics'].update(retrieved_lesson_count=True),
    lambda o: o['metrics'].update(retrieved_lesson_count=0),
    lambda o: o['metrics'].update(used_lesson_count=1),
    lambda o: o['metrics'].update(causal_quality_improved=True),
    lambda o: o['engram'].update(state='unknown', code=None),
    lambda o: o['engram'].update(state='recorded', code='engram_timeout_no_retry'),
    lambda o: o['engram'].update(state='not_attempted', code='engram_timeout_no_retry'),
    lambda o: o['engram'].update(event_id='msty-auto-' + '0' * 64),
    lambda o: o['engram'].update(success=True),
])
def test_invalid_outcome_does_not_complete_or_inherit_previous_success(mutate):
    contract = updated({**verify_receipt(), 'learning': outcome()})
    value = outcome()
    mutate(value)
    result = updated({**verify_receipt(), 'learning': value}, contract)
    assert result['status'] == 'blocked'
    assert result['whole_task_completion_verified'] is False
    assert 'learning_outcome' not in result


@pytest.mark.parametrize('field', ['learning', 'unexpected'])
def test_optional_field_is_not_a_wildcard(field):
    assert task._plan(plan_call(), {**plan_receipt(), field: None}) is None
    assert updated({**verify_receipt(), field: None})['status'] == 'blocked'


def test_new_learning_cannot_weaken_existing_base_or_provenance_guards():
    receipt = {**verify_receipt(), 'learning': outcome()}
    for mutate in (lambda r: r.update(whole_task_completion_verified='false'),
                   lambda r: r.update(criteria_origin='owner_approved'),
                   lambda r: r.update(plan_sha256='0' * 64),
                   lambda r: r['source_hashes'][0].update(sha256='0' * 64),
                   lambda r: r.update(state='failed')):
        bad = deepcopy(receipt)
        mutate(bad)
        assert updated(bad)['status'] == 'blocked'
    assert task.observe({}, {}, [{'tool_call_id': 'unissued', 'content': json.dumps(receipt)}]) is None
    old_result = updated(receipt)
    old_only = updated(verify_receipt(), old_result)
    assert old_only['status'] == 'verified_against_observations'
    assert 'learning_outcome' not in old_only  # No stale learning success.


def test_compiled_native_cycle_accepts_additive_receipts_and_preserves_unknown_projection(monkeypatch):
    seen = install(monkeypatch, [AIMessage(content='Plan checks.', tool_calls=[plan_call()], usage_metadata=USAGE),
                                'Candidate finished.', 'Declared artifact checks passed.'], [100, 110, 120])
    async def scenario():
        graph = msty.builder.compile(checkpointer=InMemorySaver())
        cfg = {'configurable': {'thread_id': 'native-learning-synthetic'}}
        first = await graph.ainvoke(initial(tools=deepcopy(TOOLS)), cfg)
        second = await graph.ainvoke(Command(resume={first['__interrupt__'][0].id:
            payload(first, {**plan_receipt(), 'learning': context()})}), cfg)
        assert second['result']['tool_calls'][0]['name'] == VERIFY
        assert second['task_contract']['learning_context']['retrieved_lesson_count'] == 1
        value = outcome()
        value['engram'].update(state='unknown', code='engram_timeout_no_retry')
        third = await graph.ainvoke(Command(resume={second['__interrupt__'][0].id:
            payload(second, {**verify_receipt(), 'learning': value}, 'b1_second')}), cfg)
        assert len(seen['requests']) == 3
        assert third['execution']['status'] == 'verified_against_observations'
        assert third['task_contract']['whole_task_completion_verified'] is False
        assert third['task_contract']['learning_outcome']['engram']['state'] == 'unknown'
        assert not third.get('__interrupt__')
    asyncio.run(scenario())
