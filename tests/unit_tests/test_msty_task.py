"""Real native plan→verification graph cycle with synthetic client observations."""
import asyncio
from copy import deepcopy
import json

import pytest
from langchain_core.messages import AIMessage
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.types import Command

from deep_agent import msty, msty_task as task, msty_execution as execution, msty_models
from tests.unit_tests.test_msty_compaction import initial, install, USAGE

PLAN = 'sanare_admin_msty_task_plan'
VERIFY = 'sanare_admin_msty_task_verify'
ARGS = {'project_slug': 'llm', 'objective': 'Check synthetic artifact.',
        'requirements': ['Fixture contains expected value.'],
        'checks': [{'requirement_index': 0, 'path': 'work/fixture.txt',
                    'kind': 'text_contains', 'expected': 'expected'}]}
TOOLS = [
    {'type': 'function', 'function': {'name': PLAN, 'parameters': {
        'type': 'object', 'properties': {'project_slug': {'type': 'string'}, 'objective': {'type': 'string'},
            'requirements': {'type': 'array', 'items': {'type': 'string'}},
            'checks': {'type': 'array', 'items': {'type': 'object'}}},
        'required': list(ARGS), 'additionalProperties': False}}},
    {'type': 'function', 'function': {'name': VERIFY, 'parameters': {
        'type': 'object', 'properties': {'plan_id': {'type': 'string'}},
        'required': ['plan_id'], 'additionalProperties': False}}},
]
MEMORY = {'type': 'function', 'function': {'name': 'msty_admin_memory_search',
    'parameters': {'type': 'object', 'properties': {'query': {'type': 'string'}},
                   'required': ['query'], 'additionalProperties': False}}}


def plan_receipt():
    return {'schema': 'msty.task.plan.v1', 'state': 'planned', 'plan_id': 'taskplan-' + 'a' * 32,
            'plan_sha256': 'b' * 64, 'project_slug': 'llm', 'requirements_count': 1, 'checks_count': 1,
            'criteria_origin': 'model_proposed', 'whole_task_completion_verified': False}


def verify_receipt():
    return {**plan_receipt(), 'schema': 'msty.task.verification.v1', 'state': 'passed',
        'checks': [{'index': 0, 'requirement_index': 0, 'kind': 'text_contains', 'status': 'passed',
                    'code': 'match', 'artifact_sha256': 'c' * 64, 'size_bytes': 8}],
        'source_hashes': [{'path': 'work/fixture.txt', 'sha256': 'c' * 64, 'size_bytes': 8}],
        'receipt_path': 'project-governance/changes/synthetic.json', 'receipt_sha256': 'd' * 64}


def plan_call():
    return {'id': 'planned-call', 'name': PLAN, 'args': deepcopy(ARGS)}


def payload(first, receipt, client_id='b1_synthetic'):
    incoming = {key: deepcopy(first[key]) for key in ('messages', 'tools', 'max_tokens',
        'tool_choice', 'execution_protocol', 'execution_task_id', 'task_budget_binding', 'compaction_protocol')}
    incoming.update(result={}, context_budget_check=None)
    call = first['execution']['pending']['calls'][0]
    incoming['messages'] += [
        {'role': 'assistant', 'content': first['result']['content'], 'tool_calls': [
            {'id': client_id, 'type': 'function', 'function': {
                'name': call['name'], 'arguments': json.dumps(call['args'])}}]},
        {'role': 'tool', 'tool_call_id': client_id, 'content': json.dumps(receipt)},
    ]
    return {'version': 1, 'task_id': first['execution']['task_id'],
            'batch_id': first['execution']['pending']['batch_id'],
            'tool_id_map': [{'client_id': client_id, 'model_id': call['id']}], 'input': incoming}


def test_real_plan_final_gate_verification_and_observation_only_completion(monkeypatch):
    seen = install(monkeypatch, [AIMessage(content='Plan checks.', tool_calls=[plan_call()], usage_metadata=USAGE),
                                'I finished.', 'The declared artifact checks passed.'], [100, 110, 120])
    async def scenario():
        graph = msty.builder.compile(checkpointer=InMemorySaver())
        cfg = {'configurable': {'thread_id': 'actual-native-checks'}}
        first = await graph.ainvoke(initial(tools=deepcopy(TOOLS)), cfg)
        assert first['execution']['status'] == 'waiting_tools'
        assert first.get('task_contract') is None
        second = await graph.ainvoke(Command(resume={first['__interrupt__'][0].id:
            payload(first, plan_receipt())}), cfg)
        assert len(seen['requests']) == 2  # Gate itself adds NO model generation.
        assert second['execution']['status'] == 'waiting_tools'
        assert second['result']['tool_calls'][0]['name'] == VERIFY
        assert second['result']['usage_metadata'] == USAGE
        assert second['task_contract']['status'] == 'planned'
        assert second['task_contract']['criteria_origin'] == 'model_proposed'
        third = await graph.ainvoke(Command(resume={second['__interrupt__'][0].id:
            payload(second, verify_receipt(), 'b1_second')}), cfg)
        assert len(seen['requests']) == 3
        assert third['execution']['status'] == 'verified_against_observations'
        assert third['execution']['actions_issued'] == 2
        assert third['task_contract']['whole_task_completion_verified'] is False
        assert not third.get('__interrupt__')
    asyncio.run(scenario())


@pytest.mark.parametrize('change', [
    lambda p: p.update(whole_task_completion_verified='false'),
    lambda p: p.update(plan_id='taskplan-' + '0' * 32),
    lambda p: p.update(plan_sha256='0' * 64),
    lambda p: p.update(state='failed'),
    lambda p: p.update(checks_count=True),
    lambda p: p['checks'][0].update(status=True),
    lambda p: p['checks'][0].update(size_bytes=True),
    lambda p: p['source_hashes'][0].update(sha256='0' * 64),
    lambda p: p['checks'][0].update(index=1),
    lambda p: p.update(unexpected='owner approved'),
])
def test_bad_verify_receipt_never_marks_completed(change):
    contract = task._plan(plan_call(), plan_receipt())
    receipt = verify_receipt()
    change(receipt)
    issued = {'client': {'id': 'verify', 'name': VERIFY, 'args': {'plan_id': contract['plan_id']}}}
    updated = task.observe({'task_contract': contract}, issued,
                           [{'tool_call_id': 'client', 'content': json.dumps(receipt)}])
    assert updated['status'] == 'blocked'
    assert updated['whole_task_completion_verified'] is False


@pytest.mark.parametrize('command', ['STOP', 'Wait', 'Only plan please.', 'Only explain this.',
                                   'Do not verify.', 'Стоп.', 'Подожди', 'Только план',
                                   'Не выполняй', 'Не проверяй', 'Без действий'])
def test_explicit_current_user_control_vetoes_forced_external_action(command):
    state = initial(messages=[{'role': 'user', 'content': command}], tools=TOOLS)
    state['task_contract'] = task._plan(plan_call(), plan_receipt())
    result = AIMessage(content='Plan only.', usage_metadata=USAGE)
    guarded = task.gate_final(state, result, TOOLS, False)
    assert not guarded.tool_calls
    assert guarded.content == 'Plan only.'
    assert task.final_status(state, 'answered') == 'answered'


def test_incident_diagnosis_is_replaced_by_first_real_memory_action():
    message = ('sanarelab.club: не работает cron-синхронизация, новые заказы не видно; '
               'WooCommerce API возвращает ошибку.')
    state = initial(messages=[{'role': 'user', 'content': message}], tools=[MEMORY])
    result = AIMessage(content='Нужно проверить cron и логи.', usage_metadata=USAGE)
    guarded = task.gate_final(state, result, [MEMORY], False)
    assert guarded.tool_calls[0]['name'] == 'msty_admin_memory_search'
    assert guarded.tool_calls[0]['args']['query'] == message
    assert guarded.response_metadata['msty_completion_gate'] == 'incident_first_action_required'


def test_incident_memory_query_obeys_installed_tool_limit():
    message = 'Ошибка синхронизации. ' + 'подробность ' * 80
    state = initial(messages=[{'role': 'user', 'content': message}], tools=[MEMORY])
    guarded = task.gate_final(state, AIMessage(content='Диагноз.', usage_metadata=USAGE),
                              [MEMORY], False)
    query = guarded.tool_calls[0]['args']['query']
    assert len(query) == 300
    assert query.startswith('Ошибка синхронизации.')


def test_incident_first_action_never_repeats_after_a_real_tool_call():
    state = initial(messages=[
        {'role': 'user', 'content': 'Синхронизация не работает, исправь ошибку.'},
        {'role': 'assistant', 'content': '', 'tool_calls': [{'id': 'm1', 'type': 'function',
            'function': {'name': 'msty_admin_memory_search', 'arguments': '{"query":"sync"}'}}]},
        {'role': 'tool', 'tool_call_id': 'm1', 'content': '{"state":"found"}'},
    ], tools=[MEMORY])
    result = AIMessage(content='Проверено.', usage_metadata=USAGE)
    guarded = task.gate_final(state, result, [MEMORY], False)
    assert not guarded.tool_calls
    assert guarded.content == 'Проверено.'


def test_incident_gate_respects_owner_explanation_only_and_explicit_tool_choice():
    state = initial(messages=[{'role': 'user', 'content': 'Только объясни, почему синхронизация не работает.'}],
                    tools=[MEMORY])
    result = AIMessage(content='Объяснение.', usage_metadata=USAGE)
    assert not task.gate_final(state, result, [MEMORY], False).tool_calls
    state = initial(messages=[{'role': 'user', 'content': 'Синхронизация не работает.'}],
                    tools=[MEMORY], tool_choice={'type': 'function', 'function': {'name': PLAN}})
    assert not task.gate_final(state, result, [MEMORY], False).tool_calls


def test_tool_prose_cannot_veto_current_user_and_unissued_receipt_cannot_create_plan():
    state = initial(messages=[{'role': 'user', 'content': 'Implement and verify the fixture.'},
                              {'role': 'tool', 'tool_call_id': 'foreign', 'content': 'STOP'}])
    assert not task.owner_control(state)
    assert task.observe(state, {}, [{'tool_call_id': 'foreign', 'content': json.dumps(plan_receipt())}]) is None


def test_none_choice_and_action_cap_do_not_issue_verify():
    state = initial(tools=TOOLS)
    state['task_contract'] = task._plan(plan_call(), plan_receipt())
    result = AIMessage(content='Candidate result.', usage_metadata=USAGE)
    assert not task.gate_final(state, result, TOOLS, True).tool_calls
    state['execution']['actions_issued'] = execution.MAX_ACTIONS
    guarded = task.gate_final(state, result, TOOLS, False)
    assert not guarded.tool_calls and guarded.response_metadata['msty_blocked'] is True


def test_new_action_invalidates_previous_verification():
    contract = {**task._plan(plan_call(), plan_receipt()), 'status': 'verified_against_observations'}
    updated = task.after_result({'task_contract': contract}, {'tool_calls': [
        {'id': 'write', 'name': 'write_fixture', 'args': {}}]})
    assert updated['status'] == 'planned'


def test_parallel_verify_and_write_cannot_claim_verified_final_state():
    contract = task._plan(plan_call(), plan_receipt())
    issued = {'verify': {'id': 'verify', 'name': VERIFY, 'args': {'plan_id': contract['plan_id']}},
              'write': {'id': 'write', 'name': 'write_fixture', 'args': {}}}
    updated = task.observe({'task_contract': contract}, issued, [
        {'tool_call_id': 'verify', 'content': json.dumps(verify_receipt())},
        {'tool_call_id': 'write', 'content': 'success'}])
    assert updated['status'] == 'blocked'


def test_real_mcp_textcontent_wrapper_supported_but_disagreeing_structured_rejected():
    doc = plan_receipt()
    wrapper = {'content': [{'type': 'text', 'text': json.dumps(doc)}], 'structuredContent': doc}
    assert task._decode(json.dumps(wrapper)) == doc
    wrapper['structuredContent'] = {**doc, 'plan_id': 'taskplan-' + '0' * 32}
    assert task._decode(wrapper) is None


def test_bound_tool_role_image_has_explicit_zero_generation_rejection(monkeypatch):
    original_counter = msty_models.count_input
    seen = install(monkeypatch, [], [])
    # Restore the real local counter; its refusal is not an API/model call.
    monkeypatch.setattr(msty_models, 'count_input', original_counter)
    result = asyncio.run(msty.graph.ainvoke(initial(messages=[{'role': 'user', 'content': 'Read tool output.'},
        {'role': 'tool', 'tool_call_id': 'synthetic', 'content': [
        {'type': 'text', 'text': 'Synthetic screenshot.'},
        {'type': 'image_url', 'image_url': {'url': 'https://example.invalid/synthetic.png'}}]}])))
    assert not seen['requests']
    assert 'результате инструмента' in result['result']['content']
    assert result['result']['usage_metadata']['total_tokens'] == 0
    assert result['execution']['status'] == 'blocked'


def test_explicit_other_tool_choice_is_not_overridden():
    state = initial(tools=TOOLS, tool_choice={'type': 'function', 'function': {'name': PLAN}})
    state['task_contract'] = task._plan(plan_call(), plan_receipt())
    guarded = task.gate_final(state, AIMessage(content='Candidate'), TOOLS, False)
    assert not guarded.tool_calls and guarded.response_metadata['msty_blocked'] is True


@pytest.mark.parametrize('field,value', [('execution_task_id', 'another'),
    ('task_budget_binding', {}), ('compaction_protocol', None)])
def test_native_resume_cannot_rebind_task_budget_or_compaction(monkeypatch, field, value):
    seen = install(monkeypatch, [AIMessage(content='Plan.', tool_calls=[plan_call()], usage_metadata=USAGE)], [100])
    async def scenario():
        graph = msty.builder.compile(checkpointer=InMemorySaver())
        cfg = {'configurable': {'thread_id': 'immutable-native'}}
        first = await graph.ainvoke(initial(tools=TOOLS), cfg)
        value_to_resume = payload(first, plan_receipt())
        value_to_resume['input'][field] = value
        with pytest.raises(execution.ExecutionProtocolError):
            await graph.ainvoke(Command(resume={first['__interrupt__'][0].id: value_to_resume}), cfg)
        assert len(seen['requests']) == 1
    asyncio.run(scenario())


def transport_receipt(receipt):
    return {'status': 200, 'body': [{'type': 'text', 'text': json.dumps(receipt)}]}


def failed_contract():
    contract = task._plan(plan_call(), plan_receipt())
    failed = verify_receipt()
    failed.update(state='error', source_hashes=[])
    failed['checks'][0].update(status='error', code='artifact_unavailable_or_non_utf8',
                               artifact_sha256=None, size_bytes=None)
    call = {'id': 'verify-failed', 'name': VERIFY, 'args': {'plan_id': contract['plan_id']}}
    return task.observe({'task_contract': contract}, {'failed': call},
                        [{'tool_call_id': 'failed', 'content': json.dumps(transport_receipt(failed))}])


def test_msty_transport_receipt_preserves_plan_and_verification_identity():
    for receipt in (plan_receipt(), verify_receipt()):
        assert task._decode(json.dumps(transport_receipt(receipt))) == receipt
    contract = task.observe({}, {'planned': plan_call()}, [{
        'tool_call_id': 'planned', 'content': json.dumps(transport_receipt(plan_receipt()))}])
    assert contract['plan_id'] == plan_receipt()['plan_id']
    assert task._verified(contract, {'args': {'plan_id': contract['plan_id']}},
                          task._decode(json.dumps(transport_receipt(verify_receipt()))))


@pytest.mark.parametrize('status', [True, '200', 199, 300, 403, 500])
def test_failed_or_malformed_transport_status_never_attests_receipt(status):
    wrapper = transport_receipt(verify_receipt())
    wrapper['status'] = status
    assert task._decode(wrapper) is None


def test_transport_and_structured_disagreement_never_attests_receipt():
    wrapper = transport_receipt(verify_receipt())
    wrapper['structuredContent'] = plan_receipt()
    assert task._decode(wrapper) is None


def test_transport_receipt_nesting_is_bounded():
    wrapper = plan_receipt()
    for _ in range(10):
        wrapper = {'status': 200, 'body': wrapper}
    assert task._decode(wrapper) is None


@pytest.mark.parametrize('name', ['native_write_todos', 'native_read_file', 'native_write_file', 'write_fixture'])
def test_failed_verification_survives_unverified_followup_actions(name):
    contract = failed_contract()
    assert contract['status'] == 'blocked'
    updated = task.after_result({'task_contract': contract}, {'tool_calls': [
        {'id': 'followup', 'name': name, 'args': {}}]})
    assert updated == contract
    assert updated['verification_observation_sha256'] != execution.canonical_digest(None)


def test_only_valid_followup_verification_recovers_failed_contract_and_todo_preserves_it():
    contract = failed_contract()
    call = {'id': 'verify-recovery', 'name': VERIFY, 'args': {'plan_id': contract['plan_id']}}
    recovered = task.observe({'task_contract': contract}, {'recovery': call}, [{
        'tool_call_id': 'recovery', 'content': json.dumps(transport_receipt(verify_receipt()))}])
    assert recovered['status'] == 'verified_against_observations'
    assert recovered['whole_task_completion_verified'] is False
    after_todo = task.after_result({'task_contract': recovered}, {'tool_calls': [
        {'id': 'todo-done', 'name': 'native_write_todos', 'args': {}}]})
    assert after_todo == recovered
    state = initial(task_contract=recovered)
    assert task.final_status(state, 'answered') == 'verified_against_observations'


@pytest.mark.parametrize('choice', ['auto', 'none', {'type': 'none'}])
def test_already_verified_observation_is_not_downgraded_by_action_veto(choice):
    contract = {**task._plan(plan_call(), plan_receipt()), 'status': 'verified_against_observations'}
    state = initial(task_contract=contract, tool_choice=choice,
                    messages=[{'role': 'user', 'content': 'Создай файл, но не запускай публикацию.'}])
    assert task.owner_control(state)
    assert task.final_status(state, 'answered') == 'verified_against_observations'
    guarded = task.gate_final(state, AIMessage(content='Recorded check passed.'), TOOLS, False)
    assert not guarded.tool_calls


@pytest.mark.parametrize('command,choice', [
    ('Создай файл, но не запускай публикацию.', 'auto'), ('STOP', 'auto'),
    ('Finish the artifact.', 'none'), ('Finish the artifact.', {'type': 'none'}),
])
def test_failed_contract_blocks_success_prose_even_with_owner_veto_or_disabled_tools(command, choice):
    state = initial(messages=[{'role': 'user', 'content': command}], tool_choice=choice,
                    task_contract=failed_contract())
    raw = AIMessage(content='FALSE_SUCCESS_file_created_and_verified', usage_metadata=USAGE)
    guarded = task.gate_final(state, raw, TOOLS, choice != 'auto')
    assert 'FALSE_SUCCESS' not in guarded.content
    assert guarded.response_metadata['msty_blocked'] is True
    assert not guarded.tool_calls
    assert guarded.usage_metadata == USAGE
    assert task.final_status(state, 'answered') == 'blocked'


def test_real_native_failed_verify_todo_cannot_erase_failure_or_publish_success(monkeypatch):
    from deep_agent import msty_native
    from langgraph.store.memory import InMemoryStore

    verify = {'id': 'verify', 'name': VERIFY, 'args': {'plan_id': plan_receipt()['plan_id']}}
    todo = {'id': 'todo', 'name': 'native_write_todos', 'args': {'todos': [
        {'content': 'Synthetic task marked complete', 'status': 'completed'}]}}
    seen = install(monkeypatch, [
        AIMessage(content='Plan.', tool_calls=[plan_call()], usage_metadata=USAGE),
        AIMessage(content='Verify.', tool_calls=[verify], usage_metadata=USAGE),
        AIMessage(content='Checklist.', tool_calls=[todo], usage_metadata=USAGE),
        'FALSE_SUCCESS_file_created_and_verified'], [100, 110, 120, 130])

    def external_payload(state, receipt, client):
        state = {**state, 'messages': deepcopy(state['native_protocol_messages'])}
        return payload(state, transport_receipt(receipt), client)

    async def scenario():
        graph = msty_native.build_graph(checkpointer=InMemorySaver(), store=InMemoryStore())
        cfg = {'configurable': {'thread_id': 'failed-native-todo'}, 'recursion_limit': 64}
        async def invoke(value):
            output = await graph.ainvoke(value, cfg)
            saved = (await graph.aget_state(cfg)).values
            return {**saved, '__interrupt__': output.get('__interrupt__', ())}

        first = await invoke(initial(tools=deepcopy(TOOLS),
            messages=[{'role': 'user', 'content': 'Создай файл, но не запускай публикацию.'}]))
        second = await invoke(Command(resume={first['__interrupt__'][0].id:
            external_payload(first, plan_receipt(), 'b1_plan')}))
        assert second['task_contract']['status'] == 'planned'
        failed = verify_receipt()
        failed.update(state='error', source_hashes=[])
        failed['checks'][0].update(status='error', code='artifact_unavailable_or_non_utf8',
                                   artifact_sha256=None, size_bytes=None)
        third = await invoke(Command(resume={second['__interrupt__'][0].id:
            external_payload(second, failed, 'b1_failed')}))
        assert third['execution']['status'] == 'waiting_native'
        assert third['task_contract']['status'] == 'blocked'
        assert third['task_contract']['plan_id'] == plan_receipt()['plan_id']
        pending = third['__interrupt__'][0]
        final = await invoke(Command(resume={pending.id: {
            **pending.value, 'type': 'msty_native_resume'}}))
        assert final['execution']['status'] == 'blocked'
        assert final['task_contract'] == third['task_contract']
        assert 'FALSE_SUCCESS' not in final['result']['content']
        assert final['result']['usage_metadata'] == USAGE
        assert not final.get('__interrupt__')
        assert len(seen['requests']) == 4

    asyncio.run(scenario())


SITE_FILE = 'sanare_admin_msty_site_file'
SITE_PATCH = 'sanare_admin_msty_site_patch'
SITE_STATUS = 'sanare_admin_msty_site_status'
SITE_TOOLS = [
    {'type': 'function', 'function': {'name': SITE_FILE, 'parameters': {'type': 'object', 'properties': {
        'job_id': {'type': 'string'}, 'path': {'type': 'string'}, 'operation': {'type': 'string'},
        'content': {'type': 'string'}, 'expected_sha256': {'type': 'string'}},
        'required': ['job_id', 'path']}}},
    {'type': 'function', 'function': {'name': SITE_PATCH, 'parameters': {'type': 'object', 'properties': {
        'job_id': {'type': 'string'}, 'path': {'type': 'string'}, 'expected_sha256': {'type': 'string'},
        'old': {'type': 'string'}, 'new': {'type': 'string'}, 'replace_all': {'type': 'boolean'}},
        'required': ['job_id', 'path', 'expected_sha256', 'old', 'new']}}},
    {'type': 'function', 'function': {'name': SITE_STATUS, 'parameters': {'type': 'object', 'properties': {
        'job_id': {'type': 'string'}, 'wait_seconds': {'type': 'integer'}}, 'required': ['job_id']}}},
]
JOB = 'site-' + 'a' * 32


def site_call(name, args, call_id='b1_site'):
    return {'role': 'assistant', 'content': '', 'tool_calls': [{'id': call_id, 'type': 'function',
            'function': {'name': name, 'arguments': json.dumps(args)}}]}


def site_receipt(payload, call_id='b1_site'):
    return {'role': 'tool', 'tool_call_id': call_id, 'content': json.dumps(payload)}


def site_view(**changes):
    return {'schema': 'msty.site.job.v1', 'job_id': JOB, 'state': 'ready', 'checks': {},
            'unchecked_writes': True, **changes}


def edited(messages_extra=()):
    state = initial(tools=SITE_TOOLS)
    state['messages'] = [{'role': 'user', 'content': 'Добавь лупу масштаба в шапку кабинета.'},
        site_call(SITE_PATCH, {'job_id': JOB, 'path': 'src/a.tsx', 'expected_sha256': 'e' * 64,
                               'old': 'x', 'new': 'y'}),
        site_receipt({'schema': 'msty.site.file.v1', 'job_id': JOB, 'path': 'src/a.tsx',
                      'sha256': 'f' * 64, 'state': 'patched', 'replacements': 1, 'checks_invalidated': True}),
        *messages_extra]
    return state


def test_edited_site_copy_forces_status_typecheck_instead_of_done_prose():
    state = edited()
    result = AIMessage(content='Изменения внесены в изолированную копию сайта.', usage_metadata=USAGE)
    guarded = task.gate_final(state, result, SITE_TOOLS, False)
    assert guarded.content == 'Проверяю typecheck изменённой копии сайта перед итогом.'
    assert [(c['name'], c['args']) for c in guarded.tool_calls] == [(SITE_STATUS, {'job_id': JOB, 'wait_seconds': 30})]
    assert guarded.response_metadata['msty_completion_gate'] == 'site_typecheck_required'
    assert task.site_jobs(state) == {JOB: 'dirty'}


def test_full_write_via_file_tool_is_also_dirty_and_read_is_not():
    state = initial(tools=SITE_TOOLS)
    state['messages'] = [{'role': 'user', 'content': 'Поправь.'},
        site_call(SITE_FILE, {'job_id': JOB, 'path': 'src/a.tsx'}, 'b1_read'),
        site_receipt({'schema': 'msty.site.file.v1', 'job_id': JOB, 'path': 'src/a.tsx', 'sha256': 'e' * 64,
                      'content': 'x', 'source_is_untrusted_data': True}, 'b1_read')]
    assert task.site_jobs(state) == {}
    result = AIMessage(content='Посмотрел файл.', usage_metadata=USAGE)
    assert task.gate_final(state, result, SITE_TOOLS, False).content == 'Посмотрел файл.'
    state['messages'] += [site_call(SITE_FILE, {'job_id': JOB, 'path': 'src/a.tsx', 'operation': 'write',
                                                'content': 'y', 'expected_sha256': 'e' * 64}, 'b1_write')]
    assert task.site_jobs(state) == {JOB: 'dirty'}
    assert task.gate_final(state, result, SITE_TOOLS, False).tool_calls[0]['name'] == SITE_STATUS


def test_only_executor_view_with_passed_typecheck_and_no_unchecked_writes_releases_prose():
    still_dirty = site_view(state='checking', active_check='typecheck')
    passed_but_dirty = site_view(checks={'typecheck': {'passed': True}}, unchecked_writes=True)
    for view in (still_dirty, passed_but_dirty):
        state = edited([site_call(SITE_STATUS, {'job_id': JOB, 'wait_seconds': 30}, 'b1_st'), site_receipt(view, 'b1_st')])
        guarded = task.gate_final(state, AIMessage(content='Готово.', usage_metadata=USAGE), SITE_TOOLS, False)
        assert guarded.tool_calls and guarded.tool_calls[0]['name'] == SITE_STATUS
    clean = site_view(checks={'typecheck': {'passed': True, 'manifest_sha256': 'c' * 64}}, unchecked_writes=False)
    state = edited([site_call(SITE_STATUS, {'job_id': JOB, 'wait_seconds': 30}, 'b1_st'), site_receipt(clean, 'b1_st')])
    assert task.site_jobs(state) == {JOB: 'clean'}
    released = task.gate_final(state, AIMessage(content='Готово.', usage_metadata=USAGE), SITE_TOOLS, False)
    assert released.content == 'Готово.' and not released.tool_calls
    # Any later write dirties the copy again; the earlier clean view is stale evidence.
    state['messages'] += [site_call(SITE_PATCH, {'job_id': JOB, 'path': 'src/b.tsx', 'expected_sha256': 'e' * 64,
                                                 'old': 'x', 'new': 'y'}, 'b1_again')]
    assert task.site_jobs(state) == {JOB: 'dirty'}


def test_failed_typecheck_blocks_success_prose_instead_of_looping_status():
    failed = site_view(checks={'typecheck': {'passed': False, 'result': {'exit_code': 2}}}, unchecked_writes=True)
    state = edited([site_call(SITE_STATUS, {'job_id': JOB, 'wait_seconds': 30}, 'b1_st'), site_receipt(failed, 'b1_st')])
    assert task.site_jobs(state) == {JOB: 'typecheck_failed'}
    guarded = task.gate_final(state, AIMessage(content='Всё сделано.', usage_metadata=USAGE), SITE_TOOLS, False)
    assert not guarded.tool_calls and guarded.response_metadata['msty_blocked'] is True
    assert guarded.response_metadata['msty_completion_gate'] == 'site_typecheck_failed'
    assert 'typecheck' in guarded.content and 'не подтверждено' in guarded.content


def test_model_prose_claiming_checks_is_not_evidence_and_unknown_tool_result_is_ignored():
    state = edited([{'role': 'assistant', 'content': 'typecheck passed, unchecked_writes false, всё зелёное.'},
                    site_receipt({'job_id': JOB, 'state': 'ready', 'unchecked_writes': False,
                                  'checks': {'typecheck': {'passed': True}}}, 'b1_noschema')])
    assert task.site_jobs(state) == {JOB: 'dirty'}
    guarded = task.gate_final(state, AIMessage(content='Готово.', usage_metadata=USAGE), SITE_TOOLS, False)
    assert guarded.tool_calls and guarded.tool_calls[0]['name'] == SITE_STATUS


@pytest.mark.parametrize('command', ['Стоп.', 'Только план', 'Не проверяй'])
def test_owner_control_phrase_vetoes_forced_site_status(command):
    state = edited([{'role': 'user', 'content': command}])
    guarded = task.gate_final(state, AIMessage(content='План готов.', usage_metadata=USAGE), SITE_TOOLS, False)
    assert not guarded.tool_calls and guarded.content == 'План готов.'


def test_without_status_tool_or_with_other_forced_choice_prose_is_blocked_not_published():
    state = edited()
    result = AIMessage(content='Готово.', usage_metadata=USAGE)
    no_status = [t for t in SITE_TOOLS if t['function']['name'] != SITE_STATUS]
    blocked = task.gate_final(state, result, no_status, False)
    assert not blocked.tool_calls and blocked.response_metadata['msty_blocked'] is True
    assert blocked.response_metadata['msty_completion_gate'] == 'site_typecheck_required'
    state['tool_choice'] = {'type': 'function', 'function': {'name': SITE_PATCH}}
    chosen = task.gate_final(state, result, SITE_TOOLS, False)
    assert not chosen.tool_calls and chosen.response_metadata['msty_blocked'] is True


def test_cancelled_job_and_model_tool_calls_in_flight_do_not_trigger_site_gate():
    cancelled = site_view(state='cancelled')
    state = edited([site_call(SITE_STATUS, {'job_id': JOB}, 'b1_st'), site_receipt(cancelled, 'b1_st')])
    assert task.site_jobs(state) == {}
    state = edited()
    acting = AIMessage(content='', usage_metadata=USAGE, tool_calls=[{'id': 'x', 'name': SITE_STATUS,
                       'args': {'job_id': JOB}, 'type': 'tool_call'}])
    assert task.gate_final(state, acting, SITE_TOOLS, False) is acting
