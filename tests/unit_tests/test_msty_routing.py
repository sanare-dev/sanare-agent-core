"""Offline acceptance of server-selected profiles and bounded consultations.

Real graph/checkpoints, real preparation/binding/identity adapter; only provider
construction and inference are replaced. These tests are not quality benchmarks
or evidence that an external MCP action was executed.
"""
import asyncio
from copy import deepcopy
import json

import pytest
from langchain_core.messages import AIMessage
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.types import Command

from deep_agent import msty, msty_execution as execution, msty_models


CONSULT = 'sanare_core_msty_brain_consult'
TOOLS = [
    {'type': 'function', 'function': {'name': CONSULT, 'parameters': {
        'type': 'object', 'properties': {'question': {'type': 'string'}},
        'required': ['question'], 'additionalProperties': False}}},
    {'type': 'function', 'function': {'name': 'read_fixture', 'parameters': {
        'type': 'object', 'properties': {'name': {'type': 'string'}},
        'required': ['name'], 'additionalProperties': False}}},
]
USAGE = {'input_tokens': 100, 'output_tokens': 10, 'total_tokens': 110}


@pytest.fixture(autouse=True)
def offline_defaults(monkeypatch):
    monkeypatch.delenv('MSTY_MODEL_PROFILE', raising=False)
    monkeypatch.setenv('LANGSMITH_TRACING', 'false')
    monkeypatch.setenv('LANGCHAIN_TRACING_V2', 'false')


def initial(**overrides):
    return {'messages': [{'role': 'user', 'content': 'Compare the synthetic fixtures.'}],
            'tools': deepcopy(TOOLS), 'max_tokens': 64, 'tool_choice': 'auto',
            'result': {}, 'context_budget': None, 'context_budget_check': None,
            'execution_protocol': execution.PROTOCOL, 'execution': {}, **overrides}


def reply(content='OK', *, calls=None, metadata=None, usage=USAGE):
    return AIMessage(content=content, tool_calls=calls or [],
                     response_metadata=metadata or {}, usage_metadata=deepcopy(usage))


def operation(identifier, *, consult=True):
    return reply('Synthetic action request.', calls=[{
        'id': identifier, 'name': CONSULT if consult else 'read_fixture',
        'args': {'question': 'Review the supplied evidence.'} if consult else {'name': 'fixture.json'},
    }])


def models(monkeypatch, sequence):
    seen = {'created': [], 'bound': [], 'invocations': []}
    responses = list(sequence)

    class Model:
        def __init__(self, profile):
            self.profile = profile

        def bind_tools(self, tools, **kwargs):
            seen['bound'].append((self.profile, deepcopy(tools), deepcopy(kwargs)))
            return self

        async def ainvoke(self, messages):
            seen['invocations'].append((self.profile, deepcopy(messages)))
            assert responses, 'Unexpected additional paid-model attempt'
            return responses.pop(0)

    def construct(profile, max_tokens, effort=None):
        seen['created'].append((profile, max_tokens))
        return Model(profile)

    monkeypatch.setattr(msty_models, 'make_model', construct)
    return seen


def resume_value(state, suffix, *, shorten=False):
    incoming = {key: deepcopy(state[key]) for key in (
        'messages', 'tools', 'max_tokens', 'tool_choice', 'context_budget', 'execution_protocol')}
    if 'brain_task_role' in state:
        incoming['brain_task_role'] = state['brain_task_role']
    incoming.update(result={}, context_budget_check=None)
    if shorten:
        incoming['messages'] = [next(m for m in reversed(incoming['messages']) if m['role'] == 'user')]
    mappings, calls, observations = [], [], []
    for index, call in enumerate(state['execution']['pending']['calls']):
        client_id = f'b1_routing_{suffix}_{index}'
        mappings.append({'client_id': client_id, 'model_id': call['id']})
        calls.append({'id': client_id, 'type': 'function', 'function': {
            'name': call['name'], 'arguments': json.dumps(call['args'])}})
        observations.append({'role': 'tool', 'tool_call_id': client_id,
                             'content': 'Synthetic observation, not a verified execution receipt.'})
    incoming['messages'] += [{'role': 'assistant', 'content': state['result']['content'],
                              'tool_calls': calls}, *observations]
    return {'version': 1, 'task_id': state['execution']['task_id'],
            'batch_id': state['execution']['pending']['batch_id'],
            'tool_id_map': mappings, 'input': incoming}


def test_default_luna_one_generation_ignores_client_model_endpoint(monkeypatch):
    seen = models(monkeypatch, [reply()])
    state = initial(tools=[], model='expensive-invented-model',
                    model_profile='sonnet', base_url='https://invalid.example')
    result = asyncio.run(msty.graph.ainvoke(state))
    assert seen['created'] == [('luna', 64)]
    assert len(seen['invocations']) == 1
    assert seen['bound'] == []
    assert result['result']['content'] == 'OK'
    assert result['result']['response_metadata'] == {
        'model_name': 'gpt-6-luna', 'msty_model_name': 'gpt-6-luna',
        'msty_model_profile': 'luna', 'msty_model_provider': 'openai',
        'msty_reasoning_effort': {'version': 1, 'level': 'high', 'reason': 'deep_work',
                                  'profile': 'luna', 'provider_value': 'low',
                                  'applied_level': 'low', 'output_limit': 64}}
    assert result['result']['usage_metadata'] == USAGE
    assert result['execution']['status'] == 'answered'
    assert result['execution']['actions_issued'] == result['execution']['consultations'] == 0
    assert not result.get('__interrupt__')


def test_analyst_is_deepseek_text_only_with_2048_output_cap(monkeypatch):
    monkeypatch.setenv('MSTY_MODEL_PROFILE', 'luna')
    seen = models(monkeypatch, [reply('An evidence-limited analysis.')])
    result = asyncio.run(msty.graph.ainvoke(initial(
        brain_task_role='analyst', tools=[], max_tokens=8192)))
    assert seen['created'] == [('deepseek', 2048)]
    assert seen['bound'] == []
    assert len(seen['invocations']) == 1
    assert seen['invocations'][0][1][0].content == msty.ANALYST_POLICY
    assert result['result']['response_metadata']['model_name'] == 'deepseek-flash'
    assert result['result']['response_metadata']['msty_model_profile'] == 'deepseek'
    assert result['result']['tool_calls'] == []
    assert not result.get('__interrupt__')


@pytest.mark.parametrize('override', [
    {'tools': TOOLS},
    {'messages': [{'role': 'user', 'content': [{'type': 'text', 'text': 'not plain text'}]}]},
    {'messages': [{'role': 'user', 'content': 'x' * 96001}]},
    {'messages': [{'role': 'assistant', 'content': '', 'tool_calls': [{
        'id': 'previous', 'type': 'function',
        'function': {'name': 'read_fixture', 'arguments': '{"name":"fixture.json"}'}}]}]},
])
def test_analyst_rejects_tools_media_oversize_or_calls_before_generation(monkeypatch, override):
    seen = models(monkeypatch, [])
    state = initial(brain_task_role='analyst', tools=[])
    state.update(deepcopy(override))
    result = asyncio.run(msty.graph.ainvoke(state))
    assert seen == {'created': [], 'bound': [], 'invocations': []}
    assert result['result']['response_metadata'] == {'msty_generation': 'not_started'}
    assert result['result']['usage_metadata'] == {'input_tokens': 0, 'output_tokens': 0, 'total_tokens': 0}
    assert result['result']['tool_calls'] == []
    assert result['execution']['status'] == 'blocked'
    assert not result.get('__interrupt__')


@pytest.mark.parametrize('role', ['council', 'architect', 'provider_override', '', None, 1])
def test_arbitrary_role_is_rejected_before_generation(monkeypatch, role):
    seen = models(monkeypatch, [])
    result = asyncio.run(msty.graph.ainvoke(initial(brain_task_role=role)))
    assert seen['created'] == seen['invocations'] == []
    assert result['result']['response_metadata'] == {'msty_generation': 'not_started'}
    assert result['execution']['status'] == 'blocked'
    assert result['result']['tool_calls'] == []


def test_unknown_server_profile_is_not_silently_replaced(monkeypatch):
    monkeypatch.setenv('MSTY_MODEL_PROFILE', 'arbitrary-provider')
    seen = models(monkeypatch, [])
    result = asyncio.run(msty.graph.ainvoke(initial()))
    assert seen['created'] == []
    assert result['result']['response_metadata'] == {'msty_generation': 'not_started'}
    assert result['execution']['status'] == 'blocked'


def test_consultation_quota_survives_checkpoints_and_shortened_callback(monkeypatch):
    seen = models(monkeypatch, [operation('consult-1'), operation('consult-2'),
                               operation('read-3', consult=False), operation('consult-3')])

    async def scenario():
        saver = InMemorySaver()
        config = {'configurable': {'thread_id': 'persistent-consultation-quota'}}
        graph = msty.builder.compile(checkpointer=saver)
        first = await graph.ainvoke(initial(), config)
        assert first['execution']['consultations'] == 1
        second = await graph.ainvoke(Command(resume={first['__interrupt__'][0].id:
            resume_value(first, '1')}), config)
        assert second['execution']['consultations'] == 2
        # Recreate the graph from its checkpoint. Shortened client history must
        # neither erase the saved quota nor prevent ordinary local tool work.
        graph = msty.builder.compile(checkpointer=saver)
        third = await graph.ainvoke(Command(resume={second['__interrupt__'][0].id:
            resume_value(second, '2', shorten=True)}), config)
        assert third['execution']['consultations'] == 2
        assert third['execution']['actions_issued'] == 3
        assert third['execution']['status'] == 'waiting_tools'
        assert third['result']['tool_calls'][0]['name'] == 'read_fixture'
        final = await graph.ainvoke(Command(resume={third['__interrupt__'][0].id:
            resume_value(third, '3', shorten=True)}), config)
        assert final['execution']['consultations'] == 2
        assert final['execution']['actions_issued'] == 3
        assert final['execution']['pending'] is None
        assert final['result']['tool_calls'] == final['result']['invalid_tool_calls'] == []
        assert final['result']['additional_kwargs'] == {}
        assert final['result']['usage_metadata'] == USAGE
        assert 'двух консультаций' in final['result']['content']
        assert not final.get('__interrupt__')
        assert len(seen['invocations']) == 4
        assert all(profile == 'luna' for profile, _ in seen['invocations'])
        assert 'Лимит консультаций исчерпан' in seen['invocations'][-1][1][0].content

    asyncio.run(scenario())


def test_role_cannot_be_changed_in_callback_before_next_generation(monkeypatch):
    seen = models(monkeypatch, [operation('read-first', consult=False)])

    async def scenario():
        graph = msty.builder.compile(checkpointer=InMemorySaver())
        config = {'configurable': {'thread_id': 'immutable-role'}}
        first = await graph.ainvoke(initial(brain_task_role='lead'), config)
        resumed = resume_value(first, 'tampered')
        resumed['input']['brain_task_role'] = 'analyst'
        with pytest.raises(execution.ExecutionProtocolError, match='Роль Brain'):
            await graph.ainvoke(Command(resume={first['__interrupt__'][0].id: resumed}), config)
        assert len(seen['invocations']) == 1

    asyncio.run(scenario())


@pytest.mark.parametrize('profile,canonical,method', [
    ('luna', 'gpt-6-luna', 'tiktoken-admission-v1'),
    ('deepseek', 'deepseek-flash', 'conservative-text-v1'),
])
def test_count_receipt_identifies_method_and_model_without_changing_usage(monkeypatch, profile, canonical, method):
    monkeypatch.setenv('MSTY_MODEL_PROFILE', profile)
    seen = models(monkeypatch, [reply()])
    counts = []

    async def count(selected, model, messages, tools):
        counts.append((selected, deepcopy(messages), deepcopy(tools)))
        return 42

    monkeypatch.setattr(msty_models, 'count_input', count)
    result = asyncio.run(msty.graph.ainvoke(initial(context_budget='msty-model-count-v1')))
    assert result['context_budget_check'] == {
        'version': 1, 'status': 'accepted', 'input_tokens': 42, 'limit': 180000, 'window_admission': True,
        'method': method, 'model_profile': profile}
    assert counts[0][0] == profile
    assert counts[0][1] == seen['invocations'][0][1]
    assert counts[0][2] == TOOLS
    assert result['result']['usage_metadata'] == USAGE
    assert result['result']['response_metadata']['model_name'] == canonical


@pytest.mark.parametrize('profile,canonical', [('luna', 'gpt-6-luna'), ('deepseek', 'deepseek-flash')])
def test_raw_usage_is_preserved_with_canonical_and_provider_identity(monkeypatch, profile, canonical):
    monkeypatch.setenv('MSTY_MODEL_PROFILE', profile)
    provider_model = canonical + '-2026-09-20'
    models(monkeypatch, [reply(metadata={'model_name': provider_model, 'token_usage': {
        'prompt_tokens': 100, 'completion_tokens': 10, 'total_tokens': 110,
        'prompt_tokens_details': {'cached_tokens': 20},
        'completion_tokens_details': {'reasoning_tokens': 2}}},
        usage={'input_tokens': 0, 'output_tokens': 0, 'total_tokens': 0})])
    result = asyncio.run(msty.graph.ainvoke(initial(tools=[])))['result']
    assert result['response_metadata']['provider_model_name'] == provider_model
    assert result['response_metadata']['model_name'] == canonical
    assert result['response_metadata']['msty_model_profile'] == profile
    assert result['usage_metadata'] == {**USAGE, 'input_token_details': {'cache_read': 20},
                                       'output_token_details': {'reasoning': 2}}


def test_missing_provider_usage_is_unknown_not_sdk_zero(monkeypatch):
    models(monkeypatch, [reply(metadata={'token_usage': {}},
                              usage={'input_tokens': 0, 'output_tokens': 0, 'total_tokens': 0})])
    result = asyncio.run(msty.graph.ainvoke(initial(tools=[])))['result']
    assert result['usage_metadata'] is None
    assert result['response_metadata']['model_name'] == 'gpt-6-luna'
    assert 'msty_generation' not in result['response_metadata']


@pytest.mark.parametrize('raw_usage,expected_usage', [
    ({'prompt_tokens': 100, 'completion_tokens': 10, 'total_tokens': 110},
     {**USAGE, 'input_token_details': {}, 'output_token_details': {}}),
    ({}, None),
])
def test_wrong_provider_identity_blocks_actions_but_preserves_checked_expense(monkeypatch, raw_usage, expected_usage):
    response = operation('wrong-model-action', consult=False).model_copy(update={
        'content': 'UNTRUSTED_PROVIDER_CONTENT_SENTINEL',
        'response_metadata': {'model_name': 'UNEXPECTED_PROVIDER_MODEL_SENTINEL',
                              'token_usage': raw_usage},
        'additional_kwargs': {'tool_calls': 'UNTRUSTED_PROVIDER_RAW_SENTINEL'},
        'usage_metadata': {'input_tokens': 0, 'output_tokens': 0, 'total_tokens': 0}})
    seen = models(monkeypatch, [response])

    async def collect():
        return [event async for event in msty.graph.astream(initial(), stream_mode=['custom', 'values'])]

    events = asyncio.run(collect())
    custom = [data for mode, data in events if mode == 'custom']
    final = [data for mode, data in events if mode == 'values'][-1]
    assert custom == [{'type': 'validated_result', 'message': final['result']}]
    assert final['result']['response_metadata'] == {
        'msty_generation': 'rejected_model', 'msty_blocked': True}
    assert final['result']['usage_metadata'] == expected_usage
    assert final['result']['tool_calls'] == final['result']['invalid_tool_calls'] == []
    assert final['result']['additional_kwargs'] == {}
    assert 'SENTINEL' not in str(events)
    assert final['execution']['status'] == 'blocked'
    assert final['execution']['actions_issued'] == 0
    assert final['execution']['pending'] is None
    assert not final.get('__interrupt__')
    assert len(seen['invocations']) == 1


@pytest.mark.parametrize('choice', ['unavailable', {'type': 'unsupported'}])
def test_invalid_tool_choice_is_zero_generation_blocker(monkeypatch, choice):
    seen = models(monkeypatch, [])
    result = asyncio.run(msty.graph.ainvoke(initial(tool_choice=choice)))
    assert seen['created'] == [('luna', 64)]
    assert seen['bound'] == seen['invocations'] == []
    assert result['result']['response_metadata'] == {'msty_generation': 'not_started'}
    assert result['result']['usage_metadata'] == {'input_tokens': 0, 'output_tokens': 0, 'total_tokens': 0}
    assert result['execution']['status'] == 'blocked'
    assert result['result']['tool_calls'] == []
    assert not result.get('__interrupt__')


@pytest.mark.parametrize('profile', ['luna', 'deepseek'])
@pytest.mark.parametrize('choice', ['none', {'type': 'none'}])
def test_none_keeps_schemas_but_never_publishes_action(monkeypatch, profile, choice):
    monkeypatch.setenv('MSTY_MODEL_PROFILE', profile)
    seen = models(monkeypatch, [operation('must-not-execute', consult=False)])
    result = asyncio.run(msty.graph.ainvoke(initial(tool_choice=choice)))
    assert seen['bound'] == [(profile, TOOLS, {'tool_choice': 'none'})]
    assert result['result']['tool_calls'] == []
    assert result['result']['usage_metadata'] == USAGE
    assert result['execution']['actions_issued'] == 0
    assert not result.get('__interrupt__')


@pytest.mark.parametrize('profile', ['luna', 'deepseek'])
@pytest.mark.parametrize('reason', ['length', 'content_filter'])
def test_truncated_or_refused_output_never_publishes_action(monkeypatch, profile, reason):
    monkeypatch.setenv('MSTY_MODEL_PROFILE', profile)
    response = operation('partial', consult=False).model_copy(update={
        'response_metadata': {'finish_reason': reason},
        'additional_kwargs': {'tool_calls': 'raw partial synthetic call'}})
    models(monkeypatch, [response])
    result = asyncio.run(msty.graph.ainvoke(initial()))
    assert result['result']['tool_calls'] == result['result']['invalid_tool_calls'] == []
    assert result['result']['additional_kwargs'] == {}
    assert result['result']['usage_metadata'] == USAGE
    assert result['result']['response_metadata']['finish_reason'] == reason
    assert result['execution']['status'] == 'incomplete'
    assert result['execution']['actions_issued'] == 0
    assert not result.get('__interrupt__')
