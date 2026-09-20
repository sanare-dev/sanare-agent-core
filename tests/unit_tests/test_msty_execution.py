"""Real LangGraph checkpoints/resume with fake inference, never external tools."""
import asyncio
from copy import deepcopy
import json

import pytest
from langchain_core.messages import AIMessage
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.types import Command

from deep_agent import msty, msty_execution as execution


TOOLS = [{'type': 'function', 'function': {'name': 'read_fixture',
    'parameters': {'type': 'object', 'properties': {'name': {'type': 'string'}},
                   'required': ['name'], 'additionalProperties': False}}}]


def initial():
    return {'messages': [{'role': 'user', 'content': 'Compare the two fixtures.'}],
            'tools': deepcopy(TOOLS), 'max_tokens': 64, 'tool_choice': 'auto',
            'result': {}, 'context_budget': None, 'context_budget_check': None,
            'execution_protocol': execution.PROTOCOL, 'execution': {}}


def operation(identifier='model-one', name='first.json'):
    return AIMessage(content='Reading a fixture.', tool_calls=[
        {'id': identifier, 'name': 'read_fixture', 'args': {'name': name}}],
        usage_metadata={'input_tokens': 100, 'output_tokens': 10, 'total_tokens': 110})


def model_sequence(monkeypatch, sequence):
    seen = []

    class Model:
        def __init__(self, **kwargs):
            assert kwargs['max_retries'] == 0

        def bind_tools(self, *args, **kwargs):
            return self

        async def ainvoke(self, messages):
            seen.append(deepcopy(messages))
            assert sequence, 'Unexpected extra inference'
            return sequence.pop(0)

    monkeypatch.setattr(msty, 'ChatAnthropic', Model)
    return seen


def resume_value(state, client_suffix='1'):
    incoming = {key: deepcopy(state[key]) for key in (
        'messages', 'tools', 'max_tokens', 'tool_choice', 'context_budget', 'execution_protocol')}
    incoming.update(result={}, context_budget_check=None)
    mappings, calls, results = [], [], []
    for index, call in enumerate(state['execution']['pending']['calls']):
        client_id = f'b1_test_{client_suffix}_{index}'
        mappings.append({'client_id': client_id, 'model_id': call['id']})
        calls.append({'id': client_id, 'type': 'function', 'function': {
            'name': call['name'], 'arguments': json.dumps(call['args'])}})
        results.append({'role': 'tool', 'tool_call_id': client_id,
                        'content': 'Synthetic observed value, not an actual filesystem read.'})
    incoming['messages'] += [{'role': 'assistant', 'content': state['result']['content'],
                              'tool_calls': calls}, *results]
    return {'version': 1, 'task_id': state['execution']['task_id'],
            'batch_id': state['execution']['pending']['batch_id'],
            'tool_id_map': mappings, 'input': incoming}


def test_simple_answer_has_one_inference_no_interrupt_no_completed_claim(monkeypatch):
    seen = model_sequence(monkeypatch, [AIMessage(content='OK')])
    result = asyncio.run(msty.graph.ainvoke(initial()))
    assert len(seen) == 1
    assert result['execution']['status'] == 'answered'
    assert result['execution']['actions_issued'] == 0
    assert not result.get('__interrupt__')


def test_real_checkpoint_resume_does_not_replay_model_and_can_recreate_graph(monkeypatch):
    seen = model_sequence(monkeypatch, [operation(), operation('model-two', 'second.json'),
                                        AIMessage(content='Comparison result')])

    async def scenario():
        saver = InMemorySaver()
        config = {'configurable': {'thread_id': 'same-real-checkpoint'}}
        graph = msty.builder.compile(checkpointer=saver)
        first = await graph.ainvoke(initial(), config, durability='sync')
        assert first['execution']['status'] == 'waiting_tools'
        assert first['execution']['step'] == 1
        assert set(first['execution']['pending']['calls'][0]) == {'id', 'name', 'args'}
        assert len(seen) == 1
        waiting = first['__interrupt__'][0]
        assert waiting.value['result_sha256'] == execution.canonical_digest(first['result'])
        # Reconstruct graph against the same serialized in-memory checkpoint.
        # This is not a claim that InMemorySaver survives an OS process restart.
        resumed_graph = msty.builder.compile(checkpointer=saver)
        snapshot = await resumed_graph.aget_state(config)
        assert snapshot.next == ('wait_external',)
        second = await resumed_graph.ainvoke(
            Command(resume={waiting.id: resume_value(first)}), config, durability='sync')
        assert len(seen) == 2
        assert second['execution']['task_id'] == first['execution']['task_id']
        assert second['execution']['step'] == 2
        assert second['execution']['actions_issued'] == 2
        assert second['execution']['pending']['batch_id'] != first['execution']['pending']['batch_id']
        final = await resumed_graph.ainvoke(Command(resume={second['__interrupt__'][0].id:
            resume_value(second, '2')}), config, durability='sync')
        assert len(seen) == 3
        assert final['execution']['status'] == 'answered'
        assert final['execution']['step'] == 3
        assert final['execution']['actions_issued'] == 2
        assert final['execution']['pending'] is None
        assert final['result']['content'] == 'Comparison result'
        assert not final.get('__interrupt__')
        assert seen[-1][-1].type == 'tool'
        assert seen[-1][-1].tool_call_id == 'b1_test_2_0'

    asyncio.run(scenario())


@pytest.mark.parametrize('mutation', [
    lambda r: r.update(task_id='another-task'),
    lambda r: r.update(batch_id='another-batch'),
    lambda r: r['input'].update(execution={'status': 'completed'}),
    lambda r: r['input'].update(context_budget_check={'status': 'accepted'}),
    lambda r: r['input'].update(max_tokens=65),
    lambda r: r['input'].update(tools=[]),
    lambda r: r['input']['messages'].pop(),
    lambda r: r['input']['messages'].append(deepcopy(r['input']['messages'][-1])),
    lambda r: r['input']['messages'][-2]['tool_calls'][0]['function'].update(arguments='{"name":"other.json"}'),
    lambda r: r['input']['messages'].append({'role': 'user', 'content': 'Do a different task'}),
    lambda r: r['tool_id_map'][0].update(model_id='unknown'),
    lambda r: r['tool_id_map'].append(deepcopy(r['tool_id_map'][0])),
])
def test_invalid_resume_rejected_before_next_inference(monkeypatch, mutation):
    seen = model_sequence(monkeypatch, [operation()])

    async def scenario():
        graph = msty.builder.compile(checkpointer=InMemorySaver())
        config = {'configurable': {'thread_id': 'tamper-case'}}
        first = await graph.ainvoke(initial(), config)
        resume = resume_value(first)
        mutation(resume)
        with pytest.raises(execution.ExecutionProtocolError):
            await graph.ainvoke(Command(resume={first['__interrupt__'][0].id: resume}), config)
        assert len(seen) == 1

    asyncio.run(scenario())


def test_two_identical_chats_do_not_share_pending_task(monkeypatch):
    seen = model_sequence(monkeypatch, [operation(), operation()])

    async def scenario():
        graph = msty.builder.compile(checkpointer=InMemorySaver())
        first = await graph.ainvoke(initial(), {'configurable': {'thread_id': 'chat-A'}})
        second = await graph.ainvoke(initial(), {'configurable': {'thread_id': 'chat-B'}})
        assert first['execution']['task_id'] != second['execution']['task_id']
        assert first['__interrupt__'][0].id != second['__interrupt__'][0].id
        assert len(seen) == 2

    asyncio.run(scenario())


def test_provider_length_never_creates_executable_interrupt(monkeypatch):
    partial = operation().model_copy(update={'response_metadata': {'stop_reason': 'max_tokens'}})
    model_sequence(monkeypatch, [partial])
    result = asyncio.run(msty.graph.ainvoke(initial()))
    assert result['execution']['status'] == 'incomplete'
    assert result['execution']['actions_issued'] == 0
    assert result['execution']['pending'] is None
    assert result['result']['tool_calls'] == []
    assert result['result']['usage_metadata']['total_tokens'] == 110
    assert result['result']['response_metadata']['stop_reason'] == 'max_tokens'
    assert not result.get('__interrupt__')


def test_persisted_action_limit_cannot_be_reset_by_shortened_history(monkeypatch):
    model_sequence(monkeypatch, [operation()])
    state = initial()
    state['execution'] = {'version': 1, 'task_id': 'already-issued', 'step': 200,
                          'actions_issued': 200, 'status': 'running', 'pending': None}
    with pytest.raises(execution.ExecutionProtocolError, match='лимит'):
        asyncio.run(msty.respond(state))


@pytest.mark.parametrize('issued', [24, 199])
def test_action_25_and_action_200_keep_existing_checkpoint_counters(monkeypatch, issued):
    seen = model_sequence(monkeypatch, [operation(), AIMessage(content='Observed final.')])

    async def scenario():
        state = initial()
        state['execution'] = {'version': 1, 'task_id': 'retained-parent-task', 'step': issued + 7,
                              'actions_issued': issued, 'consultations': 1, 'status': 'running', 'pending': None}
        saver = InMemorySaver()
        graph = msty.builder.compile(checkpointer=saver)
        config = {'configurable': {'thread_id': 'raised-action-cap-' + str(issued)}, 'recursion_limit': 64}
        first = await graph.ainvoke(state, config)
        assert first['execution']['actions_issued'] == issued + 1
        assert first['execution']['step'] == issued + 8
        assert first['execution']['consultations'] == 1
        assert first['execution']['task_id'] == 'retained-parent-task'
        assert first['execution']['status'] == 'waiting_tools'
        # A rebuilt graph and a shortened callback cannot reset consumed actions.
        graph = msty.builder.compile(checkpointer=saver)
        resume = resume_value(first)
        resume['input']['messages'] = [state['messages'][0], *resume['input']['messages'][-2:]]
        final = await graph.ainvoke(Command(resume={first['__interrupt__'][0].id: resume}), config)
        assert final['execution']['actions_issued'] == issued + 1
        assert final['execution']['step'] == issued + 9
        assert final['execution']['consultations'] == 1
        assert final['execution']['task_id'] == 'retained-parent-task'
        assert final['execution']['status'] == 'answered'
        assert len(seen) == 2 and not final.get('__interrupt__')
    asyncio.run(scenario())


def test_persisted_action_201_is_rejected_without_counter_mutation():
    assert execution.MAX_ACTIONS == 200
    state = initial()
    state['execution'] = {'version': 1, 'task_id': 'retained-parent-task', 'step': 207,
                          'actions_issued': 200, 'consultations': 2, 'status': 'running', 'pending': None}
    original = deepcopy(state)
    update = {'result': operation().model_dump(mode='json')}
    with pytest.raises(execution.ExecutionProtocolError, match='лимит'):
        execution.execution_after(state, update)
    assert state == original


def test_legacy_request_is_backwards_compatible(monkeypatch):
    model_sequence(monkeypatch, [operation()])
    state = initial()
    state['execution_protocol'] = None
    result = asyncio.run(msty.graph.ainvoke(state))
    assert result['result']['tool_calls']
    assert not result.get('__interrupt__')
    assert result['execution'] == {}


def test_resume_keeps_checkpoint_root_and_only_adds_expected_observations(monkeypatch):
    seen = model_sequence(monkeypatch, [operation(), AIMessage(content='Verified comparison')])

    async def scenario():
        graph = msty.builder.compile(checkpointer=InMemorySaver())
        config = {'configurable': {'thread_id': 'immutable-root'}}
        source = initial()
        source['messages'].insert(0, {'role': 'system', 'content': 'Original project rules.'})
        first = await graph.ainvoke(source, config)
        resume = resume_value(first)
        resume['input']['messages'][0]['content'] = 'Changed root during pending action.'
        resume['input']['messages'].insert(0, {'role': 'system', 'content': 'Late client advisory.'})
        resume['input']['messages'][-2]['content'] = 'Altered previous assistant prose.'
        final = await graph.ainvoke(Command(resume={first['__interrupt__'][0].id: resume}), config)
        assert final['messages'][0] == source['messages'][0]
        assert final['messages'][-2]['content'] == 'Reading a fixture.'
        assert len(final['messages']) == len(source['messages']) + 2
        assert all('Changed root' not in str(m.content) and 'Late client' not in str(m.content)
                   and 'Altered previous' not in str(m.content) for m in seen[-1])
        assert seen[-1][-1].tool_call_id == 'b1_test_1_0'

    asyncio.run(scenario())
