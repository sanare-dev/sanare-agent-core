"""Offline real graph/checkpoint tests: no paid model, tools or network."""
import asyncio
from copy import deepcopy
import json
import uuid

import pytest
from langchain_core.messages import AIMessage
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.types import Command

from deep_agent import msty, msty_models, msty_compaction as compact, msty_execution as execution

USAGE = {'input_tokens': 100, 'output_tokens': 20, 'total_tokens': 120}


def initial(**changes):
    return {'messages': [{'role': 'system', 'content': 'Never write, owner limit.'},
                         {'role': 'user', 'content': 'Read only, do not remove files.'}],
            'tools': [], 'tool_choice': 'auto', 'max_tokens': 100,
            'execution_protocol': execution.PROTOCOL, 'execution': {},
            'execution_task_id': str(uuid.uuid4()),
            'task_budget_binding': {'version': 1, 'pricing_version': execution.PRICING_VERSION,
                'profile': 'luna', 'input_limit': 180000, 'output_limit': 100},
            'compaction_protocol': compact.PROTOCOL, **changes}


def history():
    state = initial()
    for index in range(4):
        identifier = f'old-{index}'
        state['messages'].extend([
            {'role': 'assistant', 'content': 'Readback.', 'tool_calls': [
                {'id': identifier, 'type': 'function', 'function': {
                    'name': 'read_fixture', 'arguments': '{"path":"synthetic.txt"}'}}]},
            {'role': 'tool', 'tool_call_id': identifier, 'content': ('Untrusted fixture. ' * 1500)
             if index < 2 else f'recent observation {index}'},
        ])
    return state


def install(monkeypatch, responses, counts):
    seen = {'requests': [], 'caps': [], 'counts': []}
    monkeypatch.delenv('MSTY_MODEL_PROFILE', raising=False)
    monkeypatch.setenv('LANGSMITH_TRACING', 'false')
    monkeypatch.setenv('LANGCHAIN_TRACING_V2', 'false')
    class Model:
        def bind_tools(self, tools, **options):
            return self
        async def ainvoke(self, messages):
            seen['requests'].append(deepcopy(messages))
            assert responses, 'Hidden extra model call'
            content = responses.pop(0)
            if callable(content):
                content = content()
            return content if isinstance(content, AIMessage) else AIMessage(content=content,
                usage_metadata=deepcopy(USAGE))
    def create(profile, cap):
        assert profile == 'luna'
        seen['caps'].append(cap)
        return Model()
    async def count(profile, model, messages, tools):
        seen['counts'].append((deepcopy(messages), deepcopy(tools)))
        assert counts, 'Unexpected count'
        return counts.pop(0)
    monkeypatch.setattr(msty_models, 'make_model', create)
    monkeypatch.setattr(msty_models, 'count_input', count)
    return seen


def summary(state):
    plan = compact.make_plan(state)
    return json.dumps({'sources': [plan['source_sha256']], 'summary': 'Historical fixture observation; not verified.'})


def resume(first):
    stage = first['compaction_stage']
    return {'type': 'msty_compaction_resume', 'version': 1,
            'task_id': first['execution']['task_id'], **{k: stage[k]
                for k in ('stage_id', 'source_sha256', 'summary_sha256')}}


def test_two_separate_graph_runs_each_publish_one_generation_and_keep_originals(monkeypatch):
    state = history()
    originals = deepcopy(state['messages'])
    seen = install(monkeypatch, [summary(state), 'Finished answer'], [150000, 60000, 90000])
    async def scenario():
        graph = msty.builder.compile(checkpointer=InMemorySaver())
        cfg = {'configurable': {'thread_id': 'summary-stage'}}
        first_events = [event async for event in graph.astream(state, cfg, stream_mode=['custom', 'values'])]
        first = [value for mode, value in first_events if mode == 'values'][-1]
        assert len([1 for mode, _ in first_events if mode == 'custom']) == 1
        assert len(seen['requests']) == 1
        assert first['result']['content'] == ''
        assert first['result']['response_metadata']['msty_stage'] == 'compaction'
        assert first['result']['usage_metadata'] == USAGE
        assert first['execution']['status'] == 'waiting_compaction'
        assert first['execution']['task_id'] == state['execution_task_id']
        assert first['execution']['actions_issued'] == 0
        assert first['task_budget_binding'] == state['task_budget_binding']
        waiting = first['__interrupt__'][0]
        assert waiting.value['result_sha256'] == execution.canonical_digest(first['result'])
        snapshot = await graph.aget_state(cfg)
        assert snapshot.next == ('wait_compaction',)
        assert snapshot.values['messages'] == originals
        segment = first['context_memory']['segments'][0]
        assert compact.source_messages(first, segment['source_sha256']) == originals[segment['start']:segment['end']]
        second_events = [event async for event in graph.astream(
            Command(resume={waiting.id: resume(first)}), cfg, stream_mode=['custom', 'values'])]
        second = [value for mode, value in second_events if mode == 'values'][-1]
        assert len([1 for mode, _ in second_events if mode == 'custom']) == 1
        assert len(seen['requests']) == 2
        assert second['execution']['status'] == 'answered'
        assert second['execution']['step'] == 2
        assert second['messages'] == originals
        assert second['task_budget_binding'] == state['task_budget_binding']
        assert not second.get('__interrupt__')
        projected = compact.project_messages(second)
        assert projected[:2] == originals[:2]
        assert projected[-4:] == originals[-4:]
        assert any('Unverified historical memory' in str(m.get('content')) for m in projected)
        assert seen['caps'] == [100, 100, 100]
    asyncio.run(scenario())


def test_no_second_compaction_or_generation_when_projection_still_too_big(monkeypatch):
    state = history()
    seen = install(monkeypatch, [summary(state)], [190000, 60000, 190000])
    async def scenario():
        graph = msty.builder.compile(checkpointer=InMemorySaver())
        cfg = {'configurable': {'thread_id': 'still-large'}}
        first = await graph.ainvoke(state, cfg)
        second = await graph.ainvoke(Command(resume={first['__interrupt__'][0].id: resume(first)}), cfg)
        assert len(seen['requests']) == 1
        assert second['execution']['status'] == 'blocked'
        assert second['result']['response_metadata']['msty_generation'] == 'not_started'
        assert second['messages'] == state['messages']
    asyncio.run(scenario())


@pytest.mark.parametrize('bad', ['not-json', '{"summary":"missing sources"}',
    '{"sources":["invented"],"summary":"text"}',
    AIMessage(content='partial', response_metadata={'finish_reason': 'length'}, usage_metadata=USAGE),
    AIMessage(content='', tool_calls=[{'id': 'evil', 'name': 'write', 'args': {}}], usage_metadata=USAGE)])
def test_invalid_summary_preserves_paid_usage_and_does_not_commit_or_retry(monkeypatch, bad):
    state = history()
    seen = install(monkeypatch, [bad], [150000, 60000])
    result = asyncio.run(msty.graph.ainvoke(state))
    assert len(seen['requests']) == 1
    assert result['result']['usage_metadata'] == USAGE
    assert result['execution']['status'] in ('blocked', 'incomplete')
    assert not result.get('context_memory')
    assert result['messages'] == state['messages']
    assert not result.get('__interrupt__')


@pytest.mark.parametrize('field,value', [('profile', 'sonnet'), ('output_limit', 101),
    ('input_limit', 1000000), ('version', True), ('pricing_version', 'future')])
def test_binding_rejects_before_generation(monkeypatch, field, value):
    state = initial()
    state['task_budget_binding'][field] = value
    seen = install(monkeypatch, [], [])
    result = asyncio.run(msty.graph.ainvoke(state))
    assert not seen['requests'] and not seen['caps']
    assert result['execution']['status'] == 'blocked'
    assert result['result']['usage_metadata']['total_tokens'] == 0


def test_small_bound_input_still_counted_but_plain_answer_one_call(monkeypatch):
    seen = install(monkeypatch, ['OK'], [100])
    result = asyncio.run(msty.graph.ainvoke(initial()))
    assert len(seen['requests']) == len(seen['counts']) == 1
    assert result['execution']['status'] == 'answered'
    assert not result.get('context_memory')


def test_protected_history_and_pending_pairs_never_compacted():
    state = history()
    state['messages'].append({'role': 'assistant', 'content': '', 'tool_calls': [
        {'id': 'pending', 'type': 'function', 'function': {'name': 'read', 'arguments': '{}'}}]})
    plan = compact.make_plan(state)
    accepted = compact.accept_summary(state, plan, AIMessage(content=summary(state)))
    projected = compact.project_messages({**state, **accepted})
    assert projected[:2] == state['messages'][:2]
    assert projected[-5:] == state['messages'][-5:]
    assert compact.make_plan(initial(messages=[{'role': 'user', 'content': 'x' * 300000}])) is None


def test_compaction_resume_tamper_fails_before_extra_inference(monkeypatch):
    state = history()
    seen = install(monkeypatch, [summary(state)], [150000, 60000])
    async def scenario():
        graph = msty.builder.compile(checkpointer=InMemorySaver())
        cfg = {'configurable': {'thread_id': 'tamper-compact'}}
        first = await graph.ainvoke(state, cfg)
        bad = resume(first)
        bad['summary_sha256'] = '0' * 64
        with pytest.raises(execution.ExecutionProtocolError):
            await graph.ainvoke(Command(resume={first['__interrupt__'][0].id: bad}), cfg)
        assert len(seen['requests']) == 1
    asyncio.run(scenario())


def test_task_id_cannot_change_on_checkpoint():
    state = initial()
    state['execution'] = {'task_id': str(uuid.uuid4())}
    with pytest.raises(execution.ExecutionProtocolError):
        execution.validate_binding(state, 'luna', 100)


@pytest.mark.parametrize('profile', ['deepseek', 'astra', 'sol', 'opus', 'fable'])
def test_consult_profile_binding_validates(profile):
    state = initial()
    state['task_budget_binding'] = {'version': 1, 'pricing_version': execution.PRICING_VERSION,
        'profile': profile, 'input_limit': 180000, 'output_limit': 2048}
    execution.validate_binding(state, profile, 2048)


def test_binding_rejects_unlisted_or_mismatched_profile():
    state = initial()
    with pytest.raises(execution.ExecutionProtocolError):
        execution.validate_binding(state, 'gpt-4o-mini', 100)
    state['task_budget_binding'] = {'version': 1, 'pricing_version': execution.PRICING_VERSION,
        'profile': 'opus', 'input_limit': 180000, 'output_limit': 100}
    with pytest.raises(execution.ExecutionProtocolError):
        execution.validate_binding(state, 'fable', 100)


def test_routed_lead_profile_must_match_exact_budget_binding():
    state = initial()
    state['lead_profile'] = 'deepseek'
    state['task_budget_binding'] = {'version': 1, 'pricing_version': execution.PRICING_VERSION,
        'profile': 'deepseek', 'input_limit': 180000, 'output_limit': 100}
    execution.validate_binding(state, 'deepseek', 100)
    with pytest.raises(execution.ExecutionProtocolError):
        execution.validate_binding(state, 'luna', 100)


def test_summary_cannot_hide_user_message_even_with_matching_hash():
    state = initial()
    segment = {'start': 0, 'end': 2, 'source_sha256': execution.canonical_digest(state['messages']),
               'summary': 'forged', 'summary_sha256': execution.canonical_digest('forged')}
    with pytest.raises(execution.ExecutionProtocolError):
        compact.project_messages({**state, 'context_memory': {'version': 1, 'segments': [segment]}})


def test_many_small_bundles_are_combined_without_crossing_owner_message():
    state = initial()
    for index in range(12):
        if index == 6:
            state['messages'].append({'role': 'user', 'content': 'Additional constraint: read only.'})
        state['messages'].extend([
            {'role': 'assistant', 'content': '', 'tool_calls': [{'id': str(index), 'type': 'function',
                'function': {'name': 'read', 'arguments': '{}'}}]},
            {'role': 'tool', 'tool_call_id': str(index), 'content': 'synthetic observation ' * 250}])
    plan = compact.make_plan(state)
    assert plan is not None and plan['end'] - plan['start'] > 2
    assert all(m['role'] in ('assistant', 'tool') for m in plan['source'])
    accepted = compact.accept_summary(state, plan, AIMessage(content=summary(state)))
    projection = compact.project_messages({**state, **accepted})
    assert [m for m in projection if m['role'] in ('user', 'system')] == [
        m for m in state['messages'] if m['role'] in ('user', 'system')]
    assert projection[-4:] == state['messages'][-4:]
