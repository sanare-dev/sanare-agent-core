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
    def create(profile, cap, effort=None):
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


@pytest.mark.parametrize('bad', [
    AIMessage(content='partial', response_metadata={'finish_reason': 'length'}, usage_metadata=USAGE)])
def test_unfinished_summary_preserves_paid_usage_and_does_not_commit_or_retry(monkeypatch, bad):
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


@pytest.mark.parametrize('profile', ['deepseek', 'sol6', 'opus5'])
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
        'profile': 'opus5', 'input_limit': 180000, 'output_limit': 100}
    with pytest.raises(execution.ExecutionProtocolError):
        execution.validate_binding(state, 'sol6', 100)


@pytest.mark.parametrize('profile', ['astra', 'sol', 'opus', 'fable'])
def test_retired_consult_binding_is_rejected(profile):
    state = initial()
    state['task_budget_binding'] = {'version': 1, 'pricing_version': execution.PRICING_VERSION,
        'profile': profile, 'input_limit': 180000, 'output_limit': 2048}
    with pytest.raises(execution.ExecutionProtocolError):
        execution.validate_binding(state, profile, 2048)


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


def multi_cluster_history(clusters):
    """`clusters` isolated big tool bundles (each its own compaction candidate,
    separated by a plain user message so make_plan never merges them into one
    span), followed by the usual two verbatim tail bundles."""
    state = initial()
    for cluster in range(clusters):
        identifier = f'cluster-{cluster}'
        state['messages'].extend([
            {'role': 'assistant', 'content': 'Readback.', 'tool_calls': [
                {'id': identifier, 'type': 'function', 'function': {
                    'name': 'read_fixture', 'arguments': '{"path":"synthetic.txt"}'}}]},
            {'role': 'tool', 'tool_call_id': identifier, 'content': 'Untrusted fixture. ' * 1500},
            {'role': 'user', 'content': f'Continue analysis {cluster}.'},
        ])
    for index in range(2):
        identifier = f'recent-{index}'
        state['messages'].extend([
            {'role': 'assistant', 'content': 'Readback.', 'tool_calls': [
                {'id': identifier, 'type': 'function', 'function': {
                    'name': 'read_fixture', 'arguments': '{"path":"recent.txt"}'}}]},
            {'role': 'tool', 'tool_call_id': identifier, 'content': f'recent observation {index}'}])
    return state


def test_one_paid_pass_then_deterministic_fit_answers_without_refusal(monkeypatch):
    # brain-desk #899 (live 28.09, thread a6dd00e8): three paid summary rounds
    # in one request (313K tokens) and then «больше 3 сжатий» instead of an
    # answer. Now: one paid pass, the rest is fitted without a model call and
    # the same request answers.
    state = multi_cluster_history(4)
    originals = deepcopy(state['messages'])
    responses, counts = [summary(state)], [190000, 500]
    seen = install(monkeypatch, responses, counts)
    async def scenario():
        graph = msty.builder.compile(checkpointer=InMemorySaver())
        cfg = {'configurable': {'thread_id': 'one-paid-pass'}}
        first = await graph.ainvoke(state, cfg)
        assert first['execution']['status'] == 'waiting_compaction'
        assert first['compaction_round'] == 1
        responses.append('Finished after one paid pass.')
        # still above the trigger after the paid pass → mechanical fit → recount
        counts.extend([179000, 70000])
        second = await graph.ainvoke(
            Command(resume={first['__interrupt__'][0].id: resume(first)}), cfg)
        assert second['execution']['status'] == 'answered'
        assert second['result']['content'] == 'Finished after one paid pass.'
        assert second['compaction_round'] == 0
        assert not second.get('__interrupt__')
        assert len(seen['requests']) == 2  # one summary + one answer, nothing else paid
        assert second['context_budget_check']['input_tokens'] == 70000
        segments = second['context_memory']['segments']
        assert len(segments) == 4
        assert sum(s['summary'].startswith('[Механическая выжимка: сжатие без вызова модели')
                   for s in segments) == 3
        assert compact.make_plan(second) is None
        assert second['messages'] == originals
        # the answering request saw the extracts, never the archived raw bundles
        answered = seen['requests'][-1]
        assert not any(('Untrusted fixture. ' * 100) in str(m.content) for m in answered)
        assert [m.content for m in answered if m.type == 'human'][-1] == originals[-5]['content']
    asyncio.run(scenario())


def test_repeated_continue_does_not_pay_again(monkeypatch):
    # After the fit is persisted a new «продолжи» request on the same thread
    # finds nothing new to summarise: no paid stage, straight answer.
    state = multi_cluster_history(4)
    responses, counts = [summary(state), 'First answer.'], [190000, 500, 179000, 70000]
    seen = install(monkeypatch, responses, counts)
    async def scenario():
        graph = msty.builder.compile(checkpointer=InMemorySaver())
        cfg = {'configurable': {'thread_id': 'continue-no-pay'}}
        first = await graph.ainvoke(state, cfg)
        second = await graph.ainvoke(Command(resume={first['__interrupt__'][0].id: resume(first)}), cfg)
        assert second['execution']['status'] == 'answered'
        memory = second['context_memory']
        follow = {**state, 'messages': [*second['messages'],
                                        {'role': 'user', 'content': 'Продолжай.'}],
                  'execution_task_id': str(uuid.uuid4())}
        responses.append('Continued.')
        counts.append(150000)  # above the trigger, but everything old is archived
        # Same thread: the bridge sends the whole history as new input.
        third = await graph.ainvoke({**follow, 'context_memory': memory}, cfg)
        assert third['execution']['status'] == 'answered'
        assert third['result']['content'] == 'Continued.'
        assert len(seen['requests']) == 3  # no new summary call
        assert third['context_memory']['segments'] == memory['segments']
    asyncio.run(scenario())


def test_over_limit_drops_oldest_turns_for_this_generation_only(monkeypatch):
    # No tool bundles to archive and the input exceeds the admitted limit: the
    # oldest whole turns are left out of this generation (canonical history and
    # every system message stay; the latest owner turn is never touched).
    state = initial()
    for index in range(4):
        state['messages'].extend([{'role': 'user', 'content': f'Old turn {index}.'},
                                  {'role': 'assistant', 'content': f'Long answer {index}. ' * 3000}])
    state['messages'].append({'role': 'user', 'content': 'Делай.'})
    originals = deepcopy(state['messages'])
    seen = install(monkeypatch, ['Done.'], [200000, 120000])
    result = asyncio.run(msty.graph.ainvoke(state))
    assert result['execution']['status'] == 'answered'
    assert result['result']['content'] == 'Done.'
    assert len(seen['requests']) == 1
    sent = seen['requests'][0]
    assert sent[-1].content == 'Делай.'
    assert any('не переданы модели' in str(m.content) for m in sent)
    assert 'Never write, owner limit.' in [m.content for m in sent if m.type == 'system']
    assert not any('Long answer 0.' in str(m.content) for m in sent)
    assert result['messages'] == originals
    assert result['context_budget_check']['input_tokens'] == 120000


def test_fit_projection_keeps_pairs_and_latest_turn():
    projected = [{'role': 'system', 'content': 'S'},
                 {'role': 'user', 'content': 'first'},
                 {'role': 'assistant', 'content': '', 'tool_calls': [{'id': 'a', 'type': 'function',
                     'function': {'name': 'read', 'arguments': '{}'}}]},
                 {'role': 'tool', 'tool_call_id': 'a', 'content': 'x' * 50000},
                 {'role': 'user', 'content': 'latest'},
                 {'role': 'assistant', 'content': '', 'tool_calls': [{'id': 'b', 'type': 'function',
                     'function': {'name': 'read', 'arguments': '{}'}}]},
                 {'role': 'tool', 'tool_call_id': 'b', 'content': 'y' * 90000}]
    fitted, saved = compact.fit_projection(projected, 120000)
    assert fitted[0] == projected[0]
    users = [m['content'] for m in fitted if m['role'] == 'user']
    assert users == ['latest']
    ids = [c['id'] for m in fitted for c in m.get('tool_calls') or []]
    assert ids == ['b'] and [m['tool_call_id'] for m in fitted if m['role'] == 'tool'] == ['b']
    assert len(fitted[-1]['content'].encode()) < 5000 and 'обрезан' in fitted[-1]['content']
    assert saved > 0
    assert projected[-1]['content'] == 'y' * 90000  # input untouched


def test_mechanical_fit_archives_oldest_runs_and_clips_old_summaries():
    state = multi_cluster_history(3)
    memory, saved = compact.mechanical_fit(state, 1)
    assert len(memory['segments']) == 1 and saved > 0
    memory, saved = compact.mechanical_fit(state, 10 ** 9)
    assert len(memory['segments']) == 3
    projected = compact.project_messages({**state, 'context_memory': memory})
    assert [m for m in projected if m['role'] in ('user', 'system')] == [
        m for m in state['messages'] if m['role'] in ('user', 'system')]
    assert all(len(s['summary'].encode()) <= compact.CLIPPED_SUMMARY_BYTES for s in memory['segments'])


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


def test_luna_responses_block_content_summary_is_accepted():
    # Luna on the Responses API (#35) answers with a list of blocks
    # (reasoning + text); the summary is the text blocks, not a refusal.
    state = history()
    plan = compact.make_plan(state)
    blocks = [{'type': 'reasoning', 'id': 'rs_1', 'summary': []},
              {'type': 'text', 'text': summary(state)}]
    accepted = compact.accept_summary(state, plan, AIMessage(content=blocks))
    assert accepted['compaction_stage']['status'] == 'ready'
    with pytest.raises(compact.ExecutionProtocolError):
        compact.accept_summary(state, plan, AIMessage(content=[{'type': 'reasoning', 'id': 'rs_2'}]))


def test_fenced_json_summary_is_accepted():
    # Live 27.09: Luna wrapped the summary in a ```json fence and every
    # compaction was rejected («Сводка не прошла проверку»).
    state = history()
    plan = compact.make_plan(state)
    fenced = '```json\n' + summary(state) + '\n```'
    blocks = [{'type': 'text', 'text': fenced}]
    assert compact.accept_summary(state, plan, AIMessage(content=blocks))['compaction_stage']['status'] == 'ready'
    wrapped = 'Вот сводка:\n' + summary(state)
    assert compact.accept_summary(state, plan, AIMessage(content=wrapped))['compaction_stage']['status'] == 'ready'
    with pytest.raises(compact.ExecutionProtocolError):
        compact.accept_summary(state, plan, AIMessage(content='```json\n{"summary": "x"}\n```'))


@pytest.mark.parametrize('bad', ['not-json', '{"summary":"missing sources"}',
    '{"sources":["invented"],"summary":"text"}',
    AIMessage(content='', tool_calls=[{'id': 'evil', 'name': 'write', 'args': {}}], usage_metadata=USAGE)])
def test_rejected_summary_continues_on_mechanical_extract_not_as_answer(monkeypatch, bad):
    # Live 28.09 (brain-desk thread a6dd00e8): «Сводка не прошла проверку;
    # исходники сохранены.» became the whole answer to «делай» and the turn
    # ended with no action. Now the paid call stays the compaction stage, a
    # deterministic extract replaces its text and the same run answers.
    state = history()
    originals = deepcopy(state['messages'])
    seen = install(monkeypatch, [bad, 'Finished answer'], [150000, 60000, 90000])
    async def scenario():
        graph = msty.builder.compile(checkpointer=InMemorySaver())
        cfg = {'configurable': {'thread_id': 'mechanical'}}
        first = await graph.ainvoke(state, cfg)
        assert len(seen['requests']) == 1
        assert first['result']['content'] == ''
        assert not first['result'].get('tool_calls')
        assert first['result']['response_metadata']['msty_stage'] == 'compaction'
        assert first['result']['usage_metadata'] == USAGE
        assert first['execution']['status'] == 'waiting_compaction'
        segment = first['context_memory']['segments'][0]
        assert segment['summary'].startswith('[Механическая выжимка')
        assert 'read_fixture' in segment['summary'] and 'old-0' in segment['summary']
        assert len(segment['summary'].encode('utf-8')) <= compact.summary_limit(compact.make_plan(state))
        second = await graph.ainvoke(Command(resume={first['__interrupt__'][0].id: resume(first)}), cfg)
        assert len(seen['requests']) == 2
        assert second['result']['content'] == 'Finished answer'
        assert second['execution']['status'] == 'answered'
        assert second['messages'] == originals
    asyncio.run(scenario())


def test_mechanical_summary_fits_limit_on_huge_sources():
    state = initial()
    for index in range(60):
        identifier = f'big-{index}'
        state['messages'].extend([
            {'role': 'assistant', 'content': '', 'tool_calls': [
                {'id': identifier, 'type': 'function', 'function': {
                    'name': 'web_read', 'arguments': json.dumps({'url': 'https://example.test/' + 'я' * 400})}}]},
            {'role': 'tool', 'tool_call_id': identifier, 'content': 'Данные. ' * 800},
        ])
    plan = compact.make_plan(state)
    text = compact.mechanical_summary(plan, 'Сводка не прошла проверку; исходники сохранены.')
    assert text.strip() and len(text.encode('utf-8')) <= compact.summary_limit(plan)
    assert compact.accept_mechanical(state, plan, 'x')['compaction_stage']['status'] == 'ready'
