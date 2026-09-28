"""Per-step reasoning level: a browser chain runs low between the plan and the answer.

Live 28.09 (Brain Desk, owner_browser_find → fill_form → click → type): every
step re-reasoned at the turn level chosen once from the owner message.
Offline real graph; only provider construction, counting and inference are
replaced. Not a latency benchmark.
"""
import asyncio
from copy import deepcopy
import json
import uuid

import pytest
from langchain_core.messages import AIMessage

from deep_agent import msty, msty_compaction, msty_effort as effort, msty_execution as execution, msty_models

OWNER = 'Проанализируй форму заявки на сайте поставщика, заполни её и отправь.'
CHAIN = ('owner_browser_find', 'owner_browser_fill_form', 'owner_browser_click', 'owner_browser_snapshot')
TOOLS = [{'type': 'function', 'function': {'name': name, 'parameters': {
    'type': 'object', 'properties': {'q': {'type': 'string'}}, 'additionalProperties': True}}}
    for name in (*CHAIN, 'service_call', 'read_fixture')]
DONE = {'status': 'completed', 'model_name': 'gpt-6-luna'}
FAILED = json.dumps({'isError': True, 'error': 'Element not found'})


@pytest.fixture(autouse=True)
def offline(monkeypatch):
    monkeypatch.delenv('MSTY_MODEL_PROFILE', raising=False)
    monkeypatch.setenv('LANGSMITH_TRACING', 'false')
    monkeypatch.setenv('LANGCHAIN_TRACING_V2', 'false')


def usage(output=300, input_tokens=20000):
    return {'input_tokens': input_tokens, 'output_tokens': output,
            'total_tokens': input_tokens + output}


def call(name, index, args=None):
    return {'id': f'call_{index}', 'name': name, 'args': args or {'q': str(index)}}


def reply(text='', calls=None, output=300, metadata=None):
    return AIMessage(content=text, tool_calls=calls or [], usage_metadata=usage(output),
                     response_metadata=dict(metadata or DONE))


def history(steps, results=None):
    """Owner message plus ``steps`` completed tool batches [(name, args)]."""
    messages = [{'role': 'user', 'content': OWNER}]
    for index, (name, args) in enumerate(steps):
        messages.append({'role': 'assistant', 'content': '', 'tool_calls': [{
            'id': f'call_{index}', 'type': 'function',
            'function': {'name': name, 'arguments': json.dumps(args or {'q': str(index)})}}]})
        content = (results or {}).get(index, f'Результат шага {index}.')
        messages.append({'role': 'tool', 'tool_call_id': f'call_{index}', 'content': content})
    return messages


def state(messages, max_tokens=8192, **changes):
    return {'messages': messages, 'tools': deepcopy(TOOLS), 'max_tokens': max_tokens,
            'tool_choice': 'auto', 'result': {}, 'context_budget': None, 'context_budget_check': None,
            'execution_protocol': execution.PROTOCOL, 'execution': {}, **changes}


def install(monkeypatch, responses, counts=None):
    seen = {'created': [], 'bound': [], 'invocations': []}
    queue = list(responses)

    class Model:
        def bind_tools(self, tools, **kwargs):
            seen['bound'].append(deepcopy(kwargs))
            return self

        async def ainvoke(self, messages):
            seen['invocations'].append(deepcopy(messages))
            assert queue, 'Unexpected additional paid-model attempt'
            item = queue.pop(0)
            if isinstance(item, Exception):
                raise item
            return item

    def construct(profile, max_tokens, level=None):
        seen['created'].append((profile, max_tokens, level))
        return Model()

    monkeypatch.setattr(msty_models, 'make_model', construct)
    if counts is not None:
        remaining = list(counts)

        async def count(profile, model, messages, tools):
            return remaining.pop(0)
        monkeypatch.setattr(msty_models, 'count_input', count)
    return seen


def run(value):
    return asyncio.run(msty.graph.ainvoke(value))


def record_of(result):
    return result['result']['response_metadata'][effort.METADATA_KEY]


def test_five_step_browser_chain_runs_high_low_low_low_then_high_synthesis(monkeypatch):
    levels, records = [], []
    steps = []
    for index, name in enumerate(CHAIN):
        seen = install(monkeypatch, [reply(calls=[call(name, index)])])
        result = run(state(history(steps)))
        assert result['result']['tool_calls'][0]['name'] == name
        levels.append(seen['created'][0][2])
        records.append(record_of(result))
        steps.append((name, None))
    # Step 5: the low step turns out to be the answer → one synthesis re-run.
    seen = install(monkeypatch, [reply('Черновик.', output=250), reply('Итог: заявка отправлена.', output=900)])
    result = run(state(history(steps)))
    assert [c[2] for c in seen['created']] == ['low', 'high']
    levels.append(record_of(result)['level'])
    records.append(record_of(result))

    assert levels == ['high', 'low', 'low', 'low', 'high']
    assert [r['step'] for r in records] == ['plan', 'tool_chain', 'tool_chain', 'tool_chain', 'synthesis']
    assert records[1] == {'version': 1, 'level': 'low', 'reason': 'tool_chain', 'step': 'tool_chain',
                          'task_level': 'high', 'task_reason': 'deep_work', 'chain_steps': 1,
                          'profile': 'luna', 'provider_value': 'low'}
    assert records[4]['provider_value'] == 'high' and records[4]['reason'] == 'synthesis'
    message = result['result']
    assert message['content'] == 'Итог: заявка отправлена.'
    # Both calls are paid and reported; total output stays within the stage limit.
    assert message['usage_metadata']['output_tokens'] == 1150 <= 8192
    assert message['usage_metadata']['input_tokens'] == 40000
    assert message['response_metadata'][msty.SYNTHESIS_KEY] == {
        'version': 1, 'status': 'answered', 'tools_reexecuted': 0,
        'first': {'level': 'low', 'output_tokens': 250, 'input_tokens': 20000},
        'synthesis': {'level': 'high', 'output_limit': 6144, 'output_tokens': 900,
                      'input_tokens': 20000}}
    assert seen['invocations'][0] == seen['invocations'][1]  # same messages and tools
    assert result['execution']['status'] == 'answered'


def test_two_failures_in_a_row_rethink_at_medium(monkeypatch):
    steps = [('owner_browser_find', None), ('owner_browser_click', None), ('owner_browser_click', None)]
    seen = install(monkeypatch, [reply(calls=[call('owner_browser_snapshot', 9)])])
    result = run(state(history(steps, {1: FAILED, 2: FAILED})))
    assert seen['created'][0][2] == 'medium'
    assert record_of(result)['reason'] == 'tool_failures'
    assert record_of(result)['step'] == 'failure_rethink'


def test_one_failure_keeps_the_chain_low(monkeypatch):
    steps = [('owner_browser_find', None), ('owner_browser_click', None)]
    seen = install(monkeypatch, [reply(calls=[call('owner_browser_snapshot', 9)])])
    run(state(history(steps, {1: FAILED})))
    assert seen['created'][0][2] == 'low'


def test_success_after_failures_returns_to_low(monkeypatch):
    steps = [('owner_browser_click', None), ('owner_browser_click', None), ('owner_browser_find', None)]
    seen = install(monkeypatch, [reply(calls=[call('owner_browser_click', 9)])])
    run(state(history(steps, {0: FAILED, 1: FAILED})))
    assert seen['created'][0][2] == 'low'


@pytest.mark.parametrize('name, args, level', [
    ('read_fixture', None, 'high'),                        # not a simple tool: turn level
    ('service_call', {'method': 'GET', 'path': '/x'}, 'low'),
    ('service_call', {'method': 'POST', 'path': '/x'}, 'high'),
    ('mcp__brain__owner_browser_click', None, 'low'),      # namespaced name
    ('web_read', None, 'low'),
    ('brain_desk_read_tool_result', None, 'low'),
])
def test_which_tools_lower_the_next_step(name, args, level):
    turn = {'version': 1, 'level': 'high', 'reason': 'deep_work'}
    assert effort.step_effort(state(history([(name, args)])), turn)['level'] == level


def test_owner_force_and_low_turn_are_kept():
    steps = [('owner_browser_click', None)]
    forced = {'version': 1, 'level': 'max', 'reason': 'owner_force'}
    assert effort.step_effort(state(history(steps)), forced)['level'] == 'max'
    low = {'version': 1, 'level': 'low', 'reason': 'short_question'}
    assert effort.step_effort(state(history(steps)), low) == {**low, 'step': 'task'}


def test_no_synthesis_rerun_for_a_low_turn(monkeypatch):
    seen = install(monkeypatch, [reply('Готово.')])
    run(state([{'role': 'user', 'content': 'нажми кнопку'},
               *history([('owner_browser_click', None)])[1:]]))
    assert [c[2] for c in seen['created']] == ['low']


def test_synthesis_falls_back_to_the_low_answer_when_the_rerun_is_cut(monkeypatch):
    cut = {'status': 'incomplete', 'incomplete_details': {'reason': 'max_output_tokens'},
           'model_name': 'gpt-6-luna'}
    seen = install(monkeypatch, [reply('Итог по снимку.', output=250),
                                 reply('', output=6144, metadata=cut)])
    result = run(state(history([(n, None) for n in CHAIN])))
    assert len(seen['invocations']) == 2
    message = result['result']
    assert message['content'] == 'Итог по снимку.'
    assert message['response_metadata'][msty.SYNTHESIS_KEY]['status'] == 'kept_low'
    assert message['usage_metadata']['output_tokens'] == 6394 <= 8192


def test_synthesis_rerun_counts_its_input_against_the_stage_limit(monkeypatch):
    seen = install(monkeypatch, [reply('Итог.', output=250)], counts=[100000, 100000])
    value = state(history([(n, None) for n in CHAIN]), execution_task_id=str(uuid.uuid4()),
                  task_budget_binding={'version': 1, 'pricing_version': execution.PRICING_VERSION,
                                       'profile': 'luna', 'input_limit': 180000, 'output_limit': 8192})
    result = run(value)
    assert len(seen['invocations']) == 1  # 2 × 100000 > 180000: keep the low answer
    assert result['result']['content'] == 'Итог.'
    assert result['context_budget_check']['input_tokens'] == 100000


def test_stale_browser_snapshots_are_clipped_latest_kept(monkeypatch):
    big = 'row ' * 3000
    steps = [('owner_browser_snapshot', None), ('owner_browser_click', None), ('owner_browser_snapshot', None)]
    messages = history(steps, {0: 'OLD ' + big, 1: 'ok', 2: 'NEW ' + big})
    seen = install(monkeypatch, [reply(calls=[call('owner_browser_click', 9)])])
    run(state(messages))
    tools = [m for m in seen['invocations'][0] if m.type == 'tool']
    assert tools[0].content.startswith('[Устаревший результат браузера сокращён')
    assert len(tools[0].content.encode()) < 1200
    assert tools[1].content == 'ok'
    assert tools[2].content == 'NEW ' + big


def test_clip_is_projection_only_and_skips_other_tools():
    big = 'x' * 9000
    messages = history([('read_fixture', None), ('owner_browser_snapshot', None),
                        ('owner_browser_snapshot', None)], {0: big, 1: big, 2: big})
    original = deepcopy(messages)
    clipped = msty_compaction.clip_stale_browser_results(messages)
    assert messages == original
    assert clipped[2]['content'] == big                    # read_fixture untouched
    assert clipped[4]['content'].startswith('[Устаревший')  # older snapshot
    assert clipped[6]['content'] == big                    # latest snapshot whole
