"""Per-step reasoning level: a browser chain runs low between the plan and the answer.

Live 28.09 (Brain Desk, owner_browser_find → fill_form → click → type): every
step re-reasoned at the turn level chosen once from the owner message.
Offline real graph; only provider construction, counting and inference are
replaced. Not a latency benchmark.
"""
import asyncio
from copy import deepcopy
import json

import pytest
from langchain_core.messages import AIMessage

from deep_agent import msty, msty_effort as effort, msty_execution as execution, msty_models

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


def install(monkeypatch, responses):
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
    return seen


def run(value):
    return asyncio.run(msty.graph.ainvoke(value))


def record_of(result):
    return result['result']['response_metadata'][effort.METADATA_KEY]


def test_browser_chain_runs_high_then_low_after_each_browser_result(monkeypatch):
    levels, records = [], []
    steps = []
    for index, name in enumerate(CHAIN):
        seen = install(monkeypatch, [reply(calls=[call(name, index)])])
        result = run(state(history(steps)))
        assert result['result']['tool_calls'][0]['name'] == name
        assert len(seen['created']) == 1
        levels.append(seen['created'][0][2])
        records.append(record_of(result))
        steps.append((name, None))
    seen = install(monkeypatch, [reply('Итог: заявка отправлена.', output=250)])
    result = run(state(history(steps)))
    levels.append(seen['created'][0][2])
    records.append(record_of(result))

    assert levels == ['high', 'low', 'low', 'low', 'low']
    assert [r['step'] for r in records] == ['plan', *['tool_chain'] * 4]
    assert records[1] == {'version': 1, 'level': 'low', 'reason': 'tool_chain', 'step': 'tool_chain',
                          'task_level': 'high', 'task_reason': 'deep_work', 'chain_steps': 1,
                          'profile': 'luna', 'provider_value': 'low'}
    assert records[4]['chain_steps'] == 4
    assert result['result']['content'] == 'Итог: заявка отправлена.'
    assert len(seen['invocations']) == 1  # one paid call per step, no extra attempt
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
