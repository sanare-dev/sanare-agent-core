"""Рой Brain (msty-swarm-v1): Send fan-out, допуск моста, частичный итог.

Офлайн: модели исполнителей и шаги лида подменены; настоящие LangGraph Send,
ToolNode, checkpoint/interrupt. Успех тестов не доказывает качество роя на
живых задачах, развёртывание или работу окна.
"""
import asyncio
from copy import deepcopy
import json
import socket

import pytest
from langchain_core.messages import AIMessage
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.store.memory import InMemoryStore
from langgraph.types import Command

from deep_agent import msty, msty_execution, msty_native, msty_swarm

TOOLS = [{'type': 'function', 'function': {'name': 'external_read',
    'description': 'Read one synthetic external item.',
    'parameters': {'type': 'object', 'properties': {'name': {'type': 'string'}},
                   'required': ['name'], 'additionalProperties': False}}}]
MODELS = {'deepseek': 'deepseek-flash', 'luna': 'gpt-6-luna'}


@pytest.fixture(autouse=True)
def no_network(monkeypatch):
    for name in ('LANGSMITH_TRACING', 'LANGCHAIN_TRACING', 'LANGCHAIN_TRACING_V2'):
        monkeypatch.setenv(name, 'false')

    def denied(*args, **kwargs):
        raise AssertionError('Swarm offline test attempted network access')

    monkeypatch.setattr(socket.socket, 'connect', denied)
    monkeypatch.setattr(socket, 'create_connection', denied)


def plan(count=3, **extra):
    subtasks = [{'title': f'Часть {i}', 'role': 'исследователь', 'prompt': f'Разбери аспект {i}.',
                 'profile': 'luna' if i == 2 else 'deepseek', 'max_tokens': 512}
                for i in range(1, count + 1)]
    return {'goal': 'Сравнить три подхода', 'subtasks': subtasks, **extra}


def reply(profile, text='ответ', reason='stop', tokens=(40, 12)):
    return AIMessage(content=text, usage_metadata={
        'input_tokens': tokens[0], 'output_tokens': tokens[1], 'total_tokens': sum(tokens)},
        response_metadata={'model_name': MODELS[profile], 'finish_reason': reason,
                           'token_usage': {'prompt_tokens': tokens[0], 'completion_tokens': tokens[1],
                                           'total_tokens': sum(tokens)}})


class Workers:
    """Фабрика моделей исполнителей; считает вызовы и одновременность."""

    def __init__(self, behaviour=None, delay=0.05):
        self.behaviour = behaviour or {}
        self.delay = delay
        self.calls, self.active, self.peak = [], 0, 0

    def __call__(self, profile, max_tokens):
        outer = self

        class Model:
            async def ainvoke(self, messages):
                outer.calls.append((profile, max_tokens, [m.content for m in messages]))
                outer.active += 1
                outer.peak = max(outer.peak, outer.active)
                try:
                    await asyncio.sleep(outer.delay)
                    task = messages[-1].content
                    for marker, action in outer.behaviour.items():
                        if marker in task:
                            return await action(profile)
                    return reply(profile, 'итог: ' + task[-20:])
                finally:
                    outer.active -= 1
        return Model()


def admission_for(swarm, status='admitted'):
    if status == 'rejected':
        return {'version': 1, 'swarm_id': swarm['swarm_id'], 'status': 'rejected',
                'reason': 'Общий лимит роя превышен.'}
    return {'version': 1, 'swarm_id': swarm['swarm_id'], 'status': 'admitted', 'subtasks': [
        {'id': s['id'], 'request_id': 'req-' + s['id'], 'binding': {
            'version': 1, 'pricing_version': 'test', 'profile': s['profile'],
            'input_limit': 180000, 'output_limit': s['max_tokens']}} for s in swarm['subtasks']]}


def pending_for(args, batch='b7b1c1a4-0000-4000-8000-000000000001'):
    return {'batch_id': batch, 'calls': [{'id': 'swarm-1', 'name': msty_swarm.TOOL, 'args': args}]}


# --- План и описатель ---------------------------------------------------------

@pytest.mark.parametrize('mutate, message', [
    (lambda p: p['subtasks'].pop() and p['subtasks'].pop(), 'от 2 до 5'),
    (lambda p: p['subtasks'].extend(deepcopy(p['subtasks'][:3])), 'от 2 до 5'),
    (lambda p: p['subtasks'][0].update(profile='sol'), 'вне серверного списка'),
    (lambda p: p['subtasks'][0].update(max_tokens=4096), 'max_tokens'),
    (lambda p: p['subtasks'][0].update(tools=['x']), 'неизвестные поля'),
    (lambda p: p['subtasks'][1].update(prompt=p['subtasks'][0]['prompt']), 'дублируют'),
    (lambda p: p.update(extra=1), 'только goal и subtasks'),
    (lambda p: p['subtasks'][0].update(prompt='x' * 7000), 'длиннее'),
])
def test_plan_rejects_unsafe_or_oversized_swarms(mutate, message):
    value = plan()
    mutate(value)
    with pytest.raises(msty_swarm.SwarmPlanError, match=message):
        msty_swarm.parse_plan(value)


def test_plan_defaults_cheap_executor_and_bounded_output():
    value = plan(2)
    for item in value['subtasks']:
        item.pop('profile'), item.pop('max_tokens'), item.pop('role')
    parsed = msty_swarm.parse_plan(value)
    assert [s['profile'] for s in parsed['subtasks']] == ['deepseek', 'deepseek']
    assert [s['max_tokens'] for s in parsed['subtasks']] == [1024, 1024]
    assert [s['id'] for s in parsed['subtasks']] == ['s1', 's2']


def test_descriptor_is_deterministic_and_carries_no_prompt_text():
    first = msty_swarm.descriptor(pending_for(plan()))
    assert first == msty_swarm.descriptor(pending_for(plan()))
    assert first['subtasks'][1] == {'id': 's2', 'title': 'Часть 2', 'role': 'исследователь',
                                    'profile': 'luna', 'max_tokens': 512}
    assert 'Разбери' not in json.dumps(first, ensure_ascii=False)
    other = msty_swarm.descriptor(pending_for(plan(), batch='b7b1c1a4-0000-4000-8000-000000000002'))
    assert other['swarm_id'] != first['swarm_id']
    assert msty_swarm.descriptor(pending_for({'goal': 'x', 'subtasks': []})) is None


@pytest.mark.parametrize('mutate', [
    lambda a: a.update(swarm_id='other'),
    lambda a: a['subtasks'].pop(),
    lambda a: a['subtasks'][0]['binding'].update(profile='luna'),
    lambda a: a['subtasks'][0]['binding'].update(output_limit=2048),
    lambda a: a.update(extra=True),
])
def test_admission_must_cover_exact_plan(mutate):
    swarm = msty_swarm.descriptor(pending_for(plan()))
    admission = admission_for(swarm)
    mutate(admission)
    with pytest.raises(msty_swarm.SwarmPlanError):
        msty_swarm.check_admission(swarm, admission)


# --- Send fan-out ----------------------------------------------------------------

def test_send_fan_out_runs_workers_in_parallel_with_own_prompt_and_model():
    parsed = msty_swarm.parse_plan(plan())
    swarm = msty_swarm.descriptor(pending_for(plan()))
    events, workers = [], Workers(delay=0.2)
    report = asyncio.run(msty_swarm.run(parsed, admission_for(swarm), events.append, workers))
    assert workers.peak == 3, 'все подзадачи должны идти одновременно'
    assert sorted(c[0] for c in workers.calls) == ['deepseek', 'deepseek', 'luna']
    assert all(c[1] == 512 for c in workers.calls)
    assert all('Разбери аспект' in c[2][1] and 'Сравнить три подхода' in c[2][1] for c in workers.calls)
    assert report['status'] == 'complete' and report['completed'] == report['total'] == 3
    assert [r['id'] for r in report['subtasks']] == ['s1', 's2', 's3']
    done = [e for e in events if e['status'] == 'done']
    assert len(done) == 3 and all(e['started'] and e['usage']['output_tokens'] == 12 for e in done)
    assert {e['response_metadata']['msty_model_profile'] for e in done} == {'deepseek', 'luna'}
    assert events[-1] == {'type': 'swarm_event', 'version': 1, 'swarm_id': swarm['swarm_id'],
                          'subtask': None, 'status': 'complete', 'completed': 3, 'total': 3}
    assert not any('text' in e for e in events), 'текст ответов не уходит в поток окна'


def test_failures_timeouts_and_length_make_an_explicit_partial_result(monkeypatch):
    monkeypatch.setattr(msty_swarm, 'WORKER_TIMEOUT_SECONDS', 0.3)

    async def boom(profile):
        raise RuntimeError('provider down')

    async def slow(profile):
        await asyncio.sleep(5)

    async def cut(profile):
        return reply(profile, 'обрыв', reason='length')
    value = plan(4)
    for index, marker in enumerate(('BOOM', 'SLOW', 'CUT')):
        value['subtasks'][index + 1]['prompt'] += ' ' + marker
    parsed = msty_swarm.parse_plan(value)
    swarm = msty_swarm.descriptor(pending_for(value))
    events = []
    report = asyncio.run(msty_swarm.run(parsed, admission_for(swarm), events.append,
                                        Workers({'BOOM': boom, 'SLOW': slow, 'CUT': cut})))
    statuses = [r['status'] for r in report['subtasks']]
    assert statuses == ['done', 'failed', 'timeout', 'incomplete']
    assert report['status'] == 'partial' and report['completed'] == 1
    assert 'ЧАСТИЧНЫЙ РЕЗУЛЬТАТ' in report['note'] and 's3 «Часть 3» (timeout)' in report['note']
    failed = next(e for e in events if e['subtask'] == 's2' and e['status'] == 'failed')
    assert failed['started'] is True and failed['usage'] is None and failed['error'] == 'RuntimeError'
    text = msty_swarm.tool_text(report)
    assert 'не выдавай' in text and 'непроверенные мнения' in text


def test_wrong_model_identity_is_failed_not_accepted():
    async def wrong(profile):
        return reply('luna') if profile == 'deepseek' else reply(profile)
    value = plan(2)
    value['subtasks'][0]['prompt'] += ' WRONG'
    parsed = msty_swarm.parse_plan(value)
    events = []
    report = asyncio.run(msty_swarm.run(parsed, admission_for(msty_swarm.descriptor(pending_for(value))),
                                        events.append, Workers({'WRONG': wrong})))
    assert report['subtasks'][0]['status'] == 'failed'
    event = next(e for e in events if e['subtask'] == 's1' and e['status'] == 'failed')
    assert event['error'] == 'model_identity' and event['usage']['output_tokens'] == 12


# --- Нативный граф лида ------------------------------------------------------------

def answer(content='Итог.', calls=None):
    return AIMessage(content=content, tool_calls=calls or [],
        usage_metadata={'input_tokens': 100, 'output_tokens': 10, 'total_tokens': 110})


def scripted(monkeypatch, sequence):
    seen = []

    async def step(state, *, native_system_prompt=None, native_result_filter=None):
        assert sequence, 'Лишний шаг модели лида'
        seen.append({'state': deepcopy(state), 'system': native_system_prompt})
        result = sequence.pop(0)
        if native_result_filter:
            result = native_result_filter(result)
        return msty.publish_result(result, None)

    monkeypatch.setattr(msty, '_respond_step', step)
    return seen


def initial(swarm=True):
    value = {'messages': [{'role': 'user', 'content': 'Сравни три подхода к кэшированию.'}],
             'tools': deepcopy(TOOLS), 'max_tokens': 128, 'tool_choice': 'auto',
             'result': {}, 'context_budget': None, 'context_budget_check': None,
             'execution_protocol': msty_execution.PROTOCOL, 'execution': {}}
    if swarm:
        value['swarm_protocol'] = msty_swarm.PROTOCOL
    return value


async def invoke(graph, value, config):
    custom = []
    async for mode, chunk in graph.astream(value, config, stream_mode=['custom', 'values'], durability='sync'):
        if mode == 'custom':
            custom.append(chunk)
    return await graph.aget_state(config), custom


def swarm_call(args=None, identifier='swarm-1'):
    return {'id': identifier, 'name': msty_swarm.TOOL, 'args': args or plan(), 'type': 'tool_call'}


def offered(seen, index=0):
    return msty_swarm.TOOL in msty.tool_names(seen[index]['state']['tools'])


def test_tool_is_not_offered_without_bridge_protocol(monkeypatch):
    seen = scripted(monkeypatch, [answer()])

    async def run():
        graph = msty_native.build_graph(checkpointer=InMemorySaver(), store=InMemoryStore())
        await invoke(graph, initial(swarm=False), {'configurable': {'thread_id': 'no-swarm'}})
    asyncio.run(run())
    assert not offered(seen)


def test_native_swarm_end_to_end_with_bridge_admission(monkeypatch):
    workers = Workers(delay=0.1)
    monkeypatch.setattr(msty_swarm, 'make_worker_model', workers)
    seen = scripted(monkeypatch, [answer('', [swarm_call()]), answer('Синтез роя.')])

    async def run():
        saver, store = InMemorySaver(), InMemoryStore()
        graph = msty_native.build_graph(checkpointer=saver, store=store)
        config = {'configurable': {'thread_id': 'swarm-e2e'}}
        first, _ = await invoke(graph, initial(), config)
        assert offered(seen)
        assert first.values['execution']['status'] == 'waiting_native'
        pending = first.tasks[0].interrupts[0]
        swarm = pending.value['swarm']
        assert [s['id'] for s in swarm['subtasks']] == ['s1', 's2', 's3']
        assert workers.calls == [], 'до допуска моста исполнители не вызываются'
        resume = {**pending.value, 'type': 'msty_native_resume',
                  'swarm_admission': admission_for(swarm)}
        final, custom = await invoke(graph, Command(resume={pending.id: resume}), config)
        return final, custom, swarm
    final, custom, swarm = asyncio.run(run())
    assert workers.peak == 3 and len(workers.calls) == 3
    events = [c for c in custom if c.get('type') == 'swarm_event']
    assert [e['status'] for e in events].count('running') == 3
    assert [e['status'] for e in events].count('done') == 3 and events[-1]['status'] == 'complete'
    assert all(e['swarm_id'] == swarm['swarm_id'] for e in events)
    assert final.values['result']['content'] == 'Синтез роя.'
    tool = [m for m in seen[1]['state']['messages'] if m['role'] == 'tool'][-1]
    assert tool['tool_call_id'] == 'swarm-1' and '"status": "complete"' in tool['content']
    assert not offered(seen, 1), 'второй рой в том же ходе не предлагается'


def test_rejected_swarm_makes_no_worker_calls_and_lead_continues(monkeypatch):
    workers = Workers()
    monkeypatch.setattr(msty_swarm, 'make_worker_model', workers)
    seen = scripted(monkeypatch, [answer('', [swarm_call()]), answer('Сам.')])

    async def run():
        graph = msty_native.build_graph(checkpointer=InMemorySaver(), store=InMemoryStore())
        config = {'configurable': {'thread_id': 'swarm-rejected'}}
        first, _ = await invoke(graph, initial(), config)
        pending = first.tasks[0].interrupts[0]
        resume = {**pending.value, 'type': 'msty_native_resume',
                  'swarm_admission': admission_for(pending.value['swarm'], 'rejected')}
        return await invoke(graph, Command(resume={pending.id: resume}), config)
    final, custom = asyncio.run(run())
    assert workers.calls == [] and not [c for c in custom if c.get('type') == 'swarm_event']
    tool = [m for m in seen[1]['state']['messages'] if m['role'] == 'tool'][-1]
    assert 'не допущен мостом' in tool['content']
    assert final.values['result']['content'] == 'Сам.'


@pytest.mark.parametrize('admission', ['missing', 'foreign'])
def test_resume_without_matching_admission_cannot_start_workers(monkeypatch, admission):
    workers = Workers()
    monkeypatch.setattr(msty_swarm, 'make_worker_model', workers)
    scripted(monkeypatch, [answer('', [swarm_call()]), answer()])

    async def run():
        graph = msty_native.build_graph(checkpointer=InMemorySaver(), store=InMemoryStore())
        config = {'configurable': {'thread_id': 'swarm-forged-' + admission}}
        first, _ = await invoke(graph, initial(), config)
        pending = first.tasks[0].interrupts[0]
        resume = {**pending.value, 'type': 'msty_native_resume'}
        if admission == 'foreign':
            resume['swarm_admission'] = {**admission_for(pending.value['swarm']), 'swarm_id': 'x'}
        await invoke(graph, Command(resume={pending.id: resume}), config)
    with pytest.raises(msty_execution.ExecutionProtocolError):
        asyncio.run(run())
    assert workers.calls == []


def test_invalid_plan_gets_no_descriptor_and_guard_reply(monkeypatch):
    workers = Workers()
    monkeypatch.setattr(msty_swarm, 'make_worker_model', workers)
    bad = plan()
    bad['subtasks'][0]['profile'] = 'opus'
    seen = scripted(monkeypatch, [answer('', [swarm_call(bad)]), answer()])

    async def run():
        graph = msty_native.build_graph(checkpointer=InMemorySaver(), store=InMemoryStore())
        config = {'configurable': {'thread_id': 'swarm-bad'}}
        first, _ = await invoke(graph, initial(), config)
        pending = first.tasks[0].interrupts[0]
        assert 'swarm' not in pending.value
        await invoke(graph, Command(resume={pending.id: {**pending.value, 'type': 'msty_native_resume'}}), config)
    asyncio.run(run())
    assert workers.calls == []
    tool = [m for m in seen[1]['state']['messages'] if m['role'] == 'tool'][-1]
    assert 'Рой не запущен' in tool['content']


def test_two_swarms_in_one_step_are_blocked(monkeypatch):
    scripted(monkeypatch, [answer('', [swarm_call(), swarm_call(plan(2), 'swarm-2')])])

    async def run():
        graph = msty_native.build_graph(checkpointer=InMemorySaver(), store=InMemoryStore())
        state, _ = await invoke(graph, initial(), {'configurable': {'thread_id': 'swarm-two'}})
        return state
    state = asyncio.run(run())
    assert state.values['execution']['status'] == 'blocked'
    assert not state.values['result']['tool_calls']


def test_client_cannot_shadow_swarm_tool():
    assert msty_swarm.TOOL in msty_native.RESERVED_TOOLS
