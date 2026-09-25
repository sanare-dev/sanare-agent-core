"""Open plan gate (brain-desk #443): a turn does not end with open plan items."""
import json

import pytest
from langchain_core.messages import AIMessage

from deep_agent import msty_task as task, msty_tool_routing as routing
from tests.unit_tests.test_msty_compaction import initial, USAGE

TODOS = {'type': 'function', 'function': {'name': 'native_write_todos', 'parameters': {
    'type': 'object', 'properties': {'todos': {'type': 'array', 'items': {'type': 'object'}}},
    'required': ['todos']}}}
PLAN = {'type': 'function', 'function': {'name': 'brain_task_plan', 'parameters': {
    'type': 'object', 'properties': {'items': {'type': 'array', 'items': {'type': 'object'}}},
    'required': ['items']}}}
CHECK = {'type': 'function', 'function': {'name': 'brain_task_check', 'parameters': {
    'type': 'object', 'properties': {
        'stop_attempt': {'type': 'boolean'}, 'open_items': {'type': 'array', 'items': {'type': 'string'}},
        'draft': {'type': 'string'}, 'done': {'type': 'string'}}, 'additionalProperties': False}}}
READ = {'type': 'function', 'function': {'name': 'read_file', 'parameters': {
    'type': 'object', 'properties': {'path': {'type': 'string'}}, 'required': ['path']}}}


def call(name, args, call_id):
    return {'role': 'assistant', 'content': '', 'tool_calls': [{'id': call_id, 'type': 'function',
            'function': {'name': name, 'arguments': json.dumps(args, ensure_ascii=False)}}]}


def result_of(call_id, content='ok'):
    return {'role': 'tool', 'tool_call_id': call_id, 'content': content}


def todos(*statuses):
    return {'todos': [{'content': f'шаг {i + 1}', 'status': s} for i, s in enumerate(statuses)]}


def items(*statuses):
    return {'items': [{'id': f's{i + 1}', 'title': f'пункт {i + 1}', 'criterion': 'файл есть',
                       'status': s} for i, s in enumerate(statuses)]}


def state_with(messages, tools, text='Разбери обязательства и подготовь план подачи.'):
    state = initial(tools=tools)
    state['messages'] = [{'role': 'user', 'content': text}, *messages]
    return state


def final(text='Теперь проверю сроки и подготовлю файл.'):
    return AIMessage(content=text, usage_metadata=USAGE)


def test_native_todos_open_turn_continues_with_same_list():
    args = todos('completed', 'in_progress', 'pending')
    state = state_with([call('native_write_todos', args, 'b1'), result_of('b1')], [TODOS, READ])
    gated = task.gate_final(state, final(), [TODOS, READ], False)
    assert len(gated.tool_calls) == 1
    assert gated.tool_calls[0]['name'] == 'native_write_todos'
    assert gated.tool_calls[0]['args'] == args
    assert gated.tool_calls[0]['id'].startswith(task.PLAN_GATE_PREFIX)
    assert gated.response_metadata['msty_completion_gate'] == 'open_plan_continue'
    # The next step is told why the turn goes on and which items are open.
    state['messages'] += [{'role': 'assistant', 'content': gated.content, 'tool_calls': [
        {'id': gated.tool_calls[0]['id'], 'type': 'function',
         'function': {'name': 'native_write_todos', 'arguments': json.dumps(args)}}]},
        result_of(gated.tool_calls[0]['id'], 'Updated todo list')]
    note = task.open_plan_intervention(state)
    assert note and 'MSTY_OPEN_PLAN_V1' in note and 'шаг 2' in note and 'шаг 3' in note
    assert 'шаг 1' not in note


def test_window_task_plan_continues_through_its_check_tool():
    state = state_with([call('brain_task_plan', items('done', 'pending'), 'b1'), result_of('b1')],
                       [PLAN, CHECK, READ])
    gated = task.gate_final(state, final(), [PLAN, CHECK, READ], False)
    assert [c['name'] for c in gated.tool_calls] == ['brain_task_check']
    args = gated.tool_calls[0]['args']
    assert args['stop_attempt'] is True and args['open_items'] == ['пункт 2']
    assert 'Теперь проверю' in args['draft']


def test_closed_plan_or_no_plan_ends_the_turn_as_before():
    done = state_with([call('native_write_todos', todos('completed', 'completed'), 'b1'), result_of('b1')],
                      [TODOS])
    answer = final('Готово: файл сохранён.')
    assert task.gate_final(done, answer, [TODOS], False) is answer
    plain = state_with([], [TODOS])
    assert task.gate_final(plain, answer, [TODOS], False) is answer


def test_plan_of_an_earlier_owner_turn_does_not_hold_a_new_turn():
    state = state_with([call('native_write_todos', todos('pending'), 'b1'), result_of('b1'),
                        {'role': 'assistant', 'content': 'Промежуточный ответ.'},
                        {'role': 'user', 'content': 'Спасибо, а который час?'}], [TODOS])
    answer = final('Сейчас 12:00.')
    assert task.gate_final(state, answer, [TODOS], False) is answer


@pytest.mark.parametrize('text', ['Стоп, только план.', 'только объясни, что будешь делать'])
def test_owner_control_phrase_vetoes_the_gate(text):
    state = state_with([call('native_write_todos', todos('pending'), 'b1'), result_of('b1')], [TODOS], text)
    answer = final()
    assert task.gate_final(state, answer, [TODOS], False) is answer


def test_no_second_gate_without_real_progress_and_answer_is_marked_unfinished():
    first = task.PLAN_GATE_PREFIX + 'a' * 32
    args = todos('pending')
    state = state_with([call('native_write_todos', args, 'b1'), result_of('b1'),
                        call('native_write_todos', args, first), result_of(first)], [TODOS])
    released = task.gate_final(state, final('Сделаю позже.'), [TODOS], False)
    assert not released.tool_calls
    assert released.response_metadata['msty_completion_gate'] == 'open_plan_released'
    assert 'Не закрыто в плане: шаг 1' in released.content


def test_real_action_after_a_gate_allows_another_until_the_limit():
    args = todos('in_progress')
    messages = [call('native_write_todos', args, 'b1'), result_of('b1')]
    for n in range(task.OPEN_PLAN_GATE_LIMIT):
        gate = task.PLAN_GATE_PREFIX + f'{n:032x}'
        messages += [call('native_write_todos', args, gate), result_of(gate),
                     call('read_file', {'path': f'/x/{n}'}, f'r{n}'), result_of(f'r{n}')]
        state = state_with(list(messages), [TODOS, READ])
        gated = task.gate_final(state, final(), [TODOS, READ], False)
        if n + 1 < task.OPEN_PLAN_GATE_LIMIT:
            assert gated.tool_calls, n
    assert not gated.tool_calls
    assert gated.response_metadata['msty_completion_gate'] == 'open_plan_released'


def test_disabled_tools_forced_other_choice_or_action_cap_never_gate():
    state = state_with([call('native_write_todos', todos('pending'), 'b1'), result_of('b1')], [TODOS])
    answer = final()
    assert task.gate_final(state, answer, [TODOS], True) is answer
    forced = {**state, 'tool_choice': {'type': 'function', 'function': {'name': 'read_file'}}}
    assert task.gate_final(forced, answer, [TODOS, READ], False) is answer
    capped = {**state, 'execution': {**(state.get('execution') or {}), 'actions_issued': 200}}
    assert task.gate_final(capped, answer, [TODOS], False) is answer


def test_truncated_generation_is_not_gated():
    state = state_with([call('native_write_todos', todos('pending'), 'b1'), result_of('b1')], [TODOS])
    cut = AIMessage(content='Нача', usage_metadata=USAGE, response_metadata={'stop_reason': 'max_tokens'})
    assert task.gate_final(state, cut, [TODOS], False) is cut


def test_window_task_tools_are_always_routed_for_brain_desk():
    tools = [PLAN, CHECK, READ] + [
        {'type': 'function', 'function': {'name': f'brain_task_{n}', 'description': n,
                                          'parameters': {'type': 'object', 'properties': {}}}}
        for n in ('ask', 'finish', 'save', 'spawn')]
    messages = [{'role': 'system', 'content': '[Brain Desk · правая рука владельца] …'},
                {'role': 'user', 'content': 'привет'}]
    selected, route, _ = routing.select_tools(messages, tools)
    names = {t['function']['name'] for t in selected}
    assert {'brain_task_plan', 'brain_task_check', 'brain_task_ask', 'brain_task_finish',
            'brain_task_save', 'brain_task_spawn'} <= names
    # Not the window: nothing is forced.
    selected, _, _ = routing.select_tools([{'role': 'user', 'content': 'привет'}], tools)
    assert 'brain_task_plan' not in {t['function']['name'] for t in selected}


def test_real_native_graph_turn_with_open_todos_continues_then_ends_when_closed(monkeypatch):
    """Real msty_native cycle: the premature «теперь сделаю…» becomes one more
    metered native step (bridge ticket), and the closed list ends the turn."""
    import asyncio
    from langchain_core.messages import AIMessage as Answer
    from langgraph.checkpoint.memory import InMemorySaver
    from langgraph.store.memory import InMemoryStore
    from langgraph.types import Command
    from deep_agent import msty_native
    from tests.unit_tests.test_msty_compaction import install

    open_list = {'id': 'todo1', 'name': 'native_write_todos', 'args': todos('in_progress', 'pending')}
    closed = {'id': 'todo2', 'name': 'native_write_todos', 'args': todos('completed', 'completed')}
    seen = install(monkeypatch, [
        Answer(content='План.', tool_calls=[open_list], usage_metadata=USAGE),
        'Теперь проверю сроки и подготовлю файл.',
        Answer(content='Закрываю.', tool_calls=[closed], usage_metadata=USAGE),
        'Готово: оба шага сделаны.'], [100, 110, 120, 130])

    async def scenario():
        graph = msty_native.build_graph(checkpointer=InMemorySaver(), store=InMemoryStore())
        cfg = {'configurable': {'thread_id': 'open-plan-native'}, 'recursion_limit': 64}

        async def invoke(value):
            output = await graph.ainvoke(value, cfg)
            saved = (await graph.aget_state(cfg)).values
            return {**saved, '__interrupt__': output.get('__interrupt__', ())}

        async def resume(state):
            pending = state['__interrupt__'][0]
            return await invoke(Command(resume={pending.id: {**pending.value, 'type': 'msty_native_resume'}}))

        first = await invoke(initial(tools=[], messages=[
            {'role': 'user', 'content': 'Разбери обязательства и подготовь план подачи на 3 месяца.'}]))
        assert first['execution']['status'] == 'waiting_native'
        gated = await resume(first)
        # The text answer did not end the turn: the same list is written again.
        assert gated['execution']['status'] == 'waiting_native'
        calls = gated['result']['tool_calls']
        assert calls[0]['name'] == 'native_write_todos' and calls[0]['id'].startswith(task.PLAN_GATE_PREFIX)
        after = await resume(gated)
        assert after['execution']['status'] == 'waiting_native'
        final = await resume(after)
        assert final['execution']['status'] == 'answered'
        assert final['result']['content'] == 'Готово: оба шага сделаны.'
        assert not final.get('__interrupt__')
        # The step right after the gate was told why the turn went on.
        assert 'MSTY_OPEN_PLAN_V1' in seen['requests'][2][0].text
        assert len(seen['requests']) == 4

    asyncio.run(scenario())


def test_native_swarm_cap_is_configuration_not_code(monkeypatch):
    from deep_agent import msty_swarm
    monkeypatch.delenv('MSTY_SWARM_MAX_SUBTASKS', raising=False)
    assert msty_swarm._max_subtasks() == 5
    monkeypatch.setenv('MSTY_SWARM_MAX_SUBTASKS', '20')
    assert msty_swarm._max_subtasks() == 20
    monkeypatch.setenv('MSTY_SWARM_MAX_SUBTASKS', '500')
    assert msty_swarm._max_subtasks() == 100
    monkeypatch.setenv('MSTY_SWARM_MAX_SUBTASKS', 'x')
    assert msty_swarm._max_subtasks() == 5
