"""TAU L4: recovery loop на границе исполнения и errors[] в состоянии графа.

Офлайн: scripted-шаги модели, InMemory checkpoint, никакой сети и провайдеров.
"""
import asyncio
from copy import deepcopy
import json
import socket
from types import SimpleNamespace

import pytest
from langchain_core.messages import AIMessage, ToolMessage
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.store.memory import InMemoryStore
from langgraph.types import Command

from deep_agent import msty, msty_execution, msty_native, msty_taxonomy


@pytest.fixture(autouse=True)
def no_network(monkeypatch):
    for name in ('LANGSMITH_TRACING', 'LANGCHAIN_TRACING', 'LANGCHAIN_TRACING_V2'):
        monkeypatch.setenv(name, 'false')

    def denied(*args, **kwargs):
        raise AssertionError('Recovery offline test attempted network access')

    monkeypatch.setattr(socket.socket, 'connect', denied)
    monkeypatch.setattr(socket, 'create_connection', denied)


def call(name, args, identifier='call-1'):
    return {'id': identifier, 'name': name, 'args': args}


def middleware():
    return msty_native.NativeMstyMiddleware()


def forbidden_handler(request):
    raise AssertionError('Вызов не должен доходить до исполнения')


# --- Retry транзиентов на native-границе ---------------------------------------

def test_transient_native_call_retries_with_backoff_then_succeeds():
    attempts = 0

    async def handler(request):
        nonlocal attempts
        attempts += 1
        if attempts < 3:
            raise ConnectionError('synthetic transport reset')
        return ToolMessage(content='skill body', name='native_read_file',
                           tool_call_id='call-1')

    async def run():
        request = SimpleNamespace(
            tool_call=call('native_read_file', {'file_path': '/memory/PROJECT.md'}),
            state={'native_tool_names': ['native_read_file'], 'tau_errors': []})
        result = await middleware().awrap_tool_call(request, handler)
        assert result.content == 'skill body'
        assert attempts == 3  # исходная + 2 transient-повтора (чтение идемпотентно)
    asyncio.run(run())


def test_transient_failure_of_non_idempotent_call_is_not_retried():
    attempts = 0

    async def handler(request):
        nonlocal attempts
        attempts += 1
        raise ConnectionError('synthetic transport reset')

    async def run():
        request = SimpleNamespace(
            tool_call=call('native_write_file', {'file_path': '/scratch/x.txt', 'content': 'a'}),
            state={'native_tool_names': ['native_write_file'], 'tau_errors': []})
        result = await middleware().awrap_tool_call(request, handler)
        assert attempts == 1  # запись не идемпотентна: слепой retry запрещён
        assert result.status == 'error'
        assert 'error=transient' in result.content
        event = result.additional_kwargs['tau_event']
        assert event['kind'] == 'failure' and event['class'] == 'transient'
        assert event['attempt'] == 1 and event['source'] == 'native'
    asyncio.run(run())


def test_deterministic_native_failure_never_retries_even_for_reads():
    attempts = 0

    async def handler(request):
        nonlocal attempts
        attempts += 1
        raise ValueError('synthetic deterministic failure')

    async def run():
        request = SimpleNamespace(
            tool_call=call('native_read_file', {'file_path': '/memory/PROJECT.md'}),
            state={'native_tool_names': ['native_read_file'], 'tau_errors': []})
        result = await middleware().awrap_tool_call(request, handler)
        assert attempts == 1
        assert 'error=deterministic' in result.content
    asyncio.run(run())


def test_exhausted_budget_degrades_instead_of_executing():
    failed_call = call('native_read_file', {'file_path': '/memory/PROJECT.md'})
    # turn=0: отказы текущего хода (в state нет сообщений владельца).
    errors = [msty_taxonomy.error_entry(failed_call, 'transient', 'native', 1, 0),
              msty_taxonomy.error_entry(failed_call, 'transient', 'native', 2, 0)]

    async def run():
        request = SimpleNamespace(tool_call=failed_call,
                                  state={'native_tool_names': ['native_read_file'],
                                         'tau_errors': errors})
        result = await middleware().awrap_tool_call(request, forbidden_handler)
        assert result.status == 'error'
        assert 'budget_exhausted' in result.content
        assert 'попытка 2' in result.content
        # Дегрейд не добавляет новых записей: маркера события нет.
        assert 'tau_event' not in result.additional_kwargs
    asyncio.run(run())


def test_budget_from_previous_turn_does_not_block_new_turn():
    failed_call = call('native_read_file', {'file_path': '/memory/PROJECT.md'})
    # Два отказа в ходе 1; сейчас ход 2 — чтение должно исполниться.
    errors = [msty_taxonomy.error_entry(failed_call, 'transient', 'native', 1, 1),
              msty_taxonomy.error_entry(failed_call, 'transient', 'native', 2, 1)]
    executed = []

    async def handler(request):
        executed.append(request.tool_call['id'])
        return ToolMessage(content='ok', name='native_read_file', tool_call_id='call-1')

    async def run():
        request = SimpleNamespace(tool_call=failed_call, state={
            'native_tool_names': ['native_read_file'], 'tau_errors': errors,
            'messages': [{'role': 'user', 'content': 'a'}, {'role': 'user', 'content': 'b'}]})
        result = await middleware().awrap_tool_call(request, handler)
        assert result.content == 'ok' and executed == ['call-1']
    asyncio.run(run())


# --- Классификация отказов внешних наблюдений -----------------------------------

SQL_TOOL = {'type': 'function', 'function': {'name': 'execute_sql',
    'parameters': {'type': 'object', 'properties': {'query': {'type': 'string'}}}}}


def external_state(observation, identifier='call-1', errors=()):
    return {'tools': [deepcopy(SQL_TOOL)],
            'native_external_observations': {identifier: observation},
            'tau_errors': list(errors)}


def test_external_transient_observation_is_classified_and_annotated():
    async def run():
        request = SimpleNamespace(
            tool_call=call('execute_sql', {'query': 'select 1'}),
            state=external_state('HTTP 503 Service Unavailable'))
        result = await middleware().awrap_tool_call(request, forbidden_handler)
        assert result.status == 'error'
        # execute_sql — write: временный сбой = неизвестный исход, не «повтори».
        assert result.content.startswith('tau_class=unknown_state.')
        assert 'допустим один повтор' not in result.content
        assert 'HTTP 503' in result.content  # исходное наблюдение сохранено
        event = result.additional_kwargs['tau_event']
        assert event['kind'] == 'failure' and event['class'] == 'transient'
        assert event['source'] == 'external'
    asyncio.run(run())


def test_external_unknown_tool_passthrough_is_classified():
    """Исходный дефект 2026-09-22: «Unknown tool» из execute_tool — теперь
    классифицированный отказ с политикой прямого вызова, а не сырой текст."""
    async def run():
        request = SimpleNamespace(
            tool_call=call('execute_sql', {'query': 'select 1'}),
            state=external_state('Unknown tool: msty_store_sync_status'))
        result = await middleware().awrap_tool_call(request, forbidden_handler)
        assert result.content.startswith('tau_class=unknown_tool.')
        assert 'напрямую' in result.content and 'execute_tool' in result.content
    asyncio.run(run())


def test_external_failure_beyond_budget_is_replaced_by_honest_degradation():
    failed_call = call('execute_sql', {'query': 'select 1'})
    errors = [msty_taxonomy.error_entry(failed_call, 'transient', 'external', 1, 0),
              msty_taxonomy.error_entry(failed_call, 'transient', 'external', 2, 0)]

    async def run():
        request = SimpleNamespace(
            tool_call=failed_call,
            state=external_state('HTTP 503 Service Unavailable', errors=errors))
        result = await middleware().awrap_tool_call(request, forbidden_handler)
        assert 'budget_exhausted' in result.content
        assert 'HTTP 503' not in result.content  # дегрейд подменяет зацикленный отказ
    asyncio.run(run())


def test_successful_status_read_is_recorded_as_evidence():
    async def run():
        request = SimpleNamespace(
            tool_call=call('msty_store_sync_status', {}),
            state={'tools': [{'type': 'function', 'function': {
                       'name': 'msty_store_sync_status', 'parameters': {'type': 'object'}}}],
                   'native_external_observations': {'call-1': '{"status": "ok", "freshness_s": 210}'},
                   'tau_errors': []})
        result = await middleware().awrap_tool_call(request, forbidden_handler)
        assert result.status == 'success'
        assert result.content == '{"status": "ok", "freshness_s": 210}'
        event = result.additional_kwargs['tau_event']
        assert event['kind'] == 'evidence' and event['evidence_class'] == 'status_read'
    asyncio.run(run())


# --- Свёртка событий в состояние -------------------------------------------------

def test_fold_tau_dedupes_by_tool_call_id():
    events = [{'kind': 'failure', 'tool_call_id': 'a', 'class': 'transient'},
              {'kind': 'failure', 'tool_call_id': 'a', 'class': 'transient'},
              {'kind': 'evidence', 'tool_call_id': 'b', 'evidence_class': 'status_read'}]
    errors = msty_native._fold_tau([], events, 'failure')
    evidence = msty_native._fold_tau([], events, 'evidence')
    assert len(errors) == 1 and len(evidence) == 1
    again = msty_native._fold_tau(errors, events, 'failure')
    assert len(again) == 1


# --- Сквозной контур: errors[] в checkpoint-состоянии -----------------------------

def answer(content='Done.', calls=None):
    return AIMessage(content=content, tool_calls=calls or [],
        usage_metadata={'input_tokens': 100, 'output_tokens': 10, 'total_tokens': 110})


def initial():
    return {'messages': [{'role': 'user', 'content': 'Проверь execute_sql запрос.'}],
            'tools': [deepcopy(SQL_TOOL)], 'max_tokens': 128, 'tool_choice': 'auto',
            'result': {}, 'context_budget': None, 'context_budget_check': None,
            'execution_protocol': msty_execution.PROTOCOL, 'execution': {}}


async def invoke(graph, value, config):
    async for mode, chunk in graph.astream(value, config, stream_mode=['custom', 'values'],
                                           durability='sync'):
        pass
    return await graph.aget_state(config)


def external_resume(state, observation):
    incoming = {key: deepcopy(state[key]) for key in
        ('tools', 'max_tokens', 'tool_choice', 'context_budget', 'execution_protocol')}
    incoming.update(messages=deepcopy(state['native_protocol_messages']), result={},
                    context_budget_check=None)
    pending = state['execution']['pending']['calls']
    assert len(pending) == 1
    item = pending[0]
    client = 'b1_fixture_0'
    calls = [{'id': client, 'type': 'function', 'function': {'name': item['name'],
              'arguments': json.dumps(item['args'])}}]
    results = [{'role': 'tool', 'tool_call_id': client, 'content': observation}]
    incoming['messages'] += [{'role': 'assistant', 'content': state['result']['content'],
                              'tool_calls': calls}, *results]
    return {'version': 1, 'task_id': state['execution']['task_id'],
            'batch_id': state['execution']['pending']['batch_id'],
            'tool_id_map': [{'client_id': client, 'model_id': item['id']}], 'input': incoming}


def test_repeated_external_transient_failure_lands_in_state_and_degrades(monkeypatch):
    def sql_call(identifier):
        return call('execute_sql', {'query': 'select 1'}, identifier)
    seen = []
    results = [answer('', [sql_call('e1')]), answer('', [sql_call('e2')]),
               answer('', [sql_call('e3')]), answer('Честный дегрейд.')]

    async def step(state, *, native_system_prompt=None, native_result_filter=None):
        result = results.pop(0)
        seen.append(deepcopy(state))
        return msty.publish_result(result, None)

    monkeypatch.setattr(msty, '_respond_step', step)

    async def run():
        graph = msty_native.build_graph(checkpointer=InMemorySaver(), store=InMemoryStore())
        config = {'configurable': {'thread_id': 'tau-recovery-budget'}}
        state = await invoke(graph, initial(), config)
        for _ in range(3):
            pending = state.tasks[0].interrupts[0]
            resume = external_resume(state.values, 'HTTP 503 Service Unavailable')
            state = await invoke(graph, Command(resume={pending.id: resume}), config)
        assert len(seen) == 4
        # Два отказа записаны в errors[] состояния; третий идентичный вызов
        # получил честный дегрейд без новой записи и без исполнения.
        recorded = state.values['tau_errors']
        assert len(recorded) == 2
        assert {entry['class'] for entry in recorded} == {'transient'}
        assert {entry['attempt'] for entry in recorded} == {1, 2}
        tool_messages = [m for m in state.values['messages'] if m.type == 'tool']
        assert 'budget_exhausted' in tool_messages[-1].content
        assert state.values['execution']['status'] == 'answered'
    asyncio.run(run())


def test_cancelled_probe_does_not_block_profile():
    from deep_agent import msty_breaker
    msty_breaker.reset()
    connection = 'model:test-probe'
    for _ in range(msty_breaker.FAILURE_THRESHOLD):
        msty_breaker.record_transient_failure(connection)
    state = msty_breaker._connections[connection]
    state['open_until'] = 0.1  # cooldown истёк
    remaining, token = msty_breaker.admit(connection)
    assert remaining is None and token  # выдана проба этому вызову
    assert msty_breaker.open_remaining(connection) is not None  # проба идёт
    msty_breaker.release_probe(connection)  # посторонний вызов пробу не снимает
    assert msty_breaker.open_remaining(connection) is not None
    msty_breaker.release_probe(connection, token)  # владелец: отменённая проба
    assert msty_breaker.open_remaining(connection) is None
    msty_breaker.reset()
