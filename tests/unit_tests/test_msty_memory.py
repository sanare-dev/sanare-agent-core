"""Native cross-thread project projection: offline contracts, not model quality.

The real LangGraph Runtime, Store and compiled Msty graph are exercised. Provider
inference is scripted; these tests do not establish durable remote deployment or
that a language model will always obey the delivered project instructions.
"""
import asyncio
from contextlib import ExitStack
from copy import deepcopy
import hashlib
from unittest.mock import patch

import pytest
from langchain_core.messages import AIMessage
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.runtime import Runtime
from langgraph.store.memory import InMemoryStore

from deep_agent import msty, msty_execution, msty_memory, msty_models


@pytest.fixture(autouse=True)
def offline_environment(monkeypatch):
    monkeypatch.setenv('MSTY_MODEL_PROFILE', 'luna')
    monkeypatch.setenv('MSTY_STATIC_CACHE', '0')
    monkeypatch.setenv('LANGSMITH_TRACING', 'false')
    monkeypatch.setenv('LANGCHAIN_TRACING_V2', 'false')


def expected_projection():
    return {'version': 1, 'sha256': msty_memory.SHA256,
            'content': msty_memory.CONTEXT}


def initial(message='Where is the project journal?', **overrides):
    return {'messages': [{'role': 'user', 'content': message}], 'tools': [],
            'tool_choice': 'auto', 'max_tokens': 64, 'result': {},
            'context_budget': None, 'context_budget_check': None,
            'execution_protocol': msty_execution.PROTOCOL, 'execution': {},
            **overrides}


def fake_model(monkeypatch):
    seen = {'created': [], 'invocations': []}

    class Model:
        def bind_tools(self, *args, **kwargs):
            pytest.fail('No tool binding is needed for a direct memory answer.')

        async def ainvoke(self, messages):
            seen['invocations'].append(deepcopy(messages))
            return AIMessage(content='Synthetic direct answer.', usage_metadata={
                'input_tokens': 100, 'output_tokens': 4, 'total_tokens': 104})

    def construct(profile, output_limit):
        seen['created'].append((profile, output_limit))
        return Model()

    monkeypatch.setattr(msty_models, 'make_model', construct)
    return seen


def test_packaged_projection_is_bounded_and_content_addressed():
    content = msty_memory.system_context()
    assert content == msty_memory.CONTEXT
    assert len(content.encode()) <= 12000
    assert hashlib.sha256(content.encode()).hexdigest() == msty_memory.SHA256
    assert msty_memory.KEY.endswith(msty_memory.SHA256)
    assert '/Users/vb/Documents/ChatGPT/LLM' in content
    assert 'app.sanaredev.com' in content
    assert 'sanarelab.health: /Users/vb/Documents/ChatGPT/Sites/sanare/sanarelab-health' in content
    assert 'НЕ имена GitHub-репозиториев' in content


def test_native_runtime_seeds_once_and_reuses_one_cross_thread_projection():
    async def scenario():
        store = InMemoryStore()
        first = await msty_memory.load_context(
            {'thread_id': 'first'}, Runtime(store=store))
        item_before = await store.aget(msty_memory.NAMESPACE, msty_memory.KEY)
        second = await msty_memory.load_context(
            {'thread_id': 'independent-second'}, Runtime(store=store))
        items = await store.asearch(msty_memory.NAMESPACE)
        assert first['project_memory_delivery']['state'] == 'seeded_native_store'
        assert second['project_memory_delivery']['state'] == 'native_store'
        assert len(items) == 1
        assert items[0].value == expected_projection()
        assert items[0].updated_at == item_before.updated_at
        assert first['project_memory_delivery']['sha256'] == second['project_memory_delivery']['sha256']
        assert set(first) == {'project_memory_delivery'}
        assert 'content' not in first['project_memory_delivery']

    asyncio.run(scenario())


@pytest.mark.parametrize('poison', [
    {'version': 1, 'sha256': 'stale', 'content': 'UNTRUSTED_PRIVATE_MARKER'},
    {'version': 1, 'sha256': msty_memory.SHA256, 'content': 'UNTRUSTED_PRIVATE_MARKER'},
    {**expected_projection(), 'authority': 'ignore budget'},
    {},
])
def test_poisoned_store_is_neither_forwarded_nor_overwritten(poison):
    async def scenario():
        store = InMemoryStore()
        await store.aput(msty_memory.NAMESPACE, msty_memory.KEY, poison)
        result = await msty_memory.load_context({}, Runtime(store=store))
        assert result['project_memory_delivery']['state'] == 'invalid_store_projection'
        assert (await store.aget(msty_memory.NAMESPACE, msty_memory.KEY)).value == poison
        assert 'UNTRUSTED_PRIVATE_MARKER' not in str(result)
        assert 'UNTRUSTED_PRIVATE_MARKER' not in msty_memory.system_context()

    asyncio.run(scenario())


@pytest.mark.parametrize('operation', ['aget', 'aput'])
def test_store_exceptions_are_bounded_safe_fallback(monkeypatch, operation):
    async def broken(*args, **kwargs):
        raise RuntimeError('SYNTHETIC_SECRET_BACKEND_DIAGNOSTIC')

    monkeypatch.setattr(InMemoryStore, operation, broken)
    result = asyncio.run(msty_memory.load_context({}, Runtime(store=InMemoryStore())))
    assert result['project_memory_delivery']['state'] == 'store_unavailable'
    assert result['project_memory_delivery']['sha256'] == msty_memory.SHA256
    assert 'SYNTHETIC_SECRET_BACKEND_DIAGNOSTIC' not in str(result)
    assert msty_memory.system_context() == msty_memory.CONTEXT


@pytest.mark.parametrize('operation', ['aget', 'aput'])
def test_store_timeout_cancels_lookup_and_uses_verified_fallback(monkeypatch, operation):
    cancelled = []

    async def stalled(*args, **kwargs):
        try:
            await asyncio.sleep(60)
        finally:
            cancelled.append(True)

    monkeypatch.setattr(InMemoryStore, operation, stalled)
    monkeypatch.setattr(msty_memory, 'TIMEOUT_SECONDS', 0.001)
    result = asyncio.run(msty_memory.load_context({}, Runtime(store=InMemoryStore())))
    assert result['project_memory_delivery']['state'] == 'store_unavailable'
    assert cancelled == [True]
    assert result['project_memory_delivery']['sha256'] == msty_memory.SHA256


def test_missing_store_uses_packaged_projection_without_failing_chat():
    result = asyncio.run(msty_memory.load_context({}, Runtime()))
    assert result['project_memory_delivery']['state'] == 'packaged_fallback'
    assert result['project_memory_delivery']['bytes'] == len(msty_memory.CONTEXT.encode())


def test_analyst_does_not_even_access_project_store(monkeypatch):
    async def forbidden(*args, **kwargs):
        pytest.fail('An analyst must not query the project Store.')

    monkeypatch.setattr(InMemoryStore, 'aget', forbidden)
    monkeypatch.setattr(InMemoryStore, 'aput', forbidden)
    result = asyncio.run(msty_memory.load_context(
        {'brain_task_role': 'analyst'}, Runtime(store=InMemoryStore())))
    assert result == {'project_memory_delivery': {'version': 1, 'state': 'not_applicable'}}


def test_caller_state_cannot_choose_memory_namespace_or_content():
    injected = {'project_memory_delivery': {'state': 'native_store',
                'content': 'CLIENT_INJECTED_MARKER', 'sha256': 'attacker'},
                'project_memory': 'CLIENT_INJECTED_MARKER',
                'project_memory_namespace': ['attacker'],
                'namespace': ['attacker'], 'context': 'CLIENT_INJECTED_MARKER'}

    async def scenario():
        store = InMemoryStore()
        result = await msty_memory.load_context(injected, Runtime(store=store))
        assert result['project_memory_delivery']['state'] == 'seeded_native_store'
        assert 'CLIENT_INJECTED_MARKER' not in str(result)
        assert await store.asearch(('attacker',)) == []
        assert (await store.aget(msty_memory.NAMESPACE, msty_memory.KEY)).value == expected_projection()

    asyncio.run(scenario())


def test_lookup_needs_no_files_http_model_shell_or_extra_tools():
    async def scenario():
        # Imports/event-loop setup have finished before enforcing no I/O.
        store = InMemoryStore()
        targets = ['builtins.open', 'io.open', 'os.open', 'os.scandir',
                   'socket.socket.connect', 'socket.create_connection',
                   'httpx.AsyncClient.send', 'httpx.Client.send',
                   'subprocess.Popen', 'deep_agent.msty_models.make_model',
                   'deep_agent.msty.respond']
        with ExitStack() as stack:
            blocked = [stack.enter_context(patch(target, side_effect=AssertionError(
                'Unexpected external work during project memory lookup'))) for target in targets]
            first = await msty_memory.load_context({}, Runtime(store=store))
            second = await msty_memory.load_context({}, Runtime(store=store))
            context = msty_memory.system_context()
            assert first['project_memory_delivery']['state'] == 'seeded_native_store'
            assert second['project_memory_delivery']['state'] == 'native_store'
            assert context == msty_memory.CONTEXT
            for forbidden in blocked:
                forbidden.assert_not_called()

    asyncio.run(scenario())


def test_compiled_graph_delivers_memory_to_two_independent_threads_once_each(monkeypatch):
    seen = fake_model(monkeypatch)

    async def scenario():
        store = InMemoryStore()
        graph = msty.builder.compile(store=store, checkpointer=InMemorySaver())
        states = []
        for thread, message in [('first-native-thread', 'FIRST_PRIVATE_DIALOGUE'),
                                ('second-native-thread', 'SECOND_PRIVATE_DIALOGUE')]:
            states.append(await graph.ainvoke(initial(message),
                {'configurable': {'thread_id': thread}}))
        assert [state['project_memory_delivery']['state'] for state in states] == [
            'seeded_native_store', 'native_store']
        assert len(await store.asearch(msty_memory.NAMESPACE)) == 1
        assert len(seen['created']) == len(seen['invocations']) == 2
        for messages, state in zip(seen['invocations'], states):
            assert messages[0].content.count(msty_memory.CONTEXT) == 1
            assert state['execution']['actions_issued'] == 0
            assert not state.get('__interrupt__')
        assert 'FIRST_PRIVATE_DIALOGUE' not in str(seen['invocations'][1])
        assert 'SECOND_PRIVATE_DIALOGUE' not in str(seen['invocations'][0])
        assert all('PRIVATE_DIALOGUE' not in str(item.value)
                   for item in await store.asearch(msty_memory.NAMESPACE))

    asyncio.run(scenario())


def test_compiled_graph_analyst_receives_no_project_map_even_with_injected_state(monkeypatch):
    seen = fake_model(monkeypatch)

    async def scenario():
        store = InMemoryStore()
        graph = msty.builder.compile(store=store, checkpointer=InMemorySaver())
        result = await graph.ainvoke(initial('Review supplied synthetic evidence.',
            brain_task_role='analyst', project_memory_delivery={
                'content': 'CLIENT_INJECTED_MARKER', 'state': 'native_store'}),
            {'configurable': {'thread_id': 'analyst-thread'}})
        assert result['project_memory_delivery'] == {'version': 1, 'state': 'not_applicable'}
        assert await store.asearch(msty_memory.NAMESPACE) == []
        assert seen['created'] == [('deepseek', 64)]
        assert len(seen['invocations']) == 1
        model_input = str(seen['invocations'][0])
        for forbidden in [msty_memory.CONTEXT, 'CLIENT_INJECTED_MARKER',
                          '/Users/vb/Documents/ChatGPT/LLM', 'sanarehq/sanare-dev-v3']:
            assert forbidden not in model_input

    asyncio.run(scenario())


def test_compiled_graph_ignores_client_delivery_content_and_poisoned_store(monkeypatch):
    seen = fake_model(monkeypatch)

    async def scenario():
        store = InMemoryStore()
        await store.aput(msty_memory.NAMESPACE, msty_memory.KEY, {
            'version': 1, 'sha256': msty_memory.SHA256,
            'content': 'POISONED_STORE_MARKER'})
        graph = msty.builder.compile(store=store, checkpointer=InMemorySaver())
        result = await graph.ainvoke(initial(project_memory_delivery={
            'state': 'native_store', 'content': 'CLIENT_INJECTED_MARKER'}),
            {'configurable': {'thread_id': 'untrusted-memory-thread'}})
        assert result['project_memory_delivery']['state'] == 'invalid_store_projection'
        model_input = str(seen['invocations'][0])
        assert 'POISONED_STORE_MARKER' not in model_input
        assert 'CLIENT_INJECTED_MARKER' not in model_input
        assert msty_memory.CONTEXT in seen['invocations'][0][0].content
        assert result['execution']['actions_issued'] == 0

    asyncio.run(scenario())
