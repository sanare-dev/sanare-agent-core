"""Candidate memory: writable /memories/, read-only /memory/, Store index, consolidation."""
import asyncio
import json
from pathlib import Path
import socket
from types import SimpleNamespace

import pytest
from langchain_core.language_models.fake_chat_models import FakeMessagesListChatModel
from langchain_core.messages import AIMessage, ToolMessage
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.store.memory import InMemoryStore
from deepagents import create_deep_agent

from deep_agent import consolidator, msty_native, msty_prompts
from deep_agent import msty_native_memory as memory

ROOT = Path(__file__).resolve().parents[2]


@pytest.fixture(autouse=True)
def offline(monkeypatch):
    for name in ('LANGSMITH_TRACING', 'LANGCHAIN_TRACING', 'LANGCHAIN_TRACING_V2'):
        monkeypatch.setenv(name, 'false')

    def forbidden(*args, **kwargs):
        pytest.fail('Candidate memory test attempted a network connection')

    monkeypatch.setattr(socket.socket, 'connect', forbidden)
    monkeypatch.setattr(socket, 'create_connection', forbidden)


class Model(FakeMessagesListChatModel):
    def bind_tools(self, tools, **kwargs):
        return self


def tool_call(identifier, name, **args):
    return AIMessage(content='', tool_calls=[{'id': identifier, 'name': name, 'args': args}])


def test_store_index_and_consolidator_not_registered_until_verified():
    """2026-09-23: с store.index ревизия не стартовала (core-server gRPC не поднялся
    за 60 с) — индекс и граф consolidator сняты до отдельной проверки на staging."""
    config = json.loads((ROOT / 'langgraph.json').read_text())
    assert 'store' not in config
    assert 'consolidator' not in config['graphs']
    assert config['graphs']['msty_native'] == 'deep_agent.msty_native:graph'


def test_candidates_route_uses_separate_namespace():
    assert memory.CANDIDATES_NAMESPACE == ('sanare-owner', 'knowledge', 'candidates')
    assert memory.CANDIDATES_NAMESPACE[:2] != memory.NAMESPACE[:2]
    backend = memory.backend_factory(SimpleNamespace(store=None, state={}))
    assert set(backend.routes) == {'/memory/', '/skills/', '/memories/'}
    assert isinstance(backend.routes['/memory/'], memory.ApprovedStoreBackend)
    assert type(backend.routes['/memories/']).__name__ == 'StoreBackend'


def test_real_backend_writes_candidates_but_approved_memory_stays_read_only():
    async def scenario():
        store = InMemoryStore()
        model = Model(responses=[
            tool_call('deny', 'write_file', file_path='/memory/PROJECT.md', content='BAD'),
            tool_call('cand', 'write_file', file_path='/memories/brain/memory.md',
                      content='Что: включена память. Где: sanare-agent-core. Источник: PR.'),
            tool_call('edit', 'edit_file', file_path='/memories/brain/memory.md',
                      old_string='включена', new_string='проверена'),
            AIMessage(content='done'),
        ])
        agent = create_deep_agent(model=model, backend=memory.backend_factory,
                                  store=store, checkpointer=InMemorySaver())
        result = await agent.ainvoke({'messages': [{'role': 'user', 'content': 'write'}]},
                                     {'configurable': {'thread_id': 'candidates'}})
        observations = [m for m in result['messages'] if isinstance(m, ToolMessage)]
        assert 'permission_denied' in observations[0].content
        item = await store.aget(memory.CANDIDATES_NAMESPACE, '/brain/memory.md')
        assert item is not None and 'проверена' in '\n'.join(item.value['content'])
        assert await store.aget(memory.NAMESPACE + ('memory',), '/PROJECT.md') is None
        assert not any(path.startswith('/memories/') for path in result.get('files') or {})
    asyncio.run(scenario())


@pytest.mark.parametrize('name,args', [
    ('native_write_file', {'file_path': '/memories/brain/card.md', 'content': 'x'}),
    ('native_edit_file', {'file_path': '/memories/brain/card.md', 'old_string': 'x', 'new_string': 'y'}),
    ('native_grep', {'path': '/memories/', 'pattern': 'x'}),
    ('native_glob', {'path': '/memories/', 'pattern': '**/*.md'}),
    ('native_read_file', {'file_path': '/memories/brain/card.md'}),
])
def test_harness_admits_memories_paths(name, args):
    assert msty_native._virtual_path_error(
        {'id': 't', 'name': name, 'args': args}) is None


@pytest.mark.parametrize('name,args', [
    ('native_write_file', {'file_path': '/memories/', 'content': 'x'}),
    ('native_write_file', {'file_path': '/memory/PROJECT.md', 'content': 'x'}),
    ('native_edit_file', {'file_path': '/memory/PROJECT.md', 'old_string': 'a', 'new_string': 'b'}),
    ('native_write_file', {'file_path': '/skills/x/SKILL.md', 'content': 'x'}),
    ('native_write_file', {'file_path': '/memories/../memory/PROJECT.md', 'content': 'x'}),
    ('native_write_file', {'file_path': '/memoriesx/card.md', 'content': 'x'}),
])
def test_harness_still_denies_approved_and_escape_writes(name, args):
    error = msty_native._virtual_path_error({'id': 't', 'name': name, 'args': args})
    assert error is not None and 'native_virtual_path_required' in error.content


def test_candidate_write_observation_is_labelled_reference_only():
    message = ToolMessage(content='Updated file', tool_call_id='t')
    labelled = msty_native._virtual_write_observation(message, '/memories/brain/card.md')
    assert labelled.content.startswith('VIRTUAL CANDIDATE MEMORY')
    assert 'VIRTUAL SCRATCH ONLY' in msty_native._virtual_write_observation(message, '/scratch/x').content


def test_search_tool_is_stock_langmem_and_gated_until_bridge_admits_it(monkeypatch):
    tool = msty_native.memory_search_tool()
    assert tool.name == msty_native.MEMORY_SEARCH_TOOL == 'native_search_memory'
    assert tool.__module__.startswith('langchain') or 'langmem' in tool.func.__module__
    assert msty_native.MEMORY_SEARCH_TOOL in msty_native.SERVER_EXECUTED
    assert msty_native.MEMORY_SEARCH_TOOL in msty_native.RESERVED_TOOLS
    monkeypatch.delenv('MSTY_MEMORY_SEARCH', raising=False)
    assert not msty_native.memory_search_enabled()
    monkeypatch.setenv('MSTY_MEMORY_SEARCH', 'on')
    assert msty_native.memory_search_enabled()


def test_search_tool_searches_candidates_namespace():
    async def scenario():
        store = InMemoryStore()
        await store.aput(memory.CANDIDATES_NAMESPACE, '/brain/a.md',
                         {'content': ['candidate fact'], 'created_at': 'x', 'modified_at': 'x'})
        await store.aput(memory.NAMESPACE + ('memory',), '/PROJECT.md',
                         {'content': ['approved fact'], 'created_at': 'x', 'modified_at': 'x'})
        tool = msty_native.memory_search_tool()
        from langmem.knowledge import tools as langmem_tools
        original = langmem_tools._get_store
        langmem_tools._get_store = lambda initial=None: store
        try:
            found = json.loads(await tool.ainvoke({'query': 'fact'}))
        finally:
            langmem_tools._get_store = original
        assert [item['key'] for item in found] == ['/brain/a.md']
        assert found[0]['namespace'] == list(memory.CANDIDATES_NAMESPACE)
    asyncio.run(scenario())


def test_policy_block_tells_brain_to_search_and_record_candidates():
    assert 'MSTY_CANDIDATE_MEMORY_V1' in msty_prompts.POLICY
    assert 'MSTY_CANDIDATE_MEMORY_V1' in msty_prompts.ACTIONABLE_BLOCKS
    block = next(b for b in msty_prompts.POLICY.split('\n\n') if b.startswith('MSTY_CANDIDATE_MEMORY_V1'))
    # Имя поиска не называется: пока флаг выключен, модель не должна его выдумывать (ревью PR #5).
    assert 'native_search_memory' not in block
    for needle in ('/memories/', 'поиска', 'native_grep', 'источники', 'reference_only'):
        assert needle in block
    assert '/memories/' in msty_native.VIRTUAL_FS_SCOPE and '/memories' in msty_native.VIRTUAL_ROOTS


def test_consolidator_digest_is_bounded_and_skips_tool_payloads():
    from datetime import datetime, timezone
    since = datetime(2026, 9, 23, tzinfo=timezone.utc)
    threads = [
        {'thread_id': 'old', 'updated_at': '2026-09-22T00:00:00+00:00',
         'values': {'messages': [{'type': 'human', 'content': 'OLD'}]}},
        {'thread_id': 'new', 'updated_at': '2026-09-23T05:00:00Z', 'values': {'messages': [
            {'type': 'human', 'content': 'x' * 5000},
            {'type': 'tool', 'content': 'TOOL_PAYLOAD'},
            {'type': 'ai', 'content': [{'type': 'text', 'text': 'done'}]}]}},
    ]
    digest = consolidator.format_threads(threads, since)
    assert 'OLD' not in digest and 'TOOL_PAYLOAD' not in digest
    assert 'thread new' in digest and 'ai: done' in digest
    assert len(digest) < consolidator.MAX_CHARS_PER_MESSAGE + 200


def test_consolidation_threads_are_excluded_from_digest():
    from datetime import datetime, timezone
    since = datetime(2026, 9, 23, tzinfo=timezone.utc)
    threads = [
        {'thread_id': 'cons', 'updated_at': '2026-09-23T06:00:00Z', 'values': {'messages': [
            {'type': 'human', 'content': consolidator.CONSOLIDATION_PROMPT},
            {'type': 'ai', 'content': 'нет новых фактов'}]}},
        {'thread_id': 'work', 'updated_at': '2026-09-23T06:00:00Z', 'values': {'messages': [
            {'type': 'human', 'content': 'почини сайт'}, {'type': 'ai', 'content': 'готово'}]}},
    ]
    digest = consolidator.format_threads(threads, since)
    assert 'thread cons' not in digest and 'thread work' in digest
    assert consolidator.CONSOLIDATION_MARKER in consolidator.CONSOLIDATION_PROMPT
    assert consolidator.RECENT_CONVERSATIONS_TOOL in consolidator.CONSOLIDATION_PROMPT


def test_recent_conversations_tool_is_read_only_bounded_search(monkeypatch):
    calls = []

    class Threads:
        async def search(self, **kwargs):
            calls.append(kwargs)
            return [{'thread_id': 't1', 'updated_at': '2999-01-01T00:00:00Z',
                     'values': {'messages': [{'type': 'human', 'content': 'hello'}]}}]

    monkeypatch.setattr(consolidator, 'get_client', lambda: SimpleNamespace(threads=Threads()))
    tool = consolidator.recent_conversations
    assert tool.name == 'native_recent_conversations'
    assert tool.tool_call_schema.model_json_schema().get('properties', {}) == {}
    digest = asyncio.run(tool.ainvoke({}))
    assert 'thread t1' in digest and 'human: hello' in digest
    assert calls == [{'metadata': {'graph_id': 'msty_native'}, 'sort_by': 'updated_at',
                      'sort_order': 'desc', 'limit': consolidator.MAX_THREADS,
                      'select': ['thread_id', 'updated_at', 'values']}]


def test_recent_conversations_gated_until_bridge_admits_it(monkeypatch):
    name = msty_native.RECENT_CONVERSATIONS_TOOL
    assert name in msty_native.SERVER_EXECUTED and name in msty_native.RESERVED_TOOLS
    monkeypatch.delenv('MSTY_RECENT_CONVERSATIONS', raising=False)
    assert name not in msty_native.server_executed()
    monkeypatch.setenv('MSTY_RECENT_CONVERSATIONS', 'on')
    assert name in msty_native.server_executed()


def test_recent_conversations_offered_and_executed_through_native_harness(monkeypatch):
    from langgraph.types import Command
    from tests.unit_tests.test_msty_native import answer, call, initial, invoke, scripted
    monkeypatch.setenv('MSTY_RECENT_CONVERSATIONS', 'on')

    monkeypatch.setattr(consolidator, 'get_client', lambda: SimpleNamespace(threads=SimpleNamespace(
        search=lambda **kw: _async([{'thread_id': 'x', 'updated_at': '2999-01-01T00:00:00Z',
                                      'values': {'messages': [{'type': 'ai', 'content': 'DIGEST'}]}}]))))
    seen = scripted(monkeypatch, [answer('', [call('native_recent_conversations', {}, 'r')]), answer()])

    async def run():
        graph = msty_native.build_graph(checkpointer=InMemorySaver(), store=InMemoryStore())
        config = {'configurable': {'thread_id': 'recent-on'}, 'recursion_limit': 32}
        start = initial()
        # Инструмент выдаётся только в прогоне консолидации (маркер, ревью PR #8).
        start['messages'] = [{'role': 'user', 'content': consolidator.CONSOLIDATION_PROMPT}]
        state, _ = await invoke(graph, start, config)
        names = [tool['function']['name'] for tool in seen[0]['state']['tools']]
        assert 'native_recent_conversations' in names
        ticket = state.tasks[0].interrupts[0]
        state, _ = await invoke(graph, Command(resume={ticket.id: {
            **ticket.value, 'type': 'msty_native_resume'}}), config)
        tools = [m for m in state.values['messages'] if m.type == 'tool']
        assert tools and 'ai: DIGEST' in tools[0].content
    asyncio.run(run())


async def _async(value):
    return value


def test_full_native_harness_writes_candidate_card_into_store(monkeypatch):
    from langgraph.types import Command
    from tests.unit_tests.test_msty_native import answer, call, initial, invoke, scripted
    scripted(monkeypatch, [
        answer('', [call('native_write_file', {'file_path': '/memories/brain/harness.md',
                                               'content': 'Что: проверка. Где: PR.'}, 'w')]),
        answer('', [call('native_write_file', {'file_path': '/memory/PROJECT.md', 'content': 'BAD'}, 'd')]),
        answer()])

    async def run():
        store = InMemoryStore()
        graph = msty_native.build_graph(checkpointer=InMemorySaver(), store=store)
        config = {'configurable': {'thread_id': 'harness-candidates'}, 'recursion_limit': 64}
        state, _ = await invoke(graph, initial(), config)
        for _ in range(2):
            ticket = state.tasks[0].interrupts[0]
            state, _ = await invoke(graph, Command(resume={ticket.id: {
                **ticket.value, 'type': 'msty_native_resume'}}), config)
        tools = [m for m in state.values['messages'] if m.type == 'tool']
        assert tools[0].content.startswith('VIRTUAL CANDIDATE MEMORY')
        assert 'native_virtual_path_required' in tools[1].content
        item = await store.aget(memory.CANDIDATES_NAMESPACE, '/brain/harness.md')
        assert item is not None and item.value['content'] == ['Что: проверка. Где: PR.']
        assert await store.aget(memory.NAMESPACE + ('memory',), '/PROJECT.md') is None
        assert 'native_search_memory' not in state.values['native_tool_names']
    asyncio.run(run())


def test_search_schema_offered_only_when_enabled(monkeypatch):
    from tests.unit_tests.test_msty_native import answer, initial, invoke, scripted
    monkeypatch.setenv('MSTY_MEMORY_SEARCH', 'on')
    seen = scripted(monkeypatch, [answer()])

    async def run():
        graph = msty_native.build_graph(checkpointer=InMemorySaver(), store=InMemoryStore())
        config = {'configurable': {'thread_id': 'search-on'}, 'recursion_limit': 16}
        state, _ = await invoke(graph, initial(), config)
        names = [tool['function']['name'] for tool in seen[0]['state']['tools']]
        assert 'native_search_memory' in names
        assert 'native_search_memory' in state.values['native_tool_names']
    asyncio.run(run())


def test_disabled_tools_are_not_server_executed(monkeypatch):
    """Ревью PR #5: выдуманный native_search_memory при выключенном флаге не должен
    уходить в native-прерывание (ExecutionProtocolError на продолжении)."""
    from deep_agent import msty_native
    monkeypatch.delenv('MSTY_MEMORY_SEARCH', raising=False)
    monkeypatch.delenv('MSTY_TOOL_DISPATCHER', raising=False)
    names = msty_native.server_executed()
    assert msty_native.MEMORY_SEARCH_TOOL not in names
    monkeypatch.setenv('MSTY_MEMORY_SEARCH', 'on')
    assert msty_native.MEMORY_SEARCH_TOOL in msty_native.server_executed()
