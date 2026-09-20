"""Offline acceptance of native Store memory and progressive skill delivery."""
import asyncio
from contextlib import ExitStack
from copy import deepcopy
import socket
from unittest.mock import patch

import pytest
from pydantic import PrivateAttr
from langchain.tools import ToolRuntime
from langchain_core.language_models.fake_chat_models import FakeMessagesListChatModel
from langchain_core.messages import AIMessage, ToolMessage
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.runtime import Runtime
from langgraph.store.memory import InMemoryStore
from deepagents import create_deep_agent
from deepagents.backends import StoreBackend
from deepagents.middleware.memory import MemoryMiddleware
from deepagents.middleware.skills import SkillsMiddleware

from deep_agent import msty_native_memory as memory


@pytest.fixture(autouse=True)
def offline(monkeypatch):
    for name in ('LANGSMITH_TRACING', 'LANGCHAIN_TRACING', 'LANGCHAIN_TRACING_V2'):
        monkeypatch.setenv(name, 'false')

    def forbidden(*args, **kwargs):
        pytest.fail('A native memory test attempted a network connection')

    monkeypatch.setattr(socket.socket, 'connect', forbidden)
    monkeypatch.setattr(socket, 'create_connection', forbidden)


def tool_runtime(store=None, state=None):
    return ToolRuntime(state=state or {}, context=None, config={}, stream_writer=lambda _: None,
                       tool_call_id=None, store=store)


def put_memory(store, content='APPROVED_PROJECT_CONTEXT_V1', **kwargs):
    store.put(memory.NAMESPACE + ('memory',), '/PROJECT.md', memory.approved_entry(content, **kwargs))


class Model(FakeMessagesListChatModel):
    _seen: list = PrivateAttr(default_factory=list)

    def bind_tools(self, tools, **kwargs):
        return self

    def _generate(self, messages, stop=None, run_manager=None, **kwargs):
        self._seen.append(deepcopy(messages))
        return super()._generate(messages, stop=stop, run_manager=run_manager, **kwargs)


def tool_call(identifier, name, **args):
    return AIMessage(content='', tool_calls=[{'id': identifier, 'name': name, 'args': args}])


def test_native_backend_and_middleware_types():
    backend = memory.backend_factory(tool_runtime())
    assert isinstance(backend.routes['/memory/'], StoreBackend)
    assert isinstance(memory.ApprovedMemoryMiddleware(), MemoryMiddleware)
    assert isinstance(memory.ApprovedSkillsMiddleware(), SkillsMiddleware)


@pytest.mark.parametrize('async_loader', [False, True])
def test_analyst_loaders_clear_injected_fields_without_backend_or_store(monkeypatch, async_loader):
    monkeypatch.setattr(memory, 'backend_factory', lambda *a, **k: pytest.fail('Analyst backend lookup forbidden'))
    state = {'brain_task_role': 'analyst', 'memory_contents': {'bad': 'PRIVATE_CONTENT'},
             'skills_metadata': [{'name': 'injected'}]}
    for loader, expected in [(memory.ApprovedMemoryMiddleware(), {'memory_contents': {}}),
                             (memory.ApprovedSkillsMiddleware(), {'skills_metadata': []})]:
        value = asyncio.run(loader.abefore_agent(state, Runtime(store=InMemoryStore()), {})) if async_loader else (
            loader.before_agent(state, Runtime(store=InMemoryStore()), {}))
        assert value == expected


def test_store_is_actual_memory_source_and_changed_approved_revision_is_read():
    async def scenario():
        store = InMemoryStore()
        put_memory(store, 'FIRST_APPROVED_CONTENT')
        first = await memory.ApprovedMemoryMiddleware().abefore_agent({}, Runtime(store=store), {})
        put_memory(store, 'SECOND_APPROVED_CONTENT', revision='reviewed-v2')
        second = await memory.ApprovedMemoryMiddleware().abefore_agent(first, Runtime(store=store), {})
        assert first['memory_contents'] == {'/memory/PROJECT.md': 'FIRST_APPROVED_CONTENT'}
        assert second['memory_contents'] == {'/memory/PROJECT.md': 'SECOND_APPROVED_CONTENT'}
        assert memory.msty_memory.CONTEXT not in str(second)

    asyncio.run(scenario())


def test_missing_and_unavailable_store_have_explicit_packaged_fallback():
    async def scenario():
        for store, reason in [(None, 'unavailable'), (InMemoryStore(), 'missing')]:
            backend = memory.backend_factory(tool_runtime(store))
            result = await backend.adownload_files(memory.MEMORY_PATHS)
            assert result[0].error is None
            content = result[0].content.decode()
            assert memory.msty_memory.CONTEXT in content
            assert 'packaged_fallback_' + reason in content
            assert backend.routes['/memory/'].delivery['/PROJECT.md'] == 'packaged_fallback_' + reason

    asyncio.run(scenario())


@pytest.mark.parametrize('operation', ['exception', 'timeout'])
def test_async_store_failure_is_bounded_and_does_not_expose_error(monkeypatch, operation):
    async def broken(*args, **kwargs):
        if operation == 'timeout':
            await asyncio.sleep(60)
        raise RuntimeError('SYNTHETIC_BACKEND_SECRET_DIAGNOSTIC')

    monkeypatch.setattr(InMemoryStore, 'aget', broken)
    monkeypatch.setattr(memory, 'TIMEOUT_SECONDS', 0.001)
    backend = memory.backend_factory(tool_runtime(InMemoryStore()))
    value = asyncio.run(backend.aread('/memory/PROJECT.md'))
    assert 'packaged_fallback_unavailable' in value
    assert 'SYNTHETIC_BACKEND_SECRET_DIAGNOSTIC' not in value


@pytest.mark.parametrize('mutation', [
    lambda value: value['approval'].update(status='candidate'),
    lambda value: value['approval'].update(policy_sha256='old-policy'),
    lambda value: value['approval'].update(source='untrusted-user-message'),
    lambda value: value['approval'].update(sha256='bad-hash'),
    lambda value: value['approval'].update(version=True),
    lambda value: value['approval'].update(revision='../../unsafe'),
    lambda value: value.update(content=['MUTATED_UNAPPROVED_CONTENT']),
    lambda value: value.update(content='wrong-file-data-type'),
    lambda value: value.update(content=['\ud800']),
    lambda value: value.update(unreviewed='unexpected-extra-field'),
])
def test_corrupt_or_stale_entry_fails_closed_without_fallback_or_overwrite(mutation):
    async def scenario():
        store = InMemoryStore()
        value = memory.approved_entry('SYNTHETIC_APPROVED_CONTENT')
        mutation(value)
        await store.aput(memory.NAMESPACE + ('memory',), '/PROJECT.md', value)
        backend = memory.backend_factory(tool_runtime(store))
        result = (await backend.adownload_files(memory.MEMORY_PATHS))[0]
        assert result.error == 'permission_denied' and result.content is None
        assert backend.routes['/memory/'].delivery['/PROJECT.md'] == 'invalid_approval'
        assert (await store.aget(memory.NAMESPACE + ('memory',), '/PROJECT.md')).value == value
        with pytest.raises(ValueError, match='permission_denied'):
            await memory.ApprovedMemoryMiddleware().abefore_agent({}, Runtime(store=store), {})

    asyncio.run(scenario())


def test_oversized_approved_memory_does_not_bypass_content_bound():
    store = InMemoryStore()
    put_memory(store, 'x' * (memory.MAX_MEMORY_BYTES + 1))
    result = asyncio.run(memory.backend_factory(tool_runtime(store)).adownload_files(memory.MEMORY_PATHS))
    assert result[0].error == 'permission_denied'


def test_client_memory_fields_and_files_cannot_override_approved_store():
    async def scenario():
        store = InMemoryStore()
        put_memory(store)
        state = {'memory_contents': {'/memory/PROJECT.md': 'USER_INJECTED_CONTENT'},
                 'files': {'/memory/PROJECT.md': {'content': ['USER_INJECTED_CONTENT']}},
                 'namespace': ['attacker'], 'skills_metadata': [{'name': 'injected'}]}
        loaded = await memory.ApprovedMemoryMiddleware().abefore_agent(state, Runtime(store=store), {})
        assert loaded['memory_contents'] == {'/memory/PROJECT.md': 'APPROVED_PROJECT_CONTEXT_V1'}
        skills = await memory.ApprovedSkillsMiddleware().abefore_agent(state, Runtime(store=store), {})
        assert 'injected' not in str(skills)
        assert len(skills['skills_metadata']) == 3

    asyncio.run(scenario())


@pytest.mark.parametrize('path', ['/memory/PROJECT.md', '/memory/new.md',
                                 '/skills/site-editing/SKILL.md', '/skills/new/SKILL.md',
                                 '/memory/../PROJECT.md', '/memory/../skills/site-editing/SKILL.md',
                                 '/skills/../memory/PROJECT.md', '/memory', '/skills',
                                 '/memory//PROJECT.md'])
def test_all_model_write_edit_and_upload_routes_are_denied(path):
    async def scenario():
        store = InMemoryStore()
        put_memory(store)
        backend = memory.backend_factory(tool_runtime(store))
        assert 'permission_denied' in (await backend.awrite(path, 'bad')).error
        assert 'permission_denied' in (await backend.aedit(path, 'old', 'bad')).error
        uploaded = await backend.aupload_files([(path, b'bad')])
        assert uploaded[0].error == 'permission_denied'
        assert (await backend.awrite(path, 'bad')).files_update is None
        assert (await backend.aedit(path, 'old', 'bad')).files_update is None
        assert (await store.aget(memory.NAMESPACE + ('memory',), '/PROJECT.md')).value['content'] == [
            'APPROVED_PROJECT_CONTEXT_V1']

    asyncio.run(scenario())


def test_ephemeral_workspace_remains_writable_without_approved_store_changes():
    async def scenario():
        store = InMemoryStore()
        backend = memory.backend_factory(tool_runtime(store))
        written = await backend.awrite('/workspace/draft.md', 'EPHEMERAL_DRAFT')
        assert written.error is None
        assert written.files_update['/workspace/draft.md']['content'] == ['EPHEMERAL_DRAFT']
        assert await store.asearch(memory.NAMESPACE) == []

    asyncio.run(scenario())


def test_native_skills_loader_delivers_metadata_without_full_bodies():
    async def scenario():
        store = InMemoryStore()
        loader = memory.ApprovedSkillsMiddleware()
        result = await loader.abefore_agent({}, Runtime(store=store), {})
        assert {skill['name'] for skill in result['skills_metadata']} == {
            'site-editing', 'brain-maintenance', 'evidence-learning'}
        index = loader._format_skills_list(result['skills_metadata'])
        assert '/skills/site-editing/SKILL.md' in index
        assert 'Прочитай действующие AGENTS.md' not in str(result)
        assert 'Generic repair/provisioner отключены' not in str(result)
        body = await memory.backend_factory(tool_runtime(store)).aread('/skills/site-editing/SKILL.md')
        assert 'Прочитай действующие AGENTS.md' in body

    asyncio.run(scenario())


def test_memory_and_skills_lookup_use_no_filesystem_http_shell_or_models():
    async def scenario():
        store = InMemoryStore()
        put_memory(store)
        for path, content in memory.SKILLS.items():
            await store.aput(memory.NAMESPACE + ('skills',), path, memory.approved_entry(content))
        backend = memory.backend_factory(tool_runtime(store))
        targets = ['builtins.open', 'io.open', 'os.open', 'os.scandir',
                   'httpx.AsyncClient.send', 'httpx.Client.send', 'subprocess.Popen']
        with ExitStack() as stack:
            blocked = [stack.enter_context(patch(target, side_effect=AssertionError(
                'Unexpected I/O during native Store lookup'))) for target in targets]
            result = await backend.adownload_files(memory.MEMORY_PATHS)
            assert result[0].content == b'APPROVED_PROJECT_CONTEXT_V1'
            metadata = await memory.ApprovedSkillsMiddleware().abefore_agent({}, Runtime(store=store), {})
            assert len(metadata['skills_metadata']) == 3
            for forbidden in blocked:
                forbidden.assert_not_called()
    asyncio.run(scenario())


def test_real_deep_agent_reads_shared_memory_and_reloads_admin_change_across_threads():
    async def scenario():
        store = InMemoryStore()
        put_memory(store, 'APPROVED_SHARED_FACT_ONE')
        model = Model(responses=[AIMessage(content='First synthetic answer.'),
                                 AIMessage(content='Second synthetic answer.')])
        agent = create_deep_agent(model=model, backend=memory.backend_factory,
            middleware=memory.native_middlewares(), store=store, checkpointer=InMemorySaver())
        await agent.ainvoke({'messages': [{'role': 'user', 'content': 'FIRST_DIALOGUE_ONLY'}]},
                           {'configurable': {'thread_id': 'first'}})
        put_memory(store, 'APPROVED_SHARED_FACT_TWO', revision='operator-reviewed-v2')
        await agent.ainvoke({'messages': [{'role': 'user', 'content': 'SECOND_DIALOGUE_ONLY'}]},
                           {'configurable': {'thread_id': 'second'}})
        assert len(model._seen) == 2
        assert 'APPROVED_SHARED_FACT_ONE' in str(model._seen[0])
        assert 'APPROVED_SHARED_FACT_TWO' in str(model._seen[1])
        assert 'APPROVED_SHARED_FACT_ONE' not in str(model._seen[1])
        assert 'FIRST_DIALOGUE_ONLY' not in str(model._seen[1])
        assert 'Прочитай действующие AGENTS.md' not in str(model._seen[0])
        assert len(await store.asearch(memory.NAMESPACE + ('memory',))) == 1

    asyncio.run(scenario())


def test_real_native_write_tool_cannot_edit_approved_memory_but_can_write_scratch():
    async def scenario():
        store = InMemoryStore()
        put_memory(store)
        model = Model(responses=[
            tool_call('deny-memory', 'write_file', file_path='/memory/hijack.md', content='BAD'),
            tool_call('write-scratch', 'write_file', file_path='/workspace/result.md', content='SAFE_DRAFT'),
            AIMessage(content='Synthetic result after checked observations.'),
        ])
        agent = create_deep_agent(model=model, backend=memory.backend_factory,
            middleware=memory.native_middlewares(), store=store, checkpointer=InMemorySaver())
        result = await agent.ainvoke({'messages': [{'role': 'user', 'content': 'Synthetic writes'}]},
                                    {'configurable': {'thread_id': 'write-boundary'}})
        observations = [item for item in result['messages'] if isinstance(item, ToolMessage)]
        assert 'permission_denied' in observations[0].content
        assert result['files']['/workspace/result.md']['content'] == ['SAFE_DRAFT']
        assert not any(path == '/memory' or path.startswith('/memory/')
                       or path == '/skills' or path.startswith('/skills/') for path in result['files'])
        assert await store.aget(memory.NAMESPACE + ('memory',), '/hijack.md') is None

    asyncio.run(scenario())
