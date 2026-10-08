"""Actual native graph checkpoint/resume catalog delivery, without provider calls."""
import asyncio
from copy import deepcopy
import socket

import pytest
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.store.memory import InMemoryStore
from langgraph.types import Command

from deep_agent import msty_execution, msty_native
from tests.unit_tests.test_msty_native import (
    TOOLS, answer, call, external_resume, initial, invoke, scripted,
)


@pytest.fixture(autouse=True)
def offline(monkeypatch):
    for name in ('LANGSMITH_TRACING', 'LANGCHAIN_TRACING', 'LANGCHAIN_TRACING_V2'):
        monkeypatch.setenv(name, 'false')

    def denied(*args, **kwargs):
        raise AssertionError('Catalog regression attempted network access')

    monkeypatch.setattr(socket.socket, 'connect', denied)
    monkeypatch.setattr(socket, 'create_connection', denied)


def refresh(state, tools):
    response = external_resume(state)
    response['catalog_refresh'] = {
        'version': 1, 'task_id': state['execution']['task_id'],
        'batch_id': state['execution']['pending']['batch_id'],
        'previous_sha256': msty_execution.canonical_digest(state['tools']),
        'next_sha256': msty_execution.canonical_digest(tools),
        'tools': deepcopy(tools),
    }
    return response


@pytest.mark.parametrize('kind', ['introduced_tool', 'updated_schema'])
def test_native_resume_retains_validated_catalog_in_checkpoint_and_model(monkeypatch, kind):
    tools = deepcopy(TOOLS)
    if kind == 'introduced_tool':
        introduced = deepcopy(TOOLS[0])
        introduced['function']['name'] = 'external_read_new'
        tools.append(introduced)
    else:
        tools[0]['function']['description'] = 'Updated read-only schema description'
    seen = scripted(monkeypatch, [
        answer('', [call('external_read', {'name': 'fixture'}, 'external-1')]),
        answer(),
    ])

    async def run():
        saver, store = InMemorySaver(), InMemoryStore()
        graph = msty_native.build_graph(checkpointer=saver, store=store)
        config = {'configurable': {'thread_id': 'catalog-' + kind}}
        first, _ = await invoke(graph, initial(), config)
        before = deepcopy(first.values)
        pending = first.tasks[0].interrupts[0]
        response = refresh(before, tools)
        # The original pending schemas stay immutable; only the separate envelope changes.
        assert response['input']['tools'] == before['tools']
        graph = msty_native.build_graph(checkpointer=saver, store=store)
        second, _ = await invoke(graph, Command(resume={pending.id: response}), config)
        assert second.values['tools'] == tools
        effective = {tool['function']['name']: tool for tool in seen[1]['state']['tools']}
        for tool in tools:
            assert effective[tool['function']['name']] == tool
        assert len(seen) == 2
        assert second.values['execution']['task_id'] == before['execution']['task_id']
        assert second.values['execution']['actions_issued'] == 1
        assert second.values['execution']['native_actions'] == 0
        assert second.values['max_tokens'] == before['max_tokens']
        assert second.values['execution']['status'] == 'answered'
        observations = [message for message in seen[1]['state']['messages'] if message['role'] == 'tool']
        assert observations[-1]['tool_call_id'] == 'external-1'
        assert observations[-1]['content'] == 'observed-external-1'
    asyncio.run(run())


def test_native_resume_rejects_forged_catalog_before_another_model_step(monkeypatch):
    seen = scripted(monkeypatch, [
        answer('', [call('external_read', {'name': 'fixture'}, 'external-1')]),
        answer(),
    ])

    async def run():
        graph = msty_native.build_graph(checkpointer=InMemorySaver(), store=InMemoryStore())
        config = {'configurable': {'thread_id': 'forged-catalog'}}
        first, _ = await invoke(graph, initial(), config)
        response = refresh(first.values, TOOLS)
        response['catalog_refresh']['next_sha256'] = '0' * 64
        with pytest.raises(msty_execution.ExecutionProtocolError):
            await invoke(graph, Command(resume={first.tasks[0].interrupts[0].id: response}), config)
        assert len(seen) == 1
        unchanged = await graph.aget_state(config)
        assert unchanged.values['tools'] == first.values['tools']
        assert unchanged.values['execution']['actions_issued'] == 1
    asyncio.run(run())
