"""Real native ToolNode/checkpoint tests; scripted guarded steps, no providers."""
import asyncio
from copy import deepcopy
import json
import socket

import pytest
from langchain_core.messages import AIMessage
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.store.memory import InMemoryStore
from langgraph.types import Command

from deep_agent import msty, msty_compaction, msty_execution, msty_native

TOOLS = [{'type': 'function', 'function': {'name': 'external_read',
    'description': 'Read one synthetic external item.',
    'parameters': {'type': 'object', 'properties': {'name': {'type': 'string'}},
                   'required': ['name'], 'additionalProperties': False}}}]


@pytest.fixture(autouse=True)
def no_network(monkeypatch):
    for name in ('LANGSMITH_TRACING', 'LANGCHAIN_TRACING', 'LANGCHAIN_TRACING_V2'):
        monkeypatch.setenv(name, 'false')

    def denied(*args, **kwargs):
        raise AssertionError('Native offline test attempted network access')

    monkeypatch.setattr(socket.socket, 'connect', denied)
    monkeypatch.setattr(socket, 'create_connection', denied)


def initial():
    return {'messages': [{'role': 'user', 'content': 'Read the skill then the external fixture.'}],
            'tools': deepcopy(TOOLS), 'max_tokens': 128, 'tool_choice': 'auto',
            'result': {}, 'context_budget': None, 'context_budget_check': None,
            'execution_protocol': msty_execution.PROTOCOL, 'execution': {}}


def answer(content='Done.', calls=None):
    return AIMessage(content=content, tool_calls=calls or [],
        usage_metadata={'input_tokens': 100, 'output_tokens': 10, 'total_tokens': 110})


def call(name, args, identifier='tool-1'):
    return {'id': identifier, 'name': name, 'args': args, 'type': 'tool_call'}


def scripted(monkeypatch, sequence):
    seen = []

    async def step(state, *, native_system_prompt=None, native_result_filter=None):
        assert sequence, 'More than the admitted model steps were called'
        seen.append({'state': deepcopy(state), 'system': native_system_prompt})
        result = sequence.pop(0)
        if native_result_filter:
            result = native_result_filter(result)
        return msty.publish_result(result, None)

    monkeypatch.setattr(msty, '_respond_step', step)
    return seen


async def invoke(graph, value, config):
    emitted = []
    async for mode, chunk in graph.astream(value, config, stream_mode=['custom', 'values'], durability='sync'):
        if mode == 'custom' and chunk.get('type') == 'validated_result':
            emitted.append(chunk['message'])
    state = await graph.aget_state(config)
    return state, emitted


def external_resume(state, reverse=False):
    incoming = {key: deepcopy(state[key]) for key in
        ('tools', 'max_tokens', 'tool_choice', 'context_budget', 'execution_protocol')}
    incoming.update(messages=deepcopy(state['native_protocol_messages']), result={}, context_budget_check=None)
    calls, results, mapping = [], [], []
    pending = state['execution']['pending']['calls']
    for index, item in enumerate(reversed(pending) if reverse else pending):
        client = 'b1_fixture_' + str(index)
        mapping.append({'client_id': client, 'model_id': item['id']})
        calls.append({'id': client, 'type': 'function', 'function': {'name': item['name'],
            'arguments': json.dumps(item['args'])}})
        results.append({'role': 'tool', 'tool_call_id': client, 'content': 'observed-' + item['id']})
    incoming['messages'] += [{'role': 'assistant', 'content': state['result']['content'], 'tool_calls': calls}, *results]
    return {'version': 1, 'task_id': state['execution']['task_id'],
        'batch_id': state['execution']['pending']['batch_id'], 'tool_id_map': mapping, 'input': incoming}


def test_native_skill_then_external_then_final_one_paid_step_per_resume(monkeypatch):
    seen = scripted(monkeypatch, [answer('', [call('native_read_file', {
        'file_path': '/skills/brain-maintenance/SKILL.md'})]),
        answer('', [call('external_read', {'name': 'fixture'}, 'external-1')]), answer()])

    async def run():
        saver, store = InMemorySaver(), InMemoryStore()
        graph = msty_native.build_graph(checkpointer=saver, store=store)
        config = {'configurable': {'thread_id': 'native-external'}}
        first, events = await invoke(graph, initial(), config)
        assert len(seen) == len(events) == 1
        assert first.values['execution']['status'] == 'waiting_native'
        assert first.values['execution']['actions_issued'] == 0
        assert first.values['execution']['native_actions'] == 1
        assert events[0] == first.values['result']
        assert first.values['messages'][-1].type == 'ai'
        assert not any(message.type == 'tool' for message in first.values['messages'])
        assert 'brain-maintenance' in seen[0]['system']
        assert 'native_read_file' in msty.tool_names(seen[0]['state']['tools'])
        assert seen[0]['state']['text_stream_protocol'] is None
        pending = first.tasks[0].interrupts[0]
        assert pending.value['type'] == 'msty_native_continue'
        # Recreate the native graph against the same checkpoint before resuming.
        graph = msty_native.build_graph(checkpointer=saver, store=store)
        second, events = await invoke(graph, Command(resume={pending.id: {
            **pending.value, 'type': 'msty_native_resume'}}), config)
        assert len(seen) == 2 and len(events) == 1
        assert second.values['execution']['status'] == 'waiting_tools'
        assert second.values['execution']['native_actions'] == 1
        assert second.values['execution']['actions_issued'] == 1
        observed_skill = [m for m in seen[1]['state']['messages'] if m['role'] == 'tool'][-1]
        assert 'Обслуживание действующего Brain' in observed_skill['content']
        external = second.tasks[0].interrupts[0]
        assert external.value['type'] == 'msty_local_tools'
        final, events = await invoke(graph, Command(resume={external.id: external_resume(second.values)}), config)
        assert len(seen) == 3 and len(events) == 1
        assert final.values['result']['content'] == 'Done.'
        assert final.values['execution']['status'] == 'answered'
        assert final.values['execution']['native_actions'] == 1
        assert not final.next
        observed = [m for m in seen[2]['state']['messages'] if m['role'] == 'tool'][-1]
        assert observed['tool_call_id'] == 'external-1'
        assert observed['content'] == 'observed-external-1'
    asyncio.run(run())


def test_identical_external_calls_keep_exact_ids_when_mapping_reordered(monkeypatch):
    seen = scripted(monkeypatch, [answer('', [call('external_read', {'name': 'same'}, 'a'),
        call('external_read', {'name': 'same'}, 'b')]), answer()])

    async def run():
        graph = msty_native.build_graph(checkpointer=InMemorySaver(), store=InMemoryStore())
        config = {'configurable': {'thread_id': 'mapped'}}
        first, _ = await invoke(graph, initial(), config)
        pending = first.tasks[0].interrupts[0]
        await invoke(graph, Command(resume={pending.id: external_resume(first.values, reverse=True)}), config)
        messages = seen[1]['state']['messages']
        assert {m['tool_call_id']: m['content'] for m in messages if m['role'] == 'tool'} == {
            'a': 'observed-a', 'b': 'observed-b'}
    asyncio.run(run())


@pytest.mark.parametrize('mode', ['mixed', 'limit', 'duplicate_todos'])
def test_unsafe_batches_block_before_publication_or_execution(monkeypatch, mode):
    data = initial()
    if mode == 'mixed':
        calls = [call('native_read_file', {'file_path': '/memory/PROJECT.md'}, 'native'),
                 call('external_read', {'name': 'fixture'}, 'external')]
    elif mode == 'limit':
        data['execution'] = {'actions_issued': 1, 'native_actions': 23}
        calls = [call('external_read', {'name': 'fixture'})]
    else:
        calls = [call('native_write_todos', {'todos': []}, 'a'), call('native_write_todos', {'todos': []}, 'b')]
    seen = scripted(monkeypatch, [answer('', calls)])

    async def run():
        graph = msty_native.build_graph(checkpointer=InMemorySaver(), store=InMemoryStore())
        state, events = await invoke(graph, data, {'configurable': {'thread_id': mode}})
        assert len(seen) == len(events) == 1
        assert state.values['execution']['status'] == 'blocked'
        assert not events[0]['tool_calls']
        assert events[0]['usage_metadata']['total_tokens'] == 110
        assert not state.next
        assert not any(m.type == 'tool' for m in state.values['messages'])
    asyncio.run(run())


def test_native_ticket_mismatch_cannot_trigger_next_model(monkeypatch):
    seen = scripted(monkeypatch, [answer('', [call('native_write_file', {
        'file_path': '/must-not-exist.txt', 'content': 'not admitted'})])])

    async def run():
        graph = msty_native.build_graph(checkpointer=InMemorySaver(), store=InMemoryStore())
        config = {'configurable': {'thread_id': 'bad-ticket'}}
        state, _ = await invoke(graph, initial(), config)
        pending = state.tasks[0].interrupts[0]
        bad = {**pending.value, 'type': 'msty_native_resume', 'native_actions': 0}
        with pytest.raises(msty_execution.ExecutionProtocolError):
            await invoke(graph, Command(resume={pending.id: bad}), config)
        assert len(seen) == 1
        snapshot = await graph.aget_state(config)
        assert not snapshot.values.get('files')
        assert not any(message.type == 'tool' for message in snapshot.values['messages'])
    asyncio.run(run())


def test_external_schema_cannot_shadow_native_tools(monkeypatch):
    seen = scripted(monkeypatch, [])
    data = initial()
    data['tools'][0]['function']['name'] = 'native_read_file'

    async def run():
        graph = msty_native.build_graph(checkpointer=InMemorySaver(), store=InMemoryStore())
        with pytest.raises(msty_execution.ExecutionProtocolError):
            await graph.ainvoke(data, {'configurable': {'thread_id': 'shadow'}})
        assert seen == []
    asyncio.run(run())


def test_analyst_has_no_native_tools_or_project_prompt(monkeypatch):
    seen = scripted(monkeypatch, [answer('Bounded analysis.')])
    data = initial()
    data.update(brain_task_role='analyst', tools=[])

    async def run():
        graph = msty_native.build_graph(checkpointer=InMemorySaver(), store=InMemoryStore())
        result, events = await invoke(graph, data, {'configurable': {'thread_id': 'analyst'}})
        assert result.values['execution']['status'] == 'answered'
        assert len(events) == len(seen) == 1
        assert seen[0]['state']['tools'] == []
        assert seen[0]['system'] == msty.ANALYST_POLICY
        assert msty.selected_profile(seen[0]['state']) == 'deepseek'
        assert not result.values.get('memory_contents')
        assert not result.values.get('skills_metadata')
    asyncio.run(run())


def test_validated_external_resume_preserves_none_tool_choice_and_lower_cap(monkeypatch):
    seen = scripted(monkeypatch, [answer('', [call('external_read', {'name': 'fixture'})]), answer()])

    async def run():
        graph = msty_native.build_graph(checkpointer=InMemorySaver(), store=InMemoryStore())
        config = {'configurable': {'thread_id': 'final-only'}}
        first, _ = await invoke(graph, initial(), config)
        pending = first.tasks[0].interrupts[0]
        resume = external_resume(first.values)
        resume['input'].update(tool_choice='none', max_tokens=64)
        await invoke(graph, Command(resume={pending.id: resume}), config)
        assert seen[1]['state']['tool_choice'] == 'none'
        assert seen[1]['state']['max_tokens'] == 64
    asyncio.run(run())


def test_existing_compaction_keeps_native_count_and_uses_own_resume_ticket(monkeypatch):
    seen = []

    async def step(state, **kwargs):
        seen.append(deepcopy(state))
        if len(seen) == 1:
            return msty.publish_result(answer('', [call('native_ls', {'path': '/'})]), None)
        if len(seen) == 2:
            summary = answer('')
            summary.response_metadata['msty_stage'] = 'compaction'
            return {**msty.publish_result(summary, None),
                'context_memory': {'version': 1, 'segments': []},
                'compaction_stage': {'version': 1, 'status': 'ready', 'stage_id': 'summary-step',
                    'source_sha256': 'a' * 64, 'summary_sha256': 'b' * 64}}
        assert len(seen) == 3 and state['compaction_skip_once'] is True
        return msty.publish_result(answer(), None)

    monkeypatch.setattr(msty, '_respond_step', step)

    async def run():
        graph = msty_native.build_graph(checkpointer=InMemorySaver(), store=InMemoryStore())
        config = {'configurable': {'thread_id': 'compaction-ticket'}}
        data = initial()
        data['compaction_protocol'] = msty_compaction.PROTOCOL
        first, _ = await invoke(graph, data, config)
        ticket = first.tasks[0].interrupts[0]
        second, events = await invoke(graph, Command(resume={ticket.id: {
            **ticket.value, 'type': 'msty_native_resume'}}), config)
        assert len(events) == 1 and len(seen) == 2
        assert second.values['execution']['native_actions'] == 1
        assert second.values['execution']['harness_version'] == 'msty-native-v1'
        compact = second.tasks[0].interrupts[0]
        resume = {key: value for key, value in compact.value.items() if key != 'result_sha256'}
        resume['type'] = 'msty_compaction_resume'
        final, events = await invoke(graph, Command(resume={compact.id: resume}), config)
        assert len(events) == 1 and len(seen) == 3
        assert final.values['execution']['native_actions'] == 1
        assert final.values['execution']['harness_version'] == 'msty-native-v1'
        assert final.values['execution']['status'] == 'answered'
        assert not final.next
    asyncio.run(run())


def test_real_guarded_step_consumes_native_schemas_and_prompt_without_network(monkeypatch):
    seen = []

    class Provider:
        def bind_tools(self, tools, **kwargs):
            assert {'native_read_file', 'native_write_todos', 'external_read'} <= msty.tool_names(tools)
            return self

        async def ainvoke(self, messages):
            seen.append(deepcopy(messages))
            if len(seen) == 1:
                return answer('', [call('native_read_file', {'file_path': '/skills/evidence-learning/SKILL.md'})])
            return answer()

    monkeypatch.setenv('MSTY_MODEL_PROFILE', 'luna')
    monkeypatch.setattr(msty.msty_models, 'make_model', lambda *args: Provider())
    monkeypatch.setattr(msty.msty_models, 'stamp_usage', lambda profile, result: result)

    async def run():
        graph = msty_native.build_graph(checkpointer=InMemorySaver(), store=InMemoryStore())
        config = {'configurable': {'thread_id': 'real-step'}}
        result, events = await invoke(graph, initial(), config)
        assert len(seen) == len(events) == 1
        assert result.values['execution']['status'] == 'waiting_native'
        assert result.values['execution']['native_actions'] == 1
        assert 'MSTY_NATIVE_HARNESS_V1' in seen[0][0].text
        assert 'evidence-learning' in seen[0][0].text
        assert not any(message.type == 'tool' for message in result.values['messages'])
        ticket = result.tasks[0].interrupts[0]
        final, events = await invoke(graph, Command(resume={ticket.id: {
            **ticket.value, 'type': 'msty_native_resume'}}), config)
        assert len(events) == 1 and len(seen) == 2
        assert final.values['execution']['status'] == 'answered'
        assert 'Проверяемые результаты и уроки' in seen[1][-1].content
    asyncio.run(run())


def test_real_guarded_native_and_external_read_file_are_distinct(monkeypatch):
    seen, schemas = [], []
    sequence = [answer('', [call('native_read_file', {
        'file_path': '/skills/brain-maintenance/SKILL.md'}, 'skill')]),
        answer('', [call('read_file', {'name': '/external-fixture.txt'}, 'external-read')]), answer()]

    class Provider:
        def bind_tools(self, tools, **kwargs):
            schemas.append(deepcopy(tools))
            return self

        async def ainvoke(self, messages):
            seen.append(deepcopy(messages))
            return sequence.pop(0)

    monkeypatch.setenv('MSTY_MODEL_PROFILE', 'luna')
    monkeypatch.setattr(msty.msty_models, 'make_model', lambda *args: Provider())
    monkeypatch.setattr(msty.msty_models, 'stamp_usage', lambda profile, result: result)
    data = initial()
    data['messages'][0]['content'] = 'Use external read_file for /external-fixture.txt.'
    data['tools'] = []
    for name in ('read_file', 'write_file', 'edit_file'):
        external = deepcopy(TOOLS[0])
        external['function'].update(name=name, description='External read_file/write_file/edit_file on Mac.')
        data['tools'].append(external)

    async def run():
        graph = msty_native.build_graph(checkpointer=InMemorySaver(), store=InMemoryStore())
        config = {'configurable': {'thread_id': 'real-file-name-collision'}}
        first, events = await invoke(graph, data, config)
        assert len(events) == 1 and len(seen) == 1
        assert first.values['execution']['status'] == 'waiting_native'
        assert not any(message.type == 'tool' for message in first.values['messages'])
        assert msty_native.NATIVE_TOOLS | {'read_file', 'write_file', 'edit_file'} == msty.tool_names(schemas[0])
        assert [tool for tool in schemas[0] if tool['function']['name'] in {
            'read_file', 'write_file', 'edit_file'}] == data['tools']
        assert seen[0][-1].content == data['messages'][0]['content']
        assert msty.POLICY in seen[0][0].text
        assert '`native_read_file`' in seen[0][0].text
        ticket = first.tasks[0].interrupts[0]
        second, events = await invoke(graph, Command(resume={ticket.id: {
            **ticket.value, 'type': 'msty_native_resume'}}), config)
        assert len(events) == 1 and len(seen) == 2
        assert 'Обслуживание действующего Brain' in seen[1][-1].content
        assert second.values['execution']['status'] == 'waiting_tools'
        assert second.values['result']['tool_calls'][0]['name'] == 'read_file'
        assert second.values['execution']['native_actions'] == 1
        assert second.values['execution']['actions_issued'] == 1
        external = second.tasks[0].interrupts[0]
        assert external.value['type'] == 'msty_local_tools'
        final, events = await invoke(graph, Command(resume={external.id: external_resume(second.values)}), config)
        assert len(events) == 1 and len(seen) == 3
        assert seen[2][-1].content == 'observed-external-read'
        assert seen[2][-1].name == 'read_file'
        assert final.values['execution']['status'] == 'answered'
        assert not final.next
    asyncio.run(run())


def test_namespaced_native_callbacks_keep_scratch_and_todo_runtime_injection(monkeypatch):
    todos = [{'content': 'Verify native scratch', 'status': 'completed'}]
    seen = scripted(monkeypatch, [
        answer('', [call('native_write_file', {'file_path': '/scratch.txt', 'content': 'before'}, 'write')]),
        answer('', [call('native_edit_file', {'file_path': '/scratch.txt', 'old_string': 'before',
                                            'new_string': 'after'}, 'edit')]),
        answer('', [call('native_read_file', {'file_path': '/scratch.txt'}, 'read')]),
        answer('', [call('native_write_todos', {'todos': todos}, 'todo')]), answer()])

    async def run():
        graph = msty_native.build_graph(checkpointer=InMemorySaver(), store=InMemoryStore())
        config = {'configurable': {'thread_id': 'namespaced-scratch'}, 'recursion_limit': 64}
        state, events = await invoke(graph, initial(), config)
        assert not state.values.get('files')
        for index in range(4):
            assert len(events) == 1 and len(seen) == index + 1
            assert state.values['execution']['status'] == 'waiting_native'
            assert state.values['execution']['native_actions'] == index + 1
            assert state.values['execution']['actions_issued'] == 0
            assert sum(message.type == 'tool' for message in state.values['messages']) == index
            ticket = state.tasks[0].interrupts[0]
            state, events = await invoke(graph, Command(resume={ticket.id: {
                **ticket.value, 'type': 'msty_native_resume'}}), config)
        assert len(events) == 1 and len(seen) == 5
        assert state.values['files']['/scratch.txt']['content'] == ['after']
        assert state.values['todos'] == todos
        assert state.values['execution']['status'] == 'answered'
        assert 'after' in [message for message in seen[3]['state']['messages'] if message['role'] == 'tool'][-1]['content']
        assert not state.next
    asyncio.run(run())


def test_native_template_namespacing_never_changes_approved_memory_body():
    memory = msty_native.NamespacedMemoryMiddleware()
    source = 'Use external read_file/write_file/edit_file on Mac, not native scratch.'
    prompt = memory._format_agent_memory({'/memory/PROJECT.md': source})
    assert source in prompt
    assert 'native_edit_file' in prompt
