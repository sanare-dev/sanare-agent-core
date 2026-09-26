"""Real native ToolNode/checkpoint tests; scripted guarded steps, no providers."""
import asyncio
from copy import deepcopy
import hashlib
import json
import socket
from types import SimpleNamespace

import pytest
import tiktoken
from langchain_core.messages import AIMessage
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.store.memory import InMemoryStore
from langgraph.types import Command

from deep_agent import msty, msty_compaction, msty_execution, msty_models, msty_native, msty_prompts

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


def test_mixed_native_external_batch_runs_sequentially_without_new_user_command(monkeypatch):
    seen = scripted(monkeypatch, [answer('', [
        call('native_read_file', {'file_path': '/memory/PROJECT.md'}, 'native'),
        call('external_read', {'name': 'fixture'}, 'external')]),
        answer('', [call('external_read', {'name': 'fixture'}, 'external-reissued')]),
        answer()])

    async def run():
        saver, store = InMemorySaver(), InMemoryStore()
        graph = msty_native.build_graph(checkpointer=saver, store=store)
        config = {'configurable': {'thread_id': 'mixed-sequential'}}
        first, events = await invoke(graph, initial(), config)
        assert len(seen) == len(events) == 1
        assert first.values['execution']['status'] == 'waiting_native'
        assert first.values['execution']['native_actions'] == 1
        assert first.values['execution']['actions_issued'] == 0
        assert [item['name'] for item in events[0]['tool_calls']] == ['native_read_file']
        assert events[0]['response_metadata']['msty_deferred_external_calls'] == 1

        native = first.tasks[0].interrupts[0]
        second, events = await invoke(graph, Command(resume={native.id: {
            **native.value, 'type': 'msty_native_resume'}}), config)
        assert len(seen) == 2 and len(events) == 1
        assert second.values['execution']['status'] == 'waiting_tools'
        assert second.values['execution']['native_actions'] == 1
        assert second.values['execution']['actions_issued'] == 1
        assert [item['name'] for item in events[0]['tool_calls']] == ['external_read']

        external = second.tasks[0].interrupts[0]
        final, events = await invoke(graph, Command(resume={
            external.id: external_resume(second.values)}), config)
        assert len(seen) == 3 and len(events) == 1
        assert final.values['execution']['status'] == 'answered'
        assert final.values['result']['content'] == 'Done.'
        assert not final.next

    asyncio.run(run())


_HIDDEN_TOOL = {'type': 'function', 'function': {'name': 'external_hidden',
    'description': 'Check a private status endpoint outside default routing.',
    'parameters': {'type': 'object', 'properties': {}, 'additionalProperties': False}}}


def initial_with_hidden_tool():
    data = initial()
    data['tools'] = deepcopy(TOOLS) + [deepcopy(_HIDDEN_TOOL)]
    return data


def test_requested_tool_becomes_visible_same_run_no_new_owner_message(monkeypatch):
    """brain-agency-audit-2026-09-26 #2: a tool outside the routed set is only a
    text catalog entry until native_request_tools is called; it must become a
    real schema on the NEXT model step of the SAME run (gateway admission
    resume only — no fresh message from the owner), not a following turn."""
    monkeypatch.setenv('MSTY_TOOL_DISPATCHER', 'on')
    seen = scripted(monkeypatch, [
        answer('', [call('native_request_tools', {'names': ['external_hidden']}, 'req-1')]),
        answer('', [call('external_hidden', {}, 'hidden-1')]),
        answer()])

    async def run():
        saver, store = InMemorySaver(), InMemoryStore()
        graph = msty_native.build_graph(checkpointer=saver, store=store)
        config = {'configurable': {'thread_id': 'dispatcher-same-run'}}
        first, events = await invoke(graph, initial_with_hidden_tool(), config)
        # Step 1: not requested yet, so not on the wire; only the text catalog
        # (checked at the routing layer in test_msty_dispatcher.py) names it.
        assert 'external_hidden' not in msty.tool_names(seen[0]['state']['tools'])
        assert first.values['execution']['status'] == 'waiting_native'
        pending = first.tasks[0].interrupts[0]
        assert pending.value['type'] == 'msty_native_continue'

        # The only thing that advances the graph here is the bridge's own
        # gateway-admission resume (msty_native_continue -> msty_native_resume):
        # no Command carries a new HumanMessage, and no owner input is read.
        graph = msty_native.build_graph(checkpointer=saver, store=store)
        second, events = await invoke(graph, Command(resume={pending.id: {
            **pending.value, 'type': 'msty_native_resume'}}), config)
        assert len(seen) == 2
        # Step 2, same thread/run, same owner turn: the requested schema is now
        # a real tool the model can call directly.
        assert 'external_hidden' in msty.tool_names(seen[1]['state']['tools'])
        assert second.values['execution']['status'] == 'waiting_tools'
    asyncio.run(run())


@pytest.mark.parametrize('mode', ['limit', 'duplicate_todos'])
def test_unsafe_batches_block_before_publication_or_execution(monkeypatch, mode):
    data = initial()
    if mode == 'limit':
        data['execution'] = {'actions_issued': 1, 'native_actions': 199}
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
        'file_path': '/scratch/must-not-exist.txt', 'content': 'not admitted'})])])

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


def test_native_middleware_projects_tools_and_prompt_per_user_turn(monkeypatch):
    seen = scripted(monkeypatch, [answer('Current site status checked.')])
    data = initial()
    data['messages'] = [{'role': 'user', 'content': 'Проверь статус app.sanaredev.com.'}]
    data['tools'] = []
    for name in ('msty_site_status', 'execute_sql', 'discover_tools'):
        tool = deepcopy(TOOLS[0])
        tool['function'].update(name=name, description='Synthetic ' + name)
        data['tools'].append(tool)

    async def run():
        graph = msty_native.build_graph(checkpointer=InMemorySaver(), store=InMemoryStore())
        final, _ = await invoke(graph, data, {'configurable': {'thread_id': 'dynamic-route'}})
        visible = msty.tool_names(seen[0]['state']['tools'])
        assert 'msty_site_status' in visible
        assert 'execute_sql' not in visible and 'discover_tools' not in visible
        assert 'MSTY_DYNAMIC_ROUTE_V1' in seen[0]['system']
        assert 'domains=sites' in seen[0]['system']
        route = final.values['execution']['tool_route']
        assert route['intent'] == 'read' and route['domains'] == ['sites']
        assert route['selected_count'] == 1 and route['available_count'] == 3
        assert final.values['native_tool_route']['selected_names'] == ['msty_site_status']

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


@pytest.mark.parametrize('prior_native', [0, 24, 199])
def test_existing_compaction_keeps_native_count_and_uses_own_resume_ticket(monkeypatch, prior_native):
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
                    'source_sha256': 'a' * 64, 'summary_sha256': 'b' * 64},
                'compaction_round': 1}
        assert len(seen) == 3 and state['compaction_round'] == 1
        return msty.publish_result(answer(), None)

    monkeypatch.setattr(msty, '_respond_step', step)

    async def run():
        graph = msty_native.build_graph(checkpointer=InMemorySaver(), store=InMemoryStore())
        config = {'configurable': {'thread_id': 'compaction-ticket'}}
        data = initial()
        data['execution'] = {'native_actions': prior_native, 'consultations': 1}
        data['compaction_protocol'] = msty_compaction.PROTOCOL
        first, _ = await invoke(graph, data, config)
        ticket = first.tasks[0].interrupts[0]
        second, events = await invoke(graph, Command(resume={ticket.id: {
            **ticket.value, 'type': 'msty_native_resume'}}), config)
        assert len(events) == 1 and len(seen) == 2
        assert second.values['execution']['native_actions'] == prior_native + 1
        assert second.values['execution']['consultations'] == 1
        assert second.values['execution']['harness_version'] == 'msty-native-v1'
        compact = second.tasks[0].interrupts[0]
        resume = {key: value for key, value in compact.value.items() if key != 'result_sha256'}
        resume['type'] = 'msty_compaction_resume'
        final, events = await invoke(graph, Command(resume={compact.id: resume}), config)
        assert len(events) == 1 and len(seen) == 3
        assert final.values['execution']['native_actions'] == prior_native + 1
        assert final.values['execution']['consultations'] == 1
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
        assert 'MSTY_TOOLS_AVAILABLE' in seen[0][0].text
        assert 'external_read' in seen[0][0].text
        assert 'native_read_file' in seen[0][0].text
        assert 'MSTY_TOOLS_UNAVAILABLE' not in seen[0][0].text
        assert not any(message.type == 'tool' for message in result.values['messages'])
        ticket = result.tasks[0].interrupts[0]
        final, events = await invoke(graph, Command(resume={ticket.id: {
            **ticket.value, 'type': 'msty_native_resume'}}), config)
        assert len(events) == 1 and len(seen) == 2
        assert final.values['execution']['status'] == 'answered'
        assert 'Проверяемые результаты и уроки' in seen[1][-1].content
    asyncio.run(run())


def test_minimal_native_first_payload_stays_below_context_budget(monkeypatch):
    """Protect the always-loaded prefix; external project tools are counted elsewhere."""
    seen = {}

    class Provider:
        def bind_tools(self, tools, **kwargs):
            seen['tools'] = deepcopy(tools)
            return self

        async def ainvoke(self, messages):
            seen['messages'] = deepcopy(messages)
            return answer('OK')

    monkeypatch.setenv('MSTY_MODEL_PROFILE', 'luna')
    monkeypatch.setattr(msty.msty_models, 'make_model', lambda *args: Provider())
    monkeypatch.setattr(msty.msty_models, 'stamp_usage', lambda profile, result: result)

    async def run():
        graph = msty_native.build_graph(checkpointer=InMemorySaver(), store=InMemoryStore())
        data = initial()
        data['messages'] = [{'role': 'user', 'content': 'OK'}]
        data['tools'] = []
        final, events = await invoke(graph, data, {'configurable': {'thread_id': 'context-budget'}})
        assert len(events) == 1 and final.values['execution']['status'] == 'answered'

    asyncio.run(run())
    wire = {'messages': [
        {'role': 'system', 'content': seen['messages'][0].text},
        {'role': 'user', 'content': 'OK'},
    ], 'tools': seen['tools']}
    encoded = json.dumps(wire, ensure_ascii=False, sort_keys=True)
    tokens = len(tiktoken.get_encoding('o200k_base').encode(encoded))
    assert tokens <= 5500


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
    data['messages'][0]['content'] = (
        'Use external read_file for /external-fixture.txt; keep write_file and edit_file distinct too.')
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
        # На свежем рабочем шаге (classified, intent != direct) модель видит
        # и серверный инструмент делегирования под-агентам (sub-agents).
        assert msty_native.NATIVE_TOOLS | {'read_file', 'write_file', 'edit_file',
            msty_native.msty_subagents.DELEGATE_TOOL} == msty.tool_names(schemas[0])
        assert [tool for tool in schemas[0] if tool['function']['name'] in {
            'read_file', 'write_file', 'edit_file'}] == data['tools']
        assert seen[0][-1].content == data['messages'][0]['content']
        # Policy is routed per turn (see test_policy_routing_* below); the
        # behavioural spine and the native harness contract always ship.
        for block in ('Ты — Sanare Brain', *msty_prompts.ALWAYS_BLOCKS,
                      'MSTY_NATIVE_HARNESS_V1'):
            assert block in seen[0][0].text
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
        answer('', [call('native_write_file', {'file_path': '/scratch/fixture.txt', 'content': 'before'}, 'write')]),
        answer('', [call('native_edit_file', {'file_path': '/scratch/fixture.txt', 'old_string': 'before',
                                            'new_string': 'after'}, 'edit')]),
        answer('', [call('native_read_file', {'file_path': '/scratch/fixture.txt'}, 'read')]),
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
        assert state.values['files']['/scratch/fixture.txt']['content'] == ['after']
        for message in [message for message in state.values['messages'] if message.type == 'tool'][:2]:
            assert 'VIRTUAL SCRATCH ONLY; no Mac/local file was changed.' in message.content
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
    assert 'APPROVED PROJECT MEMORY' in prompt
    assert 'native_edit_file' not in prompt


def test_secret_pii_redacts_complete_history_input_without_hiding_business_data(monkeypatch):
    secret = 'sk-proj-' + 'A' * 32
    seen = scripted(monkeypatch, [answer('Done.')])
    data = initial()
    data['messages'] = [
        {'role': 'user', 'content': f'Old credential {secret}; owner@example.com; https://example.com; 127.0.0.1'},
        {'role': 'assistant', 'content': 'Continue.'},
        {'role': 'user', 'content': 'Use the earlier project context.'},
    ]

    async def run():
        graph = msty_native.build_graph(checkpointer=InMemorySaver(), store=InMemoryStore())
        state, _ = await invoke(graph, data, {'configurable': {'thread_id': 'pii-input'}})
        admitted = '\n'.join(str(message.get('content')) for message in seen[0]['state']['messages'])
        assert secret not in admitted
        assert '[REDACTED_API_KEY]' in admitted
        assert 'owner@example.com' in admitted
        assert 'https://example.com' in admitted
        assert '127.0.0.1' in admitted
        assert secret not in str(state.values['messages'])
    asyncio.run(run())


def test_secret_pii_redacts_custom_result_and_structured_arguments_before_publication(monkeypatch):
    secret = 'lsv2_' + 'B' * 36
    seen = scripted(monkeypatch, [answer(f'Never publish {secret}', [
        call('external_read', {'name': f'credential={secret}'}, 'secret-call')])])

    async def run():
        graph = msty_native.build_graph(checkpointer=InMemorySaver(), store=InMemoryStore())
        state, events = await invoke(graph, initial(), {'configurable': {'thread_id': 'pii-output'}})
        assert len(seen) == len(events) == 1
        assert secret not in json.dumps(events[0])
        assert secret not in json.dumps(state.values['result'])
        assert events[0]['content'] == 'Never publish [REDACTED_API_KEY]'
        assert events[0]['tool_calls'][0]['args']['name'] == 'credential=[REDACTED_API_KEY]'
        assert state.values['execution']['status'] == 'waiting_tools'
    asyncio.run(run())


def test_secret_pii_redacts_external_tool_observation_before_next_model(monkeypatch):
    secret = 'ghp_' + 'C' * 36
    seen = scripted(monkeypatch, [
        answer('', [call('external_read', {'name': 'fixture'}, 'external-secret')]), answer('Done.')])

    async def run():
        graph = msty_native.build_graph(checkpointer=InMemorySaver(), store=InMemoryStore())
        config = {'configurable': {'thread_id': 'pii-tool-result'}}
        first, _ = await invoke(graph, initial(), config)
        pending = first.tasks[0].interrupts[0]
        resume = external_resume(first.values)
        resume['input']['messages'][-1]['content'] = f'observed {secret} owner@example.com'
        final, _ = await invoke(graph, Command(resume={pending.id: resume}), config)
        admitted = [message for message in seen[1]['state']['messages'] if message['role'] == 'tool'][-1]['content']
        assert secret not in admitted
        assert '[REDACTED_API_KEY]' in admitted
        assert 'owner@example.com' in admitted
        assert secret not in str(final.values['messages'])
        assert final.values['execution']['status'] == 'answered'
    asyncio.run(run())


@pytest.mark.parametrize('name,args', [
    ('native_write_file', {'file_path': '/Users/vb/Documents/ChatGPT/LLM/work/fixture.txt', 'content': 'x'}),
    ('native_write_file', {'file_path': '/Volumes/NAS/fixture.txt', 'content': 'x'}),
    ('native_write_file', {'file_path': 'work/fixture.txt', 'content': 'x'}),
    ('native_write_file', {'file_path': '/scratch/../Users/vb/fixture.txt', 'content': 'x'}),
    ('native_write_file', {'file_path': '/scratch/../skills/fake.md', 'content': 'x'}),
    ('native_write_file', {'file_path': '/scratchpad/fixture.txt', 'content': 'x'}),
    ('native_write_file', {'file_path': '//scratch/fixture.txt', 'content': 'x'}),
    ('native_write_file', {'file_path': '/scratch/fixture\x00.txt', 'content': 'x'}),
    ('native_write_file', {'file_path': '/memory/PROJECT.md', 'content': 'x'}),
    ('native_write_file', {'file_path': '/skills/fake.md', 'content': 'x'}),
    ('native_write_file', {'file_path': '/large_tool_results/fake', 'content': 'x'}),
    ('native_edit_file', {'file_path': '/Users/vb/local.txt', 'old_string': 'x', 'new_string': 'y'}),
    ('native_read_file', {'file_path': '/Users/vb/local.txt'}),
    ('native_ls', {'path': '/Volumes'}),
    ('native_glob', {'path': '/Users/vb', 'pattern': '*.txt'}),
    ('native_glob', {'path': '/scratch/', 'pattern': '/Users/vb/*.txt'}),
    ('native_glob', {'path': '/scratch/', 'pattern': '../*.txt'}),
    ('native_grep', {'path': 'work/', 'pattern': 'needle'}),
    ('native_grep', {'path': '/scratch/', 'pattern': 'needle', 'glob': '/Users/vb/*.txt'}),
    ('native_grep', {'pattern': 'needle'}),
])
def test_forbidden_virtual_path_returns_error_without_native_handler(name, args):
    async def run():
        async def forbidden_handler(request):
            raise AssertionError('Forbidden path reached native backend')

        request = SimpleNamespace(tool_call=call(name, args), state={'native_tool_names': [name]})
        result = await msty_native.NativeMstyMiddleware().awrap_tool_call(request, forbidden_handler)
        assert result.status == 'error'
        assert result.tool_call_id == 'tool-1'
        assert 'native_virtual_path_required' in result.content
        assert 'external MCP tools without native_' in result.content
        assert not request.state.get('files')
    asyncio.run(run())


def test_forbidden_local_write_corrects_to_external_write_with_verified_callback(monkeypatch, tmp_path):
    """Actual ToolNode + guarded model + protocol callback; local IO only in pytest temp."""
    target = tmp_path / 'external-verified.txt'
    body = 'Verified external fixture, not virtual scratch.\n'
    seen, schemas = [], []
    args = {'file_path': str(target), 'content': body}
    sequence = [answer('', [call('native_write_file', args, 'wrong-virtual')]),
        answer('', [call('write_file', args, 'real-external')]), answer('External callback verified.')]

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
    data['tools'] = [{'type': 'function', 'function': {'name': 'write_file',
        'description': 'External MCP write_file writes a real local file.',
        'parameters': {'type': 'object', 'properties': {
            'file_path': {'type': 'string'}, 'content': {'type': 'string'}},
            'required': ['file_path', 'content'], 'additionalProperties': False}}}]

    async def run():
        graph = msty_native.build_graph(checkpointer=InMemorySaver(), store=InMemoryStore())
        config = {'configurable': {'thread_id': 'virtual-then-real-write'}}
        first, events = await invoke(graph, data, config)
        assert len(events) == 1 and not target.exists() and not first.values.get('files')
        ticket = first.tasks[0].interrupts[0]
        second, events = await invoke(graph, Command(resume={ticket.id: {
            **ticket.value, 'type': 'msty_native_resume'}}), config)
        assert len(events) == 1 and len(seen) == 2
        assert not target.exists() and not second.values.get('files')
        rejected = next(message for message in second.values['messages'] if message.type == 'tool')
        assert rejected.status == 'error'
        assert 'native_virtual_path_required' in rejected.content
        assert 'external MCP' in seen[1][-1].content
        assert second.values['execution']['status'] == 'waiting_tools'
        assert second.values['execution']['actions_issued'] == 1
        assert second.values['execution']['native_actions'] == 1
        assert second.values['result']['tool_calls'][0]['name'] == 'write_file'
        for schema in schemas[0]:
            function = schema['function']
            if function['name'].removeprefix('native_') in msty_native._VIRTUAL_FS_DESCRIPTIONS:
                if function['name'].startswith('native_'):
                    assert 'VIRTUAL ONLY' in function['description']
                    field = 'file_path' if 'file_path' in function['parameters']['properties'] else 'path'
                    assert 'Not a Mac path' in function['parameters']['properties'][field]['description']
                else:
                    assert schema == data['tools'][0]
        # Local synthetic MCP runner performs the real isolated file write only
        # after the external interrupt. Feed its factual receipt through the
        # existing callback validator and exact tool-ID mapping.
        target.write_text(body, encoding='utf-8')
        receipt = json.dumps({'status': 'ok', 'file_path': str(target),
            'sha256': hashlib.sha256(target.read_bytes()).hexdigest()})
        resume = external_resume(second.values)
        resume['input']['messages'][-1]['content'] = receipt
        external = second.tasks[0].interrupts[0]
        final, events = await invoke(graph, Command(resume={external.id: resume}), config)
        assert len(events) == 1 and len(seen) == 3
        assert seen[2][-1].content == receipt
        assert target.read_text(encoding='utf-8') == body
        assert not final.values.get('files')
        assert final.values['execution']['status'] == 'answered' and not final.next
    asyncio.run(run())


def test_native_root_lists_only_virtual_mountpoints(monkeypatch):
    seen = scripted(monkeypatch, [answer('', [call('native_ls', {'path': '/'})]), answer()])

    async def run():
        graph = msty_native.build_graph(checkpointer=InMemorySaver(), store=InMemoryStore())
        config = {'configurable': {'thread_id': 'virtual-mounts'}}
        state, _ = await invoke(graph, initial(), config)
        ticket = state.tasks[0].interrupts[0]
        final, _ = await invoke(graph, Command(resume={ticket.id: {
            **ticket.value, 'type': 'msty_native_resume'}}), config)
        content = seen[1]['state']['messages'][-1]['content']
        assert 'VIRTUAL mountpoints only (not Mac)' in content
        assert all(root + '/' in content for root in msty_native.VIRTUAL_ROOTS)
        assert not final.values.get('files')
    asyncio.run(run())


def test_large_external_observation_offload_remains_readable_in_virtual_mount(monkeypatch):
    observed = 'Synthetic external record; literal read_file stays source text.\n' * 1600
    seen = scripted(monkeypatch, [answer('', [call('external_read', {'name': 'large'}, 'large-external')]),
        answer('', [call('native_read_file', {'file_path': '/large_tool_results/large-external', 'limit': 2}, 'offload')]),
        answer()])

    async def run():
        graph = msty_native.build_graph(checkpointer=InMemorySaver(), store=InMemoryStore())
        config = {'configurable': {'thread_id': 'native-offload-scope'}}
        first, _ = await invoke(graph, initial(), config)
        pending = first.tasks[0].interrupts[0]
        resume = external_resume(first.values)
        resume['input']['messages'][-1]['content'] = observed
        second, _ = await invoke(graph, Command(resume={pending.id: resume}), config)
        assert second.values['execution']['status'] == 'waiting_native'
        assert set(second.values['files']) == {'/large_tool_results/large-external'}
        hint = seen[1]['state']['messages'][-1]['content']
        assert 'native_read_file tool' in hint
        ticket = second.tasks[0].interrupts[0]
        final, _ = await invoke(graph, Command(resume={ticket.id: {
            **ticket.value, 'type': 'msty_native_resume'}}), config)
        assert 'literal read_file stays source text.' in seen[2]['state']['messages'][-1]['content']
        assert final.values['execution']['status'] == 'answered'
        assert set(final.values['files']) == {'/large_tool_results/large-external'}
    asyncio.run(run())


@pytest.mark.parametrize('external_count,native_count', [(24, 0), (12, 12), (0, 24), (199, 0), (100, 99), (0, 199)])
def test_raised_native_action_limit_inherits_counts_task_and_budget_binding(monkeypatch, external_count, native_count):
    seen = scripted(monkeypatch, [answer('', [call('native_ls', {'path': '/'})]), answer()])
    data = initial()
    identifier = '18e447fe-1862-4d25-8448-b4522c1a63cf'
    binding = {'version': 1, 'pricing_version': msty_execution.PRICING_VERSION,
               'profile': 'luna', 'input_limit': 180000, 'output_limit': 128}
    data.update(execution_task_id=identifier, task_budget_binding=deepcopy(binding),
        execution={'version': 1, 'task_id': identifier, 'step': 209,
                   'actions_issued': external_count, 'native_actions': native_count,
                   'consultations': 2, 'status': 'running', 'pending': None})

    async def run():
        saver, store = InMemorySaver(), InMemoryStore()
        graph = msty_native.build_graph(checkpointer=saver, store=store)
        config = {'configurable': {'thread_id': f'inherited-{external_count}-{native_count}'}, 'recursion_limit': 64}
        first, events = await invoke(graph, data, config)
        assert len(events) == len(seen) == 1
        assert first.values['execution']['native_actions'] == native_count + 1
        assert first.values['execution']['actions_issued'] == external_count
        assert first.values['execution']['status'] == 'waiting_native'
        assert first.values['execution']['step'] == 210  # No hidden MAX_STEPS=200.
        assert first.values['execution']['consultations'] == 2
        assert first.values['execution']['task_id'] == identifier
        assert first.values['task_budget_binding'] == binding
        ticket = first.tasks[0].interrupts[0]
        graph = msty_native.build_graph(checkpointer=saver, store=store)
        final, events = await invoke(graph, Command(resume={ticket.id: {
            **ticket.value, 'type': 'msty_native_resume'}}), config)
        assert len(events) == 1 and len(seen) == 2
        assert final.values['execution']['native_actions'] == native_count + 1
        assert final.values['execution']['actions_issued'] == external_count
        assert final.values['execution']['step'] == 211
        assert final.values['execution']['consultations'] == 2
        assert final.values['execution']['task_id'] == identifier
        assert final.values['task_budget_binding'] == binding
        assert final.values['execution']['status'] == 'answered'
        assert not final.next
    asyncio.run(run())


@pytest.mark.parametrize('external_count,native_count', [(200, 0), (1, 199), (100, 100), (0, 200)])
@pytest.mark.parametrize('tool', ['native_ls', 'external_read'])
def test_action_201_is_blocked_for_combined_native_external_counters(monkeypatch, external_count, native_count, tool):
    args = {'path': '/'} if tool == 'native_ls' else {'name': 'fixture'}
    seen = scripted(monkeypatch, [answer('', [call(tool, args)])])
    data = initial()
    data['execution'] = {'actions_issued': external_count, 'native_actions': native_count,
                         'consultations': 2, 'step': 209, 'task_id': 'retained-shared-task'}

    async def run():
        graph = msty_native.build_graph(checkpointer=InMemorySaver(), store=InMemoryStore())
        final, events = await invoke(graph, data, {'configurable': {
            'thread_id': f'cap-201-{external_count}-{native_count}-{tool}'}})
        assert len(events) == len(seen) == 1
        assert not events[0]['tool_calls']
        assert final.values['execution']['status'] == 'blocked'
        assert final.values['execution']['actions_issued'] == external_count
        assert final.values['execution']['native_actions'] == native_count
        assert final.values['execution']['consultations'] == 2
        assert final.values['execution']['task_id'] == 'retained-shared-task'
        assert final.values['execution']['step'] == 210
        assert not any(message.type == 'tool' for message in final.values['messages'])
        assert not final.next
    asyncio.run(run())


def test_200_checkpointed_native_actions_then_201_blocks_at_recursion_limit_64(monkeypatch):
    assert msty_execution.MAX_ACTIONS == 200
    seen = scripted(monkeypatch, [answer('', [call('native_ls', {'path': '/'}, f'action-{index}')])
                                  for index in range(1, 202)])

    async def run():
        saver, store = InMemorySaver(), InMemoryStore()
        graph = msty_native.build_graph(checkpointer=saver, store=store)
        config = {'configurable': {'thread_id': 'all-200-checkpointed-actions'}, 'recursion_limit': 64}
        state, events = await invoke(graph, initial(), config)
        task_id = state.values['execution']['task_id']
        for count in range(1, 201):
            assert len(events) == 1 and len(seen) == count
            assert state.values['execution']['status'] == 'waiting_native'
            assert state.values['execution']['native_actions'] == count
            assert state.values['execution']['actions_issued'] == 0
            assert state.values['execution']['step'] == count
            assert state.values['execution']['task_id'] == task_id
            assert sum(message.type == 'tool' for message in state.values['messages']) == count - 1
            if count == 24:
                graph = msty_native.build_graph(checkpointer=saver, store=store)
            ticket = state.tasks[0].interrupts[0]
            state, events = await invoke(graph, Command(resume={ticket.id: {
                **ticket.value, 'type': 'msty_native_resume'}}), config)
        assert len(events) == 1 and len(seen) == 201
        assert state.values['execution']['status'] == 'blocked'
        assert state.values['execution']['native_actions'] == 200
        assert state.values['execution']['actions_issued'] == 0
        assert state.values['execution']['step'] == 201
        assert sum(message.type == 'tool' for message in state.values['messages']) == 200
        assert not state.values['result']['tool_calls'] and not state.next
        # The independent 512-message preparation bound admits this ordinary
        # 200-action tool chain without changing token/context safety limits.
        prepared = msty_models.prepare_messages('luna', seen[-1]['state']['messages'], seen[-1]['state']['tools'])
        assert len(prepared) == 401 < msty_models.MAX_MESSAGES
    asyncio.run(run())


def test_analyst_consult_profile_allowlist():
    state = {'brain_task_role': 'analyst', 'tools': [],
             'messages': [{'role': 'user', 'content': 'synthetic brief'}]}
    assert msty.selected_profile(state) == 'deepseek'
    assert msty.selected_profile({**state, 'consult_profile': None}) == 'deepseek'
    for profile in ('sol6', 'opus5'):
        assert msty.selected_profile({**state, 'consult_profile': profile}) == profile
    # Lead-only, premium autonomous and unknown profiles are never a valid
    # consultation target.
    for bad in ('luna', 'astra', 'sol', 'opus', 'fable', 'sonnet', 'unknown', '', 42):
        with pytest.raises(msty_models.ModelAdapterError):
            msty.selected_profile({**state, 'consult_profile': bad})


def test_server_routed_lead_profile_is_narrow_and_overrides_legacy_env(monkeypatch):
    monkeypatch.setenv('MSTY_MODEL_PROFILE', 'luna')
    state = {'brain_task_role': 'lead', 'messages': [{'role': 'user', 'content': 'lookup'}]}
    assert msty.selected_profile({**state, 'lead_profile': 'deepseek'}) == 'deepseek'
    assert msty.selected_profile({**state, 'lead_profile': 'luna'}) == 'luna'
    for bad in ('sol', 'opus', 'sonnet', 'unknown', '', 42):
        with pytest.raises(msty_models.ModelAdapterError):
            msty.selected_profile({**state, 'lead_profile': bad})


def test_policy_routing_keeps_the_spine_and_drops_unusable_contracts():
    """A step carries only the policy blocks its route can act on."""
    full = msty.POLICY + '\n' + msty_native.NATIVE_POLICY
    spine = ('Ты — Sanare Brain', *msty_prompts.ALWAYS_BLOCKS, 'MSTY_NATIVE_HARNESS_V1')

    plain = msty_prompts.select_policy(full, {'intent': 'direct', 'domains': []}, [])
    for block in spine:
        assert block in plain
    # A plain question cannot act, so no execution or domain contract ships.
    for block in (*msty_prompts.ACTIONABLE_BLOCKS, *msty_prompts.DOMAIN_BLOCKS,
                  *msty_prompts.TOOL_BLOCKS):
        assert block not in plain
    assert len(plain) < len(full) / 2

    site = msty_prompts.select_policy(
        full, {'intent': 'mutate', 'domains': ['sites', 'files']}, ['msty_site_status'])
    for block in (*spine, *msty_prompts.ACTIONABLE_BLOCKS,
                  'MSTY_CONTEXT_REUSE_V1', 'MSTY_SOURCE_SELECTION_V1'):
        assert block in site
    # Discovery and self-improvement belong to other routes.
    assert 'MSTY_TOOL_DISCOVERY_V1' not in site
    assert 'MSTY_CONTINUOUS_IMPROVEMENT_V1' not in site

    # The discovery contract follows the tool, including a namespaced schema.
    for names in (['discover_tools'], ['sanare_admin_discover_tools']):
        routed = msty_prompts.select_policy(full, {'intent': 'mutate', 'domains': ['pressable']}, names)
        assert 'MSTY_TOOL_DISCOVERY_V1' in routed
    assert 'MSTY_TOOL_DISCOVERY_V1' not in msty_prompts.select_policy(
        full, {'intent': 'mutate', 'domains': ['pressable']}, ['execute_tool'])

    brain = msty_prompts.select_policy(full, {'intent': 'mutate', 'domains': ['brain']}, [])
    assert 'MSTY_CONTINUOUS_IMPROVEMENT_V1' in brain and 'Внешние MCP/skills/knowledge' in brain


def test_policy_routing_never_rewrites_unknown_or_unrouted_text():
    """Only the named optional blocks may be dropped; everything else is kept."""
    full = msty.POLICY + '\n' + msty_native.NATIVE_POLICY + '\n\nOWNER_APPENDED_SENTINEL.'
    assert msty_prompts.select_policy(full, None, []) == full
    assert msty_prompts.select_policy(full, {'intent': 'bogus', 'domains': []}, []) == full
    assert msty_prompts.select_policy('', {'intent': 'direct', 'domains': []}, []) == ''
    for route in ({'intent': 'direct', 'domains': []}, {'intent': 'mutate', 'domains': ['brain']}):
        routed = msty_prompts.select_policy(full, route, [])
        assert 'OWNER_APPENDED_SENTINEL.' in routed
        # Every kept block is verbatim from the approved text.
        for block in routed.split('\n\n'):
            assert block in full


ROUTING_TOOLS = [{'type': 'function', 'function': {
    'name': name, 'description': f'Official {name} operation',
    'parameters': {'type': 'object', 'properties': {}}}}
    for name in sorted(msty_native.msty_tool_routing._KNOWN)]


def _routed_step(monkeypatch, text):
    """Run one real graph step and return what the provider actually received."""
    seen = scripted(monkeypatch, [answer('OK')])

    async def run():
        graph = msty_native.build_graph(checkpointer=InMemorySaver(), store=InMemoryStore())
        data = initial()
        data['messages'] = [{'role': 'user', 'content': text}]
        data['tools'] = deepcopy(ROUTING_TOOLS)
        await invoke(graph, data, {'configurable': {'thread_id': f'routed-{abs(hash(text))}'}})

    asyncio.run(run())
    assert len(seen) == 1
    step = seen[0]
    names = {tool['function']['name'] for tool in step['state']['tools']
             if tool.get('type') == 'function'}
    # state['tools'] carries the native virtual-filesystem schemas too; the
    # route only governs the external Toolset.
    return step['system'], names - msty_native.NATIVE_TOOLS


def test_graph_step_delivers_a_routed_policy_and_a_routed_toolset(monkeypatch):
    """End-to-end proof of the wiring, not just of select_policy in isolation."""
    system, names = _routed_step(monkeypatch, 'опубликуй job site-c7228bab8ae54e8cbf0c0b5a8f2573c3')

    # The whole server-side Toolset is never what the model sees.
    assert 0 < len(names) <= msty_native.msty_tool_routing.MAX_SELECTED_TOOLS < len(ROUTING_TOOLS)
    assert 'msty_site_release' in names
    assert not names & msty_native.msty_tool_routing._SUPABASE_WRITE

    for block in ('Ты — Sanare Brain', *msty_prompts.ALWAYS_BLOCKS,
                  *msty_prompts.ACTIONABLE_BLOCKS, 'MSTY_SOURCE_SELECTION_V1',
                  'MSTY_NATIVE_HARNESS_V1', 'MSTY_DYNAMIC_ROUTE_V1'):
        assert block in system, block
    # Contracts belonging to other routes stay off the wire.
    for block in ('MSTY_CONTINUOUS_IMPROVEMENT_V1', 'MSTY_TOOL_DISCOVERY_V1'):
        assert block not in system, block


def test_graph_step_for_a_plain_question_carries_neither_tools_nor_extra_policy(monkeypatch):
    system, names = _routed_step(monkeypatch, 'Объясни кратко, что такое vault.')

    assert names == set()
    for block in ('Ты — Sanare Brain', *msty_prompts.ALWAYS_BLOCKS, 'MSTY_NATIVE_HARNESS_V1'):
        assert block in system, block
    for block in (*msty_prompts.ACTIONABLE_BLOCKS, *msty_prompts.DOMAIN_BLOCKS,
                  *msty_prompts.TOOL_BLOCKS):
        assert block not in system, block


@pytest.mark.parametrize('content,expected', [
    # brain-desk #309: живой ZodError Supabase с блоком самовосстановления окна.
    ('Ошибка инструмента: {"error":{"name":"ZodError","message":"ref must be exactly 20 '
     'characters long"}}\n\n[Brain Desk · самовосстановление] Класс: validation (неверный '
     'аргумент). Вызови list_projects и возьми project_id оттуда.', 'invalid_args'),
    ('[{"name":"inbox_events"}]', None),
])
def test_failed_client_tool_result_is_classified_and_next_step_gets_recovery_note(
        monkeypatch, content, expected):
    """Отказ из окна не засчитывается успехом: ToolMessage размечен классом, а
    следующий шаг модели получает корректирующую заметку до ответа владельцу."""
    seen = scripted(monkeypatch, [answer('', [call('external_read', {'name': 'x' * 19}, 'ext-1')]),
                                  answer()])

    async def run():
        graph = msty_native.build_graph(checkpointer=InMemorySaver(), store=InMemoryStore())
        config = {'configurable': {'thread_id': f'recovery-{expected}'}}
        first, _ = await invoke(graph, initial(), config)
        external = first.tasks[0].interrupts[0]
        resume = external_resume(first.values)
        resume['input']['messages'][-1]['content'] = content
        final, _ = await invoke(graph, Command(resume={external.id: resume}), config)
        assert final.values['execution']['status'] == 'answered'
        return final

    final = asyncio.run(run())
    assert len(seen) == 2
    observed = [m for m in seen[1]['state']['messages'] if m['role'] == 'tool'][-1]
    system = seen[1]['system']
    if expected is None:
        assert observed['content'] == content
        assert 'TOOL_ERROR_RECOVERY_NOTE' not in system
        assert not final.values.get('tau_errors')
        return
    assert observed['content'].startswith(f'tau_class={expected}.')
    assert 'ref must be exactly' in observed['content']
    assert 'TOOL_ERROR_RECOVERY_NOTE' in system
    assert f'external_read: tau_class={expected}' in system
    assert 'TOOL_ERROR_RECOVERY_NOTE' not in seen[0]['system']
    # Превентивное правило (не угадывать id) шло уже с первым шагом внешнего вызова.
    assert 'TOOL_ERROR_RECOVERY_V1' in seen[0]['system']
    assert [entry['class'] for entry in final.values['tau_errors']] == [expected]


def test_tool_error_recovery_block_is_always_on_the_wire_and_reconciles_reuse():
    """brain-desk #309: правило восстановления после ошибки инструмента — в любом
    маршруте, а правила повторного использования id уступают отклонённому id."""
    full = msty.POLICY + '\n' + msty_native.NATIVE_POLICY
    block = next(b for b in msty.POLICY.split('\n\n') if b.startswith('TOOL_ERROR_RECOVERY_V1'))
    for needle in ('до ответа', 'не угадывай', 'list_projects', 'Переподключить',
                   'Урок Brain Desk', 'без изменений не повторяй'):
        assert needle in block, needle
    # Блок следует за внешними инструментами на любом маршруте, включая direct.
    for route in ({'intent': 'direct', 'domains': []},
                  {'intent': 'read', 'domains': ['supabase']},
                  {'intent': 'mutate', 'domains': ['brain']}):
        assert block in msty_prompts.select_policy(
            full, route, ['native_read_file', 'list_tables'], external_names=['list_tables'])
        # Без внешних инструментов вызывать нечего — префикс не раздувается.
        assert block not in msty_prompts.select_policy(
            full, route, ['native_read_file'], external_names=[])
    # Неизвестный набор внешних (старый вызывающий) — консервативно сохраняется.
    assert block in msty_prompts.select_policy(full, {'intent': 'direct', 'domains': []},
                                               ['list_tables'])
    reuse = next(b for b in msty.POLICY.split('\n\n') if b.startswith('MSTY_CONTEXT_REUSE_V1'))
    # brain-agency-audit-2026-09-26 #1: the two blocks now state one rule instead
    # of contradicting each other — known/confirmed id reused directly, rejected/
    # unknown id gets a fresh live check, worded the same way in both blocks.
    assert 'id отклонён инструментом, неизвестен' in reuse
    assert 'см. TOOL_ERROR_RECOVERY_V1' in reuse
    assert 'MSTY_CONTEXT_REUSE_V1' in block.replace('\n', ' ')
    from deep_agent import msty_memory
    assert 'если инструмент отклонил\nproject_id — вызови list_projects' in msty_memory.system_context()
