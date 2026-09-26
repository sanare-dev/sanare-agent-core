"""Offline regressions for the exact graph exposed to Msty, not the template agent."""

import asyncio
import builtins
from copy import deepcopy
import socket
import urllib.request

import pytest
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, AIMessageChunk
from langchain_core.outputs import ChatGeneration, ChatGenerationChunk, ChatResult

from deep_agent import msty


@pytest.fixture(autouse=True)
def no_tracing(monkeypatch):
    monkeypatch.setenv('LANGSMITH_TRACING', 'false')
    monkeypatch.setenv('LANGCHAIN_TRACING_V2', 'false')


def tool(schema):
    return {'type': 'function', 'function': {'name': 'inspect', 'parameters': schema}}


def call(arguments, *, name='inspect', identifier='call-1'):
    return {'id': identifier, 'name': name, 'args': arguments}


def stamped_sonnet(response, level='low', reason='short_question'):
    """Exact new publication contract; independent of the implementation helper."""
    expected = response.model_dump()
    expected['response_metadata'] = {
        **response.response_metadata,
        'model_name': 'claude-sonnet-4-6',
        'msty_model_name': 'claude-sonnet-4-6',
        'msty_model_profile': 'sonnet',
        'msty_model_provider': 'anthropic',
        # Per-task effort record (short owner text -> low); Sonnet ignores it.
        'msty_reasoning_effort': {'version': 1, 'level': level, 'reason': reason,
                                  'profile': 'sonnet', 'provider_value': None},
    }
    return expected


def install_model(monkeypatch, calls, *, invalid_calls=None):
    seen = {'generations': 0}
    response = AIMessage(
        content='SYNTHETIC_UNVERIFIED_CLAIM', tool_calls=calls,
        invalid_tool_calls=invalid_calls or [],
        additional_kwargs={'tool_calls': 'SYNTHETIC_RAW_CALL'},
        response_metadata={'stop_reason': 'tool_use'},
        usage_metadata={'input_tokens': 20, 'output_tokens': 7, 'total_tokens': 27},
    )

    class Model:
        def __init__(self, **kwargs):
            pass

        def bind_tools(self, tools, **kwargs):
            seen['bound'] = deepcopy(tools)
            return self

        async def ainvoke(self, messages):
            seen['generations'] += 1
            return response

    monkeypatch.setattr(msty, 'ChatAnthropic', Model)
    return seen, response


@pytest.mark.parametrize('schema,arguments', [
    ({'type': 'object', 'required': ['path']}, {}),
    ({'type': 'object', 'properties': {'path': {'type': 'string'}}}, {'path': 12}),
    ({'type': 'object', 'additionalProperties': False}, {'extra': True}),
    ({'properties': {'mode': {'enum': ['read']}}}, {'mode': 'write'}),
    ({'properties': {'paths': {'type': 'array', 'items': {'type': 'string'}}}},
     {'paths': ['README.md', 3]}),
    ({'properties': {'item': {'type': 'object', 'required': ['path']}}}, {'item': {}}),
    ({'properties': {'count': {'type': 'integer', 'minimum': 1}}}, {'count': 0}),
    ({'properties': {'path': {'pattern': '^safe/'}}}, {'path': 'outside/file'}),
    ({'oneOf': [{'required': ['a']}, {'required': ['b']}]}, {'a': 1, 'b': 1}),
    ({'$defs': {'path': {'type': 'string'}},
      'properties': {'path': {'$ref': '#/$defs/path'}}}, {'path': 9}),
    ({'properties': {'address': {'type': 'string', 'format': 'ipv4'}}},
     {'address': 'not-an-ip'}),
    (False, {}),
    ({'type': 'not-a-json-schema-type'}, {}),
    ({'$schema': 'https://invalid.example/unknown-draft'}, {}),
    ({'$ref': 'https://invalid.example/schema.json'}, {}),
    ({'$ref': 'file:///must-not-be-read.json'}, {}),
])
def test_arguments_are_validated_against_full_schema(monkeypatch, schema, arguments):
    seen, response = install_model(monkeypatch, [call(arguments)])
    state = {'messages': [{'role': 'user', 'content': 'inspect'}], 'tools': [tool(schema)]}
    before = deepcopy(state)
    result = asyncio.run(msty.respond(state))['result']
    assert result['tool_calls'] == []
    assert result['invalid_tool_calls'] == []
    assert 'не выполнен' in result['content']
    assert 'SYNTHETIC_UNVERIFIED_CLAIM' not in str(result)
    assert 'SYNTHETIC_RAW_CALL' not in str(result)
    assert result['usage_metadata'] == response.usage_metadata
    assert result['response_metadata'] == stamped_sonnet(response)['response_metadata']
    assert seen['generations'] == 1  # No hidden LLM repair / retry.
    assert state == before


@pytest.mark.parametrize('schema,arguments', [
    ({'$defs': {'path': {'type': 'string'}}, 'type': 'object',
      'properties': {'path': {'$ref': '#/$defs/path'}}, 'required': ['path'],
      'additionalProperties': False}, {'path': 'README.md'}),
    ({'$schema': 'http://json-schema.org/draft-07/schema#',
      'properties': {'paths': {'type': 'array', 'items': {'type': 'string'}, 'minItems': 1}}},
     {'paths': ['README.md']}),
    ({'allOf': [{'required': ['mode']}, {'properties': {'mode': {'const': 'read'}}}]},
     {'mode': 'read'}),
    (True, {}),
])
def test_valid_arguments_and_local_refs_are_unchanged(monkeypatch, schema, arguments):
    seen, response = install_model(monkeypatch, [call(arguments)])
    result = asyncio.run(msty.respond({'messages': [], 'tools': [tool(schema)]}))
    assert result['result'] == stamped_sonnet(response, 'medium', 'no_owner_text')
    assert seen['generations'] == 1


def test_invalid_batch_rejects_valid_siblings_and_ambiguous_duplicate_schema(monkeypatch):
    install_model(monkeypatch, [call({'path': 'README.md'}),
                                call({'path': 42}, identifier='call-2')])
    schema = {'properties': {'path': {'type': 'string'}}}
    result = asyncio.run(msty.respond({'messages': [], 'tools': [tool(schema)]}))
    assert result['result']['tool_calls'] == []
    install_model(monkeypatch, [call({'path': 'README.md'})])
    result = asyncio.run(msty.respond({'messages': [], 'tools': [tool(schema), tool({})]}))
    assert result['result']['tool_calls'] == []


def test_invalid_parsed_call_keeps_usage_and_clears_raw_fields(monkeypatch):
    _, response = install_model(monkeypatch, [], invalid_calls=[{
        'name': 'inspect', 'args': '{invalid', 'id': 'broken', 'error': 'parse failure'}])
    result = asyncio.run(msty.respond({'messages': [], 'tools': [tool({})]}))['result']
    assert result['tool_calls'] == result['invalid_tool_calls'] == []
    assert result['additional_kwargs'] == {}
    assert result['usage_metadata'] == response.usage_metadata


def test_external_refs_do_not_attempt_network_or_file_reads(monkeypatch):
    attempted_io = []

    def forbidden(*args, **kwargs):
        attempted_io.append(True)
        raise AssertionError('External schema access is forbidden')

    monkeypatch.setattr(socket.socket, 'connect', forbidden)
    monkeypatch.setattr(urllib.request, 'urlopen', forbidden)
    monkeypatch.setattr(builtins, 'open', forbidden)
    response = AIMessage(content='', tool_calls=[call({})])
    for reference in ('https://invalid.example/schema.json', 'file:///must-not-be-read.json'):
        assert not msty.valid_tool_calls(response, [tool({'$ref': reference})])
    assert attempted_io == []


def test_missing_parameters_keeps_existing_unconstrained_schema_contract(monkeypatch):
    _, response = install_model(monkeypatch, [call({'path': 'README.md'})])
    result = asyncio.run(msty.respond({'messages': [], 'tools': [
        {'type': 'function', 'function': {'name': 'inspect'}}]}))
    assert result['result'] == stamped_sonnet(response, 'medium', 'no_owner_text')


@pytest.mark.parametrize('stage', ['get_writer', 'write'])
def test_runtime_writer_errors_are_not_silently_suppressed(monkeypatch, stage):
    def failure(*args):
        raise RuntimeError('synthetic stream error')

    monkeypatch.setattr(msty, 'get_stream_writer', failure if stage == 'get_writer' else lambda: failure)
    with pytest.raises(RuntimeError, match='synthetic stream error'):
        msty.publish_result(AIMessage(content='ok'), None)


class SyntheticStreamingModel(BaseChatModel):
    """A real LangChain model interface, with no provider client or network I/O."""

    @property
    def _llm_type(self):
        return 'offline-synthetic'

    def bind_tools(self, tools, **kwargs):
        return self

    def _generate(self, messages, stop=None, run_manager=None, **kwargs):
        return ChatResult(generations=[ChatGeneration(message=AIMessage(
            content='SYNTHETIC_UNVERIFIED_CLAIM', tool_calls=[call({'path': 42})]))])

    async def _astream(self, messages, stop=None, run_manager=None, **kwargs):
        yield ChatGenerationChunk(message=AIMessageChunk(content='SYNTHETIC_UNVERIFIED_CLAIM'))
        yield ChatGenerationChunk(message=AIMessageChunk(content='', tool_call_chunks=[{
            'name': 'inspect', 'args': '{"path":42}', 'id': 'call-1', 'index': 0}]))


@pytest.mark.parametrize('modes', [['custom', 'values'], ['messages', 'custom', 'values']])
def test_actual_graph_custom_stream_contains_only_validated_result(monkeypatch, modes):
    monkeypatch.setattr(msty, 'ChatAnthropic', lambda **kwargs: SyntheticStreamingModel())

    async def collect():
        return [event async for event in msty.graph.astream({
            'messages': [{'role': 'user', 'content': 'inspect'}],
            'tools': [tool({'properties': {'path': {'type': 'string'}}})],
        }, stream_mode=modes)]

    events = asyncio.run(collect())
    custom = [data for mode, data in events if mode == 'custom']
    final = [data for mode, data in events if mode == 'values'][-1]
    assert custom == [{'type': 'validated_result', 'message': final['result']}]
    assert final['result']['tool_calls'] == []
    assert 'SYNTHETIC_UNVERIFIED_CLAIM' not in str(custom)
    assert 'не выполнен' in custom[0]['message']['content']
    assert [mode for mode, _ in events].index('custom') < len(events) - 1
    if 'messages' in modes:
        # Native model events are private/unvalidated, never the publication API.
        assert 'SYNTHETIC_UNVERIFIED_CLAIM' in str([x for mode, x in events if mode == 'messages'])


def test_actual_graph_budget_blocker_is_also_published_without_generation(monkeypatch):
    class Model:
        def __init__(self, **kwargs):
            pass

        async def ainvoke(self, messages):
            raise AssertionError('Rejected budget must not generate')

    monkeypatch.setattr(msty, 'ChatAnthropic', Model)

    async def collect():
        return [event async for event in msty.graph.astream({
            'messages': [], 'tools': [], 'context_budget': 'unsupported-version',
        }, stream_mode=['custom', 'values'])]

    events = asyncio.run(collect())
    custom = [data for mode, data in events if mode == 'custom']
    final = [data for mode, data in events if mode == 'values'][-1]
    assert final['context_budget_check']['status'] == 'rejected'
    assert custom == [{'type': 'validated_result', 'message': final['result']}]
    assert custom[0]['message']['usage_metadata']['total_tokens'] == 0


@pytest.mark.parametrize('calls', [[], [call({'path': 'README.md'})]])
def test_actual_graph_publishes_valid_text_and_tool_result_once_unchanged(monkeypatch, calls):
    seen, response = install_model(monkeypatch, calls)

    async def collect():
        return [event async for event in msty.graph.astream({
            'messages': [{'role': 'user', 'content': 'inspect'}],
            'tools': [tool({'properties': {'path': {'type': 'string'}}})],
        }, stream_mode=['custom', 'values'])]

    events = asyncio.run(collect())
    custom = [data for mode, data in events if mode == 'custom']
    final = [data for mode, data in events if mode == 'values'][-1]
    assert custom == [{'type': 'validated_result', 'message': stamped_sonnet(response)}]
    assert final['result'] == stamped_sonnet(response)
    assert final['context_budget_check'] is None
    assert seen['generations'] == 1


@pytest.mark.parametrize('choice', ['none', {'type': 'none'}])
def test_cap_final_keeps_history_schemas_and_uses_real_sdk_none_choice(monkeypatch, choice):
    sdk_bind = msty.ChatAnthropic.bind_tools
    seen = {}

    class Model:
        thinking = None
        bind_tools = sdk_bind

        def __init__(self, **kwargs):
            pass

        def bind(self, **kwargs):
            seen['provider_options'] = kwargs
            return self

        async def ainvoke(self, messages):
            seen['system'], seen['messages'] = msty._format_messages(messages)
            return AIMessage(content='Confirmed partial work; action limit reached.')

    monkeypatch.setattr(msty, 'ChatAnthropic', Model)
    calls = [{'id': f'completed-{i}', 'type': 'function',
              'function': {'name': 'inspect', 'arguments': '{}'}} for i in range(24)]
    history = [{'role': 'user', 'content': 'inspect'},
               {'role': 'assistant', 'content': '', 'tool_calls': calls}]
    history += [{'role': 'tool', 'tool_call_id': c['id'], 'content': 'confirmed result'} for c in calls]
    state = {'messages': history, 'tools': [tool({'type': 'object'})], 'tool_choice': choice}
    before = deepcopy(state)
    result = asyncio.run(msty.respond(state))
    assert state == before
    assert seen['provider_options']['tool_choice'] == {'type': 'none'}
    assert seen['provider_options']['tools'] == [{'name': 'inspect', 'input_schema': {'type': 'object'}}]
    assert sum(block.get('type') == 'tool_use' for m in seen['messages']
               for block in m['content'] if isinstance(block, dict)) == 24
    assert sum(block.get('type') == 'tool_result' for m in seen['messages']
               for block in m['content'] if isinstance(block, dict)) == 24
    assert result['result']['content'] == 'Confirmed partial work; action limit reached.'
    assert result['result']['tool_calls'] == []


@pytest.mark.parametrize('choice', ['none', {'type': 'none'}])
def test_explicit_none_rejects_even_schema_valid_calls_before_publication(monkeypatch, choice):
    seen, response = install_model(monkeypatch, [call({})])

    async def collect():
        return [event async for event in msty.graph.astream({
            'messages': [{'role': 'user', 'content': 'summarize; no further actions'}],
            'tools': [tool({'type': 'object'})], 'tool_choice': choice,
        }, stream_mode=['custom', 'values'])]

    events = asyncio.run(collect())
    custom = [data for mode, data in events if mode == 'custom']
    final = [data for mode, data in events if mode == 'values'][-1]
    assert custom == [{'type': 'validated_result', 'message': final['result']}]
    assert final['result']['tool_calls'] == []
    assert final['result']['invalid_tool_calls'] == []
    assert final['result']['additional_kwargs'] == {}
    assert 'отключено' in final['result']['content']
    assert final['result']['usage_metadata'] == response.usage_metadata
    assert final['result']['response_metadata'] == stamped_sonnet(response)['response_metadata']
    assert seen['generations'] == 1
