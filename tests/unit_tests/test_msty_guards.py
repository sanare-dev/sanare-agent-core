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
    assert result['response_metadata'] == response.response_metadata
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
    assert result['result'] == response.model_dump()
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
    assert result['result'] == response.model_dump()


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
    assert custom == [{'type': 'validated_result', 'message': response.model_dump()}]
    assert final['result'] == response.model_dump()
    assert final['context_budget_check'] is None
    assert seen['generations'] == 1
