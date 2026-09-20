"""Offline compiled-graph streaming contract; no network, keys or inference."""
import asyncio
from copy import deepcopy

import pytest
from langchain_core.messages import AIMessage, AIMessageChunk
from langgraph.checkpoint.memory import InMemorySaver

from deep_agent import msty, msty_models, msty_task, msty_stream, msty_execution
from tests.unit_tests.test_msty_compaction import initial, history, summary, USAGE
from tests.unit_tests.test_msty_task import TOOLS, plan_call, plan_receipt
from tests.unit_tests.test_msty_task import payload

PROTOCOL = 'msty-text-delta-v1'


def chunks(*texts, reason='stop', model='gpt-5.6-luna'):
    return [*[AIMessageChunk(content=t) for t in texts],
            AIMessageChunk(content='', response_metadata={'finish_reason': reason, 'model_name': model},
                           usage_metadata=deepcopy(USAGE))]


def install(monkeypatch, pieces, *, legacy='legacy', gate=None, counts=None, sonnet=False):
    seen = {'astream': 0, 'ainvoke': 0, 'finished': False, 'options': []}
    monkeypatch.setenv('LANGSMITH_TRACING', 'false')
    monkeypatch.setenv('LANGCHAIN_TRACING_V2', 'false')
    monkeypatch.delenv('MSTY_MODEL_PROFILE', raising=False)
    monkeypatch.setattr(msty_models.msty_gateway, 'enabled', lambda: False)
    class Model:
        def bind_tools(self, tools, **kwargs):
            return self
        def get_num_tokens_from_messages(self, *args, **kwargs):
            return 100
        async def ainvoke(self, messages):
            seen['ainvoke'] += 1
            return AIMessage(content=legacy, usage_metadata=deepcopy(USAGE))
        async def astream(self, messages, **kwargs):
            seen['astream'] += 1
            seen['options'].append(kwargs)
            for index, piece in enumerate(pieces):
                if gate is not None and index == 1:
                    await gate.wait()
                if isinstance(piece, Exception):
                    raise piece
                yield deepcopy(piece)
            seen['finished'] = True
    async def count(*args):
        return counts.pop(0) if counts is not None else 100
    monkeypatch.setattr(msty_models, 'make_model', lambda *args: Model())
    monkeypatch.setattr(msty_models, 'count_input', count)
    monkeypatch.setattr(msty, 'ChatAnthropic', lambda **kwargs: Model())
    if sonnet:
        monkeypatch.setenv('MSTY_MODEL_PROFILE', 'sonnet')
    return seen


async def events(state):
    return [event async for event in msty.graph.astream(state, stream_mode=['custom', 'values'])]


def custom(items):
    return [value for mode, value in items if mode == 'custom']


def value(items):
    return [value for mode, value in items if mode == 'values'][-1]


def test_incremental_unicode_reaches_caller_before_provider_finishes(monkeypatch):
    async def scenario():
        released = asyncio.Event()
        seen = install(monkeypatch, chunks('Привет ', '🌍!'), gate=released)
        stream = msty.graph.astream(initial(text_stream_protocol=PROTOCOL), stream_mode=['custom', 'values'])
        received = []
        async for mode, event in stream:
            received.append((mode, event))
            if mode == 'custom' and event.get('type') == 'text_delta':
                if event['seq'] == 0:
                    assert not seen['finished']
                    released.set()
        out = custom(received)
        assert out[:2] == [{'type': 'text_delta', 'version': 1, 'seq': 0, 'text': 'Привет '},
                           {'type': 'text_delta', 'version': 1, 'seq': 1, 'text': '🌍!'}]
        assert out[-1]['type'] == 'validated_result'
        final = value(received)['result']
        assert final['content'] == 'Привет 🌍!'
        assert final['usage_metadata'] == USAGE
        assert final['response_metadata']['model_name'] == 'gpt-5.6-luna'
        assert seen['options'] == [{'stream_usage': True}]
        assert seen['astream'] == 1 and seen['ainvoke'] == 0
    asyncio.run(asyncio.wait_for(scenario(), 2))


def test_legacy_callers_keep_one_ainvoke_and_no_new_custom_events(monkeypatch):
    seen = install(monkeypatch, [])
    result = asyncio.run(events(initial()))
    assert [e['type'] for e in custom(result)] == ['validated_result']
    assert value(result)['result']['content'] == 'legacy'
    assert seen['ainvoke'] == 1 and not seen['astream']


def test_tool_argument_fragments_never_exported(monkeypatch):
    pieces = [AIMessageChunk(content='Reading.', tool_call_chunks=[
        {'name': 'read', 'id': 'call', 'index': 0, 'args': '{"path":'}]),
        AIMessageChunk(content='', tool_call_chunks=[{'name': None, 'id': None, 'index': 0, 'args': '"fixture"}'}]),
        *chunks(reason='tool_calls')]
    seen = install(monkeypatch, pieces)
    tools = [{'type': 'function', 'function': {'name': 'read', 'parameters': {'type': 'object',
        'properties': {'path': {'type': 'string'}}, 'required': ['path'], 'additionalProperties': False}}}]
    result = asyncio.run(events(initial(text_stream_protocol=PROTOCOL, tools=tools)))
    emitted = custom(result)
    assert emitted[0] == {'type': 'text_delta', 'version': 1, 'seq': 0, 'text': 'Reading.'}
    assert len(emitted) == 2
    assert emitted[-1]['message']['tool_calls'][0]['args'] == {'path': 'fixture'}
    assert value(result)['execution']['status'] == 'waiting_tools'
    assert seen['astream'] == 1


@pytest.mark.parametrize('status', ['planned', 'blocked', 'verified_against_observations'])
def test_artifact_contract_never_streams_provisional_completion(monkeypatch, status):
    install(monkeypatch, chunks('Everything is done.'))
    contract = msty_task._plan(plan_call(), plan_receipt())
    contract['status'] = status
    state = initial(text_stream_protocol=PROTOCOL, tools=deepcopy(TOOLS), task_contract=contract,
                    messages=[{'role': 'user', 'content': 'Create the requested artifact.'}])
    result = asyncio.run(events(state))
    assert [event['type'] for event in custom(result)] == ['validated_result']
    if status == 'planned':
        assert value(result)['result']['content'] != 'Everything is done.'
        assert value(result)['result']['tool_calls'][0]['name'].endswith('msty_task_verify')


@pytest.mark.parametrize('pieces', [chunks('Wrong model.', model='unexpected'),
    [AIMessageChunk(content='Oops.'), RuntimeError('DO_NOT_EXPOSE_PROVIDER_DATA')],
    [AIMessageChunk(content='Text.', tool_call_chunks=[{'id': 'bad', 'name': 'unregistered', 'args': '{}', 'index': 0}]),
     *chunks(reason='tool_calls')]])
def test_rejected_or_failed_stream_invalidates_and_never_executes(monkeypatch, pieces):
    seen = install(monkeypatch, pieces)
    result = asyncio.run(events(initial(text_stream_protocol=PROTOCOL)))
    emitted = custom(result)
    assert emitted[-2] == {'type': 'text_invalidated', 'version': 1}
    assert emitted[-1]['type'] == 'validated_result'
    assert value(result)['result']['tool_calls'] == []
    assert 'DO_NOT_EXPOSE' not in repr(result)
    assert seen['astream'] == 1 and not seen['ainvoke']


@pytest.mark.parametrize('reason', ['length', 'max_tokens', 'model_context_window_exceeded', 'content_filter', 'refusal'])
def test_known_incomplete_terminal_preserves_partial_text_usage_and_finish(monkeypatch, reason):
    install(monkeypatch, chunks('Verified partial output.', reason=reason))
    result = asyncio.run(events(initial(text_stream_protocol=PROTOCOL)))
    assert [event['type'] for event in custom(result)] == ['text_delta', 'validated_result']
    final = value(result)
    assert final['result']['content'] == 'Verified partial output.'
    assert final['result']['usage_metadata'] == USAGE
    assert final['result']['response_metadata']['finish_reason'] == reason
    assert final['execution']['status'] == 'incomplete'
    assert not final['result']['tool_calls']


def test_compaction_stays_one_nonstream_generation(monkeypatch):
    state = history()
    state['text_stream_protocol'] = PROTOCOL
    seen = install(monkeypatch, [], legacy=summary(state), counts=[150000, 60000])
    result = asyncio.run(events(state))
    assert [event['type'] for event in custom(result)] == ['validated_result']
    assert value(result)['result']['response_metadata']['msty_stage'] == 'compaction'
    assert seen['ainvoke'] == 1 and not seen['astream']


def test_unknown_protocol_rejected_before_any_generation(monkeypatch):
    seen = install(monkeypatch, [])
    result = asyncio.run(events(initial(text_stream_protocol='unknown')))
    assert not seen['astream'] and not seen['ainvoke']
    assert value(result)['result']['response_metadata']['msty_generation'] == 'not_started'


def test_sonnet_text_blocks_usage_and_stop_are_aggregated(monkeypatch):
    install(monkeypatch, [AIMessageChunk(content=[{'type': 'text', 'text': 'One ', 'index': 0}],
        usage_metadata={'input_tokens': 10, 'output_tokens': 0, 'total_tokens': 10}),
        AIMessageChunk(content=[{'type': 'text', 'text': 'two', 'index': 0}]),
        AIMessageChunk(content=[], response_metadata={'stop_reason': 'end_turn', 'model': 'claude-sonnet-4-6'},
            usage_metadata={'input_tokens': 0, 'output_tokens': 3, 'total_tokens': 3})], sonnet=True)
    result = asyncio.run(events({'messages': [{'role': 'user', 'content': 'Hello'}],
                                'tools': [], 'text_stream_protocol': PROTOCOL}))
    emitted = custom(result)
    assert [event['text'] for event in emitted if event['type'] == 'text_delta'] == ['One ', 'two']
    assert emitted[-1]['message']['usage_metadata'] == {'input_tokens': 10, 'output_tokens': 3, 'total_tokens': 13}
    assert emitted[-1]['message']['response_metadata']['stop_reason'] == 'end_turn'


@pytest.mark.parametrize('pieces', [[AIMessageChunk(content='Premature EOF')], [],
    [AIMessageChunk(content='Known prefix'), AIMessageChunk(content='X' * (256 * 1024))]])
def test_missing_terminal_empty_and_oversize_stream_are_unknown_not_zero(monkeypatch, pieces):
    seen = install(monkeypatch, pieces)
    result = asyncio.run(events(initial(text_stream_protocol=PROTOCOL)))
    final = value(result)
    assert final['result']['usage_metadata'] is None
    assert final['result']['response_metadata']['msty_generation'] == 'stream_failed'
    assert final['execution']['status'] == 'blocked'
    assert not final['result']['tool_calls']
    assert seen['astream'] == 1 and not seen['ainvoke']


@pytest.mark.parametrize('raw,expected', [
    ({'prompt_tokens': 11, 'completion_tokens': 3, 'total_tokens': 14,
      'prompt_tokens_details': {'cached_tokens': 5}, 'completion_tokens_details': {'reasoning_tokens': 1}},
     {'input_tokens': 11, 'output_tokens': 3, 'total_tokens': 14,
      'input_token_details': {'cache_read': 5}, 'output_token_details': {'reasoning': 1}}),
    ({'completion_tokens': 3, 'total_tokens': 3}, None),
    ({'prompt_tokens': 11, 'completion_tokens': 3, 'total_tokens': 99}, None),
])
def test_real_sdk_converter_preserves_raw_stream_usage_never_materializes_unknown_zero(raw, expected):
    model = msty_models.ChatOpenAI(model='gpt-5.6-luna', api_key='offline-test-not-a-secret',
                                   base_url='https://example.invalid/v1', use_responses_api=False)
    converted = model._convert_chunk_to_generation_chunk(
        {'id': 'synthetic', 'choices': [], 'model': 'gpt-5.6-luna', 'usage': deepcopy(raw)}, AIMessageChunk, {})
    assert converted.message.response_metadata['token_usage'] == raw
    assert msty_models.checked_usage('luna', converted.message) == expected


def test_duplicate_usage_receipts_fail_unknown(monkeypatch):
    one = AIMessageChunk(content='', response_metadata={'token_usage': {
        'prompt_tokens': 10, 'completion_tokens': 2, 'total_tokens': 12}, 'finish_reason': 'stop'},
        usage_metadata={'input_tokens': 10, 'output_tokens': 2, 'total_tokens': 12})
    install(monkeypatch, [AIMessageChunk(content='Prefix.'), one, one])
    result = asyncio.run(events(initial(text_stream_protocol=PROTOCOL)))
    assert value(result)['result']['usage_metadata'] is None
    assert custom(result)[-2]['type'] == 'text_invalidated'


def test_render_protocol_can_enable_disable_on_callback_but_not_accept_unknown(monkeypatch):
    install(monkeypatch, [], legacy='Plan.')
    # A genuine pending checkpoint is obtained through the existing ainvoke path.
    from tests.unit_tests.test_msty_compaction import install as install_legacy
    install_legacy(monkeypatch, [AIMessage(content='Plan.', tool_calls=[plan_call()], usage_metadata=USAGE)], [100])
    first = asyncio.run(msty.graph.ainvoke(initial(tools=deepcopy(TOOLS))))
    resume = payload(first, plan_receipt())
    resume['input']['text_stream_protocol'] = PROTOCOL
    enabled = msty_execution.validate_resume(first, resume)
    assert enabled['text_stream_protocol'] == PROTOCOL
    first['text_stream_protocol'] = PROTOCOL
    resume['input']['text_stream_protocol'] = None
    assert msty_execution.validate_resume(first, resume)['text_stream_protocol'] is None
    del resume['input']['text_stream_protocol']
    assert msty_execution.validate_resume(first, resume)['text_stream_protocol'] is None
    resume['input']['text_stream_protocol'] = 'unknown'
    with pytest.raises(msty_execution.ExecutionProtocolError):
        msty_execution.validate_resume(first, resume)


def test_cancel_invalidates_and_propagates_without_retry(monkeypatch):
    emitted = []
    monkeypatch.setattr(msty_stream, 'get_stream_writer', lambda: emitted.append)
    class Model:
        async def astream(self, messages, **kwargs):
            yield AIMessageChunk(content='Partial.')
            raise asyncio.CancelledError()
    with pytest.raises(asyncio.CancelledError):
        asyncio.run(msty_stream.TextStream({}).invoke(Model(), []))
    assert emitted[-1] == {'type': 'text_invalidated', 'version': 1}


def test_real_async_adapter_requests_usage_and_preserves_raw_final_metadata(monkeypatch):
    monkeypatch.setenv('LANGSMITH_TRACING', 'false')
    monkeypatch.setenv('LANGCHAIN_TRACING_V2', 'false')
    sent = []
    raw_usage = {'prompt_tokens': 12, 'completion_tokens': 2, 'total_tokens': 14,
                 'prompt_tokens_details': {'cached_tokens': 4}}
    class Response:
        async def __aenter__(self):
            return self
        async def __aexit__(self, *args):
            return False
        async def __aiter__(self):
            for packet in [
                {'choices': [{'delta': {'role': 'assistant', 'content': 'First '}, 'finish_reason': None}]},
                {'choices': [{'delta': {'content': 'second'}, 'finish_reason': None}]},
                {'choices': [{'delta': {}, 'finish_reason': 'stop'}], 'model': 'gpt-5.6-luna'},
                {'choices': [], 'usage': deepcopy(raw_usage)}]:
                yield packet
    class Client:
        async def create(self, **payload):
            sent.append(deepcopy(payload))
            return Response()
    model = msty_models.ChatOpenAI(model='gpt-5.6-luna', api_key='offline-test-not-a-secret',
        async_client=Client(), base_url='https://example.invalid/v1', use_responses_api=False, stream_usage=False)
    emitted = []
    monkeypatch.setattr(msty_stream, 'get_stream_writer', lambda: emitted.append)
    final = asyncio.run(msty_stream.TextStream({}).invoke(model, [{'role': 'user', 'content': 'Synthetic.'}]))
    assert sent[0]['stream'] is True and sent[0]['stream_options'] == {'include_usage': True}
    assert final.content == 'First second'
    assert final.response_metadata['token_usage'] == raw_usage
    assert final.response_metadata['finish_reason'] == 'stop'
    assert final.response_metadata['model_name'] == 'gpt-5.6-luna'
    assert msty_models.checked_usage('luna', final)['input_token_details']['cache_read'] == 4
    assert [event['text'] for event in emitted] == ['First ', 'second']


def test_independent_fresh_task_does_not_inherit_old_artifact_buffer(monkeypatch):
    install(monkeypatch, chunks('Short answer.'))
    async def scenario():
        graph = msty.builder.compile(checkpointer=InMemorySaver())
        old = initial(text_stream_protocol=PROTOCOL, task_contract={
            'status': 'verified_against_observations', 'whole_task_completion_verified': False})
        first = [event async for event in graph.astream(old,
            {'configurable': {'thread_id': 'old-artifact'}}, stream_mode=['custom', 'values'])]
        assert [event['type'] for event in custom(first)] == ['validated_result']
        second = [event async for event in graph.astream(initial(text_stream_protocol=PROTOCOL),
            {'configurable': {'thread_id': 'new-user-task'}}, stream_mode=['custom', 'values'])]
        assert [event['type'] for event in custom(second)] == ['text_delta', 'validated_result']
    asyncio.run(scenario())


def test_consultation_limit_still_applies_after_streaming(monkeypatch):
    name = 'sanare_admin_msty_brain_consult'
    install(monkeypatch, [AIMessageChunk(content='Ask consultant.', tool_call_chunks=[
        {'id': 'third', 'name': name, 'index': 0, 'args': '{}'}]), *chunks(reason='tool_calls')])
    state = initial(text_stream_protocol=PROTOCOL,
        tools=[{'type': 'function', 'function': {'name': name, 'parameters': {'type': 'object'}}}],
        execution={'consultations': 2})
    result = asyncio.run(events(state))
    assert not value(result)['result']['tool_calls']
    assert 'Лимит двух консультаций' in value(result)['result']['content']
    assert custom(result)[-2]['type'] == 'text_invalidated'
    assert value(result)['result']['usage_metadata'] == USAGE
