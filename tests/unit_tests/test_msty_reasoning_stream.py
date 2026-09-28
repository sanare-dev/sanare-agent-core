"""Live reasoning summary relay (msty-reasoning-delta-v1): offline, no keys or inference."""
import asyncio
from copy import deepcopy
import json as _json
import pathlib as _pathlib

import httpx
import pytest
from langchain_core.messages import AIMessage, AIMessageChunk
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.store.memory import InMemoryStore

from deep_agent import msty_models, msty_native, msty_stream
from tests.unit_tests.test_msty_compaction import initial, USAGE
from tests.unit_tests.test_msty_stream import install, events, custom, value

REASONING = 'msty-reasoning-delta-v1'
TEXT = 'msty-text-delta-v1'
FAKE_KEY = 'sk-proj-' + 'A1b2C3d4E5f6G7h8I9j0K1l2'


def thought(text, index=0):
    """One Responses API summary delta as langchain-openai 1.1.11 converts it."""
    return AIMessageChunk(content=[{'type': 'reasoning', 'index': index,
                                    'summary': [{'index': 0, 'type': 'summary_text', 'text': text}]}])


def finish(text='Готово.'):
    return [AIMessageChunk(content=[{'type': 'text', 'text': text, 'index': 1}]),
            AIMessageChunk(content='', response_metadata={'finish_reason': 'stop', 'model_name': 'gpt-6-luna'},
                           usage_metadata=deepcopy(USAGE))]


def deltas(items):
    return [e for e in custom(items) if e.get('type') == 'reasoning_delta']


def test_summary_streams_while_answer_text_stays_withheld(monkeypatch):
    """Native harness: text_stream_protocol is None, yet thoughts reach the window."""
    seen = install(monkeypatch, [thought('Сначала проверю '), thought('счета.\n'), *finish()])
    result = asyncio.run(events(initial(reasoning_stream_protocol=REASONING)))
    kinds = [e['type'] for e in custom(result)]
    assert 'text_delta' not in kinds and kinds[-1] == 'validated_result'
    assert ''.join(e['text'] for e in deltas(result)) == 'Сначала проверю счета.\n'
    assert [e['seq'] for e in deltas(result)] == list(range(len(deltas(result))))
    assert seen['astream'] == 1 and not seen['ainvoke']
    assert msty_stream.text_content(value(result)['result']['content']) == 'Готово.'


def test_summary_and_text_protocols_are_independent(monkeypatch):
    install(monkeypatch, [thought('План: '), thought('один шаг. '), *finish('Ответ')])
    result = asyncio.run(events(initial(reasoning_stream_protocol=REASONING, text_stream_protocol=TEXT)))
    kinds = [e['type'] for e in custom(result)]
    assert kinds.index('reasoning_delta') < kinds.index('text_delta')
    assert ''.join(e['text'] for e in custom(result) if e['type'] == 'text_delta') == 'Ответ'


def test_without_opt_in_no_reasoning_event_and_no_summary_request(monkeypatch):
    install(monkeypatch, [thought('секретная мысль '), *finish()])
    requested = []
    original = msty_models.make_model
    monkeypatch.setattr(msty_models, 'make_model', lambda *args: requested.append(args) or original(*args))
    result = asyncio.run(events(initial(text_stream_protocol=TEXT)))
    assert not deltas(result)
    assert requested and all(len(args) == 3 for args in requested)  # no summary flag


def test_credential_split_across_deltas_is_redacted_whole(monkeypatch):
    install(monkeypatch, [thought('Ключ ' + FAKE_KEY[:9]), thought(FAKE_KEY[9:] + ' не нужен.'), *finish()])
    result = asyncio.run(events(initial(reasoning_stream_protocol=REASONING)))
    shown = ''.join(e['text'] for e in deltas(result))
    assert FAKE_KEY not in shown and FAKE_KEY[9:] not in shown
    assert '[REDACTED_API_KEY]' in shown and shown.endswith('не нужен.')


def test_oversized_reasoning_stops_relay_but_answer_completes(monkeypatch):
    big = ('слово ' * 3000)
    install(monkeypatch, [*[thought(big) for _ in range(5)], *finish()])
    result = asyncio.run(events(initial(reasoning_stream_protocol=REASONING)))
    shown = sum(len(e['text'].encode()) for e in deltas(result))
    assert 0 < shown <= msty_stream.MAX_REASONING_BYTES
    assert msty_stream.text_content(value(result)['result']['content']) == 'Готово.'


def test_unknown_reasoning_protocol_rejected_before_generation(monkeypatch):
    seen = install(monkeypatch, [*finish()])
    result = asyncio.run(events(initial(reasoning_stream_protocol='msty-reasoning-delta-v9')))
    assert not seen['astream'] and not seen['ainvoke']
    assert 'потока рассуждений' in value(result)['result']['content']


@pytest.mark.parametrize('chunk,expected', [
    (thought('responses'), 'responses'),
    (AIMessageChunk(content=[{'type': 'reasoning', 'reasoning': 'v1 string'}]), 'v1 string'),
    (AIMessageChunk(content=[{'type': 'thinking', 'thinking': 'anthropic'}]), 'anthropic'),
    (AIMessageChunk(content='', additional_kwargs={'reasoning_content': 'deepseek/qwen'}), 'deepseek/qwen'),
    (AIMessageChunk(content='', additional_kwargs={'thought': 'gemini'}), 'gemini'),
    (AIMessageChunk(content='', additional_kwargs={'reasoning': {
        'type': 'reasoning', 'summary': [{'type': 'summary_text', 'text': 'v0 item'}]}}), 'v0 item'),
    # Encrypted / redacted reasoning is never text.
    (AIMessageChunk(content=[{'type': 'reasoning', 'summary': [], 'encrypted_content': 'gAAA'}]), ''),
    (AIMessageChunk(content=[{'type': 'redacted_thinking', 'data': 'opaque'}]), ''),
    (AIMessageChunk(content='plain answer'), ''),
])
def test_one_rule_for_every_provider_reasoning_field(chunk, expected):
    assert msty_stream.reasoning_text(chunk) == expected


def test_real_responses_converter_summary_delta_is_picked_up():
    from openai.types.responses import ResponseReasoningSummaryTextDeltaEvent
    from langchain_openai.chat_models.base import _convert_responses_chunk_to_generation_chunk
    event = ResponseReasoningSummaryTextDeltaEvent(
        type='response.reasoning_summary_text.delta', delta='Думаю над счетом', item_id='rs_1',
        output_index=0, summary_index=0, sequence_number=3, obfuscation='x')
    *_, generation = _convert_responses_chunk_to_generation_chunk(event, 0, 0, 0)
    assert msty_stream.reasoning_text(generation.message) == 'Думаю над счетом'


def test_luna_requests_summary_only_on_opt_in(monkeypatch):
    for name in ('OPENAI_API_KEY', 'DEEPSEEK_API_KEY'):
        monkeypatch.setenv(name, 'synthetic-offline-not-a-real-key')
    monkeypatch.setattr(msty_models.msty_gateway, 'enabled', lambda: False)
    assert msty_models.make_model('luna', 4096, 'high').reasoning == {'effort': 'high'}
    assert msty_models.make_model('luna', 4096, 'high', True).reasoning == {'effort': 'high', 'summary': 'auto'}
    # DeepSeek keeps thinking disabled: the flag adds no provider reasoning or cost.
    assert msty_models.make_model('deepseek', 4096, 'high', True).extra_body['thinking'] == {'type': 'disabled'}


def test_native_progress_names_tool_and_filters_plan(monkeypatch):
    sent = []
    monkeypatch.setattr(msty_native, 'get_stream_writer', lambda: sent.append)
    state = {'reasoning_stream_protocol': REASONING}
    msty_native._publish_progress(state, {'name': 'native_write_todos', 'args': {'todos': [
        {'content': 'Сверить счета ' + FAKE_KEY, 'status': 'in_progress'},
        {'content': 'x' * 500, 'status': 'odd'}, 'не словарь']}})
    msty_native._publish_progress(state, {'name': 'delegate', 'args': {'task': 'секретные аргументы'}})
    msty_native._publish_progress({}, {'name': 'delegate', 'args': {}})  # no opt-in: nothing
    assert sent[0]['type'] == 'step_progress' and sent[0]['tool'] == 'native_write_todos'
    assert sent[0]['plan'][0] == {'content': 'Сверить счета [REDACTED_API_KEY]', 'status': 'in_progress'}
    assert sent[0]['plan'][1]['status'] == 'pending' and len(sent[0]['plan'][1]['content']) == 200
    assert sent[1] == {'type': 'step_progress', 'version': 1, 'tool': 'delegate', 'status': 'start'}
    assert len(sent) == 2


# --- Real Responses API wire (prod regression 28.09.2026) -------------------
# Recorded live through the LangSmith Gateway (gpt-6-luna, effort=medium,
# summary=auto → "detailed"); encrypted reasoning blobs shortened. The first
# prod turn with the protocol on failed before any request: langchain-openai
# forwarded astream(stream_usage=True) into responses.create() → TypeError →
# «Поток модели не завершён». These tests drive the real ChatOpenAI + openai
# SDK parser over that exact byte stream.
FIXTURE = _pathlib.Path(__file__).with_name('fixtures_luna_responses_stream_20260928.sse')


def recorded_summary():
    text = ''
    for line in FIXTURE.read_text().splitlines():
        if line.startswith('data:'):
            event = _json.loads(line[5:])
            if event.get('type') == 'response.reasoning_summary_text.delta':
                text += event['delta']
    return text


def real_luna(monkeypatch, requests):
    """make_model's own ChatOpenAI, only the HTTP transport replaced."""
    monkeypatch.setenv('OPENAI_API_KEY', 'synthetic-offline-not-a-real-key')
    monkeypatch.setenv('MSTY_LLM_GATEWAY_ENABLED', '0')

    def handler(request):
        requests.append(_json.loads(request.content))
        return httpx.Response(200, headers={'content-type': 'text/event-stream'},
                              content=FIXTURE.read_bytes())
    original = msty_models.ChatOpenAI
    monkeypatch.setattr(msty_models, 'ChatOpenAI', lambda **kwargs: original(
        **{**kwargs, 'http_async_client': httpx.AsyncClient(transport=httpx.MockTransport(handler))}))


def test_real_responses_stream_relays_summary_without_stream_usage_kwarg(monkeypatch):
    requests, emitted = [], []
    real_luna(monkeypatch, requests)
    monkeypatch.setattr(msty_stream, 'get_stream_writer', lambda: emitted.append)
    model = msty_models.bind_tools('luna', msty_models.make_model('luna', 1024, 'medium', True), [], 'auto')
    stream = msty_stream.TextStream({'reasoning_stream_protocol': REASONING}, text=False)
    final = asyncio.run(stream.invoke(model, [{'role': 'user', 'content': 'Сколько будет 17*23?'}]))
    assert len(requests) == 1 and requests[0]['stream'] is True
    assert 'stream_usage' not in requests[0]
    assert requests[0]['reasoning'] == {'effort': 'medium', 'summary': 'auto'}
    assert msty_stream.text_content(final.content) == '391'
    assert final.usage_metadata['output_tokens'] == 27  # summary text is not billed output
    relayed = [e for e in emitted if e['type'] == 'reasoning_delta']
    assert ''.join(e['text'] for e in relayed) == recorded_summary()
    assert 1 < len(relayed) < 10  # coalesced, not one event per word (73 deltas)


def test_chat_completions_profiles_still_request_stream_usage():
    class Chat:
        use_responses_api = False
    class Bound:
        bound = Chat()
    assert not msty_stream.uses_responses_api(Bound())
    Chat.use_responses_api = True
    assert msty_stream.uses_responses_api(Bound())


def test_native_harness_real_responses_stream_with_protocol_on(monkeypatch):
    """The exact prod path: msty_native graph, Luna, protocol on, real wire."""
    requests = []
    real_luna(monkeypatch, requests)
    monkeypatch.setenv('MSTY_MODEL_PROFILE', 'luna')
    for name in ('LANGSMITH_TRACING', 'LANGCHAIN_TRACING', 'LANGCHAIN_TRACING_V2'):
        monkeypatch.setenv(name, 'false')

    async def count(*args):
        return 100
    monkeypatch.setattr(msty_models, 'count_input', count)
    from deep_agent import msty_execution, msty_native

    async def run():
        graph = msty_native.build_graph(checkpointer=InMemorySaver(), store=InMemoryStore())
        config = {'configurable': {'thread_id': 'real-responses-stream'}}
        value = {'messages': [{'role': 'user', 'content': 'Сколько будет 17*23?'}],
                 'tools': [], 'max_tokens': 1024, 'tool_choice': 'auto', 'result': {},
                 'context_budget': None, 'context_budget_check': None,
                 'execution_protocol': msty_execution.PROTOCOL, 'execution': {},
                 'reasoning_stream_protocol': REASONING}
        items = [item async for item in graph.astream(value, config, stream_mode=['custom', 'values'],
                                                        durability='sync')]
        return items, await graph.aget_state(config)
    items, state = asyncio.run(run())
    events = custom(items)
    kinds = [e['type'] for e in events]
    assert 'text_delta' not in kinds  # native harness still withholds answer text
    assert kinds[-1] == 'validated_result'
    assert ''.join(e['text'] for e in events if e['type'] == 'reasoning_delta') == recorded_summary()
    result = events[-1]['message']
    assert msty_stream.text_content(result['content']) == '391', result['content']
    assert not result['response_metadata'].get('msty_blocked')
    assert state.values['execution']['status'] == 'answered'
    assert 'stream_usage' not in requests[0] and requests[0]['reasoning']['summary'] == 'auto'
    assert isinstance(AIMessage.model_validate(result), AIMessage)
