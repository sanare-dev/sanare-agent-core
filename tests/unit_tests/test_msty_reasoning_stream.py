"""Live reasoning summary relay (msty-reasoning-delta-v1): offline, no keys or inference."""
import asyncio
from copy import deepcopy

import pytest
from langchain_core.messages import AIMessageChunk

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
