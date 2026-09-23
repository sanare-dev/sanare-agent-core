"""Offline adapter/protocol tests: fake keys, mock transports, no provider calls."""
import asyncio
from copy import deepcopy
import json

import httpx
import pytest
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage
from langchain_openai import ChatOpenAI

from deep_agent import msty_models as adapter


TOOLS = [{'type': 'function', 'function': {'name': 'read_fixture', 'parameters': {
    'type': 'object', 'properties': {'path': {'type': 'string'}}, 'required': ['path']}}}]


@pytest.fixture(autouse=True)
def fake_environment(monkeypatch):
    for name in ('OPENAI_API_KEY', 'DEEPSEEK_API_KEY', 'ANTHROPIC_API_KEY'):
        monkeypatch.setenv(name, 'synthetic-offline-not-a-real-key')
    monkeypatch.setenv('LANGSMITH_TRACING', 'false')
    monkeypatch.setenv('LANGCHAIN_TRACING_V2', 'false')


@pytest.mark.parametrize('profile,model,endpoint', [
    ('luna', 'gpt-6-luna', 'https://api.openai.com/v1'),
    ('deepseek', 'deepseek-flash', 'https://api.deepseek.com/v1'),
    ('sonnet', 'claude-sonnet-4-6', 'https://api.anthropic.com'),
])
def test_fixed_model_endpoint_and_no_retries(profile, model, endpoint, monkeypatch):
    monkeypatch.setenv('OPENAI_BASE_URL', 'https://not-a-provider.invalid')
    monkeypatch.setenv('ANTHROPIC_BASE_URL', 'https://not-a-provider.invalid')
    obj = adapter.make_model(profile, 123)
    assert obj.model == model if profile == 'sonnet' else obj.model_name == model
    assert obj.max_retries == 0
    assert obj.default_request_timeout == 120 if profile == 'sonnet' else obj.request_timeout == 120
    assert obj.anthropic_api_url == endpoint if profile == 'sonnet' else obj.openai_api_base == endpoint
    assert obj.temperature is None
    if profile != 'sonnet':
        assert obj.use_responses_api is False
    if profile == 'luna':
        assert obj.reasoning_effort == 'none'
    if profile == 'deepseek':
        assert obj.extra_body == {'thinking': {'type': 'disabled'}, 'max_tokens': 123}


@pytest.mark.parametrize('profile', ['unknown', 'gpt-6-astra', 'https://evil.invalid', None, {}, ['luna']])
def test_unknown_profile_rejected_without_sdk(profile, monkeypatch):
    monkeypatch.setattr(adapter, 'ChatOpenAI', lambda **kwargs: pytest.fail('SDK must not be called'))
    with pytest.raises(adapter.ModelAdapterError, match='профиль'):
        adapter.make_model(profile)


@pytest.mark.parametrize('limit', [0, -1, 8193, True, 2.5, '200'])
def test_output_bound_is_strict(limit):
    with pytest.raises(adapter.ModelAdapterError):
        adapter.make_model('luna', limit)


def test_missing_provider_key_never_falls_back(monkeypatch):
    monkeypatch.delenv('DEEPSEEK_API_KEY')
    with pytest.raises(adapter.ModelAdapterError, match='Ключ'):
        adapter.make_model('deepseek')


def test_profiles_are_immutable():
    with pytest.raises(TypeError):
        adapter.PROFILES['evil'] = adapter.PROFILES['luna']
    with pytest.raises(AttributeError):
        adapter.PROFILES['luna'].endpoint = 'https://evil.invalid'


def test_anthropic_blocks_to_openai_keep_text_calls_results_and_do_not_mutate():
    messages = [SystemMessage(content=[{'type': 'text', 'text': 'Policy',
                                       'cache_control': {'type': 'ephemeral'}}]),
                HumanMessage(content='Task'),
                AIMessage(content=[{'type': 'text', 'text': 'Read it'},
                    {'type': 'tool_use', 'id': 'call_a', 'name': 'read_fixture',
                     'input': {'path': '/synthetic/a'}}]),
                HumanMessage(content=[{'type': 'tool_result', 'tool_use_id': 'call_a',
                                       'content': 'result 17'}])]
    original = deepcopy(messages)
    result = adapter.prepare_messages('luna', messages, TOOLS)
    assert result[0].content == 'Policy'
    assert result[1].content == 'Task'
    assert result[2].content == 'Read it'
    assert result[2].tool_calls == [{'name': 'read_fixture', 'args': {'path': '/synthetic/a'},
                                    'id': 'call_a', 'type': 'tool_call'}]
    assert isinstance(result[3], ToolMessage)
    assert result[3].tool_call_id == 'call_a' and result[3].content == 'result 17'
    assert messages == original


def test_duplicate_native_and_content_call_deduplicates_exactly_once():
    call = {'name': 'read_fixture', 'args': {'path': 'a'}, 'id': 'a', 'type': 'tool_call'}
    message = AIMessage(content=[{'type': 'tool_use', 'id': 'a', 'name': 'read_fixture', 'input': {'path': 'a'}}],
                        tool_calls=[call])
    result = adapter.prepare_messages('deepseek', [message], TOOLS)
    assert result[0].tool_calls == [call]


def test_conflicting_same_id_call_rejected():
    message = AIMessage(content=[{'type': 'tool_use', 'id': 'a', 'name': 'read_fixture', 'input': {'path': 'b'}}],
                        tool_calls=[{'name': 'read_fixture', 'args': {'path': 'a'}, 'id': 'a'}])
    with pytest.raises(adapter.ModelAdapterError, match='Конфликт'):
        adapter.prepare_messages('luna', [message], TOOLS)


def test_malformed_tool_call_in_plain_text_message_rejected():
    message = AIMessage(content='text', invalid_tool_calls=[
        {'name': 'read_fixture', 'args': '{broken', 'id': 'a', 'error': 'invalid'}])
    with pytest.raises(adapter.ModelAdapterError, match='некорректный'):
        adapter.prepare_messages('luna', [message], TOOLS)


def test_tool_result_error_flag_remains_visible():
    message = HumanMessage(content=[{'type': 'tool_result', 'tool_use_id': 'a',
                                     'content': 'Permission denied', 'is_error': True}])
    result = adapter.prepare_messages('luna', [message], TOOLS)
    assert json.loads(result[0].content) == {'is_error': True, 'content': 'Permission denied'}


def test_thinking_text_preserved_without_signature():
    message = AIMessage(content=[{'type': 'thinking', 'thinking': 'Historical reasoning.',
                                 'signature': 'not-portable'}, {'type': 'text', 'text': 'Answer'}])
    result = adapter.prepare_messages('luna', [message], [])
    assert result[0].content == 'Historical reasoning.\nAnswer'


@pytest.mark.parametrize('block', [
    {'type': 'redacted_thinking', 'data': 'synthetic'},
    {'type': 'compaction', 'content': 'synthetic'},
    {'type': 'file', 'file': {'file_data': 'synthetic'}},
    {'type': 'text'}, {'type': 'mystery', 'text': 'SECRET_SENTINEL'},
])
def test_unsupported_history_fails_without_echoing_content(block):
    with pytest.raises(adapter.ModelAdapterError) as exc:
        adapter.prepare_messages('luna', [AIMessage(content=[block])], [])
    assert 'SECRET_SENTINEL' not in str(exc.value)


def test_mixed_human_text_and_tool_result_not_reordered():
    with pytest.raises(adapter.ModelAdapterError, match='Смешанные'):
        adapter.prepare_messages('luna', [HumanMessage(content=[
            {'type': 'text', 'text': 'new instruction'},
            {'type': 'tool_result', 'tool_use_id': 'a', 'content': 'data'}])], [])


@pytest.mark.parametrize('profile', ['luna', 'deepseek'])
@pytest.mark.parametrize('block', [
    {'type': 'image_url', 'image_url': {'url': 'https://example.invalid/synthetic.png', 'detail': 'low'}},
    {'type': 'image', 'source': {'type': 'base64', 'media_type': 'image/png', 'data': 'AAA='}},
])
def test_images_preserved_with_luna_envelope_other_profiles_fail_closed(profile, block):
    history = [HumanMessage(content=[{'type': 'text', 'text': 'Look'}, block])]
    result = adapter.prepare_messages(profile, history, [])
    assert result[0].content[0] == {'type': 'text', 'text': 'Look'}
    assert result[0].content[1]['type'] == 'image_url'
    if profile == 'luna':
        original = deepcopy(result)
        assert asyncio.run(adapter.count_input(profile, object(), result, [])) > 4000
        assert result == original
    else:
        with pytest.raises(adapter.ModelAdapterError, match='изображениями'):
            asyncio.run(adapter.count_input(profile, object(), result, []))


def test_sonnet_history_is_preserved_copy():
    history = [SystemMessage(content=[{'type': 'text', 'text': 'a', 'cache_control': {'type': 'ephemeral'}}])]
    prepared = adapter.prepare_messages('sonnet', history, [])
    assert prepared == history and prepared is not history
    assert prepared[0] is not history[0]


@pytest.mark.parametrize('profile,choice,expected', [
    ('luna', {'type': 'none'}, 'none'), ('deepseek', 'any', 'required'),
    ('sonnet', 'none', {'type': 'none'}), ('sonnet', 'required', 'any'),
    ('luna', {'type': 'function', 'function': {'name': 'read_fixture'}}, 'read_fixture'),
])
def test_tool_choice_translation_and_schema_preservation(profile, choice, expected):
    class Model:
        def bind_tools(self, tools, tool_choice):
            return tools, tool_choice
    tools = deepcopy(TOOLS)
    tools[0]['cache_control'] = {'type': 'ephemeral'}
    tools[0]['function']['parameters']['properties']['cache_control'] = {'type': 'string'}
    prepared, actual = adapter.bind_tools(profile, Model(), tools, choice)
    assert actual == expected
    assert prepared[0]['function']['parameters'] == tools[0]['function']['parameters']
    assert ('cache_control' in prepared[0]) is (profile == 'sonnet')
    assert tools[0]['cache_control'] == {'type': 'ephemeral'}


@pytest.mark.parametrize('choice', ['missing_tool', True, {'type': 'web_search'}, {'type': 'function', 'function': {'name': 'missing'}}])
def test_unavailable_tool_cannot_be_requested(choice):
    with pytest.raises(adapter.ModelAdapterError):
        adapter.bind_tools('luna', object(), TOOLS, choice)


def test_no_tools_is_not_fabricated():
    model = object()
    assert adapter.bind_tools('luna', model, [], 'none') is model
    with pytest.raises(adapter.ModelAdapterError):
        adapter.bind_tools('luna', model, [], 'required')


def test_duplicate_schema_rejected():
    with pytest.raises(adapter.ModelAdapterError):
        adapter.prepare_messages('luna', [], TOOLS * 2)


@pytest.mark.parametrize('profile', ['luna', 'deepseek'])
def test_conservative_text_charge_includes_schemas_and_utf8_not_chars_div4(profile):
    messages = [SystemMessage(content='p'), HumanMessage(content='Щ'*100)]
    def counter(ms, ts):
        return asyncio.run(adapter.count_input(profile, object(), ms, ts))
    plain = counter(messages, [])
    assert plain > len(('Щ'*100).encode())
    assert counter(messages, TOOLS) > plain + (512 if profile == 'luna' else 2048)
    assert counter([HumanMessage(content='a'*100)], []) < counter([HumanMessage(content='Щ'*100)], [])
    if profile == 'deepseek':
        assert counter([HumanMessage(content='a'*200000)], []) > 180000


def test_luna_long_text_not_rejected_merely_for_crossing_byte_trigger():
    text = 'synthetic scoped request. ' * 10000
    assert len(text.encode()) > 200000
    count = asyncio.run(adapter.count_input('luna', object(), [HumanMessage(content=text)], TOOLS))
    assert 20000 < count < 180000
    assert adapter.COUNT_METHODS['luna'] == 'tiktoken-admission-v1'


def test_unknown_tokenizer_fails_closed_without_family_fallback(monkeypatch):
    # Токенизатор задан явно (o200k_base); его сбой закрывает допуск без утечки.
    def unknown(name):
        assert name == 'o200k_base'
        raise KeyError('SECRET_SENTINEL')
    monkeypatch.setattr(adapter.tiktoken, 'get_encoding', unknown)
    with pytest.raises(adapter.ModelAdapterError) as exc:
        asyncio.run(adapter.count_input('luna', object(), [HumanMessage(content='x')], []))
    assert 'SECRET_SENTINEL' not in str(exc.value)


def test_sonnet_counter_passes_the_block_system_to_real_counter_interface():
    history = [SystemMessage(content=[{'type': 'text', 'text': 'p', 'cache_control': {'type': 'ephemeral'}}]), HumanMessage(content='q')]
    class Model:
        def get_num_tokens_from_messages(self, messages, tools, **kwargs):
            assert messages == history and tools == TOOLS
            assert kwargs['system'] == history[0].content
            assert kwargs['timeout'] == 20
            return 67
    assert asyncio.run(adapter.count_input('sonnet', Model(), history, TOOLS)) == 67


@pytest.mark.parametrize('value', [-1, True, '5', None])
def test_invalid_exact_counter_fails_closed(value):
    class Model:
        def get_num_tokens_from_messages(self, *args, **kwargs):
            return value
    with pytest.raises(adapter.ModelAdapterError):
        asyncio.run(adapter.count_input('sonnet', Model(), [HumanMessage(content='a')], []))


def test_exact_counter_failure_is_content_free():
    class Model:
        def get_num_tokens_from_messages(self, *args, **kwargs):
            raise RuntimeError('SECRET_SENTINEL')
    with pytest.raises(adapter.ModelAdapterError) as exc:
        asyncio.run(adapter.count_input('sonnet', Model(), [HumanMessage(content='a')], []))
    assert 'SECRET_SENTINEL' not in str(exc.value)


@pytest.mark.parametrize('profile', ['luna', 'deepseek', 'sonnet'])
def test_stamp_preserves_usage_and_canonical_identity(profile):
    name = adapter.PROFILES[profile].model
    result = AIMessage(content='a', response_metadata={'model_name': name + '-2026-09-01', 'finish_reason': 'stop'},
                       usage_metadata={'input_tokens': 42, 'output_tokens': 3, 'total_tokens': 45,
                                       'input_token_details': {'cache_read': 30}})
    stamped = adapter.stamp_usage(profile, result)
    assert stamped.response_metadata['model_name'] == name
    assert stamped.response_metadata['provider_model_name'] == name + '-2026-09-01'
    assert stamped.response_metadata['msty_model_profile'] == profile
    assert stamped.usage_metadata == result.usage_metadata
    assert result.response_metadata['model_name'].endswith('-2026-09-01')


def test_stamp_never_invents_zero_usage_and_rejects_wrong_identity():
    message = AIMessage(content='a')
    assert adapter.stamp_usage('luna', message).usage_metadata is None
    with pytest.raises(adapter.ModelAdapterError):
        adapter.stamp_usage('luna', AIMessage(content='a', response_metadata={'model_name': 'gpt-6-astra'}))


@pytest.mark.parametrize('raw', [{}, {'prompt_tokens': 1},
    {'prompt_tokens': True, 'completion_tokens': 1, 'total_tokens': 2},
    {'prompt_tokens': 3, 'completion_tokens': 1, 'total_tokens': 9},
    {'prompt_tokens': 3, 'completion_tokens': 1, 'total_tokens': 4,
     'prompt_tokens_details': {'cached_tokens': 4}},
    {'prompt_tokens': 3, 'completion_tokens': 1, 'total_tokens': 4,
     'prompt_tokens_details': 'bad'},
])
def test_sdk_zero_fill_does_not_turn_missing_or_invalid_usage_into_free_call(raw):
    message = AIMessage(content='a', response_metadata={'token_usage': raw},
                        usage_metadata={'input_tokens': 0, 'output_tokens': 0, 'total_tokens': 0})
    assert adapter.stamp_usage('luna', message).usage_metadata is None


def test_cache_write_and_deepseek_top_level_hits_not_discarded():
    raw = {'prompt_tokens': 100, 'completion_tokens': 4, 'total_tokens': 104,
           'prompt_tokens_details': {'cached_tokens': 40, 'cache_write_tokens': 20},
           'completion_tokens_details': {'reasoning_tokens': 1}}
    message = AIMessage(content='a', response_metadata={'token_usage': raw})
    usage = adapter.stamp_usage('luna', message).usage_metadata
    assert usage['input_token_details'] == {'cache_read': 40, 'cache_creation': 20}
    assert usage['output_token_details'] == {'reasoning': 1}
    deepseek = {'prompt_tokens': 100, 'completion_tokens': 4, 'total_tokens': 104,
                'prompt_cache_hit_tokens': 70, 'prompt_cache_miss_tokens': 30}
    usage = adapter.stamp_usage('deepseek', AIMessage(content='a', response_metadata={'token_usage': deepseek})).usage_metadata
    assert usage['input_token_details']['cache_read'] == 70


def test_conflicting_cache_read_counts_not_silently_selected():
    raw = {'prompt_tokens': 100, 'completion_tokens': 4, 'total_tokens': 104,
           'prompt_tokens_details': {'cached_tokens': 40}, 'prompt_cache_hit_tokens': 60}
    assert adapter.stamp_usage('deepseek', AIMessage(content='a', response_metadata={'token_usage': raw})).usage_metadata is None


@pytest.mark.parametrize('profile', ['luna', 'deepseek'])
def test_actual_sdk_mock_http_payload_and_tool_result_roundtrip(profile, monkeypatch):
    captured = []
    def handler(request):
        captured.append((str(request.url), json.loads(request.content)))
        return httpx.Response(200, json={'id': 'synthetic', 'object': 'chat.completion', 'created': 1,
            'model': adapter.PROFILES[profile].model,
            'choices': [{'index': 0, 'finish_reason': 'stop', 'message': {'role': 'assistant', 'content': '17'}}],
            'usage': {'prompt_tokens': 55, 'completion_tokens': 2, 'total_tokens': 57,
                      'prompt_tokens_details': {'cached_tokens': 20}}})
    transport = httpx.MockTransport(handler)
    sync_client, async_client = httpx.Client(transport=transport), httpx.AsyncClient(transport=transport)
    monkeypatch.setattr(adapter, 'ChatOpenAI', lambda **kwargs: ChatOpenAI(
        **kwargs, http_client=sync_client, http_async_client=async_client))
    model = adapter.make_model(profile, 40)
    history = adapter.prepare_messages(profile, [HumanMessage(content='read'),
        AIMessage(content=[{'type': 'text', 'text': 'reading'}, {'type': 'tool_use',
            'id': 'a', 'name': 'read_fixture', 'input': {'path': '/synthetic/a'}}]),
        ToolMessage(content='17', tool_call_id='a')], TOOLS)
    async def execute():
        try:
            return await adapter.bind_tools(profile, model, TOOLS, 'none').ainvoke(history)
        finally:
            await async_client.aclose()
            sync_client.close()
    result = adapter.stamp_usage(profile, asyncio.run(execute()))
    assert len(captured) == 1
    url, payload = captured[0]
    assert url == adapter.PROFILES[profile].endpoint + '/chat/completions'
    assert payload['model'] == adapter.PROFILES[profile].model
    assert payload['tool_choice'] == 'none' and payload['tools'] == TOOLS
    assert payload['messages'][1]['tool_calls'][0]['id'] == 'a'
    assert payload['messages'][2]['content'] == '17'
    assert 'temperature' not in payload
    if profile == 'luna':
        assert payload['reasoning_effort'] == 'none' and payload['max_completion_tokens'] == 40
        assert 'thinking' not in payload
    else:
        assert payload['thinking'] == {'type': 'disabled'} and payload['max_tokens'] == 40
        assert 'max_completion_tokens' not in payload
        assert 'reasoning_effort' not in payload
    assert result.usage_metadata is not None, result.response_metadata.get('token_usage')
    assert result.usage_metadata['input_token_details']['cache_read'] == 20
    assert result.usage_metadata['input_tokens'] == 55
