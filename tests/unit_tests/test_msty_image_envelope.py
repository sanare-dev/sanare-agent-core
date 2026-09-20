"""Official Luna image cap, offline only; no image decode/fetch/inference."""
import asyncio
from copy import deepcopy

import pytest
from langchain_core.messages import HumanMessage, ToolMessage
from langchain_openai import ChatOpenAI

from deep_agent import msty, msty_models as models, msty_compaction as compact
from tests.unit_tests.test_msty_compaction import initial, install


def image(detail=None, url='https://example.invalid/fixture.png'):
    spec = {'url': url}
    if detail is not None:
        spec['detail'] = detail
    return {'type': 'image_url', 'image_url': spec}


@pytest.mark.parametrize('detail,expected', [('low', 437), ('high', 3129),
    ('original', 36129), ('auto', 36129), (None, 36129)])
def test_official_patch_cap_multiplier_rounding_and_framing(detail, expected):
    source = [{'role': 'user', 'content': [{'type': 'text', 'text': 'Keep this owner constraint'}, image(detail)]}]
    original = deepcopy(source)
    counting, envelope = models._image_counting_projection('luna', source)
    assert envelope == expected
    assert source == original
    assert counting[0]['content'][0] == source[0]['content'][0]
    assert counting[0]['content'][1]['image_url']['url'] == '[image content counted separately]'
    assert models.count_method('luna', source) == 'tiktoken-image-envelope-v1'


def test_multiple_images_are_summed_without_touching_text_or_tools():
    source = [{'role': 'user', 'content': [image('low'), image('high'), image()]}]
    assert models._image_counting_projection('luna', source)[1] == 437 + 3129 + 36129
    assert models.count_method('luna', [HumanMessage(content='text')]) == 'tiktoken-admission-v1'


@pytest.mark.parametrize('bad', [image('xhigh'), image(True), {'type': 'image_url', 'image_url': 'https://example.invalid/a.png'},
    {'type': 'image_url', 'image_url': {'url': 'https://example.invalid/a.png', 'unknown': 'hidden'}},
    {'type': 'image_url', 'image_url': {'url': 'https://example.invalid/a.png'}, 'extra': 'hidden'},
    image(url='file:///private/example.png'), image(url='data:image/svg+xml;base64,AAAA')])
def test_unsupported_or_ambiguous_images_fail_closed(bad):
    with pytest.raises(models.ModelAdapterError):
        models._image_counting_projection('luna', [{'role': 'user', 'content': [bad]}])


def test_image_count_and_unsupported_provider_are_bounded():
    with pytest.raises(models.ModelAdapterError, match='32'):
        models._image_counting_projection('luna', [{'role': 'user', 'content': [image('low')] * 33}])
    with pytest.raises(models.ModelAdapterError):
        models._image_counting_projection('deepseek', [{'role': 'user', 'content': [image()]}])
    with pytest.raises(models.ModelAdapterError, match='результате инструмента'):
        models._image_counting_projection('luna', [{'role': 'tool', 'content': [image()]}])


def test_base64_removed_only_from_counting_and_original_sdk_payload_preserved():
    uri = 'data:image/png;base64,' + 'ABCD' * 100000
    history = [HumanMessage(content=[{'type': 'text', 'text': 'Instruction stays.'}, image(url=uri)])]
    prepared = models.prepare_messages('luna', history, [])
    small = [HumanMessage(content=[{'type': 'text', 'text': 'Instruction stays.'}, image(url='data:image/png;base64,AAAA')])]
    count = asyncio.run(models.count_input('luna', object(), prepared, []))
    assert count == asyncio.run(models.count_input('luna', object(), small, []))
    assert 36000 < count < 50000  # Not base64 chars→tokens.
    model = ChatOpenAI(model='gpt-5.6-luna', api_key='offline-test-not-a-secret',
                       use_responses_api=False, reasoning_effort='none', store=False)
    schema = [{'type': 'function', 'function': {'name': 'read', 'parameters': {'type': 'object'}}}]
    payload = model._get_request_payload(prepared, tools=deepcopy(schema))
    assert payload['messages'][0]['content'][1]['image_url']['url'] == uri
    assert payload['messages'][0]['content'][0]['text'] == 'Instruction stays.'
    assert payload['tools'] == schema
    assert 'input' not in payload


def test_bound_user_image_allowed_one_generation_and_method_is_not_exact(monkeypatch):
    counter = models.count_input
    seen = install(monkeypatch, ['Synthetic image answer.'], [])
    monkeypatch.setattr(models, 'count_input', counter)
    block = image('high')
    state = initial(messages=[{'role': 'user', 'content': [{'type': 'text', 'text': 'Read only.'}, block]}])
    result = asyncio.run(msty.graph.ainvoke(state))
    assert len(seen['requests']) == 1
    assert seen['requests'][0][-1].content[-1] == block
    assert result['context_budget_check']['method'] == 'tiktoken-image-envelope-v1'
    assert result['execution']['status'] == 'answered'


def test_total_text_plus_images_limit_rejects_before_generation(monkeypatch):
    counter = models.count_input
    seen = install(monkeypatch, [], [])
    monkeypatch.setattr(models, 'count_input', counter)
    state = initial(messages=[{'role': 'user', 'content': [image()] * 5}])
    result = asyncio.run(msty.graph.ainvoke(state))
    assert not seen['requests']
    assert result['context_budget_check']['input_tokens'] > 180000
    assert result['execution']['status'] == 'blocked'


def test_binary_tool_bundles_are_never_serialized_into_text_summaries():
    messages = []
    for i in range(4):
        messages.extend([{'role': 'assistant', 'content': '', 'tool_calls': [
            {'id': str(i), 'type': 'function', 'function': {'name': 'screenshot', 'arguments': '{}'}}]},
            {'role': 'tool', 'tool_call_id': str(i), 'content': [image(url='data:image/png;base64,' + 'A' * 20000)]}])
    assert compact.make_plan({'messages': messages}) is None
