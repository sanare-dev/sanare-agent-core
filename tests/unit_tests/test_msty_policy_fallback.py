"""Offline: отказ политики провайдера (OpenAI `invalid_prompt`) до генерации.

Живой инцидент 2026-09-23 (run 01a0cf13): gpt-6-luna вернула 400 invalid_prompt,
граф упал, владелец увидел только «Ошибка выполнения LangGraph». Здесь реальный
граф/checkpoint и адаптер; заменены только конструирование и вызов провайдера.
"""
import asyncio
from copy import deepcopy

import pytest
from langchain_core.messages import AIMessage

from deep_agent import msty, msty_execution as execution, msty_models, msty_stream, msty_taxonomy


USAGE = {'input_tokens': 100, 'output_tokens': 10, 'total_tokens': 110}
IMAGE = {'type': 'image_url', 'image_url': {'url': 'data:image/png;base64,iVBORw0KGgo='}}


class PolicyRejected(Exception):
    """Форма openai.BadRequestError: status_code, code, body — без текста промпта."""
    def __init__(self, code='invalid_prompt', status=400, body=None):
        super().__init__('Error code: 400 - flagged')
        self.status_code, self.code = status, code
        self.body = body if body is not None else {'code': code, 'type': 'invalid_request_error'}


@pytest.fixture(autouse=True)
def offline_defaults(monkeypatch):
    monkeypatch.delenv('MSTY_MODEL_PROFILE', raising=False)
    monkeypatch.setenv('LANGSMITH_TRACING', 'false')
    monkeypatch.setenv('LANGCHAIN_TRACING_V2', 'false')


def initial(content='Что делают мои агенты?', **overrides):
    return {'messages': [{'role': 'user', 'content': content}],
            'tools': [], 'max_tokens': 64, 'tool_choice': 'auto',
            'result': {}, 'context_budget': None, 'context_budget_check': None,
            'execution_protocol': execution.PROTOCOL, 'execution': {}, **overrides}


def providers(monkeypatch, behaviour):
    seen = {'created': [], 'invocations': []}

    class Model:
        def __init__(self, profile):
            self.profile = profile

        def bind_tools(self, tools, **kwargs):
            return self

        async def ainvoke(self, messages):
            seen['invocations'].append((self.profile, deepcopy(messages)))
            outcome = behaviour[self.profile]
            if isinstance(outcome, Exception):
                raise outcome
            return deepcopy(outcome)

    def construct(profile, max_tokens):
        seen['created'].append(profile)
        return Model(profile)

    monkeypatch.setattr(msty_models, 'make_model', construct)
    return seen


def test_luna_policy_rejection_falls_back_to_deepseek_once(monkeypatch):
    seen = providers(monkeypatch, {
        'luna': PolicyRejected(),
        'deepseek': AIMessage(content='Агенты: учёт, заказы.', usage_metadata=deepcopy(USAGE))})
    result = asyncio.run(msty.graph.ainvoke(initial()))
    assert seen['created'] == ['luna', 'deepseek']
    meta = result['result']['response_metadata']
    assert meta['msty_model_profile'] == 'deepseek'
    assert meta[msty.POLICY_FALLBACK_KEY] == {
        'version': 1, 'from': 'luna', 'to': 'deepseek', 'reason': 'invalid_prompt'}
    assert result['result']['content'] == 'Агенты: учёт, заказы.'
    assert result['result']['usage_metadata'] == USAGE
    assert result['execution']['status'] == 'answered'
    # Резервная модель знает, почему отвечает она.
    policy = seen['invocations'][1][1][0].content
    policy = policy if isinstance(policy, str) else ''.join(b.get('text', '') for b in policy)
    assert 'отвечаешь ты как резервная модель' in policy
    assert '↪ Ответ резервной модели DeepSeek' in policy


def test_fallback_replaces_images_with_explicit_text_not_silent_drop(monkeypatch):
    seen = providers(monkeypatch, {
        'luna': PolicyRejected(),
        'deepseek': AIMessage(content='Изображение мне не видно.', usage_metadata=deepcopy(USAGE))})
    state = initial(content=[{'type': 'text', 'text': 'Посмотри скрин'}, IMAGE])
    result = asyncio.run(msty.graph.ainvoke(state))
    assert result['result']['response_metadata']['msty_model_profile'] == 'deepseek'
    luna_user = seen['invocations'][0][1][-1].content
    assert any(isinstance(b, dict) and b.get('type') == 'image_url' for b in luna_user)
    deepseek_user = seen['invocations'][1][1][-1].content
    blocks = deepseek_user if isinstance(deepseek_user, list) else [{'type': 'text', 'text': deepseek_user}]
    assert not any(isinstance(b, dict) and b.get('type') in ('image_url', 'image') for b in blocks)
    assert 'Изображение не передано резервной модели' in str(deepseek_user)
    assert 'Посмотри скрин' in str(deepseek_user)


def test_double_rejection_is_an_honest_blocked_answer_not_a_graph_error(monkeypatch):
    seen = providers(monkeypatch, {'luna': PolicyRejected(), 'deepseek': PolicyRejected()})
    result = asyncio.run(msty.graph.ainvoke(initial()))
    assert seen['created'] == ['luna', 'deepseek']
    content = result['result']['content']
    assert '`invalid_prompt`' in content and 'расхода нет' in content
    assert result['result']['tool_calls'] == []
    assert result['result']['usage_metadata'] == {'input_tokens': 0, 'output_tokens': 0, 'total_tokens': 0}
    assert result['result']['response_metadata']['msty_generation'] == 'not_started'
    assert result['execution']['status'] == 'blocked'


def test_deepseek_lead_has_no_fallback_and_is_not_retried(monkeypatch):
    seen = providers(monkeypatch, {'deepseek': PolicyRejected()})
    result = asyncio.run(msty.graph.ainvoke(initial(lead_profile='deepseek')))
    assert seen['created'] == ['deepseek']
    assert '`invalid_prompt`' in result['result']['content']
    assert result['execution']['status'] == 'blocked'


def test_other_bad_request_still_fails_closed_without_fallback(monkeypatch):
    seen = providers(monkeypatch, {'luna': PolicyRejected(code='context_length_exceeded')})
    with pytest.raises(PolicyRejected):
        asyncio.run(msty.graph.ainvoke(initial()))
    assert seen['created'] == ['luna']


@pytest.mark.parametrize('error, expected', [
    (PolicyRejected(), 'invalid_prompt'),
    (PolicyRejected(code='content_policy_violation'), 'content_policy_violation'),
    (PolicyRejected(code=None, body={'error': {'code': 'invalid_prompt'}}), 'invalid_prompt'),
    (PolicyRejected(code='invalid_prompt', status=429), None),
    (PolicyRejected(code='context_length_exceeded'), None),
    (ValueError('invalid_prompt'), None),
])
def test_policy_rejection_code_reads_only_status_and_code(error, expected):
    assert msty_taxonomy.policy_rejection_code(error) == expected


def test_stream_failure_carries_policy_code_before_first_fragment():
    class Model:
        async def astream(self, messages, **kwargs):
            raise PolicyRejected()
            yield  # pragma: no cover

    stream = msty_stream.TextStream({})
    with pytest.raises(msty_stream.StreamFailure) as caught:
        asyncio.run(stream.invoke(Model(), []))
    assert caught.value.policy_rejection == 'invalid_prompt'
    assert caught.value.emitted is False
