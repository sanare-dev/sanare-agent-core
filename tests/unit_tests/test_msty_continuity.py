"""Offline policy/delivery contracts, not a claim of model behavioral accuracy."""
import asyncio

import pytest
from langchain_core.messages import AIMessage

from deep_agent import msty


TOOLS = [{'type': 'function', 'function': {
    'name': 'read_test_file',
    'parameters': {'type': 'object', 'properties': {'path': {'type': 'string'}},
                   'required': ['path'], 'additionalProperties': False},
}}]


@pytest.mark.parametrize('tools', [[], TOOLS])
def test_continuity_rule_is_present_with_or_without_client_tools(tools):
    policy = msty.policy_for_tools(tools)
    assert 'MSTY_TASK_CONTINUITY_V1' in policy
    assert 'не новая задача и не новое разрешение' in policy
    assert 'не проси «дай команду»' in policy
    assert 'только один шаг' in policy
    assert 'не увеличивай лимиты' in policy
    assert 'только план' in policy


@pytest.mark.parametrize('choice', ['auto', 'none'])
def test_every_callback_gets_continuity_without_forcing_actions_or_retry(monkeypatch, choice):
    seen = {'calls': 0}

    class Model:
        def __init__(self, **kwargs):
            seen['config'] = kwargs

        def bind_tools(self, tools, **kwargs):
            seen['choice'] = kwargs['tool_choice']
            return self

        async def ainvoke(self, messages):
            seen['messages'] = messages
            seen['calls'] += 1
            return AIMessage(content='Контрольный ответ; модель в этом тесте подменена.')

    monkeypatch.setattr(msty, 'ChatAnthropic', Model)
    monkeypatch.setenv('MSTY_STATIC_CACHE', '0')
    history = [
        {'role': 'system', 'content': 'Выполни минимальное разрешённое действие.'},
        {'role': 'user', 'content': 'Прочитай два тестовых файла и сопоставь содержимое.'},
        {'role': 'assistant', 'content': '', 'tool_calls': [{
            'id': 'read1', 'type': 'function', 'function': {
                'name': 'read_test_file', 'arguments': '{"path":"first.txt"}'}}]},
        {'role': 'tool', 'tool_call_id': 'read1', 'content': 'synthetic first result'},
        {'role': 'user', 'content': 'Стоп. Теперь только объясни, ничего не выполняй.'},
    ]
    asyncio.run(msty.respond({'messages': history, 'tools': TOOLS, 'tool_choice': choice}))
    assert 'MSTY_TASK_CONTINUITY_V1' in seen['messages'][0].content
    assert seen['messages'][1].content == history[0]['content']
    assert seen['messages'][-1].content == history[-1]['content']
    assert seen['messages'][-2].tool_call_id == 'read1'
    assert seen['messages'][-2].content == 'synthetic first result'
    assert seen['calls'] == 1
    assert seen['config']['max_retries'] == 0
    assert seen['choice'] == ('auto' if choice == 'auto' else {'type': 'none'})
