import asyncio
from copy import deepcopy
import threading
from types import SimpleNamespace

import pytest
from langchain_core.messages import AIMessage
from deep_agent import msty


@pytest.fixture(autouse=True)
def isolated_cache_default(monkeypatch):
    monkeypatch.delenv('MSTY_STATIC_CACHE', raising=False)


def test_external_component_policy_present_with_and_without_tools():
    no_tools = msty.policy_for_tools([])
    with_tools = msty.policy_for_tools([
        {'type': 'function', 'function': {'name': 'msty_project_read',
                                        'parameters': {'type': 'object'}}},
    ])
    for policy in (no_tools, with_tools):
        assert 'Smithery' in policy
        assert 'Arcade' in policy
        assert 'нельзя установить или зарегистрировать внутри диалога' in policy
        assert 'манифест' in policy
        assert 'лицензия' in policy
        assert 'версия' in policy
        assert 'область действия' in policy


def test_access_answers_policy_present_with_and_without_tools():
    no_tools = msty.policy_for_tools([])
    with_tools = msty.policy_for_tools([
        {'type': 'function', 'function': {'name': 'msty_project_read',
                                        'parameters': {'type': 'object'}}},
    ])
    for policy in (no_tools, with_tools):
        assert 'MSTY_ACCESS_ANSWERS_V1' in policy
        assert 'своими credentials' in policy
        assert 'не чтением\nхранилища' in policy
        assert 'оправдание' in policy
        assert 'пересохранить тулсет' in policy


def test_no_tools_policy_is_explicit_and_does_not_advertise_supervisor():
    policy = msty.policy_for_tools([])
    assert 'MSTY_TOOLS_UNAVAILABLE' in policy
    assert 'инструменты не подключены' in policy
    assert 'tool_call' in policy and 'tool_response' in policy
    assert 'msty_brain_delegate' not in policy
    assert 'msty_brain_job' not in policy
    assert 'msty_brain_lessons' not in policy
    assert 'msty_brain_verify' not in policy


def test_tool_policy_uses_only_current_schemas():
    policy = msty.policy_for_tools([
        {'type': 'function', 'function': {'name': 'read_file',
                                        'parameters': {'type': 'object'}}},
    ])
    assert 'MSTY_TOOLS_AVAILABLE' in policy
    assert 'read_file' in policy
    assert 'msty_brain_delegate' not in policy
    assert 'msty_brain_job' not in policy
    assert 'установленным' in policy


def test_delegation_is_not_advertised_without_actual_schema():
    absent = msty.policy_for_tools([])
    present = msty.policy_for_tools([
        {'type': 'function', 'function': {'name': 'msty_brain_delegate1',
                                        'parameters': {'type': 'object'}}},
    ])
    assert 'msty_brain_delegate1' not in absent
    assert 'msty_brain_delegate1' in present
    assert 'msty_brain_job' not in present
    assert 'отключён' in present


def test_empty_tools_are_not_bound_and_receive_unavailable_policy(monkeypatch):
    seen = {}
    class Model:
        def __init__(self, **kw):
            pass
        def bind_tools(self, *args, **kw):
            raise AssertionError('No tools may be bound for an empty tools list')
        async def ainvoke(self, messages):
            seen['messages'] = messages
            return AIMessage(content='Инструменты не подключены; файл не проверен.')
    monkeypatch.setattr(msty, 'ChatAnthropic', Model)
    result = asyncio.run(msty.respond({'messages': [
        {'role': 'user', 'content': 'Прочитай файл'},
    ], 'tools': [], 'max_tokens': 32}))
    assert 'MSTY_TOOLS_UNAVAILABLE' in seen['messages'][0].content[0]['text']
    assert result['result']['tool_calls'] == []
    assert 'не проверен' in result['result']['content']


def test_out_of_schema_tool_call_is_never_returned_to_client(monkeypatch):
    class Model:
        def __init__(self, **kw):
            pass
        def bind_tools(self, *args, **kw):
            return self
        async def ainvoke(self, messages):
            return AIMessage(content='Всё сделал', tool_calls=[
                {'id': 'invented-1', 'name': 'msty_brain_delegate', 'args': {}}],
                usage_metadata={'input_tokens': 10, 'output_tokens': 4, 'total_tokens': 14})
    monkeypatch.setattr(msty, 'ChatAnthropic', Model)
    result = asyncio.run(msty.respond({'messages': [{'role': 'user', 'content': 'Проверь'}],
        'tools': [{'type': 'function', 'function': {'name': 'read_file',
                                                   'parameters': {'type': 'object'}}}]}))
    assert result['result']['tool_calls'] == []
    assert 'не выполнен' in result['result']['content']
    assert 'Всё сделал' not in result['result']['content']
    assert result['result']['usage_metadata']['total_tokens'] == 14


def test_no_tools_sentinel_replaces_hallucinated_structured_call(monkeypatch):
    class Model:
        def __init__(self, **kw):
            pass
        async def ainvoke(self, messages):
            return AIMessage(content='', tool_calls=[
                {'id': 'invented-1', 'name': 'read_file', 'args': {}}])
    monkeypatch.setattr(msty, 'ChatAnthropic', Model)
    result = asyncio.run(msty.respond({'messages': [{'role': 'user', 'content': 'Прочитай'}], 'tools': []}))
    assert result['result']['tool_calls'] == []
    assert 'инструменты не подключены' in result['result']['content']


def test_valid_tool_call_is_preserved_for_real_msty_execution(monkeypatch):
    class Model:
        def __init__(self, **kw):
            pass
        def bind_tools(self, *args, **kw):
            return self
        async def ainvoke(self, messages):
            return AIMessage(content='', tool_calls=[
                {'id': 'real-1', 'name': 'read_file', 'args': {'path': 'README.md'}}])
    monkeypatch.setattr(msty, 'ChatAnthropic', Model)
    result = asyncio.run(msty.respond({'messages': [{'role': 'user', 'content': 'Прочитай'}],
        'tools': [{'type': 'function', 'function': {'name': 'read_file',
                                                   'parameters': {'type': 'object'}}}]}))
    assert result['result']['tool_calls'][0]['name'] == 'read_file'
    assert result['result']['tool_calls'][0]['args'] == {'path': 'README.md'}

def test_client_tool_roundtrip(monkeypatch):
    seen = {}
    class Model:
        def __init__(self, **kw):
            seen['config'] = kw
        def bind_tools(self, tools, **kw):
            seen['tools'] = tools
            return self
        async def ainvoke(self, messages):
            seen['messages'] = messages
            return AIMessage(content='verified')
    monkeypatch.setattr(msty, 'ChatAnthropic', Model)
    result = asyncio.run(msty.respond({'messages': [
        {'role': 'user', 'content': 'inspect'},
        {'role': 'assistant', 'content': '', 'tool_calls': [{'id': 'call1', 'type': 'function', 'function': {'name': 'inspect', 'arguments': '{}'}}]},
        {'role': 'tool', 'tool_call_id': 'call1', 'content': 'verified result'},
    ], 'tools': [{'type': 'function', 'function': {'name': 'inspect', 'parameters': {'type': 'object'}}}], 'max_tokens': 10}))
    assert seen['messages'][-1].type == 'tool'
    assert seen['messages'][-1].content == 'verified result'
    assert seen['tools'][0]['function']['name'] == 'inspect'
    assert result['result']['type'] == 'ai'
    assert seen['config']['max_retries'] == 0


def budget_model(monkeypatch, tokens=120000, count_error=None):
    seen = {'events': []}
    class Model:
        def __init__(self, **kwargs):
            seen['config'] = kwargs
        def get_num_tokens_from_messages(self, messages, *, tools, **kwargs):
            seen['events'].append('count')
            seen['count_thread'] = threading.get_ident()
            seen['count_messages'] = deepcopy(messages)
            seen['count_tools'] = deepcopy(tools)
            seen['count_options'] = kwargs
            if count_error is not None:
                raise count_error
            return tokens
        def bind_tools(self, tools, **kwargs):
            seen['events'].append('bind')
            seen['generation_tools'] = deepcopy(tools)
            return self
        async def ainvoke(self, messages):
            seen['events'].append('generate')
            seen['generation_messages'] = deepcopy(messages)
            return AIMessage(content='verified', usage_metadata={
                'input_tokens': 120010, 'output_tokens': 3, 'total_tokens': 120013})
    monkeypatch.setattr(msty, 'ChatAnthropic', Model)
    return seen


def test_large_context_is_counted_in_a_worker_before_generation_without_truncation(monkeypatch):
    seen = budget_model(monkeypatch)
    current_thread = threading.get_ident()
    tools = [{'type': 'function', 'function': {'name': 'inspect',
              'parameters': {'type': 'object', 'properties': {'path': {'type': 'string'}}}}}]
    state = {'messages': [{'role': 'system', 'content': 'KEEP ALL POLICY'},
                           {'role': 'user', 'content': 'я' * 105000}],
             'tools': tools, 'max_tokens': 8192}
    before = deepcopy(state)
    result = asyncio.run(msty.respond(state))
    assert seen['events'] == ['count', 'bind', 'generate']
    assert seen['count_thread'] != current_thread
    assert seen['count_options']['timeout'] == 20.0
    assert seen['count_messages'] == seen['generation_messages']
    assert seen['count_messages'][0].content == [{'type': 'text',
        'text': msty.policy_for_tools(tools) + '\n\n' + msty.msty_memory.system_context(),
        'cache_control': {'type': 'ephemeral', 'ttl': '5m'}}]
    assert seen['count_messages'][1].content == 'KEEP ALL POLICY'
    assert seen['count_messages'][-1].content == state['messages'][-1]['content']
    assert seen['count_tools'] == seen['generation_tools'] == tools
    assert state == before
    assert seen['config']['max_tokens'] == 8192
    assert seen['config']['max_retries'] == 0
    assert result['context_budget_check'] == {
        'version': 1, 'status': 'accepted', 'input_tokens': 120000, 'limit': 180000,
        'method': 'anthropic-exact-v1', 'model_profile': 'sonnet'}
    assert result['result']['usage_metadata']['input_tokens'] == 120010


def test_required_budget_flag_counts_even_a_short_request(monkeypatch):
    seen = budget_model(monkeypatch, tokens=42)
    result = asyncio.run(msty.respond({'messages': [{'role': 'user', 'content': 'ok'}],
                                       'tools': [], 'context_budget': 'anthropic-count-v1'}))
    assert seen['events'] == ['count', 'generate']
    assert result['context_budget_check']['input_tokens'] == 42
    assert result['context_budget_check']['status'] == 'accepted'


def test_short_request_does_not_count_and_clears_stale_check(monkeypatch):
    seen = budget_model(monkeypatch, count_error=AssertionError('must not count'))
    result = asyncio.run(msty.respond({'messages': [{'role': 'user', 'content': 'ok'}],
        'tools': [], 'context_budget': None,
        'context_budget_check': {'version': 1, 'status': 'accepted', 'input_tokens': 999}}))
    assert seen['events'] == ['generate']
    assert result['context_budget_check'] is None


@pytest.mark.parametrize('tokens,accepted', [(180000, True), (180001, False)])
def test_input_token_limit_is_inclusive_and_overflow_never_generates(monkeypatch, tokens, accepted):
    seen = budget_model(monkeypatch, tokens=tokens)
    result = asyncio.run(msty.respond({'messages': [{'role': 'user', 'content': 'ok'}],
        'tools': [], 'context_budget': 'anthropic-count-v1'}))
    expected_check = {
        'version': 1, 'status': 'accepted' if accepted else 'rejected',
        'input_tokens': tokens, 'limit': 180000}
    if accepted:
        expected_check.update(method='anthropic-exact-v1', model_profile='sonnet')
    assert result['context_budget_check'] == expected_check
    if accepted:
        assert seen['events'] == ['count', 'generate']
    else:
        assert seen['events'] == ['count']
        assert result['result']['type'] == 'ai'
        assert '180000' in result['result']['content']
        assert result['result']['tool_calls'] == []
        assert result['result']['usage_metadata'] == {
            'input_tokens': 0, 'output_tokens': 0, 'total_tokens': 0}


def test_count_failure_is_fail_closed_without_raw_exception_or_generation(monkeypatch):
    seen = budget_model(monkeypatch, count_error=RuntimeError('PRIVATE_PROVIDER_RESPONSE_SENTINEL'))
    result = asyncio.run(msty.respond({'messages': [{'role': 'user', 'content': 'ok'}],
        'tools': [], 'context_budget': 'anthropic-count-v1'}))
    assert seen['events'] == ['count']
    assert result['context_budget_check'] == {
        'version': 1, 'status': 'rejected', 'input_tokens': None, 'limit': 180000}
    assert 'PRIVATE_PROVIDER_RESPONSE_SENTINEL' not in str(result)
    assert result['result']['usage_metadata']['total_tokens'] == 0
    assert 'не запущена' in result['result']['content']


@pytest.mark.parametrize('tokens', [None, '120000', -1, True])
def test_invalid_token_count_never_generates(monkeypatch, tokens):
    seen = budget_model(monkeypatch, tokens=tokens)
    result = asyncio.run(msty.respond({'messages': [{'role': 'user', 'content': 'ok'}],
        'tools': [], 'context_budget': 'anthropic-count-v1'}))
    assert seen['events'] == ['count']
    assert result['context_budget_check']['status'] == 'rejected'
    assert result['context_budget_check']['input_tokens'] is None


def test_large_tool_schema_counts_even_when_messages_are_short(monkeypatch):
    seen = budget_model(monkeypatch, tokens=80000)
    tools = [{'type': 'function', 'function': {'name': 'inspect',
        'description': 'x' * 205000, 'parameters': {'type': 'object'}}}]
    result = asyncio.run(msty.respond({'messages': [{'role': 'user', 'content': 'ok'}], 'tools': tools}))
    assert seen['events'] == ['count', 'bind', 'generate']
    assert seen['count_tools'] == tools
    assert result['context_budget_check']['status'] == 'accepted'


def test_large_graph_policy_is_part_of_preflight_size_and_count(monkeypatch):
    seen = budget_model(monkeypatch, tokens=80000)
    monkeypatch.setattr(msty, 'POLICY', 'POLICY ' * 30000)
    result = asyncio.run(msty.respond({'messages': [{'role': 'user', 'content': 'ok'}], 'tools': []}))
    assert seen['events'] == ['count', 'generate']
    assert seen['count_messages'][0].content[0]['text'].startswith(msty.POLICY)
    assert result['context_budget_check']['status'] == 'accepted'


def test_real_sdk_counter_receives_all_system_blocks_including_project_policy(monkeypatch):
    # Exercise the installed real conversion/count method with only its network
    # client replaced. Its native implementation omits list-form system prompts
    # unless our adapter supplies them explicitly as keyword arguments.
    sdk_counter = msty.ChatAnthropic.get_num_tokens_from_messages
    seen = {'events': []}
    def count_tokens(**kwargs):
        seen['events'].append('count')
        seen['request'] = kwargs
        return SimpleNamespace(input_tokens=1400)
    class Model:
        model = 'claude-sonnet-4-6'
        context_management = None
        betas = None
        get_num_tokens_from_messages = sdk_counter
        def __init__(self, **kwargs):
            self._client = SimpleNamespace(messages=SimpleNamespace(count_tokens=count_tokens))
        def bind_tools(self, *args, **kwargs):
            return self
        async def ainvoke(self, messages):
            seen['events'].append('generate')
            seen['generation_system'], seen['generation_messages'] = msty._format_messages(messages)
            return AIMessage(content='ok')
    monkeypatch.setattr(msty, 'ChatAnthropic', Model)
    result = asyncio.run(msty.respond({
        'messages': [{'role': 'system', 'content': 'PROJECT_POLICY_SENTINEL'},
                     {'role': 'system', 'content': [{'type': 'text', 'text': 'BLOCK_POLICY_SENTINEL'}]},
                     {'role': 'user', 'content': 'USER_SENTINEL'}],
        'tools': [{'type': 'function', 'function': {'name': 'inspect',
                   'parameters': {'type': 'object', 'properties': {}}}}],
        'context_budget': 'anthropic-count-v1',
    }))
    assert seen['events'] == ['count', 'generate']
    assert seen['request']['system'] == seen['generation_system']
    assert isinstance(seen['request']['system'], list)
    assert any('PROJECT_POLICY_SENTINEL' == block['text'] for block in seen['request']['system'])
    assert any('BLOCK_POLICY_SENTINEL' == block['text'] for block in seen['request']['system'])
    assert seen['request']['system'][0]['text'].startswith(msty.POLICY)
    assert seen['request']['system'][0]['cache_control'] == {'type': 'ephemeral', 'ttl': '5m'}
    assert all('cache_control' not in block for block in seen['request']['system'][1:])
    assert seen['request']['messages'] == seen['generation_messages']
    assert seen['request']['tools'][0]['name'] == 'inspect'
    assert seen['request']['timeout'] == 20.0
    assert result['context_budget_check']['input_tokens'] == 1400


def test_unknown_budget_protocol_is_fail_closed(monkeypatch):
    seen = budget_model(monkeypatch)
    result = asyncio.run(msty.respond({'messages': [{'role': 'user', 'content': 'ok'}],
        'tools': [], 'context_budget': 'unknown-v999'}))
    assert seen['events'] == []
    assert result['context_budget_check']['status'] == 'rejected'
    assert result['result']['usage_metadata']['total_tokens'] == 0
