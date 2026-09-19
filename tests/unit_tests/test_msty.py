import asyncio
from langchain_core.messages import AIMessage
from deep_agent import msty


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
    assert 'MSTY_TOOLS_UNAVAILABLE' in seen['messages'][0].content
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
