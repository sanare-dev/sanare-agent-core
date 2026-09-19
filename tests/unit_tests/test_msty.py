import asyncio
from langchain_core.messages import AIMessage
from deep_agent import msty

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
