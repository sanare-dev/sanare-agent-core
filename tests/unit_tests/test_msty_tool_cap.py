"""Предел входа 256 схем при маршрутизации <=28 к модели (brain-desk #183).

Офлайн: провайдер лида подменён без сети; настоящие abefore_agent, роутер,
адаптер схем (msty_models) и valid_tool_calls. Успех тестов не доказывает
развёртывание или поведение живых коннекторов.
"""
import asyncio
from copy import deepcopy
import socket

import pytest
from langchain_core.messages import AIMessage
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.store.memory import InMemoryStore

from deep_agent import msty, msty_execution, msty_models, msty_native, msty_tool_routing


@pytest.fixture(autouse=True)
def no_network(monkeypatch):
    for name in ('LANGSMITH_TRACING', 'LANGCHAIN_TRACING', 'LANGCHAIN_TRACING_V2'):
        monkeypatch.setenv(name, 'false')

    def denied(*args, **kwargs):
        raise AssertionError('Tool cap offline test attempted network access')

    monkeypatch.setattr(socket.socket, 'connect', denied)
    monkeypatch.setattr(socket, 'create_connection', denied)


def tools(count):
    return [{'type': 'function', 'function': {
        'name': f'connector{index // 50}_item_{index:03d}',
        'description': f'Synthetic connector item {index}.',
        'parameters': {'type': 'object', 'properties': {'name': {'type': 'string'}},
                       'required': ['name'], 'additionalProperties': False}}}
            for index in range(count)]


def test_limits_keep_input_above_provider_cap_and_routing_below_it():
    assert msty_models.MAX_TOOLS == 256
    assert msty_models.MAX_MODEL_TOOLS == 128
    assert msty_tool_routing.MAX_SELECTED_TOOLS == 28 < msty_models.MAX_MODEL_TOOLS
    assert msty_tool_routing.MAX_CATALOG >= msty_models.MAX_TOOLS


def test_input_contract_accepts_256_and_rejects_257():
    msty_models.check_tools(tools(256))
    with pytest.raises(msty_models.ModelAdapterError, match='больше 256'):
        msty_models.check_tools(tools(257))


@pytest.mark.parametrize('mutate', [
    lambda items: items[200]['function'].update(name=items[0]['function']['name']),
    lambda items: items[200]['function'].update(parameters='not-a-schema'),
    lambda items: items[200].update(type='retrieval'),
    lambda items: items[200]['function'].update(name=''),
    lambda items: items[200]['function']['parameters'].update(default=float('nan')),
])
def test_input_contract_still_checks_uniqueness_and_schemas(mutate):
    items = tools(256)
    mutate(items)
    with pytest.raises(msty_models.ModelAdapterError):
        msty_models.check_tools(items)


def test_one_generation_stays_within_provider_cap():
    class Model:
        def bind_tools(self, schemas, **kwargs):
            self.schemas = schemas
            return self
    model = Model()
    msty_models.bind_tools('deepseek', model, tools(128), 'auto')
    assert len(model.schemas) == 128
    with pytest.raises(msty_models.ModelAdapterError, match='больше 128'):
        msty_models.bind_tools('deepseek', Model(), tools(129), 'auto')


def test_catalog_lists_every_unselected_schema_up_to_input_limit():
    items = tools(256)
    text = msty_tool_routing.catalog_prompt(items, [items[0]['function']['name']])
    assert 'connector5_item_255' in text and 'connector0_item_000' not in text
    assert text.count('\n- ') == 255


def initial(count):
    return {'messages': [{'role': 'user', 'content': 'Прочитай connector5_item_255 для отчёта.'}],
            'tools': tools(count), 'max_tokens': 128, 'tool_choice': 'auto',
            'result': {}, 'context_budget': None, 'context_budget_check': None,
            'execution_protocol': msty_execution.PROTOCOL, 'execution': {}, 'lead_profile': 'deepseek'}


def install(monkeypatch):
    seen = {'bound': [], 'calls': 0}

    class Provider:
        def bind_tools(self, schemas, **kwargs):
            seen['bound'].append(deepcopy(schemas))
            return self

        async def ainvoke(self, messages):
            seen['calls'] += 1
            return AIMessage(content='Готово.', usage_metadata={
                'input_tokens': 10, 'output_tokens': 2, 'total_tokens': 12},
                response_metadata={'model_name': 'deepseek-flash', 'finish_reason': 'stop',
                                   'token_usage': {'prompt_tokens': 10, 'completion_tokens': 2,
                                                   'total_tokens': 12}})
    monkeypatch.setattr(msty.msty_models, 'make_model', lambda *args: Provider())
    return seen


async def run(value, thread):
    graph = msty_native.build_graph(checkpointer=InMemorySaver(), store=InMemoryStore())
    config = {'configurable': {'thread_id': thread}}
    async for _ in graph.astream(value, config, stream_mode=['custom', 'values'], durability='sync'):
        pass
    return await graph.aget_state(config)


def test_native_accepts_256_schemas_and_routes_at_most_28_external(monkeypatch):
    seen = install(monkeypatch)
    state = asyncio.run(run(initial(256), 'cap-256'))
    assert seen['calls'] == 1 and state.values['execution']['status'] == 'answered'
    client = {tool['function']['name'] for tool in tools(256)}
    external = [tool for tool in seen['bound'][0] if tool['function']['name'] in client]
    assert 0 < len(external) <= msty_tool_routing.MAX_SELECTED_TOOLS
    assert len(seen['bound'][0]) <= msty_models.MAX_MODEL_TOOLS
    assert 'connector5_item_255' in {tool['function']['name'] for tool in external}
    route = state.values['execution']['tool_route']
    assert route['available_count'] == 256 and route['selected_count'] == len(external)


def test_native_rejects_257_schemas_before_any_model_call(monkeypatch):
    seen = install(monkeypatch)
    with pytest.raises(msty_execution.ExecutionProtocolError, match='больше 256'):
        asyncio.run(run(initial(257), 'cap-257'))
    assert seen['calls'] == 0 and not seen['bound']


def test_native_rejects_invalid_schema_inside_large_input(monkeypatch):
    seen = install(monkeypatch)
    value = initial(200)
    value['tools'][150]['function']['parameters'] = ['not', 'a', 'schema']
    with pytest.raises(msty_execution.ExecutionProtocolError):
        asyncio.run(run(value, 'cap-invalid'))
    assert seen['calls'] == 0
