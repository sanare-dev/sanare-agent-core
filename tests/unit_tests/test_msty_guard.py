"""TAU L3: Guard — корректирующие сообщения вместо сырых отказов исполнения.

Позитив (валидный вызов проходит), негативы: unknown_tool с fuzzy-кандидатами,
invalid_args со списком нарушений схемы, alias-match — вызов по алиасу резолвится
в каноническое имя реестра. Интеграционные проверки идут через реальную обёртку
NativeMstyMiddleware.awrap_tool_call без провайдеров и сети.
"""
import asyncio
from types import SimpleNamespace

import pytest

from deep_agent import msty_execution, msty_guard, msty_native, msty_registry


SQL_SCHEMA = {'type': 'object',
              'properties': {'query': {'type': 'string'}},
              'required': ['query'], 'additionalProperties': False}


def call(name, args, identifier='call-1'):
    return {'id': identifier, 'name': name, 'args': args}


def tool_schema(name, parameters):
    return {'type': 'function', 'function': {'name': name, 'parameters': parameters}}


def middleware():
    return msty_native.NativeMstyMiddleware()


def forbidden_handler(request):
    raise AssertionError('Отклонённый Guard вызов не должен доходить до исполнения')


# --- Чистая функция валидации -------------------------------------------------

def test_valid_call_passes_guard():
    assert msty_guard.guard_validate(
        call('execute_sql', {'query': 'select 1'}), schema=SQL_SCHEMA) is None
    # Статусные чтения без аргументов тоже проходят.
    assert msty_guard.guard_validate(call('msty_store_sync_status', {})) is None


def test_unknown_tool_returns_fuzzy_candidates_not_a_raw_error():
    result = msty_guard.guard_validate(call('msty_store_sync_stats', {}))
    assert result is not None and result.status == 'error'
    assert 'unknown_tool' in result.content
    assert "'msty_store_sync_stats'" in result.content
    assert 'msty_store_sync_status' in result.content  # fuzzy top-3 из реестра
    assert 'один раз' in result.content  # правило одной смены способа вызова
    assert result.tool_call_id == 'call-1'


def test_invalid_args_return_schema_violations_to_the_model():
    result = msty_guard.guard_validate(call('execute_sql', {'query': 42}),
                                       schema=SQL_SCHEMA)
    assert result is not None and result.status == 'error'
    assert 'invalid_arguments' in result.content
    assert 'query' in result.content
    # Отсутствие обязательного поля — тоже invalid_arguments.
    missing = msty_guard.guard_validate(call('execute_sql', {}), schema=SQL_SCHEMA)
    assert missing is not None and 'invalid_arguments' in missing.content


def test_alias_call_resolves_to_canonical_name():
    # Прямая резолюция алиаса.
    entry = msty_registry.find('store_sync_status')
    assert entry is not None and entry.name == 'msty_store_sync_status'
    # Вызов по алиасу проходит Guard как канонический инструмент.
    assert msty_guard.guard_validate(call('store_sync_status', {})) is None
    # Суффиксный неймспейсинг клиента тоже резолвится, а нарушение схемы
    # сообщает каноническое имя записи.
    result = msty_guard.guard_validate(call('sanare_admin_execute_sql', {'query': 42}),
                                       schema=SQL_SCHEMA)
    assert result is not None and 'invalid_arguments' in result.content
    assert "'execute_sql'" in result.content


def test_unverifiable_schema_is_not_a_call_failure():
    # Чужой диалект/битый $ref: Guard не блокирует догадкой, исполнитель
    # вернёт собственную ошибку.
    assert msty_guard.guard_validate(
        call('execute_sql', {}), schema={'$ref': 'file:///must-not-be-read.json'}) is None
    assert msty_guard.guard_validate(call('execute_sql', {}), schema=None) is None


# --- Врезка в обёртку исполнения (перед ToolNode) ------------------------------

def test_middleware_corrects_unknown_name_without_execution():
    async def run():
        request = SimpleNamespace(tool_call=call('msty_store_sync_stats', {}),
                                  state={'tools': []})
        result = await middleware().awrap_tool_call(request, forbidden_handler)
        assert result.status == 'error'
        assert 'unknown_tool' in result.content
        assert 'msty_store_sync_status' in result.content
    asyncio.run(run())


def test_middleware_still_raises_for_registry_tool_not_admitted_on_this_step():
    # Имя из реестра, но не допущенное на шаге — рассогласование конвейера,
    # а не «unknown tool»: протокольная граница сохранена.
    async def run():
        request = SimpleNamespace(tool_call=call('msty_store_sync_status', {}),
                                  state={'tools': []})
        with pytest.raises(msty_execution.ExecutionProtocolError):
            await middleware().awrap_tool_call(request, forbidden_handler)
    asyncio.run(run())


def test_middleware_never_discards_result_of_already_executed_external_call():
    # Допущенный внешний вызов уже исполнен клиентом Msty (аргументы сверены до
    # отправки). Повторная проверка после исполнения выбрасывала бы реальный
    # результат записи и толкала модель на повтор — двойной побочный эффект.
    async def run():
        request = SimpleNamespace(
            tool_call=call('execute_sql', {'query': 42}),
            state={'tools': [tool_schema('execute_sql', SQL_SCHEMA)],
                   'native_external_observations': {'call-1': 'rows: 1'}})
        result = await middleware().awrap_tool_call(request, forbidden_handler)
        assert result.content == 'rows: 1'
        assert result.status != 'error'
    asyncio.run(run())


def test_guard_messages_do_not_echo_argument_values():
    secret = 'sk-proj-SECRETVALUE0123456789'
    schema = {'type': 'object', 'properties': {'role': {'enum': ['a', 'b']}},
              'additionalProperties': False}
    result = msty_guard.guard_arguments(call('execute_sql', {'role': secret, 'x': secret}),
                                        schema=schema)
    assert result is not None and secret not in result.content


def test_middleware_passes_valid_call_to_verified_observation():
    async def run():
        request = SimpleNamespace(
            tool_call=call('execute_sql', {'query': 'select 1'}),
            state={'tools': [tool_schema('execute_sql', SQL_SCHEMA)],
                   'native_external_observations': {'call-1': 'rows: 1'}})
        result = await middleware().awrap_tool_call(request, forbidden_handler)
        assert result.content == 'rows: 1'
        assert result.tool_call_id == 'call-1'
    asyncio.run(run())


def test_middleware_permits_client_tools_outside_the_registry():
    # Будущие коннекторы Msty допустимы, пока клиент передал их схему: реестр
    # судит только покрытые им инструменты.
    async def run():
        request = SimpleNamespace(
            tool_call=call('future_inventory_lookup', {'sku': 'A-1'}),
            state={'tools': [tool_schema('future_inventory_lookup',
                                         {'type': 'object'})],
                   'native_external_observations': {'call-1': 'sku ok'}})
        result = await middleware().awrap_tool_call(request, forbidden_handler)
        assert result.content == 'sku ok'
    asyncio.run(run())
