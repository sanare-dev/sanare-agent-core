"""Диспетчер инструментов: живой дефект 2026-09-23 — клиент передал 116 схем,
роутер выдал модели 4, Brain ответил «нет инструментов» на вопрос о Telegram-ботах."""
import asyncio
import json
from pathlib import Path
from types import SimpleNamespace

from deep_agent import msty_native, msty_tool_routing as routing

REAL = json.loads((Path(__file__).parent / 'fixtures_real_toolset_20260923.json').read_text())
TOOLS = [{'type': 'function', 'function': {'name': t['name'], 'description': t['description'],
          'parameters': {'type': 'object', 'properties': {}}}} for t in REAL]
QUESTION = 'Проверь систему. Какие Telegram-боты у меня, видишь? Если у тебя к ним доступ, установи.'


def test_live_telegram_question_gets_system_tools_and_catalog():
    _, route, _ = routing.select_tools([{'role': 'user', 'content': QUESTION}], TOOLS)
    assert {'msty_system_overview', 'msty_admin_system_map', 'msty_codex_start'} <= set(route['selected_names'])
    assert route['catalog'] is True
    catalog = routing.catalog_prompt(TOOLS, route['selected_names'])
    assert 'msty_codex_start' in catalog and 'native_request_tools' in catalog


def test_requested_tools_become_visible_next_step():
    history = [{'role': 'user', 'content': QUESTION},
               {'role': 'assistant', 'content': '', 'tool_calls': [{'id': 'r1', 'type': 'function',
                'function': {'name': 'native_request_tools',
                             'arguments': json.dumps({'names': ['msty_codex_start', 'search_files']})}}]},
               {'role': 'tool', 'tool_call_id': 'r1', 'content': 'Подключено'}]
    _, route, _ = routing.select_tools(history, TOOLS)
    assert {'msty_codex_start', 'search_files'} <= set(route['selected_names'])
    assert route['requested'] == ['msty_codex_start', 'search_files']


def test_requested_tools_reset_on_new_owner_turn():
    history = [{'role': 'user', 'content': QUESTION},
               {'role': 'assistant', 'content': '', 'tool_calls': [{'id': 'r1', 'type': 'function',
                'function': {'name': 'native_request_tools',
                             'arguments': json.dumps({'names': ['apply_migration']})}}]},
               {'role': 'tool', 'tool_call_id': 'r1', 'content': 'Подключено'},
               {'role': 'assistant', 'content': 'Готово'},
               {'role': 'user', 'content': 'Спасибо'}]
    _, route, _ = routing.select_tools(history, TOOLS)
    assert route['requested'] == []


def test_execute_tool_with_direct_name_exposes_direct_tool():
    history = [{'role': 'user', 'content': 'Проверь синхру'},
               {'role': 'assistant', 'content': '', 'tool_calls': [{'id': 'e1', 'type': 'function',
                'function': {'name': 'execute_tool',
                             'arguments': json.dumps({'tool_name': 'msty_store_sync_status'})}}]},
               {'role': 'tool', 'tool_call_id': 'e1', 'content': 'Unknown tool'}]
    _, route, _ = routing.select_tools(history, TOOLS)
    assert 'msty_store_sync_status' in route['selected_names']


def test_explain_question_does_not_carry_catalog():
    _, route, _ = routing.select_tools([{'role': 'user', 'content': 'Объясни кратко, что такое vault.'}], TOOLS)
    assert route['catalog'] is False


def test_dispatcher_is_off_by_default(monkeypatch):
    monkeypatch.delenv('MSTY_TOOL_DISPATCHER', raising=False)
    assert routing.dispatcher_enabled() is False
    monkeypatch.setenv('MSTY_TOOL_DISPATCHER', 'on')
    assert routing.dispatcher_enabled() is True


def test_request_tools_call_enables_only_toolset_names():
    call = {'name': 'native_request_tools', 'id': 'r1',
            'args': {'names': ['msty_codex_start', 'telegram_magic']}}
    request = SimpleNamespace(tool_call=call, state={
        'native_tool_names': ['native_request_tools'], 'tools': TOOLS, 'messages': []})

    async def forbidden(request):
        raise AssertionError('серверный инструмент не исполняется handler-ом')
    result = asyncio.run(msty_native.NativeMstyMiddleware().awrap_tool_call(request, forbidden))
    assert 'Подключено: msty_codex_start' in result.content
    assert 'telegram_magic' in result.content and result.status == 'success'
