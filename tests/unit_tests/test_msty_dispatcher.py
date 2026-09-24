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
    assert {'msty_system_overview', 'msty_admin_system_map'} <= set(route['selected_names'])
    assert route['catalog'] is True
    catalog = routing.catalog_prompt(TOOLS, route['selected_names'])
    assert 'msty_codex_start' in catalog and 'native_request_tools' in catalog


def test_requested_tools_become_visible_next_step(monkeypatch):
    monkeypatch.setenv('MSTY_TOOL_DISPATCHER', 'on')
    history = [{'role': 'user', 'content': QUESTION},
               {'role': 'assistant', 'content': '', 'tool_calls': [{'id': 'r1', 'type': 'function',
                'function': {'name': 'native_request_tools',
                             'arguments': json.dumps({'names': ['msty_codex_start', 'search_files']})}}]},
               {'role': 'tool', 'tool_call_id': 'r1', 'content': 'Подключено'}]
    _, route, _ = routing.select_tools(history, TOOLS)
    assert {'msty_codex_start', 'search_files'} <= set(route['selected_names'])
    assert route['requested'] == ['msty_codex_start', 'search_files']


def test_requested_tools_reset_on_new_owner_turn(monkeypatch):
    monkeypatch.setenv('MSTY_TOOL_DISPATCHER', 'on')
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


def test_dispatcher_is_on_by_default_and_can_be_switched_off(monkeypatch):
    # 24.09: keyword routing alone hid the owner's tools; the bridge allows
    # native_request_tools, so the catalog + request is the default now.
    monkeypatch.delenv('MSTY_TOOL_DISPATCHER', raising=False)
    assert routing.dispatcher_enabled() is True
    monkeypatch.setenv('MSTY_TOOL_DISPATCHER', 'off')
    assert routing.dispatcher_enabled() is False


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


import pytest  # noqa: E402


@pytest.mark.parametrize('question', [
    'Установи цену 1990 на магнезиум', 'Установи напоминание на завтра', 'Установи приоритет высокий',
    'Set up a meeting with Anna tomorrow', 'Разверни мысль подробнее', 'Разверни список задач',
    'Which packages are installed?', 'What is our current setup for Amazon ads?',
    'Установи обновление Brain', 'Установи расширение pgvector в Supabase',
    'Установи SSL на Pressable сайте', 'Установи плагин на сайт sanarelab.com',
])
def test_install_route_does_not_leak_codex(question):
    _, route, _ = routing.select_tools([{'role': 'user', 'content': question}], TOOLS)
    assert 'msty_codex_start' not in route['selected_names']


@pytest.mark.parametrize('question', [
    'Установи Telegram-бота на Mac', 'Разверни локально MCP-сервер для Telegram',
    'Install the telegram bot package', 'Подключи бота к Msty'])
def test_install_route_gives_codex_for_software(question):
    _, route, _ = routing.select_tools([{'role': 'user', 'content': question}], TOOLS)
    assert 'msty_codex_start' in route['selected_names']


@pytest.mark.parametrize('target', ['file', 'status', 'start', 'cancel'])
def test_short_execute_tool_names_expose_nothing(target):
    history = [{'role': 'user', 'content': 'Проверь статус'},
               {'role': 'assistant', 'content': '', 'tool_calls': [{'id': 'e1', 'type': 'function',
                'function': {'name': 'execute_tool', 'arguments': json.dumps({'tool_name': target})}}]},
               {'role': 'tool', 'tool_call_id': 'e1', 'content': 'Unknown tool'}]
    assert routing.requested_names(history, {t['function']['name'] for t in TOOLS}) == set()


def test_write_tool_via_execute_tool_needs_mutation_intent():
    history = [{'role': 'user', 'content': 'Покажи таблицы'},
               {'role': 'assistant', 'content': '', 'tool_calls': [{'id': 'e1', 'type': 'function',
                'function': {'name': 'execute_tool', 'arguments': json.dumps({'tool_name': 'apply_migration'})}}]},
               {'role': 'tool', 'tool_call_id': 'e1', 'content': 'Unknown tool'}]
    _, route, _ = routing.select_tools(history, TOOLS)
    assert 'apply_migration' not in route['selected_names']


def test_request_history_ignored_when_dispatcher_off(monkeypatch):
    monkeypatch.setenv('MSTY_TOOL_DISPATCHER', 'off')
    history = [{'role': 'user', 'content': 'Привет'},
               {'role': 'assistant', 'content': '', 'tool_calls': [{'id': 'r1', 'type': 'function',
                'function': {'name': 'native_request_tools',
                             'arguments': json.dumps({'names': ['execute_sql', 'apply_migration']})}}]},
               {'role': 'tool', 'tool_call_id': 'r1', 'content': 'x'}]
    _, route, _ = routing.select_tools(history, TOOLS)
    assert not {'execute_sql', 'apply_migration'} & set(route['selected_names'])


def test_execute_tool_never_exposes_foreign_write_even_on_install():
    history = [{'role': 'user', 'content': 'Установи бота на Mac'},
               {'role': 'assistant', 'content': '', 'tool_calls': [{'id': 'e1', 'type': 'function',
                'function': {'name': 'execute_tool', 'arguments': json.dumps({'tool_name': 'apply_migration'})}}]},
               {'role': 'tool', 'tool_call_id': 'e1', 'content': 'Unknown tool'}]
    _, route, _ = routing.select_tools(history, TOOLS)
    assert 'apply_migration' not in route['selected_names']


@pytest.mark.parametrize('question', [
    'Можно ли установить приложение на Mac?', 'Какой сервер установить на Mac?',
    'How to install the telegram bot package?'])
def test_install_questions_do_not_get_codex(question):
    _, route, _ = routing.select_tools([{'role': 'user', 'content': question}], TOOLS)
    assert 'msty_codex_start' not in route['selected_names']
