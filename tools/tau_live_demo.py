"""TAU live demo: все слои собранной системы на РЕАЛЬНОМ графе msty_native.

Opt-in демонстрация для владельца («запусти и посмотри, как оно работает»):
in-process граф без сервера, скриптованная модель вместо team.brain (живых
ключей локально нет), никакой сети и секретов. Каждая сцена показывает один
слой TAU и завершается явным вердиктом PASS/FAIL; любой FAIL → exit 1.

Запуск отдельной явной командой (юнит-тестами и release-гейтом не запускается):

    uv run python tools/tau_live_demo.py
"""
import asyncio
from copy import deepcopy
import json
import os
import sys

from langchain_core.messages import AIMessage, ToolMessage
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.store.memory import InMemoryStore
from langgraph.types import Command

from deep_agent import (msty, msty_execution, msty_models, msty_native,
                        msty_subagents, msty_taxonomy, msty_tool_routing)


# --- Хелперы графа (те же паттерны, что и офлайн-тесты) -----------------------

SQL_TOOL = {'type': 'function', 'function': {'name': 'execute_sql',
    'parameters': {'type': 'object', 'properties': {'query': {'type': 'string'}}}}}
STATUS_TOOL = {'type': 'function', 'function': {'name': 'msty_store_sync_status',
    'parameters': {'type': 'object', 'properties': {}}}}


def answer(content='', calls=None):
    return AIMessage(content=content, tool_calls=calls or [],
        usage_metadata={'input_tokens': 100, 'output_tokens': 10, 'total_tokens': 110})


def call(name, args, identifier):
    return {'id': identifier, 'name': name, 'args': args}


def initial(text, tools):
    return {'messages': [{'role': 'user', 'content': text}],
            'tools': deepcopy(tools), 'max_tokens': 512, 'tool_choice': 'auto',
            'result': {}, 'context_budget': None, 'context_budget_check': None,
            'execution_protocol': msty_execution.PROTOCOL, 'execution': {}}


async def invoke(graph, value, config):
    async for _mode, _chunk in graph.astream(value, config, stream_mode=['custom', 'values'],
                                             durability='sync'):
        pass
    return await graph.aget_state(config)


def external_resume(values, observation):
    """Resume внешнего pending-вызова ровно по протоколу (как клиент Msty)."""
    incoming = {key: deepcopy(values[key]) for key in
        ('tools', 'max_tokens', 'tool_choice', 'context_budget', 'execution_protocol')}
    incoming.update(messages=deepcopy(values['native_protocol_messages']), result={},
                    context_budget_check=None)
    item = values['execution']['pending']['calls'][0]
    client = 'b1_demo_0'
    calls = [{'id': client, 'type': 'function', 'function': {'name': item['name'],
              'arguments': json.dumps(item['args'])}}]
    results = [{'role': 'tool', 'tool_call_id': client, 'content': observation}]
    incoming['messages'] += [{'role': 'assistant', 'content': values['result']['content'],
                              'tool_calls': calls}, *results]
    return {'version': 1, 'task_id': values['execution']['task_id'],
            'batch_id': values['execution']['pending']['batch_id'],
            'tool_id_map': [{'client_id': client, 'model_id': item['id']}], 'input': incoming}


class Scene:
    """Скриптование модели прямым присваиванием (в демо нет monkeypatch)."""

    def __init__(self, title):
        self.title = title
        self._originals = []

    def patch(self, owner, name, replacement):
        self._originals.append((owner, name, getattr(owner, name)))
        setattr(owner, name, replacement)

    def script_model(self, sequence):
        seen = []

        async def step(state, *, native_system_prompt=None, native_result_filter=None):
            assert sequence, 'Скрипт модели исчерпан: лишний шаг'
            seen.append({'state': deepcopy(state), 'system': native_system_prompt})
            result = sequence.pop(0)
            if native_result_filter:
                result = native_result_filter(result)
            return msty.publish_result(result, None)

        self.patch(msty, '_respond_step', step)
        return seen

    def close(self):
        for owner, name, original in reversed(self._originals):
            setattr(owner, name, original)


def brief(value, limit=280):
    text = value if isinstance(value, str) else json.dumps(value, ensure_ascii=False)
    return text if len(text) <= limit else text[:limit] + '…'


def show(label, value, limit=280):
    print(f'  {label}: {brief(value, limit)}')


# --- Сцены ----------------------------------------------------------------------

def scene_guard():
    """L3 Guard: опечатка в имени инструмента перехватывается до исполнения."""
    print('═' * 78)
    print('СЦЕНА 1. L3 Guard: опечатка «msty_store_synch_status» не доходит до модели')
    print('═' * 78)
    scene = Scene('guard')
    try:
        scene.script_model([
            answer('', [call('msty_store_synch_status', {}, 'g1')]),      # опечатка
            answer('', [call('msty_store_sync_status', {}, 'g2')]),       # исправление
            answer('Синхронизация работает: свежесть данных 210 секунд.')])

        async def run():
            graph = msty_native.build_graph(checkpointer=InMemorySaver(), store=InMemoryStore())
            config = {'configurable': {'thread_id': 'demo-guard'}}
            state = await invoke(graph, initial('Проверь синхронизацию магазина.',
                                                [STATUS_TOOL]), config)
            # Клиент Msty на опечатку ответил бы сырым Unknown tool — показываем,
            # что Guard подменяет это корректирующим ToolMessage.
            pending = state.tasks[0].interrupts[0]
            show('клиент получил вызов', state.values['execution']['pending']['calls'])
            state = await invoke(graph, Command(resume={pending.id: external_resume(
                state.values, 'Unknown tool: msty_store_synch_status')}), config)
            pending = state.tasks[0].interrupts[0]
            state = await invoke(graph, Command(resume={pending.id: external_resume(
                state.values, '{"status": "ok", "freshness_s": 210}')}), config)
            return state

        state = asyncio.run(run())
        tools = [m for m in state.values['messages'] if m.type == 'tool']
        correction = tools[0].content
        show('Guard вернул модели вместо сырого отказа', correction, 400)
        show('успешное чтение после исправления', tools[1].content)
        show('финал модели', state.values['result']['content'])
        ok = ('msty_store_sync_status' in correction
              and 'Unknown tool' not in correction
              and tools[0].status == 'error'
              and 'freshness_s' in tools[1].content
              and state.values['execution']['status'] == 'answered')
        assert ok, 'корректирующее ToolMessage не сработало как ожидалось'
        # tau-событие Guard классифицировано и лежит в состоянии.
        show('tau_errors в состоянии', state.values.get('tau_errors'))
        assert any(e['class'] == msty_taxonomy.UNKNOWN_TOOL
                   for e in state.values['tau_errors']), 'отказ Guard не классифицирован'
        print('  >>> Сырой «Unknown tool» до модели НЕ дошёл: Guard подменил его')
        print('      коррекцией с кандидатом «msty_store_sync_status», модель')
        print('      исправилась и дочитала статус.')
    finally:
        scene.close()


def scene_taxonomy():
    """L4: retry идемпотентного native + бюджет на внешний transient."""
    print('═' * 78)
    print('СЦЕНА 2. L4 Таксономия: retry идемпотентного чтения, бюджет 2 на transient')
    print('═' * 78)
    # Часть А: middleware — идемпотентное чтение, два transient-сбоя, retry ×2.
    print('  Часть А. native_read_file (идемпотентен по манифесту):')
    attempts = 0

    async def flaky_handler(request):
        nonlocal attempts
        attempts += 1
        if attempts < 3:
            raise ConnectionError('synthetic transport reset')
        return ToolMessage(content='# PROJECT\nBrain контур', name='native_read_file',
                           tool_call_id='demo-r1')

    from types import SimpleNamespace
    request = SimpleNamespace(
        tool_call=call('native_read_file', {'file_path': '/memory/PROJECT.md'}, 'demo-r1'),
        state={'native_tool_names': ['native_read_file'], 'tau_errors': []})
    result = asyncio.run(msty_native.NativeMstyMiddleware()
                         .awrap_tool_call(request, flaky_handler))
    show(f'попыток исполнения: {attempts} (2 сбоя ConnectionError + успех)', result.content)
    assert attempts == 3 and 'Brain контур' in result.content, 'retry идемпотента не сработал'

    # Часть Б: граф — внешний transient дважды, третий вызов деградирует.
    print('  Часть Б. execute_sql через клиента: два «HTTP 503», третий — дегрейд:')
    scene = Scene('taxonomy')
    try:
        scene.script_model([
            answer('', [call('execute_sql', {'query': 'select 1'}, 'e1')]),
            answer('', [call('execute_sql', {'query': 'select 1'}, 'e2')]),
            answer('', [call('execute_sql', {'query': 'select 1'}, 'e3')]),
            answer('Запрос не выполнен: контур БД временно недоступен, '
                   'бюджет повторов исчерпан; действий не произведено.')])

        async def run():
            graph = msty_native.build_graph(checkpointer=InMemorySaver(), store=InMemoryStore())
            config = {'configurable': {'thread_id': 'demo-taxonomy'}}
            state = await invoke(graph, initial('Выполни execute_sql: select 1.',
                                                [SQL_TOOL]), config)
            for _ in range(3):
                pending = state.tasks[0].interrupts[0]
                state = await invoke(graph, Command(resume={pending.id: external_resume(
                    state.values, 'HTTP 503 Service Unavailable')}), config)
            return state

        state = asyncio.run(run())
        errors = state.values['tau_errors']
        tools = [m for m in state.values['messages'] if m.type == 'tool']
        show('tau_errors (две классифицированные записи)', errors, 420)
        show('третий вызов — честный дегрейд', tools[-1].content, 200)
        show('финал модели', state.values['result']['content'])
        assert (len(errors) == 2
                and {e['class'] for e in errors} == {'transient'}
                and {e['attempt'] for e in errors} == {1, 2}
                and 'budget_exhausted' in tools[-1].content
                and state.values['execution']['status'] == 'answered'), \
            'бюджет попыток или классификация не сработали'
        print('  >>> Отказы классифицированы как transient, записаны в tau_errors')
        print('      состояния графа; третий идентичный вызов получил дегрейд')
        print('      budget_exhausted БЕЗ исполнения — расхода и нагрузки нет.')
    finally:
        scene.close()


def scene_evidence_gate():
    """L5: негативный финал без статус-чтения переписывается Gate'ом."""
    print('═' * 78)
    print('СЦЕНА 3. L5 Evidence Gate: «не настроена» без доказательства — не диагноз')
    print('═' * 78)
    scene = Scene('gate')
    claimed = 'Синхронизация магазина не настроена: cron отсутствует.'

    class Provider:
        def bind_tools(self, tools, **kwargs):
            return self

        async def ainvoke(self, messages):
            return answer(claimed)

    try:
        # Настоящий _respond_step: модель подменена провайдером-заглушкой,
        # Gate — штатная точка врезки перед stream.finish.
        scene.patch(msty_models, 'make_model', lambda *args: Provider())
        scene.patch(msty_models, 'stamp_usage', lambda profile, result: result)
        os.environ['MSTY_MODEL_PROFILE'] = 'luna'

        async def run():
            graph = msty_native.build_graph(checkpointer=InMemorySaver(), store=InMemoryStore())
            config = {'configurable': {'thread_id': 'demo-gate'}}
            return await invoke(graph, initial('Как там синхронизация магазина?', []),
                                config)

        state = asyncio.run(run())
        rewritten = state.values['result']['content']
        show('модель утверждала', claimed)
        show('Gate выдал владельцу', rewritten, 420)
        assert ('Не могу подтвердить негативный вывод' in rewritten
                and claimed in rewritten
                and 'НЕ подтверждена' in rewritten), 'Gate не переписал неподтверждённый негатив'
        print('  >>> Негативное утверждение без успешного статус-чтения')
        print('      переписано как неподтверждённое: владелец видит, ЧТО')
        print('      проверялось, а не уверенную выдумку модели.')
    finally:
        os.environ.pop('MSTY_MODEL_PROFILE', None)
        scene.close()


def scene_subagents():
    """Sub-agents: делегирование researcher'у, отчёт, глубина 1."""
    print('═' * 78)
    print('СЦЕНА 4. Sub-agents: Brain поручает исследование под-агенту researcher')
    print('═' * 78)
    loadout = msty_subagents.role_loadout('researcher', ['brain'])
    show('loadout researcher: серверные инструменты', loadout['server_tools'])
    show('loadout researcher: внешние через родителя', loadout['delegated'][:6], 220)
    assert 'msty_delegate_task' not in loadout['delegated'], 'глубина 1 нарушена в loadout'
    assert not ({'native_write_file', 'native_edit_file'} & set(loadout['server_tools'])), \
        'researcher получил write'

    scene = Scene('subagents')
    sub_calls = []

    class SubProvider:
        def bind_tools(self, tools, **kwargs):
            sub_calls.extend(tool['function']['name'] for tool in tools
                             if tool.get('type') == 'function')
            return self

        async def ainvoke(self, messages):
            if len(sub_calls) and not getattr(self, '_listed', False):
                self._listed = True
                return answer('', [call('native_ls', {'path': '/memory'}, 's1')])
            return answer('Память содержит approved-инструкции; устаревших не найдено.')

    try:
        seen = scene.script_model([
            answer('', [call('msty_delegate_task', {
                'goal': 'Обзор /memory: какие инструкции хранятся, нет ли устаревших.',
                'role': 'researcher', 'domains': ['brain'], 'max_steps': 3}, 'd1')]),
            answer('Под-агент закончил обзор памяти: устаревших инструкций нет.')])
        provider = SubProvider()
        scene.patch(msty_models, 'make_model', lambda *args: provider)
        scene.patch(msty_models, 'bind_tools',
                    lambda profile, model, tools, choice: model.bind_tools(tools))

        async def run():
            graph = msty_native.build_graph(checkpointer=InMemorySaver(), store=InMemoryStore())
            config = {'configurable': {'thread_id': 'demo-subagents'}}
            state = await invoke(graph, initial(
                'Исследуй память проекта через под-агента и доложи.', [STATUS_TOOL]), config)
            # waiting_native: билет продолжения серверного шага, как в протоколе.
            ticket = state.tasks[0].interrupts[0]
            state = await invoke(graph, Command(resume={ticket.id: {
                **ticket.value, 'type': 'msty_native_resume'}}), config)
            return state

        state = asyncio.run(run())
        tools = [m for m in state.values['messages'] if m.type == 'tool']
        report = json.loads(tools[0].content)
        show('родитель выдал схему msty_delegate_task',
             'msty_delegate_task' in msty.tool_names(seen[0]['state']['tools']))
        show('отчёт под-агента (укорочен)', {k: report[k] for k in
             ('status', 'role', 'steps_used', 'budget', 'evidence')})
        show('findings', report['findings'], 220)
        show('errors[] под-прогона', report['errors'])
        show('под-агенту выданы схемы', sorted(set(sub_calls)))
        show('финал родителя', state.values['result']['content'])
        assert (report['status'] == 'done' and report['steps_used'] == 2
                and 'msty_delegate_task' not in sub_calls  # глубина 1 в рантайме
                and 'native_write_file' not in sub_calls   # researcher без write
                and 'устаревших не найдено' in report['findings']
                and state.values['execution']['status'] == 'answered'), \
            'под-прогон или отчёт не соответствуют контракту'
        print('  >>> Под-агент отработал в серверном контуре: свой стек сообщений,')
        print('      бюджет 3 шага (израсходовано 2), read-only loadout по манифесту,')
        print('      msty_delegate_task ему НЕ выдан — глубина 1 соблюдена.')
    finally:
        scene.close()


def scene_semantic():
    """L2 semantic: честный disabled-режим, маршрут идентичен lexical-only."""
    print('═' * 78)
    print('СЦЕНА 5. L2 Semantic: слой отключён честно, маршрут = lexical-only')
    print('═' * 78)
    # «заказы не синхронизируются» — лексический триггер промахивается (нет
    # словоформы «синхронизац»); ровно такие запросы семантика и ловит после
    # прогрева модели на деплое.
    messages = [{'role': 'user', 'content': 'Почему заказы не синхронизируются?'}]
    tools = [deepcopy(STATUS_TOOL), deepcopy(SQL_TOOL)]
    selected_on, route_on, _ = msty_tool_routing.select_tools(messages, tools)
    previous = os.environ.get('MSTY_SEMANTIC')
    os.environ['MSTY_SEMANTIC'] = 'off'
    try:
        selected_off, route_off, _ = msty_tool_routing.select_tools(messages, tools)
    finally:
        if previous is None:
            del os.environ['MSTY_SEMANTIC']
        else:
            os.environ['MSTY_SEMANTIC'] = previous
    show('route[semantic] (без fastembed в окружении)', route_on['semantic'], 320)
    show('выбрано инструментов', route_on['selected_names'] or '(ничего)')
    assert route_on['semantic']['status'] in {'disabled', 'skipped'}, \
        'ожидался честный disabled/s skipped'
    assert route_on['selected_names'] == route_off['selected_names'], \
        'маршрут с недоступным слоем ОБЯЗАН совпадать с lexical-only'
    print('  >>> Слой недоступен (fastembed не установлен) — диагностика честно')
    print('      помечает disabled, выбор инструментов байт-в-байт тот же, что')
    print('      и у lexical-only. После деплоя extra + warmup такой запрос')
    print('      даст top-K с msty_store_sync_status поверх проекции.')


SCENES = [('L3 Guard', scene_guard),
          ('L4 Таксономия + бюджет', scene_taxonomy),
          ('L5 Evidence Gate', scene_evidence_gate),
          ('Sub-agents', scene_subagents),
          ('L2 Semantic (disabled)', scene_semantic)]


def main():
    print('TAU LIVE DEMO: реальный граф msty_native, скриптованная модель,')
    print('без сервера, сети и секретов. По одной сцене на слой системы.\n')
    failures = []
    for title, scene in SCENES:
        try:
            scene()
        except Exception as error:
            failures.append(title)
            print(f'  !!! Сбой сцены: {type(error).__name__}: {brief(str(error), 400)}')
        print(f'  ВЕРДИКТ: {"FAIL" if title in failures else "PASS"} — {title}\n')
    print('═' * 78)
    passed = len(SCENES) - len(failures)
    print(f'СВОДКА: {passed}/{len(SCENES)} сцен PASS' +
          (f'; провалены: {", ".join(failures)}' if failures else ' — все слои работают.'))
    return 1 if failures else 0


if __name__ == '__main__':
    sys.exit(main())
