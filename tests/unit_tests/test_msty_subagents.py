"""Sub-agents: ограниченные под-прогоны — офлайн-тесты.

Скриптовая модель и in-memory backend: ни сети, ни провайдеров, ни ФС.
"""
import asyncio
from copy import deepcopy
import fnmatch
import json
from types import SimpleNamespace

from langchain_core.messages import AIMessage

from deep_agent import msty_native, msty_registry, msty_subagents, msty_taxonomy


class ScriptedModel:
    """Модель-очередь: отдаёт заранее заданные AIMessage, запоминает вход."""
    def __init__(self, results):
        self.results = list(results)
        self.seen = []

    async def ainvoke(self, messages):
        self.seen.append(deepcopy(messages))
        return self.results.pop(0)


class FakeBackend:
    """In-memory виртуальная ФС: тот же интерфейс, что у CompositeBackend."""
    def __init__(self, files):
        self.files = dict(files)

    async def als_info(self, path):
        prefix = path.rstrip('/') + '/'
        return [{'path': p} for p in self.files if p.startswith(prefix)]

    async def aread(self, file_path, offset=0, limit=2000):
        if file_path not in self.files:
            raise FileNotFoundError(file_path)
        return self.files[file_path]

    async def aglob_info(self, pattern, path='/'):
        return [{'path': p} for p in self.files if fnmatch.fnmatch(p, pattern)]

    async def agrep_raw(self, pattern, path=None, glob=None):
        return [{'path': p, 'line': 1, 'text': pattern}
                for p, text in self.files.items() if pattern in text]


def answer(content='', calls=None):
    return AIMessage(content=content, tool_calls=calls or [],
        usage_metadata={'input_tokens': 10, 'output_tokens': 5, 'total_tokens': 15})


def call(name, args, identifier='c1'):
    return {'id': identifier, 'name': name, 'args': args}


FILES = {'/memory/PROJECT.md': '# Проект\nBrain и его контур.',
         '/scratch/note.txt': 'синхронизация настроена, cron активен'}


def run_subagent(model, role='researcher', domains=('brain',), max_steps=4,
                 goal='Проверь заметку о синхронизации.', backend=None):
    return asyncio.run(msty_subagents.run(
        goal=goal, role=role, domains=list(domains), max_steps=max_steps,
        report_format='', model_factory=lambda schemas: model,
        backend=backend or FakeBackend(FILES)))


# --- Loadout по ролям: манифест, не доверие к аргументам ----------------------

def test_loadout_researcher_is_read_only():
    loadout = msty_subagents.role_loadout('researcher', ['brain'])
    writes = [name for name in loadout['delegated']
              if msty_registry.find(name).access == 'write']
    assert writes == []
    assert not ({'native_write_file', 'native_edit_file'} & set(loadout['server_tools']))


def test_loadout_operator_gets_write_but_delegate_is_never_granted():
    loadout = msty_subagents.role_loadout('operator', ['brain'])
    assert {'native_write_file', 'native_edit_file'} <= set(loadout['server_tools'])
    writes = [name for name in loadout['delegated']
              if msty_registry.find(name).access == 'write']
    assert writes  # operator получает write-записи домена
    for role in ('researcher', 'operator', 'auditor'):
        assert 'msty_delegate_task' not in msty_subagents.role_loadout(role, [])['delegated']


def test_loadout_auditor_only_status_read():
    loadout = msty_subagents.role_loadout('auditor', ['brain', 'commerce'])
    classes = {msty_registry.find(name).evidence_class for name in loadout['delegated']}
    assert classes == {'status_read'}
    assert 'msty_store_sync_status' in loadout['delegated']


def test_loadout_domains_filter():
    loadout = msty_subagents.role_loadout('researcher', ['supabase'])
    domains = {dom for name in loadout['delegated']
               for dom in msty_registry.find(name).domains}
    assert domains and domains <= {'supabase'}


# --- Под-прогон: глубина, бюджет, отчёт ----------------------------------------

def test_depth_guard_rejects_nested_delegation():
    model = ScriptedModel([
        answer('', [call('msty_delegate_task', {'goal': 'вложенный'})]),
        answer('Готово.')])
    report = run_subagent(model)
    assert report['status'] == 'done'
    assert len(report['errors']) == 1
    assert report['errors'][0]['class'] == msty_taxonomy.UNKNOWN_TOOL
    refusal = model.seen[1][-1]
    assert refusal.type == 'tool' and 'глубина 1' in refusal.content


def test_budget_exhaustion_gives_partial_report():
    model = ScriptedModel([
        answer('', [call('native_ls', {'path': '/memory'}, 'c1')]),
        answer('', [call('native_ls', {'path': '/scratch'}, 'c2')]),
        answer('Частичный итог: прочитана только память.')])
    report = run_subagent(model, max_steps=2)
    assert report['status'] == 'partial' and report['steps_used'] == 2
    assert report['budget'] == 2
    # После исчерпания бюджета — принудительный запрос отчёта без инструментов.
    assert 'Бюджет шагов исчерпан' in model.seen[-1][-1].content
    assert 'Частичный итог' in report['findings']


def test_env_budget_cap(monkeypatch):
    monkeypatch.setenv('MSTY_SUBAGENT_MAX_STEPS', '3')
    model = ScriptedModel([answer('ok')])
    report = run_subagent(model, max_steps=12)
    assert report['budget'] == 3


def test_report_shape_and_evidence_gate_on_negative_claim():
    model = ScriptedModel([answer('Синхронизация магазина не работает.')])
    report = run_subagent(model)
    for key in ('status', 'role', 'findings', 'evidence', 'errors',
                'steps_used', 'recommended_calls', 'artifacts'):
        assert key in report
    assert report['status'] == 'done'
    # Негатив без успешных статус-чтений переписывается как неподтверждённый.
    assert 'Не могу подтвердить негативный вывод' in report['findings']
    assert 'НЕ подтверждена' in report['findings']


def test_errors_classified_with_budget_two():
    def bad(i):
        return call('native_read_file', {'file_path': '/etc/passwd'}, f'c{i}')
    model = ScriptedModel([answer('', [bad(1)]), answer('', [bad(2)]),
                           answer('', [bad(3)]), answer('Сдаюсь.')])
    report = run_subagent(model, max_steps=4)
    deterministic = [e for e in report['errors']
                     if e['class'] == msty_taxonomy.DETERMINISTIC]
    assert len(deterministic) == 2  # два отказа записаны, третий — budget_exhausted
    assert 'budget_exhausted' in model.seen[3][-1].content
    assert report['errors'][0]['source'] == 'subagent'


def test_researcher_write_is_rejected():
    model = ScriptedModel([
        answer('', [call('native_write_file', {'file_path': '/scratch/x.txt', 'content': 'a'})]),
        answer('Понял, писать нельзя.')])
    report = run_subagent(model, role='researcher')
    assert report['artifacts'] == {}
    assert report['errors'][0]['class'] == msty_taxonomy.UNKNOWN_TOOL


def test_operator_scratch_isolated_from_parent_backend():
    backend = FakeBackend(FILES)
    model = ScriptedModel([
        answer('', [call('native_write_file', {'file_path': '/scratch/new.txt',
                                               'content': 'черновик отчёта'})]),
        answer('Записал черновик.')])
    report = run_subagent(model, role='operator', backend=backend)
    assert '/scratch/new.txt' in report['artifacts']
    assert '/scratch/new.txt' not in backend.files  # checkpoint родителя не тронут


def test_recommended_calls_only_from_loadout():
    model = ScriptedModel([
        answer('', [call('native_request_external', {
            'name': 'msty_store_sync_status', 'arguments': {}, 'reason': 'свежий статус'})]),
        answer('', [call('native_request_external', {
            'name': 'execute_sql', 'arguments': {'query': 'drop'}, 'reason': 'вне роли'})]),
        answer('Запросил статус через родителя.')])
    report = run_subagent(model, role='auditor', domains=('commerce',))
    assert report['recommended_calls'] == [
        {'name': 'msty_store_sync_status', 'args': {}, 'reason': 'свежий статус'}]
    assert report['errors'][0]['class'] == msty_taxonomy.UNKNOWN_TOOL


def test_unknown_role_fails_safely():
    report = run_subagent(ScriptedModel([answer('никогда')]), role='hacker')
    assert report['status'] == 'failed'
    assert report['errors'][0]['class'] == msty_taxonomy.INVALID_ARGS


def test_model_failure_is_classified_not_raised():
    class BrokenModel:
        async def ainvoke(self, messages):
            raise ConnectionError('synthetic transport reset')

    report = run_subagent(BrokenModel())
    assert report['status'] == 'failed'
    assert report['errors'][0]['class'] == msty_taxonomy.TRANSIENT
    assert 'Под-прогон прерван' in report['findings']


# --- Врезка в middleware родителя ------------------------------------------------

def _delegate_request(args, errors=()):
    return SimpleNamespace(
        tool_call=call('msty_delegate_task', args),
        state={'native_tool_names': ['msty_delegate_task'], 'tau_errors': list(errors)},
        runtime=None)


def test_middleware_returns_report_message(monkeypatch):
    report = {'version': 1, 'status': 'done', 'role': 'researcher', 'goal': 'g',
              'findings': 'проверено', 'evidence': ['native_read_file'], 'errors': [],
              'steps_used': 2, 'budget': 6, 'recommended_calls': [], 'artifacts': {}}

    async def fake_run(self, call, request):
        return report

    monkeypatch.setattr(msty_native.NativeMstyMiddleware, '_run_delegate', fake_run)

    async def forbidden(request):
        raise AssertionError('delegate не исполняется ToolNode')

    async def run():
        request = _delegate_request({'goal': 'g', 'role': 'researcher'})
        return await msty_native.NativeMstyMiddleware().awrap_tool_call(request, forbidden)

    result = asyncio.run(run())
    assert result.type == 'tool' and result.status == 'success'
    assert json.loads(result.content)['findings'] == 'проверено'
    assert (result.additional_kwargs or {}).get('tau_event') is None


def test_middleware_failed_report_marks_tau_error(monkeypatch):
    report = {'version': 1, 'status': 'failed', 'role': 'operator', 'goal': 'g',
              'findings': 'прерван', 'evidence': [],
              'errors': [{'class': msty_taxonomy.TRANSIENT}],
              'steps_used': 1, 'budget': 6, 'recommended_calls': [], 'artifacts': {}}

    async def fake_run(self, call, request):
        return report

    monkeypatch.setattr(msty_native.NativeMstyMiddleware, '_run_delegate', fake_run)

    async def forbidden(request):
        raise AssertionError('delegate не исполняется ToolNode')

    async def run():
        request = _delegate_request({'goal': 'g', 'role': 'operator'})
        return await msty_native.NativeMstyMiddleware().awrap_tool_call(request, forbidden)

    result = asyncio.run(run())
    event = (result.additional_kwargs or {}).get('tau_event')
    assert result.status == 'error'
    assert event['kind'] == 'failure' and event['class'] == msty_taxonomy.TRANSIENT


def test_middleware_guard_rejects_invalid_args():
    async def forbidden(request):
        raise AssertionError('невалидный вызов не должен исполняться')

    async def run():
        request = _delegate_request({'goal': 'g', 'role': 'hacker'})
        return await msty_native.NativeMstyMiddleware().awrap_tool_call(request, forbidden)

    result = asyncio.run(run())
    event = (result.additional_kwargs or {}).get('tau_event')
    # Значение аргумента в корректирующее сообщение не попадает, только поле.
    assert result.status == 'error' and 'role' in result.content
    assert 'hacker' not in result.content
    assert event['class'] == msty_taxonomy.INVALID_ARGS


def test_middleware_budget_exhausted_without_execution():
    delegate_call = call('msty_delegate_task', {'goal': 'g', 'role': 'researcher'})
    errors = [msty_taxonomy.error_entry(delegate_call, msty_taxonomy.TRANSIENT, 'subagent', 1, 0),
              msty_taxonomy.error_entry(delegate_call, msty_taxonomy.TRANSIENT, 'subagent', 2, 0)]

    async def forbidden(request):
        raise AssertionError('бюджет исчерпан: исполнения быть не должно')

    async def run():
        request = _delegate_request({'goal': 'g', 'role': 'researcher'}, errors=errors)
        return await msty_native.NativeMstyMiddleware().awrap_tool_call(request, forbidden)

    result = asyncio.run(run())
    assert result.status == 'error' and 'budget_exhausted' in result.content
    assert (result.additional_kwargs or {}).get('tau_event') is None
