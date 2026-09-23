"""Sub-agents Brain: ограниченные под-прогоны с делегированием от родителя.

Фундамент под поручение владельца «создавать ботов, которые делают кучу
работы». Brain вызывает `msty_delegate_task` (запись манифеста L1, kind:
native), а этот модуль исполняет под-агента ВНУТРИ серверного контура:
свой стек сообщений, своя модель по профилю, бюджет шагов, жёсткий
предохранитель глубины (под-агент не получает msty_delegate_task).

Граница архитектуры (честно, не «сделано»): внешние msty_* инструменты
исполняет клиент Msty через interrupt/resume — сервер их цикл крутить не
может. Поэтому loadout под-агента делится на две части:
- server_tools — серверно-исполняемые (виртуальная ФС, todos): под-агент
  вызывает их напрямую в под-прогоне;
- delegated — записи манифеста нужных доменов и access-класса: под-агент
  НЕ получает их схем, а возвращает их в отчёте как recommended_calls
  через псевдо-инструмент native_request_external; исполняет их родитель
  штатным протоколом, проверяя Guard'ом.

Стек TAU действует внутри под-прогона: имена/аргументы проверяются по
манифесту (не по доверию к аргументам), отказы классифицируются таксономией
L4 с бюджетом 2 попыток на точный вызов, финальный отчёт проходит evidence
gate — негативное утверждение без успешных статус-чтений переписывается
как неподтверждённое.

Изоляция: под-агент читает родительскую виртуальную ФС через backend, но
пишет в собственный in-memory scratch — его мутации не попадают в
checkpoint родителя; письменные артефакты возвращаются в отчёте
(`artifacts`), публикует их родитель, если сочтёт нужным.
"""
from __future__ import annotations

import fnmatch
import json
import os

from langchain_core.messages import HumanMessage, SystemMessage, ToolMessage

from . import msty_evidence, msty_registry, msty_taxonomy


DELEGATE_TOOL = 'msty_delegate_task'
#: Псевдо-инструмент декомпозиции: под-агент просит родителя исполнить
#: внешний (клиентский) вызов. Не исполняется, только записывается в отчёт.
RECOMMEND_TOOL = 'native_request_external'

DEFAULT_MAX_STEPS = 6       # консервативный бюджет под-прогона
#: Общий предел времени под-прогона (все шаги модели и инструментов).
RUN_TIMEOUT_SECONDS = 300
HARD_MAX_STEPS = 12         # потолок и для аргумента, и для env
ATTEMPT_BUDGET = msty_taxonomy.ATTEMPT_BUDGET
MAX_FINDINGS_CHARS = 4000   # отчёт не должен раздувать контекст родителя
MAX_ARTIFACT_CHARS = 500

_FS_READ = ('native_ls', 'native_read_file', 'native_glob', 'native_grep')
_FS_WRITE = ('native_write_file', 'native_edit_file')
_TODOS = 'native_write_todos'

# Шаблоны ролей как код: поведенческий слой (дисциплина планирования) поверх
# жёстких механизмов (loadout по манифесту, бюджеты, предохранитель глубины).
_ROLES = {
    'researcher': {
        'access': frozenset({'read'}),
        'server_write': False,
        'evidence_only': False,
        'prompt': (
            'Ты researcher — под-агент Brain для ДЛИННЫХ read-only исследований. '
            'Изучай виртуальную файловую систему (/memory, /skills, /memories, /scratch) '
            'методично: сначала ls/glob, затем чтение по делу. Писать файлы тебе '
            'запрещено ролью. Внешние инструменты исполняет родитель: если нужен '
            'вызов из списка доступных — запроси его через native_request_external '
            'с точными аргументами и причиной. Негативные выводы («не работает», '
            '«отсутствует») делай только при наличии проверенных чтений.'),
    },
    'operator': {
        'access': frozenset({'read', 'write'}),
        'server_write': True,
        'evidence_only': False,
        'prompt': (
            'Ты operator — под-агент Brain для КОРОТКИХ атомарных задач. Делай '
            'ровно поставленное, без расширения скоупа. Твои записи идут в '
            'изолированный scratch: они видны тебе, но не коммитятся — итоговые '
            'тексты файлов верни в отчёте, родитель опубликует их сам. Внешние '
            'мутации (клиентские msty_*) запрашивай через native_request_external.'),
    },
    'auditor': {
        'access': frozenset({'read'}),
        'server_write': False,
        'evidence_only': True,
        'prompt': (
            'Ты auditor — под-агент Brain для ПЕРЕПРОВЕРКИ критичных утверждений. '
            'Проверяй цель по первоисточнику: читай указанные файлы и сверяй '
            'утверждение с содержимым. Твой статусный контур — инструменты '
            'status_read, их исполняет родитель: запрашивай через '
            'native_request_external, когда без свежего чтения вывод сделать '
            'нельзя. Утверждение без подтверждения помечай как НЕПОДТВЕРЖДЁННОЕ, '
            'не выноси его как факт.'),
    },
}


def _env_int(name: str, default: int) -> int:
    try:
        value = int(os.environ.get(name, ''))
    except ValueError:
        return default
    return value if value > 0 else default


def budget_cap() -> int:
    """Потолок бюджета шагов: env MSTY_SUBAGENT_MAX_STEPS, иначе HARD_MAX."""
    return min(_env_int('MSTY_SUBAGENT_MAX_STEPS', HARD_MAX_STEPS), HARD_MAX_STEPS)


def delegate_schema() -> dict:
    """OpenAI-схема инструмента делегирования из записи манифеста L1."""
    entry = msty_registry.find(DELEGATE_TOOL)
    return {'type': 'function', 'function': {
        'name': DELEGATE_TOOL, 'description': entry.description,
        'parameters': entry.schema}}


def _server_schemas(write: bool) -> dict[str, dict]:
    """Статические схемы серверно-исполняемых инструментов под-агента.

    Зеркалят native-контракт родителя: пути строго виртуальные, записи только
    в /scratch/ (у под-агента — изолированный in-memory scratch).
    """
    path_desc = ('Абсолютный ВИРТУАЛЬНЫЙ путь в /scratch/, /memory/, /skills/, /memories/, '
                 '/large_tool_results/; записи только /scratch/. Не путь Mac.')
    schemas = {
        'native_ls': {'path': {'type': 'string', 'description': path_desc}},
        'native_read_file': {'file_path': {'type': 'string', 'description': path_desc}},
        'native_glob': {'pattern': {'type': 'string'},
                        'path': {'type': 'string', 'description': path_desc}},
        'native_grep': {'pattern': {'type': 'string'},
                        'path': {'type': 'string', 'description': path_desc}},
        _TODOS: {'todos': {'type': 'array', 'items': {'type': 'string'}}},
        RECOMMEND_TOOL: {
            'name': {'type': 'string', 'description': 'Имя внешнего инструмента из списка.'},
            'arguments': {'type': 'object'},
            'reason': {'type': 'string', 'description': 'Зачем этот вызов нужен отчёту.'}},
    }
    if write:
        schemas.update({
            'native_write_file': {'file_path': {'type': 'string', 'description': path_desc},
                                  'content': {'type': 'string'}},
            'native_edit_file': {'file_path': {'type': 'string', 'description': path_desc},
                                 'old_string': {'type': 'string'},
                                 'new_string': {'type': 'string'}}})
    required = {'native_ls': ['path'], 'native_read_file': ['file_path'],
                'native_glob': ['pattern', 'path'], 'native_grep': ['pattern', 'path'],
                _TODOS: ['todos'], RECOMMEND_TOOL: ['name', 'arguments', 'reason'],
                'native_write_file': ['file_path', 'content'],
                'native_edit_file': ['file_path', 'old_string', 'new_string']}
    return {name: {'type': 'function', 'function': {
                'name': name,
                'description': f'Серверный инструмент под-агента: {name}.',
                'parameters': {'type': 'object', 'properties': props,
                               'required': required[name], 'additionalProperties': False}}}
            for name, props in schemas.items()}


def role_loadout(role: str, domains: list[str]) -> dict:
    """Детерминированный loadout под-агента по манифесту L1, не по аргументам.

    Режет дважды: access-класс роли (researcher/auditor — только read) и
    пересечение доменов записи с запрошенными. msty_delegate_task исключён
    всегда — предохранитель глубины 1 на уровне манифеста.
    """
    spec = _ROLES[role]
    wanted = set(domains or ())
    delegated = []
    for entry in msty_registry.TOOLS:
        if entry.name == DELEGATE_TOOL or entry.name.startswith('native_'):
            continue
        if entry.access not in spec['access']:
            continue
        if spec['evidence_only'] and entry.evidence_class != 'status_read':
            continue
        if wanted and not (set(entry.domains) & wanted):
            continue
        delegated.append(entry.name)
    server = list(_FS_READ) + [_TODOS, RECOMMEND_TOOL]
    if spec['server_write']:
        server = list(_FS_READ) + list(_FS_WRITE) + [_TODOS, RECOMMEND_TOOL]
    return {'role': role, 'server_tools': server, 'delegated': sorted(delegated),
            'schemas': list(_server_schemas(spec['server_write']).values())}


def _valid_path(path) -> bool:
    """Зеркало msty_native._valid_virtual_path: только виртуальные корни."""
    roots = ('/scratch', '/memory', '/skills', '/memories', '/large_tool_results')
    return (isinstance(path, str) and path.startswith('/') and not path.startswith('//')
            and not any(ord(char) < 32 for char in path) and '\\' not in path
            and not any(part in {'.', '..'} for part in path.split('/'))
            and any(path == root or path.startswith(root + '/') for root in roots))


def _record(errors: list, call: dict, failure_class: str) -> dict:
    """Запись отказа под-прогона по таксономии L4 (source=subagent)."""
    entry = msty_taxonomy.error_entry(call, failure_class, 'subagent',
                                      msty_taxonomy.prior_attempts(errors, call) + 1)
    errors.append(entry)
    return entry


async def _execute(call: dict, loadout: dict, backend, scratch: dict,
                   errors: list, recommended: list) -> tuple[str, bool]:
    """Исполнить один вызов под-агента. Возвращает (content, ok).

    Guard внутри под-прогона: имя вне server_tools (включая msty_delegate_task)
    — unknown_tool без исполнения; путь вне виртуальных корней — deterministic;
    сбой backend — transient/deterministic по таксономии исключения. Бюджет
    попыток — 2 на точный вызов (fingerprint), как у родителя.
    """
    name, args = call.get('name'), call.get('args') or {}
    if name == DELEGATE_TOOL:
        _record(errors, call, msty_taxonomy.UNKNOWN_TOOL)
        return ('error=unknown_tool. Под-агент не может делегировать дальше (глубина 1). '
                'Запроси нужный внешний вызов через native_request_external.'), False
    if name == RECOMMEND_TOOL:
        target = args.get('name')
        if target not in loadout['delegated']:
            _record(errors, call, msty_taxonomy.UNKNOWN_TOOL)
            return (f"error=unknown_tool. '{target}' вне loadout роли "
                    f"(доступно: {', '.join(loadout['delegated']) or 'ничего'})."), False
        recommended.append({'name': target, 'args': args.get('arguments') or {},
                            'reason': str(args.get('reason', ''))[:300]})
        return 'Записано: родитель получит этот вызов в отчёте и решит, исполнять ли.', True
    if name not in loadout['server_tools']:
        _record(errors, call, msty_taxonomy.UNKNOWN_TOOL)
        return (f'error=unknown_tool. Инструмент {name} недоступен роли под-агента. '
                f'Доступно: {", ".join(loadout["server_tools"])}.'), False
    if msty_taxonomy.prior_attempts(errors, call) >= ATTEMPT_BUDGET:
        return ('error=budget_exhausted. Этот точный вызов уже отказал дважды; '
                'измени подход или завершай с частичным отчётом.'), False
    path = args.get('file_path') if name in {'native_read_file', 'native_write_file',
                                             'native_edit_file'} else args.get('path')
    if name != _TODOS and not _valid_path(path):
        _record(errors, call, msty_taxonomy.DETERMINISTIC)
        return ('error=deterministic. Путь обязан быть виртуальным (/scratch/, /memory/, '
                '/skills/, /memories/, /large_tool_results/); записи только /scratch/. Ничего не '
                'прочитано и не записано.'), False
    try:
        if name == _TODOS:
            todos = [str(item)[:200] for item in args.get('todos') or []][:10]
            scratch['__todos__'] = todos
            return f'Список шагов обновлён ({len(todos)} позиций).', True
        if name == 'native_ls':
            entries = await backend.als_info(path)
            names = [item.get('path', '') for item in entries]
            names += [p for p in scratch if not p.startswith('__') and p.startswith(path.rstrip('/') + '/')]
            return 'Виртуальный листинг:\n' + ('\n'.join(sorted(names)) or '(пусто)'), True
        if name == 'native_read_file':
            if path in scratch:
                return scratch[path][:8000], True
            return (await backend.aread(path))[:8000], True
        if name == 'native_glob':
            pattern = str(args.get('pattern') or '')
            entries = await backend.aglob_info(pattern, path)
            names = [item.get('path', '') for item in entries]
            names += [p for p in scratch if not p.startswith('__') and fnmatch.fnmatch(p, pattern)]
            return '\n'.join(sorted(names)) or '(совпадений нет)', True
        if name == 'native_grep':
            pattern = str(args.get('pattern') or '')
            matches = await backend.agrep_raw(pattern, path)
            lines = matches if isinstance(matches, str) else json.dumps(matches, ensure_ascii=False)[:4000]
            own = [p for p, text in scratch.items()
                   if not p.startswith('__') and pattern in text]
            return (lines + ('\nscratch: ' + ', '.join(own) if own else ''))[:4000] or '(нет совпадений)', True
        if name == 'native_write_file':
            if not path.startswith('/scratch/'):
                _record(errors, call, msty_taxonomy.DETERMINISTIC)
                return 'error=deterministic. Записи под-агента только в /scratch/ (изолированный).', False
            scratch[path] = str(args.get('content') or '')
            return 'Записано в изолированный scratch под-агента (родителю уйдёт в artifacts).', True
        if name == 'native_edit_file':
            old, new = str(args.get('old_string') or ''), str(args.get('new_string') or '')
            current = scratch.get(path)
            if current is None or old not in current:
                _record(errors, call, msty_taxonomy.DETERMINISTIC)
                return 'error=deterministic. Файл не найден в scratch или фрагмент отсутствует.', False
            scratch[path] = current.replace(old, new, 1)
            return 'Scratch-файл под-агента обновлён.', True
    except Exception as error:
        failure_class = (msty_taxonomy.TRANSIENT if msty_taxonomy.is_transient_exception(error)
                         else msty_taxonomy.DETERMINISTIC)
        _record(errors, call, failure_class)
        return (f'error={failure_class}. Чтение виртуальной ФС не удалось; детали в отчёте '
                'родителю не раскрываются.'), False
    _record(errors, call, msty_taxonomy.UNKNOWN_TOOL)
    return f'error=unknown_tool. {name} не исполняется под-прогоном.', False


def _gate_findings(findings: str, successful_reads: list, errors: list) -> str:
    """Evidence gate на отчёте под-агента: негатив без статус-чтения — не факт.

    Под-прогон не исполняет статус-чтения (status_read — внешние инструменты,
    их исполняет родитель по recommended_calls); чтение виртуальной ФС
    доказательством состояния системы не является. Поэтому диагноз-негатив
    под-агента всегда помечается неподтверждённым.
    """
    if not msty_evidence.has_negative_claim(findings):
        return findings
    checked = ', '.join(sorted(set(successful_reads))) or 'ничего'
    failed = ', '.join(sorted({str(entry.get('tool')) for entry in errors})) or 'нет'
    return ('Не могу подтвердить негативный вывод: под-агент не выполнял профильных '
            f'статус-чтений. Прочитано: {checked}; отказы: {failed}. Ниже — оценка '
            'под-агента, она НЕ подтверждена; для проверки исполни recommended_calls.'
            f'\n\n{findings}')


async def run(*, goal: str, role: str, domains: list, max_steps, report_format,
              model_factory, backend) -> dict:
    """Ограниченный под-прогон под-агента; никогда не падает в граф родителя.

    model_factory: callable(schemas) -> bound model (в тестах — скриптовая
    модель). Бюджет: запрошенный, но не выше budget_cap(); исчерпание —
    честный partial с принудительным запросом отчёта.
    """
    if role not in _ROLES:
        # Guard отклоняет это по enum схемы раньше; здесь — defence in depth,
        # чтобы вызов в обход Guard не уронил граф KeyError'ом.
        return {'version': 1, 'status': 'failed', 'role': str(role), 'goal': goal[:200],
                'findings': f"Неизвестная роль под-агента: '{role}'.",
                'evidence': [], 'errors': [msty_taxonomy.error_entry(
                    {'name': DELEGATE_TOOL, 'args': {'role': role}},
                    msty_taxonomy.INVALID_ARGS, 'subagent', 1)],
                'steps_used': 0, 'budget': 0, 'recommended_calls': [], 'artifacts': {}}
    loadout = role_loadout(role, domains)
    budget = max(1, min(max_steps or DEFAULT_MAX_STEPS, budget_cap()))
    system = (_ROLES[role]['prompt'] + f'\nБюджет: не более {budget} шагов. '
              'Заверши работу финальным ответом без вызовов инструментов.')
    delegated_hint = (', '.join(loadout['delegated'])
                      or 'нет (внешние инструменты этой роли недоступны)')
    user = (f'Цель: {goal}\n\nФорма отчёта: {report_format or "свободная, по существу"}.\n'
            f'Внешние инструменты, доступные через родителя (native_request_external): '
            f'{delegated_hint}.')
    messages = [SystemMessage(content=system), HumanMessage(content=user)]
    errors: list = []
    recommended: list = []
    successful_reads: list = []
    scratch: dict = {}
    model = model_factory(loadout['schemas'])
    status, findings, steps = 'failed', '', 0
    try:
        while steps < budget:
            steps += 1
            result = await model.ainvoke(messages)
            messages.append(result)
            calls = result.tool_calls or []
            if not calls:
                status, findings = 'done', str(result.content or '')
                break
            for call in calls:
                content, ok = await _execute(call, loadout, backend, scratch,
                                             errors, recommended)
                if ok and call.get('name') in _FS_READ:
                    successful_reads.append(call['name'])
                messages.append(ToolMessage(content=content, name=call.get('name'),
                                            tool_call_id=call.get('id', f'sub-{steps}')))
        else:
            status = 'partial'
        if status == 'partial':
            # Бюджет исчерпан: честный дегрейд — принудительный частичный отчёт,
            # новых вызовов инструментов не исполняем.
            messages.append(HumanMessage(
                content='Бюджет шагов исчерпан. Немедленно дай ЧАСТИЧНЫЙ отчёт: что '
                        'проверено, что осталось непроверенным. Без вызовов инструментов.'))
            result = await model.ainvoke(messages)
            findings = str(result.content or '')
    except Exception as error:
        failure_class = (msty_taxonomy.TRANSIENT if msty_taxonomy.is_transient_exception(error)
                         else msty_taxonomy.DETERMINISTIC)
        _record(errors, {'name': 'subagent_model', 'args': {'role': role}}, failure_class)
        findings = (f'Под-прогон прерван ({failure_class}-сбой контура модели). '
                    'Частичных результатов нет; родителю решать, повторять ли.')
        status = 'failed'
    findings = _gate_findings(findings.strip(), successful_reads, errors)
    artifacts = {path: text[:MAX_ARTIFACT_CHARS] for path, text in scratch.items()
                 if not path.startswith('__')}
    return {'version': 1, 'status': status, 'role': role,
            'goal': goal[:200], 'findings': findings[:MAX_FINDINGS_CHARS],
            'evidence': sorted(set(successful_reads)), 'errors': errors,
            'steps_used': steps, 'budget': budget,
            'recommended_calls': recommended[:10], 'artifacts': artifacts}
