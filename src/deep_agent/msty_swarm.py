"""Рой Brain (msty-swarm-v1): лид делит сложную задачу на 2–5 подзадач.

Поручение владельца 24.09.2026 (brain-desk #142): «как Kimi Agent Swarm» —
Brain сам создаёт себе агентов, у каждого свой промпт и кусок работы, затем
собирает результаты. Образцы: LangGraph Send/map-reduce (fan-out + reducer),
deepagents `task` (изолированный контекст исполнителя, наверх только итог),
OpenAI Agents SDK `as_tool` (менеджер остаётся главным и синтезирует сам),
Anthropic multi-agent research (задание = цель, формат, границы; число
агентов по сложности). Ссылки — в README и issue.

Граница учёта (почему не встроенный deepagents `task`): каждая генерация
исполнителя — отдельная строка учёта моста. Поэтому рой проходит двумя
шагами одного уже существующего native-протокола:

1. Лид вызывает `native_swarm` с планом. Как любой native-вызов, пакет
   checkpoint'ится ДО исполнения; `msty_native_continue` дополнительно несёт
   `swarm` — описатель плана (id подзадач, профили, пределы; без текста
   промптов).
2. Мост проверяет план, ставит по резерву на каждую подзадачу (correlation:
   swarm_id, subtask), проверяет общий лимит роя и аварийную остановку и
   возвращает `swarm_admission`. Только тогда ToolNode запускает Send-подграф:
   исполнители параллельны, каждый делает ровно одну генерацию без
   инструментов, событие `swarm_event` несёт измеренный usage и модель для
   расчёта строки учёта. Отклонённый рой не делает ни одного вызова модели.

Исполнитель никогда не бросает исключение наружу (иначе LangGraph отбросит
обновления всего super-step): отказ, таймаут и обрыв по длине становятся
статусом подзадачи. Итог роя `complete|partial|failed`; частичный результат
явно помечается и не выдаётся за полный. Отмена запуска мостом (stop,
отключение клиента) отменяет исполнителей; мост считает их расход
неизвестным, а не нулём.
"""
from __future__ import annotations

import asyncio
import hashlib
import json
import operator
import time
import uuid
from typing import Annotated, Callable, TypedDict

from langchain_core.messages import HumanMessage, SystemMessage
from langgraph.graph import END, START, StateGraph
from langgraph.types import Send

from . import msty_breaker, msty_execution, msty_models, msty_taxonomy

PROTOCOL = 'msty-swarm-v1'
TOOL = 'native_swarm'
EVENT = 'swarm_event'
MIN_SUBTASKS = 2


def _max_subtasks() -> int:
    """Потолок подзадач роя — конфигурацией, не кодом (brain-desk #443).

    MSTY_SWARM_MAX_SUBTASKS (2..100). По умолчанию 5 — как у моста
    (brain_swarm.MAX_SUBTASKS): поднимать вместе с BRAIN_SWARM_MAX_SUBTASKS
    моста, иначе мост отвергнет план шире своего потолка.
    """
    import os
    try:
        value = int(os.environ.get('MSTY_SWARM_MAX_SUBTASKS', '5'))
    except ValueError:
        return 5
    return min(100, max(MIN_SUBTASKS, value))


MAX_SUBTASKS = _max_subtasks()
#: Дешёвые исполнители, реально доступные облачному графу (msty_models).
#: Gemini Flash живёт только локальной полосой моста и здесь недоступен.
PROFILES = ('deepseek', 'luna')
DEFAULT_PROFILE = 'deepseek'
DEFAULT_MAX_TOKENS = 1024
MIN_MAX_TOKENS = 256
MAX_OUTPUT_TOKENS = 2048        # = выходной предел analyst-допуска моста
MAX_GOAL_CHARS = 2000
MAX_TITLE_CHARS = 80
MAX_ROLE_CHARS = 60
MAX_PROMPT_CHARS = 6000
MAX_FORMAT_CHARS = 400
MAX_RESULT_CHARS = 6000         # на подзадачу в контекст лида
WORKER_TIMEOUT_SECONDS = 75     # мост ограничивает весь ход 150 с
_NAMESPACE = uuid.UUID('2f0c2d4e-6c1b-4a8e-9f3e-5b7a0c1d2e3f')
TERMINAL = ('done', 'incomplete', 'failed', 'timeout')
#: error подзадачи при открытом контуре провайдера (статус остаётся failed —
#: множество TERMINAL моста не меняется).
PROVIDER_UNAVAILABLE = 'provider_unavailable'

WORKER_POLICY = (
    'Ты — исполнитель роя Brain. Тебе поручена ОДНА подзадача общей цели; '
    'другие части делают параллельно другие исполнители, итог собирает лид. '
    'У тебя нет инструментов, файлов, браузера и памяти: работай только с '
    'текстом задания. Не выдумывай факты, источники, результаты проверок или '
    'выполненные действия; чего не знаешь — так и пиши. Отвечай по-русски, '
    'кратко и по существу, строго в пределах своей подзадачи и в заданном формате.')


class SwarmPlanError(ValueError):
    """План роя не прошёл серверную проверку; модельных вызовов не было."""


def enabled(state) -> bool:
    """Рой доступен только лиду и только при явном протоколе от моста."""
    return (state.get('swarm_protocol') == PROTOCOL and
            state.get('brain_task_role', 'lead') == 'lead')


def schema() -> dict:
    subtask = {'type': 'object', 'additionalProperties': False,
        'required': ['title', 'prompt'],
        'properties': {
            'title': {'type': 'string', 'description': 'Короткое имя подзадачи для карточки владельца.'},
            'role': {'type': 'string', 'description': 'Роль исполнителя: исследователь, критик, аналитик…'},
            'prompt': {'type': 'string', 'description': (
                'Полное самодостаточное задание: цель, нужные факты из контекста, '
                'границы (что НЕ делать). Исполнитель не видит диалог и файлы.')},
            'output_format': {'type': 'string', 'description': 'Формат ответа исполнителя.'},
            'profile': {'type': 'string', 'enum': list(PROFILES), 'description': (
                'deepseek — DeepSeek Flash (по умолчанию), luna — GPT-6 Luna.')},
            'max_tokens': {'type': 'integer', 'minimum': MIN_MAX_TOKENS,
                           'maximum': MAX_OUTPUT_TOKENS},
        }}
    return {'type': 'function', 'function': {'name': TOOL, 'description': (
        'Рой: параллельно поручить 2–5 НЕЗАВИСИМЫХ подзадач дешёвым текстовым '
        'исполнителям и получить их ответы одним сообщением; синтез делаешь ты. '
        'Только для сложной делимой задачи (разные аспекты, варианты, источники, '
        'независимая критика). Не для простого вопроса, не для последовательных '
        'шагов и не для действий: у исполнителей нет инструментов, всё нужное '
        'включи в prompt. Один рой на ход; каждый исполнитель — отдельный платный вызов.'),
        'parameters': {'type': 'object', 'additionalProperties': False,
            'required': ['goal', 'subtasks'],
            'properties': {
                'goal': {'type': 'string', 'description': 'Общая цель роя одной-двумя фразами.'},
                'subtasks': {'type': 'array', 'minItems': MIN_SUBTASKS,
                             'maxItems': MAX_SUBTASKS, 'items': subtask},
            }}}}


def _text(value, name, limit, required=True):
    if value is None and not required:
        return ''
    if not isinstance(value, str) or (required and not value.strip()):
        raise SwarmPlanError(f'Поле {name} обязательно и должно быть строкой.')
    value = value.strip()
    if len(value) > limit:
        raise SwarmPlanError(f'Поле {name} длиннее {limit} символов.')
    return value


def parse_plan(args) -> dict:
    """Нормализованный план; любые лишние или неверные поля — отказ."""
    if not isinstance(args, dict) or set(args) - {'goal', 'subtasks'}:
        raise SwarmPlanError('План роя: допустимы только goal и subtasks.')
    goal = _text(args.get('goal'), 'goal', MAX_GOAL_CHARS)
    items = args.get('subtasks')
    if not isinstance(items, list) or not MIN_SUBTASKS <= len(items) <= MAX_SUBTASKS:
        raise SwarmPlanError(f'Рой: от {MIN_SUBTASKS} до {MAX_SUBTASKS} подзадач.')
    subtasks = []
    for index, item in enumerate(items, 1):
        if not isinstance(item, dict) or set(item) - {
                'title', 'role', 'prompt', 'output_format', 'profile', 'max_tokens'}:
            raise SwarmPlanError('Подзадача роя содержит неизвестные поля.')
        profile = item.get('profile', DEFAULT_PROFILE)
        if profile not in PROFILES:
            raise SwarmPlanError('Исполнитель роя вне серверного списка: ' + ', '.join(PROFILES) + '.')
        limit = item.get('max_tokens', DEFAULT_MAX_TOKENS)
        if type(limit) is not int or not MIN_MAX_TOKENS <= limit <= MAX_OUTPUT_TOKENS:
            raise SwarmPlanError(f'max_tokens подзадачи: {MIN_MAX_TOKENS}–{MAX_OUTPUT_TOKENS}.')
        subtasks.append({'id': 's' + str(index),
            'title': _text(item.get('title'), 'title', MAX_TITLE_CHARS),
            'role': _text(item.get('role'), 'role', MAX_ROLE_CHARS, required=False) or 'исполнитель',
            'prompt': _text(item.get('prompt'), 'prompt', MAX_PROMPT_CHARS),
            'output_format': _text(item.get('output_format'), 'output_format', MAX_FORMAT_CHARS,
                                   required=False),
            'profile': profile, 'max_tokens': limit})
    if len({s['prompt'] for s in subtasks}) != len(subtasks):
        raise SwarmPlanError('Подзадачи роя дублируют друг друга (одинаковый prompt).')
    return {'goal': goal, 'subtasks': subtasks}


def _digest(value) -> str:
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True,
        separators=(',', ':')).encode('utf-8')).hexdigest()


def descriptor(pending) -> dict | None:
    """Описатель роя для билета продолжения; детерминирован по checkpoint.

    Содержит только id/заголовки/роли/профили/пределы — не тексты заданий.
    Неверный план описателя не получает: мост ничего не резервирует, а
    ToolNode вернёт лиду ошибку плана без вызова модели.
    """
    if not isinstance(pending, dict):
        return None
    calls = [c for c in pending.get('calls') or [] if c.get('name') == TOOL]
    if len(calls) != 1:
        return None
    try:
        plan = parse_plan(calls[0].get('args'))
    except SwarmPlanError:
        return None
    return {'version': 1,
        'swarm_id': str(uuid.uuid5(_NAMESPACE, str(pending['batch_id']) + ':' + str(calls[0]['id']))),
        'tool_call_id': calls[0]['id'], 'plan_sha256': _digest(plan),
        'subtasks': [{key: item[key] for key in ('id', 'title', 'role', 'profile', 'max_tokens')}
                     for item in plan['subtasks']]}


def check_admission(swarm: dict, admission) -> dict:
    """Решение моста по описателю: admitted (все подзадачи) или rejected."""
    if not isinstance(admission, dict) or admission.get('version') != 1 or \
            type(admission.get('version')) is not int or admission.get('swarm_id') != swarm['swarm_id']:
        raise SwarmPlanError('Допуск роя не соответствует плану.')
    if admission.get('status') == 'rejected':
        reason = admission.get('reason')
        if set(admission) != {'version', 'swarm_id', 'status', 'reason'} or \
                not isinstance(reason, str) or not 0 < len(reason) <= 300:
            raise SwarmPlanError('Отказ в рое оформлен неверно.')
        return admission
    if admission.get('status') != 'admitted' or set(admission) != {
            'version', 'swarm_id', 'status', 'subtasks'}:
        raise SwarmPlanError('Допуск роя оформлен неверно.')
    items = admission['subtasks']
    if not isinstance(items, list) or [i.get('id') if isinstance(i, dict) else None
                                       for i in items] != [s['id'] for s in swarm['subtasks']]:
        raise SwarmPlanError('Допуск роя покрывает не все подзадачи плана.')
    for item, planned in zip(items, swarm['subtasks']):
        if (set(item) != {'id', 'request_id', 'binding'} or not isinstance(item['request_id'], str) or
                not 0 < len(item['request_id']) <= 128 or not _binding_matches(item['binding'], planned)):
            raise SwarmPlanError('Тарифный допуск подзадачи не совпадает с планом.')
    return admission


def _binding_matches(binding, planned) -> bool:
    """Та же сверка, что у лида (msty_execution.validate_binding): версия и
    тарифный манифест, профиль и выход из плана, вход — в окне профиля."""
    expected = {'version': 1, 'pricing_version': msty_execution.PRICING_VERSION,
                'profile': planned['profile'], 'output_limit': planned['max_tokens']}
    return (isinstance(binding, dict) and set(binding) == {*expected, 'input_limit'} and
            all(binding[key] == value for key, value in expected.items()) and
            all(type(binding[key]) is int for key in ('version', 'input_limit', 'output_limit')) and
            planned['profile'] in msty_execution.CONTEXT_WINDOWS and
            msty_execution.LEGACY_INPUT_LIMIT <= binding['input_limit'] <=
            msty_execution.window_input_limit(planned['profile']))


def admission_record(swarm: dict, admission: dict) -> dict:
    """Проверенный допуск вместе с описателем, по которому он выдан.

    Хранится приватно в checkpoint; при исполнении verify_execution сверяет
    с ним исполняемый план (вторая линия к структурной гарантии пакета).
    """
    return {**admission, 'descriptor': {key: swarm[key] for key in
                                        ('swarm_id', 'tool_call_id', 'plan_sha256', 'subtasks')}}


def verify_execution(call: dict, plan: dict, record) -> None:
    """Исполняемый план = допущенный: swarm_id, вызов, plan_sha256, подзадачи."""
    descriptor = record.get('descriptor') if isinstance(record, dict) else None
    if (not isinstance(descriptor, dict) or record.get('swarm_id') != descriptor.get('swarm_id') or
            descriptor.get('tool_call_id') != call.get('id') or
            descriptor.get('plan_sha256') != _digest(plan) or
            descriptor.get('subtasks') != [{key: item[key] for key in ('id', 'title', 'role', 'profile', 'max_tokens')}
                                           for item in plan['subtasks']]):
        raise SwarmPlanError('Исполняемый план роя не совпадает с допуском моста.')
    if record.get('status') == 'admitted' and [s.get('id') for s in record.get('subtasks') or []] != \
            [s['id'] for s in plan['subtasks']]:
        raise SwarmPlanError('Исполняемый план роя не совпадает с допуском моста.')


def _content_text(content) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return ''.join(part.get('text', '') for part in content
                       if isinstance(part, dict) and part.get('type') == 'text')
    return ''


def _identity(metadata) -> dict | None:
    if not isinstance(metadata, dict):
        return None
    picked = {key: metadata[key] for key in ('model_name', 'msty_model_profile', 'msty_model_provider')
              if isinstance(metadata.get(key), str)}
    return picked or None


class _WorkerState(TypedDict, total=False):
    swarm_id: str
    goal: str
    subtask: dict
    subtasks: list[dict]
    results: Annotated[list[dict], operator.add]


def make_worker_model(profile: str, max_tokens: int):
    """Точка подмены в тестах; в работе — тот же адаптер, что у лида."""
    return msty_models.make_model(profile, max_tokens)


class _GuardedEmit:
    """Поток событий, который не роняет super-step.

    Сбой stream writer в ветке Send отбросил бы результаты и итоговые
    usage-события всех исполнителей. Потерянное событие считается, мост
    закрывает такую строку как неизвестный расход (не ноль).
    """

    def __init__(self, emit: Callable):
        self.emit, self.lost = emit, 0

    def __call__(self, event: dict) -> None:
        try:
            self.emit(event)
        except Exception:  # noqa: BLE001 — CancelledError (BaseException) проходит
            self.lost += 1


def _event(swarm_id, sub, status, **extra) -> dict:
    return {'type': EVENT, 'version': 1, 'swarm_id': swarm_id, 'subtask': sub['id'],
            'status': status, 'profile': sub['profile'], **extra}


def _result(sub, status, error, reason, usage, text) -> dict:
    return {'id': sub['id'], 'title': sub['title'], 'role': sub['role'],
            'profile': sub['profile'], 'status': status, 'error': error,
            'finish_reason': reason if isinstance(reason, str) else None,
            'usage': ({k: usage.get(k) for k in ('input_tokens', 'output_tokens')}
                      if isinstance(usage, dict) else None),
            'text': text[:MAX_RESULT_CHARS] + ('…(усечено)' if len(text) > MAX_RESULT_CHARS else '')}


async def _worker_call(swarm_id: str, goal: str, sub: dict, emit: Callable,
                       model_factory: Callable) -> dict:
    started, usage, metadata, reason, text = False, None, None, None, ''
    status, error = 'failed', None
    begun = time.monotonic()
    # TAU L4: тот же контур, что у лида ('model:'+профиль). Открытый контур —
    # подзадача не вызывается: статус failed/provider_unavailable, started=False,
    # мост закрывает строку нулём (not_started), платного вызова нет.
    connection = 'model:' + sub['profile']
    remaining, probe_token = msty_breaker.admit(connection)
    if remaining is not None:
        emit(_event(swarm_id, sub, 'failed', started=False, usage=None, response_metadata=None,
                    finish_reason=None, chars=0, elapsed_ms=0, error=PROVIDER_UNAVAILABLE))
        return _result(sub, 'failed', PROVIDER_UNAVAILABLE, None, None, '')
    emit(_event(swarm_id, sub, 'running'))
    try:
        model = model_factory(sub['profile'], sub['max_tokens'])
        task = (f'Общая цель роя: {goal}\n\nТвоя подзадача «{sub["title"]}» (роль: {sub["role"]}):\n'
                f'{sub["prompt"]}')
        if sub.get('output_format'):
            task += '\n\nФормат ответа: ' + sub['output_format']
        messages = [SystemMessage(content=WORKER_POLICY + ' Твоя роль: ' + sub['role'] + '.'),
                    HumanMessage(content=task)]
        started = True
        try:
            raw = await asyncio.wait_for(model.ainvoke(messages), timeout=WORKER_TIMEOUT_SECONDS)
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            if msty_taxonomy.is_transient_exception(exc):
                msty_breaker.record_transient_failure(connection)
            else:
                msty_breaker.record_success(connection)  # провайдер ответил: контур жив
            raise
        msty_breaker.record_success(connection)
        try:
            usage = msty_models.checked_usage(sub['profile'], raw)
        except Exception:
            usage = None  # неизвестный расход, не ноль
        stamped = msty_models.stamp_usage(sub['profile'], raw)
        metadata = stamped.response_metadata
        reason = msty_models.finish_reason(metadata)
        text = _content_text(stamped.content).strip()
        if reason in ('length', 'max_tokens'):
            status = 'incomplete'
        elif not text:
            status, error = 'failed', 'empty'
        else:
            status = 'done'
    except asyncio.CancelledError:
        raise  # отмена запуска мостом: исход неизвестен, мост так и учтёт
    except (asyncio.TimeoutError, TimeoutError):
        status, error = 'timeout', 'timeout'
    except msty_models.ModelAdapterError:
        status, error = 'failed', 'model_identity' if started else 'model_setup'
    except Exception as exc:  # noqa: BLE001 — исполнитель не роняет super-step
        status, error = 'failed', type(exc).__name__[:60]
    finally:
        # Проба полуоткрытого контура не держится 150 с после отмены/сбоя.
        msty_breaker.release_probe(connection, probe_token)
    event = _event(swarm_id, sub, status, started=started, usage=usage,
                   response_metadata=_identity(metadata),
                   finish_reason=reason if isinstance(reason, str) else None,
                   chars=len(text), elapsed_ms=int((time.monotonic() - begun) * 1000))
    if error:
        event['error'] = error
    emit(event)
    return _result(sub, status, error, reason, usage, text)


def build_graph(emit: Callable, model_factory: Callable | None = None):
    """plan → Send(worker)×N → collect. Без checkpointer: живёт внутри ToolNode."""
    factory = model_factory or make_worker_model
    emit = emit if isinstance(emit, _GuardedEmit) else _GuardedEmit(emit)

    def fan_out(state):
        return [Send('worker', {'swarm_id': state['swarm_id'], 'goal': state['goal'], 'subtask': sub})
                for sub in state['subtasks']]

    async def worker(state):
        result = await _worker_call(state['swarm_id'], state['goal'], state['subtask'], emit, factory)
        return {'results': [result]}

    graph = StateGraph(_WorkerState)
    graph.add_node('worker', worker)
    # Fan-in: reducer operator.add уже склеил результаты; порядок — в run().
    graph.add_node('collect', lambda state: {})
    graph.add_conditional_edges(START, fan_out, ['worker'])
    graph.add_edge('worker', 'collect')
    graph.add_edge('collect', END)
    return graph.compile(name='msty_swarm')


def summary(swarm_id: str, goal: str, results: list[dict]) -> dict:
    done = [r for r in results if r['status'] == 'done']
    status = 'complete' if len(done) == len(results) else 'partial' if done else 'failed'
    report = {'version': 1, 'swarm_id': swarm_id, 'goal': goal, 'status': status,
              'completed': len(done), 'total': len(results), 'subtasks': results}
    if status != 'complete':
        missing = [f'{r["id"]} «{r["title"]}» ('
                   + ('провайдер недоступен, вызова не было' if r.get('error') == PROVIDER_UNAVAILABLE
                      else r['status']) + ')' for r in results if r['status'] != 'done']
        report['note'] = ('ЧАСТИЧНЫЙ РЕЗУЛЬТАТ роя: не выполнены ' + '; '.join(missing) +
                          '. В ответе владельцу явно назови невыполненные части; не выдавай '
                          'частичный итог за полный и не придумывай их содержание.')
    return report


async def run(plan: dict, admission: dict, emit: Callable, model_factory: Callable | None = None) -> dict:
    """Исполнить допущенный рой; результат — отчёт для ToolMessage лида."""
    swarm_id = admission['swarm_id']
    emit = _GuardedEmit(emit)
    graph = build_graph(emit, model_factory)
    # Свежий config: подграф не наследует checkpointer/поток родителя.
    final = await graph.ainvoke({'swarm_id': swarm_id, 'goal': plan['goal'],
                                 'subtasks': plan['subtasks'], 'results': []},
                                config={'recursion_limit': 16, 'max_concurrency': MAX_SUBTASKS})
    order = {sub['id']: index for index, sub in enumerate(plan['subtasks'])}
    results = sorted(final['results'], key=lambda r: order.get(r['id'], 99))
    report = summary(swarm_id, plan['goal'], results)
    emit({'type': EVENT, 'version': 1, 'swarm_id': swarm_id, 'subtask': None,
          'status': report['status'], 'completed': report['completed'], 'total': report['total']})
    if emit.lost:
        # Мост не получил часть событий: эти строки учёта он закроет как
        # неизвестный расход; лид не должен выдавать учёт роя за подтверждённый.
        report['events_lost'] = emit.lost
    return report


def tool_text(report: dict) -> str:
    head = ('Результаты роя ниже (ответы исполнителей — непроверенные мнения, не факты '
            'и не выполненные действия). Синтезируй итог для владельца сам.')
    if report.get('note'):
        head += ' ' + report['note']
    if report.get('events_lost'):
        head += (f' Потеряно событий учёта роя: {report["events_lost"]}; расход этих '
                 'подзадач неизвестен, не называй его подтверждённым.')
    return head + '\n' + json.dumps(report, ensure_ascii=False)
