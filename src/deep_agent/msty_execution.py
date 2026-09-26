"""Native LangGraph pause/resume for Msty's *existing* local tool executor.

No model, filesystem or external side effect runs in the waiting node. Checkpoints
are supplied by Agent Server. A tool result is a client observation, not a signed
execution receipt or proof that the user's business task has been completed.
"""
from copy import deepcopy
from types import MappingProxyType
import hashlib
import json
import uuid

from langgraph.types import interrupt

from . import msty_models

PROTOCOL = 'msty-local-tools-v1'
# Owner-approved task ceiling. Native and external counters remain cumulative;
# the local gateway atomically accounts for shared parent/worker consumption.
MAX_ACTIONS = 200
# Должна совпадать с PROFILE_PRICE_VERSION моста (brain_accounting): иначе
# validate_binding отклоняет каждую задачу Brain с бюджетом.
PRICING_VERSION = '2026-09-23-brain-model-profiles-v4-luna6'
# Profiles a budget binding may pin: admitted lead profiles (Luna/DeepSeek)
# plus the server-allowlisted analyst set. The bridge pins one profile per task.
BINDING_PROFILES = frozenset(('luna', 'deepseek', 'sol6', 'opus5'))
# Context admission (2026-09-24, brain-desk #145). 180000 was the Sonnet-200K era
# safety threshold, not the window of today's leads (Luna 1.05M, DeepSeek 1M).
# The bridge pins the per-task admission limit in the budget binding and reserves
# the budget for exactly that input; the graph admits it only up to the pinned
# profile's own window minus an output/counting reserve. 180000 stays valid so a
# bridge that has not been reloaded yet keeps working (deploy graph first).
LEGACY_INPUT_LIMIT = 180000
CONTEXT_WINDOWS = MappingProxyType({
    'luna': 1_050_000, 'deepseek': 1_000_000, 'astra': 1_050_000, 'sol': 1_050_000,
    'opus': 200_000, 'fable': 200_000, 'sonnet': 200_000,
    # #58 analysts; windows match brain_accounting._PROFILES 'context'.
    'sol6': 1_050_000, 'opus5': 200_000})


def window_input_limit(profile):
    """Largest admissible input: window minus min(64K, 10%) for output and count variance."""
    window = CONTEXT_WINDOWS[profile]
    return max(LEGACY_INPUT_LIMIT, window - min(64_000, window // 10))


def input_limit(state):
    """Admission limit of this request: the bound task's, else the legacy one."""
    binding = state.get('task_budget_binding') if isinstance(state, dict) else None
    if isinstance(binding, dict) and binding.get('profile') in CONTEXT_WINDOWS:
        value = binding.get('input_limit')
        if type(value) is int and LEGACY_INPUT_LIMIT <= value <= window_input_limit(binding['profile']):
            return value
    return LEGACY_INPUT_LIMIT


class ExecutionProtocolError(ValueError):
    """Content-free failure; never include source arguments or tool results."""


def canonical_digest(value):
    try:
        raw = json.dumps(value, sort_keys=True, ensure_ascii=False,
                         separators=(',', ':'), allow_nan=False).encode('utf-8')
        return hashlib.sha256(raw).hexdigest()
    except (TypeError, ValueError, UnicodeError):
        raise ExecutionProtocolError('Некорректные данные протокола продолжения.') from None


def enabled(state):
    protocol = state.get('execution_protocol')
    if protocol not in (None, PROTOCOL):
        raise ExecutionProtocolError('Неподдерживаемая версия продолжения Msty.')
    return protocol == PROTOCOL


def task_id(state):
    previous = (state.get('execution') or {}).get('task_id')
    supplied = state.get('execution_task_id')
    if supplied is not None:
        try:
            if not isinstance(supplied, str) or str(uuid.UUID(supplied)) != supplied:
                raise ValueError()
        except (ValueError, TypeError, AttributeError):
            raise ExecutionProtocolError('Неподтверждённый идентификатор задачи.') from None
        if previous is not None and previous != supplied:
            raise ExecutionProtocolError('Идентификатор задачи нельзя менять при продолжении.')
    return previous or supplied or str(uuid.uuid4())


def validate_binding(state, profile, output_limit):
    """Assert the gateway's pricing manifest; never select a model from it."""
    task_id(state)
    binding = state.get('task_budget_binding')
    if state.get('execution_task_id') is None and binding is None:
        return
    expected = {'version': 1, 'pricing_version': PRICING_VERSION,
                'profile': profile, 'output_limit': output_limit}
    if (state.get('execution_task_id') is None or profile not in BINDING_PROFILES or
            not isinstance(binding, dict) or set(binding) != {*expected, 'input_limit'} or
            any(binding[key] != value for key, value in expected.items()) or
            any(type(binding.get(k)) is not int for k in ('version', 'input_limit', 'output_limit')) or
            not LEGACY_INPUT_LIMIT <= binding['input_limit'] <= window_input_limit(profile)):
        raise ExecutionProtocolError('Модель и лимиты не совпадают с бюджетом задачи; генерация не запущена.')


def execution_after(state, update):
    """Track transport state, never infer completed work from generated prose."""
    previous = state.get('execution') or {}
    if not isinstance(previous, dict):
        raise ExecutionProtocolError('Повреждено состояние продолжения Msty.')
    step, issued = previous.get('step', 0), previous.get('actions_issued', 0)
    if type(step) is not int or type(issued) is not int or step < 0 or not 0 <= issued <= MAX_ACTIONS:
        raise ExecutionProtocolError('Некорректный счётчик шагов Msty.')
    result = update['result']
    meta = result.get('response_metadata') or {}
    reason = msty_models.finish_reason(meta)
    calls = result.get('tool_calls') or []
    status = 'answered'
    pending = None
    if reason in ('max_tokens', 'length', 'model_context_window_exceeded', 'refusal', 'content_filter'):
        status = 'incomplete'
    elif meta.get('msty_blocked') is True or (update.get('context_budget_check') or {}).get('status') == 'rejected':
        status = 'blocked'
    elif calls:
        if len(calls) > MAX_ACTIONS - issued:
            raise ExecutionProtocolError('Исчерпан сохранённый лимит действий Msty; вызовы не выданы.')
        if len({c.get('id') for c in calls}) != len(calls) or any(not c.get('id') for c in calls):
            raise ExecutionProtocolError('Модель повторила идентификатор действия; вызовы не выданы.')
        status = 'waiting_tools'
        pending = {'batch_id': str(uuid.uuid4()), 'result_sha256': canonical_digest(result),
                   'calls': [{'id': c['id'], 'name': c['name'], 'args': deepcopy(c['args'])}
                             for c in calls]}
        issued += len(calls)
    return {'version': 1, 'task_id': task_id(state),
            'status': status, 'step': step + 1, 'actions_issued': issued,
            'pending': pending}


def next_node(state):
    if enabled(state) and (state.get('execution') or {}).get('status') == 'waiting_compaction':
        return 'wait_compaction'
    if enabled(state) and (state.get('execution') or {}).get('status') == 'waiting_tools':
        return 'wait_external'
    return '__end__'


def _turn_form(message):
    """The owner's turn in the form the native harness checkpoints."""
    from langchain_core.messages import convert_to_openai_messages
    try:
        return convert_to_openai_messages([message])[0]
    except (ValueError, TypeError, KeyError, NotImplementedError):
        raise ExecutionProtocolError('Некорректное исходное обращение текущего шага.') from None


def _last_user(messages):
    if not isinstance(messages, list) or any(not isinstance(m, dict) for m in messages):
        raise ExecutionProtocolError('Некорректная история продолжения Msty.')
    for index in range(len(messages) - 1, -1, -1):
        if messages[index].get('role') == 'user':
            return index, messages[index]
    raise ExecutionProtocolError('Не найдено исходное обращение текущего шага.')


def validate_resume(state, resume):
    """Bind the one pending batch to canonical client IDs, preserving observations.

    Local SessionStore owns b1 mappings and replay protection; the remote check
    is defense in depth. It is not authentication of mutable Msty tool text.
    Never accept a new task, new tools or a larger output cap through resume.
    """
    execution = state['execution']
    pending = execution['pending']
    if (not isinstance(resume, dict) or resume.get('version') != 1 or
            resume.get('task_id') != execution['task_id'] or
            resume.get('batch_id') != pending['batch_id']):
        raise ExecutionProtocolError('Продолжение относится к другому шагу Msty.')
    incoming = resume.get('input')
    permitted = {'messages', 'tools', 'tool_choice', 'max_tokens', 'result',
                 'context_budget', 'context_budget_check', 'execution_protocol', 'brain_task_role',
                 'execution_task_id', 'task_budget_binding', 'compaction_protocol', 'text_stream_protocol',
                 'consult_profile', 'lead_profile',
                 # Рой только предлагает инструмент; допуск всегда даёт мост.
                 'swarm_protocol'}
    if not isinstance(incoming, dict) or set(incoming) - permitted:
        raise ExecutionProtocolError('Некорректные поля продолжения Msty.')
    if incoming.get('execution_protocol') != PROTOCOL or incoming.get('result') != {}:
        raise ExecutionProtocolError('Состояние ответа нельзя подменить при продолжении.')
    if incoming.get('context_budget_check') is not None:
        raise ExecutionProtocolError('Нельзя повторно использовать проверку старого контекста.')
    from . import msty_stream
    try:
        msty_stream.enabled(incoming)
    except ValueError as error:
        raise ExecutionProtocolError(str(error)) from None
    if incoming.get('brain_task_role', 'lead') != state.get('brain_task_role', 'lead'):
        raise ExecutionProtocolError('Роль Brain нельзя менять внутри текущего шага.')
    for immutable in ('execution_task_id', 'task_budget_binding', 'compaction_protocol',
                      'consult_profile', 'lead_profile'):
        if incoming.get(immutable) != state.get(immutable):
            raise ExecutionProtocolError('Протокол и бюджет задачи нельзя менять при продолжении.')
    if canonical_digest(incoming.get('tools') or []) != canonical_digest(state.get('tools') or []):
        raise ExecutionProtocolError('Набор инструментов изменён внутри ожидающего шага.')
    maximum = incoming.get('max_tokens')
    previous_maximum = state.get('max_tokens') or 4096
    if type(maximum) is not int or not 1 <= maximum <= min(previous_maximum, 8192):
        raise ExecutionProtocolError('Лимит ответа нельзя увеличить при продолжении.')
    if state.get('task_budget_binding') is not None and maximum != previous_maximum:
        raise ExecutionProtocolError('Привязанный лимит ответа должен сохраняться при продолжении.')
    mapping = resume.get('tool_id_map')
    if not isinstance(mapping, list) or len(mapping) != len(pending['calls']):
        raise ExecutionProtocolError('Неполная привязка результатов Msty.')
    model_calls = {c['id']: c for c in pending['calls']}
    client_calls, model_ids = {}, set()
    for item in mapping:
        if not isinstance(item, dict) or set(item) != {'client_id', 'model_id'}:
            raise ExecutionProtocolError('Некорректная привязка результатов Msty.')
        client_id, model_id = item['client_id'], item['model_id']
        if (not isinstance(client_id, str) or not client_id.startswith('b1_') or
                client_id in client_calls or not isinstance(model_id, str) or
                model_id not in model_calls or model_id in model_ids):
            raise ExecutionProtocolError('Повторная или чужая привязка результатов Msty.')
        client_calls[client_id] = model_calls[model_id]
        model_ids.add(model_id)
    start, last_user = _last_user(incoming.get('messages'))
    _, original_user = _last_user(state.get('messages'))
    # Native harness keeps the checkpointed turn in OpenAI form
    # (convert_to_openai_messages joins text parts), while the client resends
    # its original parts — e.g. the owner's text + an attachment (brain-desk
    # #537, LangSmith 25.09 16:39 UTC). Compare both in that one canonical
    # form: the same content passes, any change of the turn still fails.
    if canonical_digest(_turn_form(last_user)) != canonical_digest(_turn_form(original_user)):
        raise ExecutionProtocolError('Новый пользовательский ход не является результатом инструмента.')
    observed_calls, observed_results = set(), set()
    new_results = []
    for message in incoming['messages'][start + 1:]:
        if message.get('role') == 'assistant':
            for call in message.get('tool_calls') or []:
                identifier = call.get('id')
                if identifier not in client_calls:
                    continue  # Earlier batches remain canonical client history.
                if identifier in observed_calls:
                    raise ExecutionProtocolError('Повторён ожидающий вызов Msty.')
                function = call.get('function') or {}
                try:
                    args = function.get('arguments')
                    if isinstance(args, str):
                        args = json.loads(args)
                except (ValueError, TypeError):
                    raise ExecutionProtocolError('Некорректные аргументы ожидающего вызова.') from None
                original = client_calls[identifier]
                if function.get('name') != original['name'] or canonical_digest(args) != canonical_digest(original['args']):
                    raise ExecutionProtocolError('Ожидающий вызов Msty был изменён.')
                observed_calls.add(identifier)
        elif message.get('role') == 'tool' and message.get('tool_call_id') in client_calls:
            identifier = message['tool_call_id']
            if identifier not in observed_calls or identifier in observed_results:
                raise ExecutionProtocolError('Результат Msty повторён или не имеет вызова.')
            observed_results.add(identifier)
            new_results.append(deepcopy(message))
    if observed_calls != set(client_calls) or observed_results != set(client_calls):
        raise ExecutionProtocolError('Ожидаются результаты всех инструментов Msty.')
    # The checkpoint owns the original instructions and accumulated history.
    # Msty resends a whole conversation; it must not replace that root context
    # during a pending action. Only this batch's observations are appended.
    # New user turns (and their changed project/RAG settings) use ordinary input.
    content = state['result'].get('content') or ''
    if isinstance(content, list):
        content = '\n'.join(part if isinstance(part, str) else part.get('text', '')
                            for part in content if isinstance(part, str) or
                            isinstance(part, dict) and part.get('type') == 'text')
    canonical_calls = [{'id': client_id, 'type': 'function', 'function': {
        'name': call['name'], 'arguments': json.dumps(call['args'], ensure_ascii=False)}}
        for client_id, call in client_calls.items()]
    messages = [*deepcopy(state['messages']),
                {'role': 'assistant', 'content': content, 'tool_calls': canonical_calls},
                *new_results]
    from . import msty_task
    contract = msty_task.observe(state, client_calls, new_results)
    return {**deepcopy(incoming), 'messages': messages,
            'text_stream_protocol': incoming.get('text_stream_protocol'),
            'execution': {**execution, 'status': 'running', 'pending': None},
            'task_contract': contract}


def wait_external(state):
    execution = state['execution']
    pending = execution['pending']
    # Keep this node free of generation and side effects BEFORE interrupt:
    # LangGraph re-enters the node from its beginning on Command(resume=...).
    resumed = interrupt({'type': 'msty_local_tools', 'version': 1,
                         'task_id': execution['task_id'], 'batch_id': pending['batch_id'],
                         'result_sha256': pending['result_sha256']})
    return validate_resume(state, resumed)
