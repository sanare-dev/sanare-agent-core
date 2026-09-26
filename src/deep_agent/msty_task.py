"""Observation-bound completion gate for real native Msty plan/verify tools.

Coverage is model_proposed, never owner-approved completeness. Hashes bind data
integrity, not client authenticity. A passed local artifact check proves only
verified_against_observations, not business delivery or universal correctness.
No shell, model invocation, local I/O or new autonomous controller lives here.
Optional learning receipts are bounded reference-only observations: retrieval is
not demonstrated use, and unknown Engram delivery is never recorded success.
"""
from copy import deepcopy
import json
import re
import uuid

from .msty_execution import MAX_ACTIONS, canonical_digest
from . import msty_models

PLAN_SUFFIX = 'msty_task_plan'
VERIFY_SUFFIX = 'msty_task_verify'
SITE_FILE_SUFFIX = 'msty_site_file'
SITE_PATCH_SUFFIX = 'msty_site_patch'
SITE_STATUS_SUFFIX = 'msty_site_status'
SITE_CHECK_SUFFIX = 'msty_site_check'
SITE_JOB_SCHEMA = 'msty.site.job.v1'
SITE_FILE_SCHEMA = 'msty.site.file.v1'
PLAN_KEYS = {'schema', 'state', 'plan_id', 'plan_sha256', 'project_slug',
             'requirements_count', 'checks_count', 'criteria_origin', 'whole_task_completion_verified'}
VERIFY_KEYS = PLAN_KEYS | {'checks', 'source_hashes', 'receipt_path', 'receipt_sha256'}
CONTEXT_KEYS = {'schema', 'state', 'authority', 'retrieved_lesson_count', 'used_lesson_count',
                'lessons', 'quarantined_count', 'scan_truncated', 'reuse_receipt_path',
                'reuse_receipt_sha256', 'code'}
LESSON_KEYS = {'id', 'lesson', 'source', 'source_sha256', 'lesson_sha256', 'state',
               'authority', 'published', 'semantic_verification'}
OUTCOME_KEYS = {'schema', 'state', 'lesson_id', 'lesson_receipt_path', 'lesson_receipt_sha256',
                'source_sha256', 'metrics', 'engram', 'code'}
METRIC_KEYS = {'outcome', 'passed_checks', 'failed_checks', 'error_checks', 'baseline_outcome',
               'baseline_source_sha256', 'recovery_observed', 'retrieved_lesson_count', 'used_lesson_count'}
JOURNAL = 'project-governance/changes/'
HASH = re.compile(r'[0-9a-f]{64}\Z')
PLAN_ID = re.compile(r'taskplan-[0-9a-f]{32}\Z')
CONTROL = re.compile(r'(?:\b(?:stop|wait|pause)\b|only\s+(?:a\s+)?(?:plan|explain)|'
                     r'(?:do\s+not|don.t)\s+(?:verify|execute|act|run)|'
                     r'\b(?:стоп|остановись|подожди|погоди)\b|'
                     r'только\s+(?:план|объясни|объяснение|расскажи)|'
                     r'не\s+(?:выполняй|проверяй|запускай|делай)|без\s+действий)', re.I)
INCIDENT = re.compile(
    r'(?:\b(?:error|failed|failure|broken|incident|missing|stuck|timeout)\b|'
    r'\bне\s+(?:работает|запускается|синхронизируется|обновляется|появляется|видно|приходит|забирает)\b|'
    r'\b(?:ошибк\w*|сбой\w*|инцидент\w*|проблем\w*|пропал\w*|завис\w*)\b)', re.I)


def _named(name, suffix):
    return isinstance(name, str) and (name == suffix or name.endswith('_' + suffix))


def _hash(value):
    return isinstance(value, str) and HASH.fullmatch(value) is not None


def _receipt_keys(receipt, required):
    return isinstance(receipt, dict) and set(receipt) in (required, required | {'learning'})


def _bounded_json(value, limit):
    try:
        return len(json.dumps(value, ensure_ascii=False, allow_nan=False,
                              separators=(',', ':')).encode('utf-8')) <= limit
    except (ValueError, TypeError, UnicodeError, RecursionError):
        return False


def _count(value, maximum):
    return type(value) is int and 0 <= value <= maximum


def _learning_context(receipt):
    """Validate the additive v1 context without promoting its untrusted text."""
    value = receipt.get('learning')
    if (not isinstance(value, dict) or set(value) != CONTEXT_KEYS or not _bounded_json(value, 8192) or
            value['schema'] != 'msty.learning.context.v1' or
            value['state'] not in ('loaded', 'unavailable') or value['authority'] != 'reference_only' or
            value['used_lesson_count'] is not None or not _count(value['retrieved_lesson_count'], 3) or
            not _count(value['quarantined_count'], 256) or type(value['scan_truncated']) is not bool or
            not isinstance(value['lessons'], list) or len(value['lessons']) != value['retrieved_lesson_count']):
        return False
    if value['state'] == 'loaded':
        if value['code'] is not None:
            return False
    elif value['code'] not in ('learning_context_unavailable', 'learning_reuse_not_persisted'):
        return False
    expected_path = JOURNAL + 'msty-learning-reuse-' + receipt['plan_id'][9:] + '.json'
    if value['reuse_receipt_path'] is None:
        if value['reuse_receipt_sha256'] is not None or value['state'] != 'unavailable':
            return False
    elif value['reuse_receipt_path'] != expected_path or not _hash(value['reuse_receipt_sha256']):
        return False
    identities = set()
    for lesson in value['lessons']:
        if (not isinstance(lesson, dict) or set(lesson) != LESSON_KEYS or
                not isinstance(lesson['id'], str) or not re.fullmatch(r'[A-Za-z0-9_-]{1,110}', lesson['id']) or
                lesson['id'] in identities or not isinstance(lesson['lesson'], str) or
                not 10 <= len(lesson['lesson']) <= 2000 or
                not isinstance(lesson['source'], str) or len(lesson['source']) > 255 or
                not re.fullmatch(r'[A-Za-z0-9_.-]+\.json', lesson['source']) or
                not _hash(lesson['source_sha256']) or not _hash(lesson['lesson_sha256']) or
                lesson['state'] != 'candidate' or lesson['authority'] != 'reference_only' or
                lesson['published'] is not False or lesson['semantic_verification'] != 'not_performed'):
            return False
        identities.add(lesson['id'])
    return True


def _learning_outcome(contract, receipt):
    """Bind counters to this observation; NAS status remains a separate fact."""
    value = receipt.get('learning')
    if (not isinstance(value, dict) or set(value) != OUTCOME_KEYS or not _bounded_json(value, 4096) or
            value['schema'] != 'msty.learning.outcome.v1'):
        return False
    if value['state'] == 'error':
        return (value['code'] == 'local_learning_not_persisted' and
                all(value[key] is None for key in ('lesson_id', 'lesson_receipt_path',
                    'lesson_receipt_sha256', 'source_sha256', 'metrics')) and
                value['engram'] == {'state': 'not_attempted', 'code': 'local_lesson_not_persisted',
                                    'event_id': None})
    source = receipt['receipt_sha256']
    lesson_id = 'msty-brain-lesson-auto-' + source
    if (value['state'] != 'recorded' or value['code'] is not None or value['source_sha256'] != source or
            value['lesson_id'] != lesson_id or value['lesson_receipt_path'] != JOURNAL + lesson_id + '.json' or
            not _hash(value['lesson_receipt_sha256'])):
        return False
    metrics, engram = value['metrics'], value['engram']
    if (not isinstance(metrics, dict) or set(metrics) != METRIC_KEYS or
            metrics['outcome'] != receipt['state'] or
            metrics['baseline_outcome'] not in ('passed', 'failed', 'error') or
            not _hash(metrics['baseline_source_sha256']) or type(metrics['recovery_observed']) is not bool or
            metrics['recovery_observed'] != (metrics['baseline_outcome'] in ('failed', 'error') and
                                            receipt['state'] == 'passed') or
            not _count(metrics['retrieved_lesson_count'], 3) or metrics['used_lesson_count'] is not None):
        return False
    context = contract.get('learning_context')
    if context is not None and metrics['retrieved_lesson_count'] != context['retrieved_lesson_count']:
        return False
    for outcome in ('passed', 'failed', 'error'):
        count = metrics[outcome + '_checks']
        if not _count(count, 24) or count != sum(check['status'] == outcome for check in receipt['checks']):
            return False
    if (not isinstance(engram, dict) or set(engram) != {'state', 'code', 'event_id'} or
            engram['event_id'] != 'msty-auto-' + source):
        return False
    if engram['state'] == 'recorded':
        return engram['code'] is None
    if engram['state'] == 'not_attempted':
        return engram['code'] in ('engram_adapter_unavailable', 'engram_adapter_error')
    return engram['state'] == 'unknown' and engram['code'] in (
        'engram_timeout_no_retry', 'engram_failed_outcome_unknown', 'engram_adapter_error',
        'engram_prior_attempt_unconfirmed', 'engram_delivery_record_unavailable')


def _decode(value, depth=0):
    # Msty serializes HTTP -> MCP TextContent -> JSON. Bound nesting while
    # admitting that real transport; a transport envelope is not a receipt.
    if depth > 8:
        return None
    if isinstance(value, str):
        try:
            return _decode(json.loads(value), depth + 1)
        except (ValueError, TypeError):
            return None
    if isinstance(value, list):
        if len(value) == 1 and isinstance(value[0], dict) and value[0].get('type') == 'text':
            return _decode(value[0].get('text'), depth + 1)
        return None
    if isinstance(value, dict):
        if value.get('isError'):
            return None
        if 'status' in value or 'body' in value:
            if (set(value) != {'status', 'body'} or type(value['status']) is not int or
                    not 200 <= value['status'] < 300):
                return None
            return _decode(value['body'], depth + 1)
        if 'schema' in value:
            return value
        structured = _decode(value['structuredContent'], depth + 1) if 'structuredContent' in value else None
        text = _decode(value['content'], depth + 1) if 'content' in value else None
        if structured is not None and text is not None and structured != text:
            return None
        return structured if structured is not None else text
    return None


def _plan(call, receipt):
    args = call.get('args') or {}
    requirements, checks = args.get('requirements'), args.get('checks')
    if (not _receipt_keys(receipt, PLAN_KEYS) or
            receipt.get('schema') != 'msty.task.plan.v1' or receipt.get('state') != 'planned' or
            receipt.get('criteria_origin') != 'model_proposed' or
            receipt.get('whole_task_completion_verified') is not False or
            not isinstance(receipt.get('plan_id'), str) or not PLAN_ID.fullmatch(receipt['plan_id']) or
            not _hash(receipt.get('plan_sha256')) or
            not isinstance(requirements, list) or not 1 <= len(requirements) <= 12 or
            not isinstance(checks, list) or not 1 <= len(checks) <= 24 or
            type(receipt.get('requirements_count')) is not int or receipt['requirements_count'] != len(requirements) or
            type(receipt.get('checks_count')) is not int or receipt['checks_count'] != len(checks) or
            receipt.get('project_slug') != args.get('project_slug')):
        return None
    summaries = []
    for index, check in enumerate(checks):
        if (not isinstance(check, dict) or type(check.get('requirement_index')) is not int or
                not 0 <= check['requirement_index'] < len(requirements) or
                check.get('kind') not in ('sha256', 'text_contains', 'json_pointer_equals') or
                not isinstance(check.get('path'), str)):
            return None
        summaries.append({'index': index, 'requirement_index': check['requirement_index'],
                          'kind': check['kind'], 'path': check['path']})
    if {c['requirement_index'] for c in summaries} != set(range(len(requirements))):
        return None
    if 'learning' in receipt and not _learning_context(receipt):
        return None
    contract = {'version': 1, 'status': 'planned', 'criteria_origin': 'model_proposed',
            'whole_task_completion_verified': False, 'plan_id': receipt['plan_id'],
            'plan_sha256': receipt['plan_sha256'], 'project_slug': receipt['project_slug'],
            'requirements_count': len(requirements), 'checks_count': len(checks),
            'checks': summaries, 'plan_call_sha256': canonical_digest(call)}
    if 'learning' in receipt:
        context = deepcopy(receipt['learning'])
        # The native message already carries candidate text. The checkpoint only
        # retains receipt references, never a second privileged instruction copy.
        context['lesson_refs'] = [{key: item[key] for key in
            ('id', 'source', 'source_sha256', 'lesson_sha256')} for item in context.pop('lessons')]
        contract['learning_context'] = context
    return contract


def _verified(contract, call, receipt):
    if (not _receipt_keys(receipt, VERIFY_KEYS) or
            receipt.get('schema') != 'msty.task.verification.v1' or receipt.get('state') != 'passed' or
            call.get('args') != {'plan_id': contract.get('plan_id')} or
            receipt.get('criteria_origin') != 'model_proposed' or
            receipt.get('whole_task_completion_verified') is not False or
            any(receipt.get(key) != contract.get(key) for key in
                ('plan_id', 'plan_sha256', 'project_slug', 'requirements_count', 'checks_count')) or
            any(type(receipt.get(key)) is not int for key in ('requirements_count', 'checks_count')) or
            not _hash(receipt.get('receipt_sha256')) or not isinstance(receipt.get('receipt_path'), str) or
            not receipt['receipt_path'].startswith('project-governance/changes/') or
            not isinstance(receipt.get('checks'), list) or len(receipt['checks']) != contract['checks_count'] or
            not isinstance(receipt.get('source_hashes'), list)):
        return False
    sources = {}
    for source in receipt['source_hashes']:
        if (not isinstance(source, dict) or set(source) != {'path', 'sha256', 'size_bytes'} or
                not isinstance(source['path'], str) or source['path'] in sources or
                not _hash(source['sha256']) or type(source['size_bytes']) is not int or source['size_bytes'] < 0):
            return False
        sources[source['path']] = source
    if set(sources) != {check['path'] for check in contract['checks']}:
        return False
    for expected, observed in zip(contract['checks'], receipt['checks']):
        if (not isinstance(observed, dict) or set(observed) != {'index', 'requirement_index', 'kind',
                'status', 'code', 'artifact_sha256', 'size_bytes'} or
                observed['status'] != 'passed' or observed['code'] != 'match' or
                type(observed['index']) is not int or type(observed['requirement_index']) is not int or
                any(observed.get(k) != expected[k] for k in ('index', 'requirement_index', 'kind')) or
                observed['artifact_sha256'] != sources[expected['path']]['sha256'] or
                type(observed['size_bytes']) is not int or observed['size_bytes'] != sources[expected['path']]['size_bytes']):
            return False
    return 'learning' not in receipt or _learning_outcome(contract, receipt)


def observe(state, issued_calls, results):
    """Only actual-issued mapped calls can update the checkpoint contract."""
    contract = deepcopy(state.get('task_contract'))
    # Two plans, or plan+verify in one batch, have ambiguous dependency order.
    plan_calls = [c for c in issued_calls.values() if _named(c.get('name'), PLAN_SUFFIX)]
    verify_calls = [c for c in issued_calls.values() if _named(c.get('name'), VERIFY_SUFFIX)]
    if (len(plan_calls) > 1 or plan_calls and verify_calls or
            verify_calls and (len(verify_calls) != 1 or len(issued_calls) != 1)):
        return {'version': 1, 'status': 'blocked', 'criteria_origin': 'model_proposed',
                'whole_task_completion_verified': False, 'reason': 'ambiguous_plan_batch'}
    for result in results:
        call = issued_calls.get(result.get('tool_call_id'))
        if not call:
            continue
        receipt = _decode(result.get('content'))
        if _named(call.get('name'), PLAN_SUFFIX):
            contract = _plan(call, receipt) or {'version': 1, 'status': 'blocked',
                'criteria_origin': 'model_proposed', 'whole_task_completion_verified': False,
                'reason': 'unconfirmed_plan_observation'}
        elif _named(call.get('name'), VERIFY_SUFFIX) and contract:
            passed = _verified(contract, call, receipt) if contract.get('plan_id') else False
            contract = {**contract, 'status': 'verified_against_observations' if passed else 'blocked',
                'verification_call_sha256': canonical_digest(call),
                'verification_observation_sha256': canonical_digest(receipt),
                'reason': None if passed else 'checks_not_confirmed'}
            # Clear a prior successful learning observation on every new verify;
            # only the current, valid, issued receipt can set it again.
            contract.pop('learning_outcome', None)
            if passed and 'learning' in receipt:
                contract['learning_outcome'] = deepcopy(receipt['learning'])
    return contract


def owner_control(state):
    """Conservative veto, not a general semantic intent/authorization parser."""
    for message in reversed(state.get('messages') or []):
        if message.get('role') == 'user':
            content = message.get('content')
            return not isinstance(content, str) or CONTROL.search(content) is not None
    return True


def _latest_user_index_and_text(state):
    messages = state.get('messages') or []
    for index in range(len(messages) - 1, -1, -1):
        message = messages[index]
        if message.get('role') == 'user':
            content = message.get('content')
            return index, content if isinstance(content, str) else None
    return None, None


def _incident_needs_first_action(state):
    """A reported failure is an implicit outcome task, not a diagnosis request."""
    index, content = _latest_user_index_and_text(state)
    if index is None or content is None or CONTROL.search(content) or not INCIDENT.search(content):
        return False, None
    for message in (state.get('messages') or [])[index + 1:]:
        if message.get('role') == 'tool' or (
                message.get('role') == 'assistant' and message.get('tool_calls')):
            return False, None
    return True, content


def _outcome_start_gate(state, result, tools, disabled):
    """Replace unsupported first-step incident diagnosis with one real read."""
    meta = result.response_metadata
    needed, content = _incident_needs_first_action(state)
    if (not needed or result.tool_calls or result.invalid_tool_calls or disabled or
            meta.get('msty_blocked') or meta.get('msty_generation') == 'not_started' or
            msty_models.finish_reason(meta) in
                ('max_tokens', 'length', 'model_context_window_exceeded', 'refusal', 'content_filter') or
            (state.get('execution') or {}).get('actions_issued', 0) >= MAX_ACTIONS):
        return None
    names = [tool.get('function', {}).get('name') for tool in tools
             if tool.get('type') == 'function' and
             _named(tool.get('function', {}).get('name'), 'msty_admin_memory_search')]
    choice = state.get('tool_choice')
    selected = choice.get('function', {}).get('name') if isinstance(choice, dict) else None
    if len(names) != 1 or selected is not None and names != [selected]:
        return None
    # The installed memory tool has a strict 300-character query contract.
    query = ' '.join(content.split())[:300]
    return result.model_copy(update={
        'content': 'Сначала проверяю сохранённый контекст этого инцидента, затем продолжаю до результата.',
        'tool_calls': [{'id': 'outcome_' + uuid.uuid4().hex, 'name': names[0],
                        'args': {'query': query}, 'type': 'tool_call'}],
        'invalid_tool_calls': [], 'additional_kwargs': {},
        'response_metadata': {**meta, 'msty_completion_gate': 'incident_first_action_required'}})


def _call_args(call):
    if not isinstance(call, dict):
        return None, None
    function = call.get('function')
    if isinstance(function, dict):
        name, raw = function.get('name'), function.get('arguments')
        if isinstance(raw, str):
            try:
                raw = json.loads(raw)
            except ValueError:
                raw = None
        return name, raw if isinstance(raw, dict) else None
    args = call.get('args')
    return call.get('name'), args if isinstance(args, dict) else None


def _current_messages(state):
    """Messages after the latest owner instruction only."""
    messages = state.get('messages') or []
    start = 0
    for index, message in enumerate(messages):
        if isinstance(message, dict) and message.get('role') == 'user':
            start = index + 1
    return messages[start:]


def repeated_observation(state):
    """Return a repeated call/result pair that has stopped making progress.

    Two byte-identical observations for the same tool and canonical arguments are
    sufficient evidence that a third identical call is not a new strategy.  The
    result text stays hashed: it is untrusted data and must not be promoted into
    the system instruction.
    """
    calls = {}
    observations = {}
    for message in _current_messages(state):
        if not isinstance(message, dict):
            continue
        if message.get('role') == 'assistant':
            for call in message.get('tool_calls') or []:
                name, args = _call_args(call)
                identifier = call.get('id') if isinstance(call, dict) else None
                if isinstance(identifier, str) and isinstance(name, str) and args is not None:
                    calls[identifier] = (name, canonical_digest(args))
        elif message.get('role') == 'tool':
            identifier = message.get('tool_call_id')
            call = calls.get(identifier)
            if call is None:
                continue
            key = (*call, canonical_digest({'content': message.get('content', ''),
                                             'name': message.get('name')}))
            observations[key] = observations.get(key, 0) + 1
            if observations[key] >= 2:
                return {'tool_name': call[0], 'arguments_sha256': call[1],
                        'observation_sha256': key[2], 'count': observations[key]}
    return None


def progress_intervention(state):
    """A bounded system intervention that asks the model to choose a new branch."""
    repeated = repeated_observation(state)
    if repeated is None:
        return None
    return (
        'MSTY_STAGNATION_INTERVENTION_V1. Инструмент '
        f'{repeated["tool_name"]} с теми же аргументами уже дважды вернул одинаковое '
        'наблюдение. Третий идентичный вызов запрещён: он не является прогрессом. '
        'Снова сопоставь исходный требуемый результат с фактами и выбери материально '
        'другую ветку: другой инструмент/источник, изменённые аргументы, исправление '
        'причины либо точный внешний блокер. Не повторяй диагностику и не выдавай '
        'промежуточное наблюдение за выполненную задачу.'
    )


# Open plan gate (brain-desk #443) — the Stop hook of agent harnesses: Codex
# CLI Stop hooks and Claude Code «decision: block» (the Ralph loop is built on
# it). A turn that tries to end while the plan written in THIS owner turn
# still has pending/in_progress items does not end: the model gets its plan
# back and continues. The plan is either the native TODO list or the Brain
# Desk window's task harness tool (brain_task_plan, «До результата»). Bounded:
# at most OPEN_PLAN_GATE_LIMIT times per owner turn, never twice without a real
# action in between, never against an owner control phrase, never past the
# shared action cap. Every continuation is an ordinary metered step (native
# ticket or the client's external round trip): the bridge budget, emergency
# stop and approvals are untouched.
NATIVE_TODOS = 'native_write_todos'
TASK_PLAN_SUFFIX = 'brain_task_plan'
TASK_CHECK_SUFFIX = 'brain_task_check'
OPEN_PLAN_GATE_LIMIT = 3
PLAN_GATE_PREFIX = 'plangate_'
OPEN_STATUSES = frozenset({'pending', 'in_progress'})
_TERMINATED = ('max_tokens', 'length', 'model_context_window_exceeded', 'refusal', 'content_filter')


def _plan_items(name, args):
    """(kind, [(title, status)]) of one plan call, None for any other call."""
    if name == NATIVE_TODOS and isinstance(args.get('todos'), list):
        return 'native', [(item.get('content'), item.get('status'))
                          for item in args['todos'] if isinstance(item, dict)]
    if _named(name, TASK_PLAN_SUFFIX) and isinstance(args.get('items'), list):
        return 'task', [(item.get('title'), item.get('status'))
                        for item in args['items'] if isinstance(item, dict)]
    return None


def _plan_bookkeeping(name):
    return name == NATIVE_TODOS or _named(name, TASK_PLAN_SUFFIX) or _named(name, TASK_CHECK_SUFFIX)


def open_plan(state):
    """The latest plan written in this owner turn: kind, arguments, open titles."""
    latest = None
    for message in _current_messages(state):
        if not isinstance(message, dict) or message.get('role') != 'assistant':
            continue
        for call in message.get('tool_calls') or []:
            name, args = _call_args(call)
            found = _plan_items(name, args) if args is not None else None
            if found is None:
                continue
            latest = {'kind': found[0], 'name': name, 'args': args, 'open': [
                ' '.join(title.split())[:120] for title, status in found[1]
                if status in OPEN_STATUSES and isinstance(title, str) and title.strip()]}
    return latest


def _gate_history(state):
    """(gates fired in this owner turn, real actions after the last one)."""
    fired = since = 0
    for message in _current_messages(state):
        if not isinstance(message, dict) or message.get('role') != 'assistant':
            continue
        for call in message.get('tool_calls') or []:
            identifier = call.get('id') if isinstance(call, dict) else None
            name, _ = _call_args(call)
            if isinstance(identifier, str) and identifier.startswith(PLAN_GATE_PREFIX):
                fired, since = fired + 1, 0
            elif isinstance(name, str) and not _plan_bookkeeping(name):
                since += 1
    return fired, since


def open_plan_intervention(state):
    """System note for the step right after a gate: why the turn goes on."""
    messages = _current_messages(state)
    last = next((m for m in reversed(messages) if isinstance(m, dict) and m.get('role') == 'assistant'), None)
    if last is None or not any(isinstance(call, dict) and str(call.get('id', '')).startswith(PLAN_GATE_PREFIX)
                               for call in last.get('tool_calls') or []):
        return None
    plan = open_plan(state)
    if plan is None or not plan['open']:
        return None
    left = '; '.join(plan['open'][:8])
    return ('MSTY_OPEN_PLAN_V1. Ты попытался закончить ход, но в твоём плане открыты пункты: '
            f'{left}. Ход не закончен. Сделай следующий пункт реальными инструментами и проверь '
            'результат. Пункт, который сделать нельзя, отметь в плане и прямо назови блокер и что '
            'нужно от владельца. Не заканчивай ход обещанием («теперь сделаю…», «сейчас проверю…»): '
            'итог — только после проверки, с доказательствами.')


def open_plan_gate(state, result, tools, disabled):
    """Continue a turn that would end with open plan items (see the note above)."""
    meta = result.response_metadata
    if (result.tool_calls or result.invalid_tool_calls or disabled or owner_control(state) or
            meta.get('msty_blocked') or meta.get('msty_generation') == 'not_started' or
            meta.get('stop_reason', meta.get('finish_reason')) in _TERMINATED or
            (state.get('execution') or {}).get('actions_issued', 0) >= MAX_ACTIONS):
        return None
    plan = open_plan(state)
    if plan is None or not plan['open']:
        return None
    fired, progress = _gate_history(state)
    if fired >= OPEN_PLAN_GATE_LIMIT or (fired and not progress):
        # No endless loop: the answer goes out, honestly marked unfinished.
        update = {'response_metadata': {**meta, 'msty_completion_gate': 'open_plan_released'}}
        if isinstance(result.content, str):
            update['content'] = (result.content.rstrip() + '\n\n_Не закрыто в плане: '
                                 + '; '.join(plan['open'][:8]) + '._')
        return result.model_copy(update=update)
    names = [tool.get('function', {}).get('name') for tool in tools
             if isinstance(tool, dict) and tool.get('type') == 'function']
    choice = state.get('tool_choice')
    selected = choice.get('function', {}).get('name') if isinstance(choice, dict) else None
    identifier = PLAN_GATE_PREFIX + uuid.uuid4().hex
    check = next((name for name in names if _named(name, TASK_CHECK_SUFFIX)), None)
    if plan['kind'] == 'task' and check and selected in (None, check):
        draft = result.content if isinstance(result.content, str) else ''
        call = {'id': identifier, 'name': check, 'type': 'tool_call', 'args': {
            'stop_attempt': True, 'open_items': plan['open'][:12], 'draft': draft[:1500]}}
    elif plan['kind'] == 'native' and NATIVE_TODOS in names and selected is None:
        call = {'id': identifier, 'name': NATIVE_TODOS, 'type': 'tool_call', 'args': deepcopy(plan['args'])}
    else:
        return None
    return result.model_copy(update={
        'content': 'Ход не закончен: в плане открыты пункты — продолжаю работу.',
        'tool_calls': [call], 'invalid_tool_calls': [], 'additional_kwargs': {},
        'response_metadata': {**meta, 'msty_completion_gate': 'open_plan_continue'}})


def _site_progress(state, job_id):
    """Latest executor view and consecutive identical views for one site job."""
    latest = None
    latest_digest = None
    repeated = 0
    for message in _current_messages(state):
        if not isinstance(message, dict) or message.get('role') != 'tool':
            continue
        receipt = _decode(message.get('content'))
        if (not isinstance(receipt, dict) or receipt.get('schema') != SITE_JOB_SCHEMA or
                receipt.get('job_id') != job_id):
            continue
        material = {key: receipt.get(key) for key in
                    ('state', 'active_check', 'unchecked_writes', 'checks', 'error',
                     'cancel_requested', 'commit_sha', 'pr', 'release')}
        digest = canonical_digest(material)
        if digest == latest_digest:
            repeated += 1
        else:
            latest_digest, repeated = digest, 1
        latest = receipt
    return latest, repeated


def site_jobs(state):
    """Site copies edited in this conversation and what the executor last said about them.

    Evidence is executor receipts only: a write/patch marks the job dirty; only a
    later status view with unchecked_writes False and a passed typecheck clears it.
    Model prose never counts. Returns {job_id: 'dirty' | 'typecheck_failed' | 'clean'}.
    """
    jobs = {}
    for message in state.get('messages') or []:
        if not isinstance(message, dict):
            continue
        if message.get('role') == 'tool':
            receipt = _decode(message.get('content'))
            if not isinstance(receipt, dict) or not isinstance(receipt.get('job_id'), str):
                continue
            job = receipt['job_id']
            if receipt.get('schema') == SITE_FILE_SCHEMA and receipt.get('state') in ('written', 'patched'):
                jobs[job] = 'dirty'
            elif receipt.get('schema') == SITE_JOB_SCHEMA:
                if receipt.get('state') == 'cancelled':
                    jobs.pop(job, None)
                    continue
                if job not in jobs:
                    continue
                typecheck = (receipt.get('checks') or {}).get('typecheck') or {}
                if receipt.get('unchecked_writes') is False and typecheck.get('passed') is True:
                    jobs[job] = 'clean'
                elif receipt.get('unchecked_writes') is True and 'typecheck' in (receipt.get('checks') or {}):
                    jobs[job] = 'typecheck_failed' if typecheck.get('passed') is not True else 'dirty'
                else:
                    jobs[job] = 'dirty'
    return jobs


def _site_gate(state, result, tools, disabled):
    """Edited site copy without a current passed typecheck cannot be reported as done."""
    meta = result.response_metadata
    if (result.tool_calls or result.invalid_tool_calls or disabled or owner_control(state) or
            meta.get('msty_blocked') or meta.get('msty_generation') == 'not_started' or
            msty_models.finish_reason(meta) in
                ('max_tokens', 'length', 'model_context_window_exceeded', 'refusal', 'content_filter')):
        return None
    pending = {job: status for job, status in site_jobs(state).items() if status != 'clean'}
    if not pending:
        return None
    if any(status == 'typecheck_failed' for status in pending.values()):
        return result.model_copy(update={
            'content': 'Выполнение не подтверждено: typecheck изменённой копии сайта не пройден. '
                       'Нужны исправление и повторная проверка через msty_site_status; '
                       'текст модели результатом не является.',
            'tool_calls': [], 'invalid_tool_calls': [], 'additional_kwargs': {},
            'response_metadata': {**meta, 'msty_blocked': True, 'msty_completion_gate': 'site_typecheck_failed'}})
    names = [tool.get('function', {}).get('name') for tool in tools
             if tool.get('type') == 'function' and _named(tool.get('function', {}).get('name'), SITE_STATUS_SUFFIX)]
    choice = state.get('tool_choice')
    selected = choice.get('function', {}).get('name') if isinstance(choice, dict) else None
    if (len(names) != 1 or selected is not None and names != [selected] or
            (state.get('execution') or {}).get('actions_issued', 0) >= MAX_ACTIONS):
        return result.model_copy(update={
            'content': 'Изменения в копии сайта не проверены (typecheck не запускался или не завершён); '
                       'завершение не подтверждено.',
            'tool_calls': [], 'invalid_tool_calls': [], 'additional_kwargs': {},
            'response_metadata': {**meta, 'msty_blocked': True, 'msty_completion_gate': 'site_typecheck_required'}})
    job = next(iter(pending))
    latest, repeated = _site_progress(state, job)
    if latest is not None:
        state_name = latest.get('state')
        # A ready executor that did not start/record typecheck will not change by
        # being read again. Switch from polling to the explicit check operation.
        if state_name == 'ready' and not (latest.get('checks') or {}).get('typecheck'):
            check_names = [tool.get('function', {}).get('name') for tool in tools
                           if tool.get('type') == 'function' and
                           _named(tool.get('function', {}).get('name'), SITE_CHECK_SUFFIX)]
            if len(check_names) == 1 and (selected is None or selected == check_names[0]):
                return result.model_copy(update={
                    'content': 'Статус не запустил проверку; запускаю typecheck отдельной операцией.',
                    'tool_calls': [{'id': 'sitecheck_' + uuid.uuid4().hex, 'name': check_names[0],
                                    'args': {'job_id': job, 'preset': 'typecheck', 'test_paths': []},
                                    'type': 'tool_call'}],
                    'invalid_tool_calls': [], 'additional_kwargs': {},
                    'response_metadata': {**meta, 'msty_completion_gate':
                                          'site_explicit_typecheck_required'}})
        # Each status call already waits up to 30 seconds. Three identical
        # in-flight observations are a stalled executor, not permission to spin.
        if state_name in ('preparing', 'checking', 'releasing') and repeated >= 3:
            return result.model_copy(update={
                'content': 'Выполнение не подтверждено: исполнитель сайта трижды вернул '
                           'одно и то же состояние без прогресса. Повторный опрос остановлен; '
                           'нужна проверка или восстановление самого исполнительного job.',
                'tool_calls': [], 'invalid_tool_calls': [], 'additional_kwargs': {},
                'response_metadata': {**meta, 'msty_blocked': True,
                                      'msty_completion_gate': 'site_executor_stalled'}})
    return result.model_copy(update={
        'content': 'Проверяю typecheck изменённой копии сайта перед итогом.',
        'tool_calls': [{'id': 'sitecheck_' + uuid.uuid4().hex, 'name': names[0],
                        'args': {'job_id': job, 'wait_seconds': 30}, 'type': 'tool_call'}],
        'invalid_tool_calls': [], 'additional_kwargs': {},
        'response_metadata': {**meta, 'msty_completion_gate': 'site_typecheck_required'}})


def gate_final(state, result, tools, disabled):
    contract = state.get('task_contract') or {}
    meta = result.response_metadata
    outcome = _outcome_start_gate(state, result, tools, disabled)
    if outcome is not None:
        return outcome
    site = _site_gate(state, result, tools, disabled)
    if site is not None:
        return site
    if contract.get('status') == 'blocked' and not result.tool_calls:
        jobs = site_jobs(state)
        if jobs and all(status == 'clean' for status in jobs.values()):
            # A generic file-plan failure must not overrule the registered site
            # executor's newer, domain-specific typecheck receipt. This recovers
            # old checkpoints while new routes no longer expose the incompatible
            # task planner to site jobs at all.
            return result.model_copy(update={'response_metadata': {
                **meta, 'msty_completion_gate': 'site_executor_verified'}})
        # A stop/control phrase suppresses automatic actions; it can never
        # convert a failed artifact check into successful delivery. Preserve
        # charged usage, but do not publish unsupported completion prose.
        return result.model_copy(update={
            'content': 'Выполнение не подтверждено: проверка артефакта не пройдена. '
                       'Нужны фактическое исправление и успешная повторная проверка; '
                       'внутренний список задач не подтверждает результат.',
            'tool_calls': [], 'invalid_tool_calls': [], 'additional_kwargs': {},
            'response_metadata': {**meta, 'msty_blocked': True,
                                  'msty_completion_gate': 'failed_verification_preserved'}})
    if (contract.get('status') != 'planned' or result.tool_calls or result.invalid_tool_calls or disabled or
            owner_control(state) or meta.get('msty_blocked') or meta.get('msty_generation') == 'not_started' or
            msty_models.finish_reason(meta) in
                ('max_tokens', 'length', 'model_context_window_exceeded', 'refusal', 'content_filter')):
        # The artifact verification above takes precedence; otherwise a turn
        # with open items of its own plan continues (brain-desk #443).
        return open_plan_gate(state, result, tools, disabled) or result
    names = [tool.get('function', {}).get('name') for tool in tools
             if tool.get('type') == 'function' and _named(tool.get('function', {}).get('name'), VERIFY_SUFFIX)]
    choice = state.get('tool_choice')
    selected = choice.get('function', {}).get('name') if isinstance(choice, dict) else None
    if selected is not None and names != [selected]:
        return result.model_copy(update={'response_metadata': {**meta, 'msty_blocked': True},
            'content': 'Текущий выбор инструмента не разрешает проверку плана; завершение не подтверждено.'})
    if len(names) != 1 or (state.get('execution') or {}).get('actions_issued', 0) >= MAX_ACTIONS:
        return result.model_copy(update={'response_metadata': {**meta, 'msty_blocked': True},
            'content': 'Проверки плана ещё не подтверждены; инструмент проверки недоступен или лимит действий исчерпан.'})
    return result.model_copy(update={'content': 'Проверю критерии плана по локальным наблюдениям.',
        'tool_calls': [{'id': 'verify_' + uuid.uuid4().hex, 'name': names[0],
                        'args': {'plan_id': contract['plan_id']}, 'type': 'tool_call'}],
        'invalid_tool_calls': [], 'additional_kwargs': {},
        'response_metadata': {**meta, 'msty_completion_gate': 'native_verification_required'}})


def after_result(state, result):
    contract = deepcopy(state.get('task_contract'))
    if contract and contract.get('status') == 'verified_against_observations' and result.get('tool_calls'):
        if any(not _named(c.get('name'), VERIFY_SUFFIX) and c.get('name') != 'native_write_todos'
               for c in result['tool_calls']):
            if contract.get('plan_id'):
                contract['status'] = 'planned'
    return contract


def final_status(state, default_status):
    contract = state.get('task_contract') or {}
    if default_status == 'answered' and contract:
        if contract.get('status') == 'blocked':
            return 'blocked'
        if contract.get('status') == 'verified_against_observations':
            return 'verified_against_observations'
        if owner_control(state) or state.get('tool_choice') == 'none' or state.get('tool_choice') == {'type': 'none'}:
            return 'answered'
        return 'blocked'
    return default_status
