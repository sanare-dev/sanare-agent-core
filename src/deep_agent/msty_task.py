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

PLAN_SUFFIX = 'msty_task_plan'
VERIFY_SUFFIX = 'msty_task_verify'
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
    if depth > 3:
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


def gate_final(state, result, tools, disabled):
    contract = state.get('task_contract') or {}
    meta = result.response_metadata
    if (contract.get('status') != 'planned' or result.tool_calls or result.invalid_tool_calls or disabled or
            owner_control(state) or meta.get('msty_blocked') or meta.get('msty_generation') == 'not_started' or
            meta.get('stop_reason', meta.get('finish_reason')) in
                ('max_tokens', 'length', 'model_context_window_exceeded', 'refusal', 'content_filter')):
        return result
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
    if contract and result.get('tool_calls'):
        if any(not _named(c.get('name'), VERIFY_SUFFIX) for c in result['tool_calls']):
            if contract.get('plan_id'):
                contract['status'] = 'planned'
    return contract


def final_status(state, default_status):
    contract = state.get('task_contract') or {}
    if default_status == 'answered' and contract:
        if owner_control(state) or state.get('tool_choice') == 'none' or state.get('tool_choice') == {'type': 'none'}:
            return 'answered'
        return ('verified_against_observations' if contract.get('status') == 'verified_against_observations'
                else 'blocked')
    return default_status
