"""Observation-bound completion gate for real native Msty plan/verify tools.

Coverage is model_proposed, never owner-approved completeness. Hashes bind data
integrity, not client authenticity. A passed local artifact check proves only
verified_against_observations, not business delivery or universal correctness.
No shell, model invocation, local I/O or new autonomous controller lives here.
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
    if (not isinstance(receipt, dict) or set(receipt) != PLAN_KEYS or
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
    return {'version': 1, 'status': 'planned', 'criteria_origin': 'model_proposed',
            'whole_task_completion_verified': False, 'plan_id': receipt['plan_id'],
            'plan_sha256': receipt['plan_sha256'], 'project_slug': receipt['project_slug'],
            'requirements_count': len(requirements), 'checks_count': len(checks),
            'checks': summaries, 'plan_call_sha256': canonical_digest(call)}


def _verified(contract, call, receipt):
    if (not isinstance(receipt, dict) or set(receipt) != VERIFY_KEYS or
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
    return True


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
