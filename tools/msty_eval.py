"""Deterministic protocol evaluators and opt-in synthetic LangSmith dataset sync.

Default CLI is offline. No provider, judge, tracing, Engine, shell/tool executor,
experiment upload, or private trace import exists here. This is not a task runner.
"""
import argparse
import asyncio
from contextlib import contextmanager
from copy import deepcopy
import hashlib
import json
import logging
import os
from pathlib import Path
import uuid


DATASET_PATH = Path(__file__).with_name('msty_eval_dataset_v1.json')
DATASET_SHA256 = '9f40dd9bdb37ac513601802897387cde87803c5fb1ccb6aa273d99254f8b715e'
CASE_IDS = ('simple', 'no_tools', 'schema_invalid', 'stop', 'multi_step', 'cross_thread', 'length')
OBSERVATION_SCHEMA = 'msty.protocol.observation.v1'
ENDPOINTS = {'https://api.smith.langchain.com', 'https://eu.api.smith.langchain.com'}


class EvalContractError(ValueError):
    """Content-free error safe to report without inputs, keys or SDK messages."""


def canonical(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(',', ':'),
                      allow_nan=False).encode()


def load_dataset():
    raw = DATASET_PATH.read_bytes()
    if len(raw) > 64_000:
        raise EvalContractError('dataset_size_invalid')
    data = json.loads(raw)
    if hashlib.sha256(canonical(data)).hexdigest() != DATASET_SHA256:
        raise EvalContractError('dataset_review_hash_mismatch')
    if (data.get('schema_version') != 'msty.synthetic.dataset.v1' or data.get('synthetic') is not True
            or tuple(c['id'] for c in data['cases']) != CASE_IDS):
        raise EvalContractError('dataset_contract_invalid')
    return data


def manifest():
    data = load_dataset()
    return {'dataset_version': data['version'], 'dataset_sha256': DATASET_SHA256,
            'evaluator_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            'cases': list(CASE_IDS), 'synthetic_only': True, 'model_calls': 0,
            'external_sync': False, 'scope': 'dataset validation, not a runtime/model evaluation'}


def _protocol_match(inputs, outputs, reference_outputs):
    """Consistency of captured structured protocol only; never action attestation."""
    if not isinstance(inputs, dict) or not isinstance(outputs, dict):
        return False
    case = next((c for c in load_dataset()['cases'] if c['inputs'] == inputs), None)
    if case is None or reference_outputs != case['outputs']:
        return False  # Unknown/unreviewed criteria must not silently pass.
    if outputs.get('schema_version') != OBSERVATION_SCHEMA:
        return False
    if outputs.get('evidence_kind') not in ('synthetic_mock', 'native_capture_unverified'):
        return False
    steps = outputs.get('steps')
    if (not isinstance(steps, list)
            or not reference_outputs.get('min_steps', 1) <= len(steps) <= reference_outputs['max_steps']):
        return False
    if outputs.get('guard_blocked') is not reference_outputs['guard_blocked']:
        return False
    from jsonschema import Draft202012Validator  # Already pinned by the graph.
    schemas = {t['function']['name']: t['function'].get('parameters', {}) for t in inputs['tools']}
    issued, pending, names, threads = {}, set(), [], set()
    for index, step in enumerate(steps):
        if not isinstance(step, dict) or not isinstance(step.get('thread_id'), str) or not step['thread_id']:
            return False
        threads.add(step['thread_id'])
        results, calls = step.get('tool_results'), step.get('tool_calls')
        if not isinstance(results, list) or not isinstance(calls, list):
            return False
        for result in results:
            if not isinstance(result, dict):
                return False
            identifier = result.get('tool_call_id')
            if not isinstance(identifier, str) or identifier not in pending or result.get('name') != issued[identifier]:
                return False
            pending.remove(identifier)
        if pending:
            return False  # No new model step until earlier callbacks are present.
        for call in calls:
            if not isinstance(call, dict):
                return False
            identifier, name = call.get('id'), call.get('name')
            if (not isinstance(identifier, str) or not identifier or identifier in issued
                    or not isinstance(name, str) or not isinstance(call.get('args'), dict)):
                return False
            if name not in schemas or not Draft202012Validator(schemas[name]).is_valid(call['args']):
                return False
            issued[identifier] = name
            pending.add(identifier)
            names.append(name)
        if index < len(steps) - 1 and (step.get('finish_reason') != 'tool_calls' or not calls):
            return False
    if pending or len(threads) != 1 or names != reference_outputs['tool_names']:
        return False
    if steps[-1].get('finish_reason') != reference_outputs['finish_reason']:
        return False
    if 'final_text' in reference_outputs and outputs.get('final_text') != reference_outputs['final_text']:
        return False
    if reference_outputs.get('cross_thread_rejected'):
        attempted = outputs.get('attempted_thread_ids')
        if (outputs.get('cross_thread_rejected') is not True or not isinstance(attempted, list)
                or not all(isinstance(x, str) and x for x in attempted) or len(set(attempted)) < 2):
            return False
    return True


def protocol_evaluator(inputs: dict, outputs: dict, reference_outputs: dict) -> list[dict]:
    """Official LangSmith code-evaluator signature; deterministic and network-free.

    Native capture is deliberately UNVERIFIED. Assistant prose, tool-result text,
    and a caller-supplied evidence label cannot prove an external action occurred.
    """
    try:
        passed = _protocol_match(inputs, outputs, reference_outputs)
    except (KeyError, TypeError, ValueError):
        passed = False
    kind = outputs.get('evidence_kind') if isinstance(outputs, dict) else None
    if kind not in ('synthetic_mock', 'native_capture_unverified'):
        kind = 'unrecognized'
    return [{'key': 'protocol_consistency', 'score': int(passed)},
            {'key': 'evidence_scope', 'value': kind},
            {'key': 'native_action_verification', 'value': 'not_performed'},
            {'key': 'model_behavior_quality', 'value': 'not_measured'}]


def _uuid(value):
    try:
        return str(uuid.UUID(value))
    except (ValueError, TypeError, AttributeError):
        raise EvalContractError('workspace_uuid_required') from None


async def sync_synthetic_dataset(client, *, workspace_id, confirm_sha256):
    """Create/read back only pinned synthetic examples; never overwrite/delete.

    The injected client is useful for offline tests. A confirmed SDK write is
    NOT model evaluation. Unknown write outcome: inspect with the same IDs next
    time, never use a new random dataset/name as a retry.
    """
    from langsmith.utils import LangSmithNotFoundError

    data = load_dataset()
    workspace_id = _uuid(workspace_id)
    if confirm_sha256 != DATASET_SHA256:
        raise EvalContractError('explicit_dataset_hash_confirmation_required')
    if _uuid(client.workspace_id) != workspace_id:
        raise EvalContractError('workspace_mismatch')
    name = f"msty-synthetic-{data['version']}-{DATASET_SHA256[:12]}"
    description = f"Synthetic protocol fixtures only; {DATASET_SHA256}; no native action proof or model quality score."
    try:
        dataset = await client.read_dataset(dataset_name=name)
    except LangSmithNotFoundError:
        dataset = await client.create_dataset(name, description=description, data_type='kv')
    if dataset.name != name or dataset.description != description:
        raise EvalContractError('dataset_name_collision_no_overwrite')
    payloads, missing = [], []
    for case in data['cases']:
        identifier = str(uuid.uuid5(uuid.NAMESPACE_URL,
            f"msty-synthetic/{workspace_id}/{DATASET_SHA256}/{case['id']}"))
        payload = {'id': identifier, 'inputs': deepcopy(case['inputs']), 'outputs': deepcopy(case['outputs']),
                   'metadata': {'synthetic': True, 'case_id': case['id'],
                                'dataset_sha256': DATASET_SHA256, 'evidence_scope': 'reference_only',
                                'dataset_split': ['base']}}
        payloads.append(payload)
        try:
            existing = await client.read_example(identifier)
        except LangSmithNotFoundError:
            missing.append(payload)
            continue
        if not _same_example(existing, payload, dataset.id):
            raise EvalContractError('example_conflict_no_overwrite')
    for payload in missing:
        await client.create_example(dataset_id=dataset.id, **payload)
    for payload in payloads:
        if not _same_example(await client.read_example(payload['id']), payload, dataset.id):
            raise EvalContractError('example_readback_mismatch')
    return {'dataset_id': str(dataset.id), 'dataset_version': data['version'], 'dataset_sha256': DATASET_SHA256,
            'created': len(missing), 'already_present': len(payloads) - len(missing), 'verified': len(payloads),
            'synthetic_only': True, 'model_calls': 0, 'experiments_created': 0,
            'scope': 'synthetic dataset delivery only; no evaluation or native execution'}


def _same_example(example, payload, dataset_id):
    return (str(example.id) == payload['id'] and str(example.dataset_id) == str(dataset_id)
            and example.inputs == payload['inputs'] and example.outputs == payload['outputs']
            and example.metadata == payload['metadata'])


@contextmanager
def _quiet_sdk_logs():
    """SDK warnings can embed raw HTTP bodies; restore prior state on every exit.

    Scoped to this explicit CLI operation and these two SDK loggers. This does
    not reconfigure root logging, other applications, or permanent log levels.
    """
    loggers = [logging.getLogger(name) for name in ('langsmith.async_client', 'langsmith.client')]
    previous = [logger.disabled for logger in loggers]
    try:
        for logger in loggers:
            logger.disabled = True
        yield
    finally:
        for logger, disabled in zip(loggers, previous):
            logger.disabled = disabled


async def _sync_cli(args):
    # Confirm locally before SDK construction, auth access, or external requests.
    load_dataset()
    if args.confirm_dataset_sha256 != DATASET_SHA256:
        raise EvalContractError('explicit_dataset_hash_confirmation_required')
    workspace = _uuid(args.workspace_id)
    if _uuid(os.environ.get('LANGSMITH_WORKSPACE_ID')) != workspace:
        raise EvalContractError('workspace_environment_mismatch')
    endpoint = os.environ.get('LANGSMITH_ENDPOINT', 'https://api.smith.langchain.com').rstrip('/')
    if endpoint not in ENDPOINTS:
        raise EvalContractError('unapproved_langsmith_endpoint')
    key = os.environ.get('LANGSMITH_API_KEY')
    if not key:
        raise EvalContractError('configured_langsmith_key_required')
    from langsmith import AsyncClient
    # This installed SDK counts TOTAL attempts, despite naming this max_retries.
    # 0 sends no request at all; 1 means one attempt and no automatic retry.
    with _quiet_sdk_logs():
        async with AsyncClient(api_url=endpoint, api_key=key, workspace_id=workspace,
                               timeout_ms=(10_000, 20_000, 20_000, 10_000),
                               retry_config={'max_retries': 1}, disable_prompt_cache=True) as client:
            return await sync_synthetic_dataset(client, workspace_id=workspace,
                                                confirm_sha256=args.confirm_dataset_sha256)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=('manifest', 'sync'), nargs='?', default='manifest')
    parser.add_argument('--workspace-id')
    parser.add_argument('--confirm-dataset-sha256')
    args = parser.parse_args(argv)
    try:
        result = manifest() if args.command == 'manifest' else asyncio.run(_sync_cli(args))
    except EvalContractError as exc:
        print(json.dumps({'state': 'blocked', 'code': str(exc)}))
        return 2
    except Exception:
        # SDK error strings may include response bodies, URLs, headers or keys.
        print(json.dumps({'state': 'unknown', 'code': 'sync_or_validation_failed_check_before_retry'}))
        return 2
    print(json.dumps(result, ensure_ascii=False))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
