"""Synthetic evaluator/sync contracts; no provider, native tools or cloud writes."""
import asyncio
from copy import deepcopy
import hashlib
import importlib.util
import json
from pathlib import Path
import socket
from types import SimpleNamespace
import uuid

import pytest
from langchain_core.messages import AIMessage
from langsmith import AsyncClient
from langsmith.utils import LangSmithNotFoundError

from deep_agent import msty


ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT / 'tools/msty_eval.py'
spec = importlib.util.spec_from_file_location('msty_eval_under_test', SCRIPT)
ev = importlib.util.module_from_spec(spec)
spec.loader.exec_module(ev)
DATASET = ev.load_dataset()
CASES = {c['id']: c for c in DATASET['cases']}
WORKSPACE = '8aa5fe31-6502-4fba-8c67-f87087db6870'  # Synthetic UUID, not an account.
REVIEWED_EVALUATOR_SHA256 = 'cfb6a51ddc89025a48946c44d6bcf0a2e9d9bb4a2f0621f7e2070332b20b401e'
REVIEWED_DATASET_SHA256 = '9f40dd9bdb37ac513601802897387cde87803c5fb1ccb6aa273d99254f8b715e'


@pytest.fixture(autouse=True)
def offline(monkeypatch):
    monkeypatch.setenv('LANGSMITH_TRACING', 'false')
    monkeypatch.setenv('LANGCHAIN_TRACING_V2', 'false')
    monkeypatch.setenv('MSTY_STATIC_CACHE', '0')
    def forbidden(*args, **kwargs):
        raise AssertionError('Network is forbidden in synthetic evaluation tests')
    monkeypatch.setattr(socket.socket, 'connect', forbidden)
    monkeypatch.setattr(socket.socket, 'connect_ex', forbidden)


def step(reason='stop', calls=None, results=None, thread='synthetic-thread-A'):
    return {'thread_id': thread, 'finish_reason': reason,
            'tool_calls': calls or [], 'tool_results': results or []}


def observation(case_id):
    case = CASES[case_id]
    value = {'schema_version': ev.OBSERVATION_SCHEMA, 'evidence_kind': 'synthetic_mock',
             'steps': [step(case['outputs']['finish_reason'])],
             'guard_blocked': case['outputs']['guard_blocked'],
             'final_text': case['outputs'].get('final_text', 'SYNTHETIC_BLOCKER')}
    if case_id == 'multi_step':
        value['steps'] = [
            step('tool_calls', [{'id': 'synthetic-write', 'name': 'write_marker',
                                'args': {'marker': 'SYNTHETIC_MARKER_V1'}}]),
            step('tool_calls', [{'id': 'synthetic-read', 'name': 'read_marker', 'args': {}}],
                 [{'tool_call_id': 'synthetic-write', 'name': 'write_marker'}]),
            step('stop', results=[{'tool_call_id': 'synthetic-read', 'name': 'read_marker'}]),
        ]
    elif case_id == 'cross_thread':
        value.update(cross_thread_rejected=True,
                     attempted_thread_ids=['synthetic-thread-A', 'synthetic-thread-B'])
    return value


def metrics(case_id, value):
    case = CASES[case_id]
    return {r['key']: r.get('score', r.get('value')) for r in
            ev.protocol_evaluator(case['inputs'], value, case['outputs'])}


def test_reviewed_dataset_and_evaluator_are_in_existing_release_fingerprint():
    # verify_release fingerprints this test; changes under tools require a
    # reviewed hash update here rather than silently retaining the old gate SHA.
    assert ev.DATASET_SHA256 == REVIEWED_DATASET_SHA256
    assert hashlib.sha256(SCRIPT.read_bytes()).hexdigest() == REVIEWED_EVALUATOR_SHA256
    assert DATASET['synthetic'] is True
    assert tuple(CASES) == ev.CASE_IDS


@pytest.mark.parametrize('case_id', ev.CASE_IDS)
def test_all_reviewed_fixture_observations_pass_only_protocol_consistency(case_id):
    actual = metrics(case_id, observation(case_id))
    assert actual == {'protocol_consistency': 1, 'evidence_scope': 'synthetic_mock',
                      'native_action_verification': 'not_performed',
                      'model_behavior_quality': 'not_measured'}


def test_prose_and_self_reported_tool_result_are_not_execution_or_protocol_proof():
    case = CASES['multi_step']
    for value in ({'content': 'Done; I wrote and verified it.'},
                  {'tool_result': {'status': 'success', 'file_written': True}},
                  {'native_action_verified': True, 'evidence_kind': 'native_capture_unverified'}):
        actual = metrics(case['id'], value)
        assert actual['protocol_consistency'] == 0
        assert actual['native_action_verification'] == 'not_performed'


def test_native_capture_label_never_upgrades_to_verified_action():
    value = observation('multi_step')
    value.update(evidence_kind='native_capture_unverified', native_action_verified=True)
    actual = metrics('multi_step', value)
    assert actual['protocol_consistency'] == 1
    assert actual['evidence_scope'] == 'native_capture_unverified'
    assert actual['native_action_verification'] == 'not_performed'


@pytest.mark.parametrize('mutation', ['missing_result', 'wrong_result_id', 'wrong_result_name',
                                     'duplicate_result', 'duplicate_call', 'cross_thread',
                                     'invalid_schema', 'incomplete', 'wrong_final_text'])
def test_multistep_corruptions_fail(mutation):
    value = observation('multi_step')
    if mutation == 'missing_result': value['steps'][1]['tool_results'] = []
    elif mutation == 'wrong_result_id': value['steps'][1]['tool_results'][0]['tool_call_id'] = 'not-issued'
    elif mutation == 'wrong_result_name': value['steps'][1]['tool_results'][0]['name'] = 'other_tool'
    elif mutation == 'duplicate_result': value['steps'][1]['tool_results'] *= 2
    elif mutation == 'duplicate_call': value['steps'][1]['tool_calls'][0]['id'] = 'synthetic-write'
    elif mutation == 'cross_thread': value['steps'][1]['thread_id'] = 'synthetic-thread-B'
    elif mutation == 'invalid_schema': value['steps'][0]['tool_calls'][0]['args']['marker'] = 'wrong'
    elif mutation == 'incomplete': value['steps'] = value['steps'][:-1]
    elif mutation == 'wrong_final_text': value['final_text'] = 'I promise to read it later.'
    assert metrics('multi_step', value)['protocol_consistency'] == 0


@pytest.mark.parametrize('case_id', ['no_tools', 'schema_invalid', 'stop', 'length'])
def test_forbidden_or_partial_action_never_passes(case_id):
    value = observation(case_id)
    value['steps'][0]['tool_calls'] = [{'id': 'partial', 'name': 'inspect_marker', 'args': {'path': '/synthetic/project/marker.txt'}}]
    assert metrics(case_id, value)['protocol_consistency'] == 0


def test_cross_thread_flag_without_observed_mismatch_does_not_pass():
    value = observation('cross_thread')
    value['attempted_thread_ids'] = ['synthetic-thread-A']
    assert metrics('cross_thread', value)['protocol_consistency'] == 0


@pytest.mark.parametrize('value', [None, [], {}, {'schema_version': 'wrong'}])
def test_missing_or_wrong_observation_fails_closed(value):
    assert metrics('simple', value)['protocol_consistency'] == 0


def test_reference_or_scenario_changes_are_not_silently_accepted():
    case = deepcopy(CASES['simple'])
    case['outputs']['final_text'] = 'unreviewed answer'
    assert ev.protocol_evaluator(case['inputs'], observation('simple'), case['outputs'])[0]['score'] == 0


@pytest.mark.parametrize('case_id', ['simple', 'no_tools', 'schema_invalid', 'stop'])
def test_evaluator_consumes_actual_guard_result_with_synthetic_model(monkeypatch, case_id):
    case = CASES[case_id]
    calls = [] if case_id == 'simple' else [{'id': 'synthetic-call', 'name': 'inspect_marker',
        'args': {'path': 42 if case_id == 'schema_invalid' else '/synthetic/project/marker.txt'}}]
    generated = AIMessage(content='SYNTHETIC_OK' if case_id == 'simple' else 'UNVERIFIED_MODEL_PROSE',
                          tool_calls=calls)
    seen = []
    class Model:
        def __init__(self, **kwargs):
            assert kwargs['max_retries'] == 0
        def bind_tools(self, tools, **kwargs): return self
        async def ainvoke(self, messages):
            seen.append(messages)
            return generated
    monkeypatch.setattr(msty, 'ChatAnthropic', Model)
    result = asyncio.run(msty.respond(case['inputs']))['result']
    value = observation(case_id)
    value['steps'][0]['tool_calls'] = result['tool_calls']
    value['final_text'] = result['content']
    value['guard_blocked'] = bool(calls) and not result['tool_calls'] and result['content'] != generated.content
    assert metrics(case_id, value)['protocol_consistency'] == 1
    assert len(seen) == 1


class FakeClient:
    """In-memory SDK-shaped double. Never creates an SDK HTTP client."""
    def __init__(self):
        self.workspace_id = WORKSPACE
        self.dataset = None
        self.examples = {}
        self.writes = []
        self.fail_after_write = False
    async def read_dataset(self, **kwargs):
        if self.dataset is None: raise LangSmithNotFoundError('synthetic missing')
        return self.dataset
    async def create_dataset(self, name, **kwargs):
        self.writes.append('dataset')
        self.dataset = SimpleNamespace(id=uuid.UUID('30dd8077-bb51-45c1-a769-01d79b3a3ccf'),
                                       name=name, description=kwargs['description'])
        return self.dataset
    async def read_example(self, identifier):
        if identifier not in self.examples: raise LangSmithNotFoundError('synthetic missing')
        return self.examples[identifier]
    async def create_example(self, **kwargs):
        self.writes.append(kwargs['id'])
        self.examples[kwargs['id']] = SimpleNamespace(**deepcopy(kwargs))
        if self.fail_after_write:
            self.fail_after_write = False
            raise TimeoutError('synthetic uncertain write')


def sync(client, **kwargs):
    return asyncio.run(ev.sync_synthetic_dataset(client, workspace_id=WORKSPACE,
        confirm_sha256=kwargs.get('confirm_sha256', ev.DATASET_SHA256)))


def test_sdk_has_the_exact_public_methods_used_without_constructing_client():
    for name in ('read_dataset', 'create_dataset', 'read_example', 'create_example', 'aclose'):
        assert callable(getattr(AsyncClient, name))


def test_synthetic_sync_is_verified_idempotent_and_does_not_overwrite():
    client = FakeClient()
    first = sync(client)
    assert first['created'] == first['verified'] == 7
    assert first['model_calls'] == first['experiments_created'] == 0
    assert len(client.writes) == 8
    payloads = deepcopy(client.examples)
    second = sync(client)
    assert second['created'] == 0 and second['already_present'] == second['verified'] == 7
    assert client.examples.keys() == payloads.keys() and len(client.writes) == 8
    assert all(e.metadata['synthetic'] is True for e in client.examples.values())


def test_uncertain_write_is_read_back_by_same_id_on_resumption():
    client = FakeClient()
    client.fail_after_write = True
    with pytest.raises(TimeoutError): sync(client)
    completed_id = next(iter(client.examples))
    result = sync(client)
    assert result['already_present'] == 1 and result['created'] == 6
    assert client.writes.count(completed_id) == 1


@pytest.mark.parametrize('conflict', ['name', 'description', 'inputs', 'outputs', 'metadata', 'dataset_id'])
def test_existing_conflict_never_overwrites(conflict):
    client = FakeClient()
    sync(client)
    if conflict in ('name', 'description'): setattr(client.dataset, conflict, 'synthetic-conflict')
    else: setattr(next(iter(client.examples.values())), conflict, 'synthetic-conflict')
    before = list(client.writes)
    with pytest.raises(ev.EvalContractError): sync(client)
    assert client.writes == before


@pytest.mark.parametrize('failure', ['confirmation', 'workspace'])
def test_sync_confirmation_and_workspace_fail_before_reads_or_writes(failure):
    client = FakeClient()
    if failure == 'workspace': client.workspace_id = '73f687a2-8f92-43c7-917c-3b6e7a3bbd74'
    with pytest.raises(ev.EvalContractError):
        sync(client, confirm_sha256='wrong' if failure == 'confirmation' else ev.DATASET_SHA256)
    assert client.dataset is None and client.writes == []


def test_manifest_cli_is_offline_and_does_not_run_evaluation(capsys):
    assert ev.main([]) == 0
    report = json.loads(capsys.readouterr().out)
    assert report['model_calls'] == 0 and report['external_sync'] is False
    assert report['dataset_sha256'] == REVIEWED_DATASET_SHA256


def test_cli_missing_confirmation_does_not_construct_client(monkeypatch, capsys):
    import langsmith
    def forbidden(**kwargs): raise AssertionError('SDK client constructed before confirmation')
    monkeypatch.setattr(langsmith, 'AsyncClient', forbidden)
    assert ev.main(['sync', '--workspace-id', WORKSPACE]) == 2
    assert json.loads(capsys.readouterr().out)['code'] == 'explicit_dataset_hash_confirmation_required'


@pytest.mark.parametrize('failure', ['workspace', 'endpoint', 'key'])
def test_cli_auth_and_destination_guards_without_using_real_credentials(monkeypatch, capsys, failure):
    import langsmith
    def forbidden(**kwargs): raise AssertionError('SDK client should not be constructed')
    monkeypatch.setattr(langsmith, 'AsyncClient', forbidden)
    monkeypatch.setenv('LANGSMITH_WORKSPACE_ID', WORKSPACE)
    monkeypatch.setenv('LANGSMITH_ENDPOINT', 'https://api.smith.langchain.com')
    monkeypatch.setenv('LANGSMITH_API_KEY', 'SYNTHETIC_NOT_A_REAL_KEY')
    if failure == 'workspace': monkeypatch.setenv('LANGSMITH_WORKSPACE_ID', 'different')
    elif failure == 'endpoint': monkeypatch.setenv('LANGSMITH_ENDPOINT', 'https://invalid.example')
    else: monkeypatch.delenv('LANGSMITH_API_KEY')
    assert ev.main(['sync', '--workspace-id', WORKSPACE, '--confirm-dataset-sha256', ev.DATASET_SHA256]) == 2
    assert 'SYNTHETIC_NOT_A_REAL_KEY' not in capsys.readouterr().out


def mock_sdk_http(monkeypatch, handler):
    """Real SDK request/retry/response path, HTTP transport mocked, no profile read."""
    import httpx
    import langsmith.async_client as sdk
    actual_http_client = httpx.AsyncClient
    monkeypatch.setattr(sdk._profiles, 'load_profile_client_config', lambda: SimpleNamespace(
        api_url=None, api_key=None, workspace_id=None, has_oauth=False))
    monkeypatch.setattr(sdk.httpx, 'AsyncClient', lambda **kwargs:
        actual_http_client(**kwargs, transport=httpx.MockTransport(handler)))
    monkeypatch.setenv('LANGSMITH_WORKSPACE_ID', WORKSPACE)
    monkeypatch.setenv('LANGSMITH_ENDPOINT', 'https://api.smith.langchain.com')
    monkeypatch.setenv('LANGSMITH_API_KEY', 'SYNTHETIC_NOT_A_REAL_KEY')


@pytest.mark.parametrize('failure', ['server_error', 'connection_error', 'rate_limit'])
def test_real_sdk_cli_makes_exactly_one_http_attempt_on_error(monkeypatch, capsys, failure):
    import httpx
    requests = []
    def handler(request):
        requests.append((request.method, request.url.path))
        if request.url.path == '/info':
            return httpx.Response(200, json={'version': '0.20.0'})
        if failure == 'connection_error':
            raise httpx.ConnectError('synthetic connection failure', request=request)
        return httpx.Response(429 if failure == 'rate_limit' else 503,
                              json={'detail': 'synthetic failure'})
    mock_sdk_http(monkeypatch, handler)
    assert ev.main(['sync', '--workspace-id', WORKSPACE,
                    '--confirm-dataset-sha256', ev.DATASET_SHA256]) == 2
    # The SDK context manager has a separate one-attempt /info capability read.
    assert requests == [('GET', '/info'), ('GET', '/datasets')]
    report = json.loads(capsys.readouterr().out)
    assert report == {'state': 'unknown', 'code': 'sync_or_validation_failed_check_before_retry'}


def test_real_sdk_cli_creates_and_reads_back_synthetic_examples(monkeypatch, capsys):
    import httpx
    requests, datasets, examples = [], {}, {}
    dataset_id = '30dd8077-bb51-45c1-a769-01d79b3a3ccf'
    timestamp = '2026-01-01T00:00:00Z'
    def handler(request):
        path = request.url.path
        requests.append((request.method, path))
        assert request.headers['x-tenant-id'] == WORKSPACE
        if request.method == 'GET' and path == '/info':
            return httpx.Response(200, json={'version': '0.20.0'})
        if request.method == 'GET' and path == '/datasets':
            return httpx.Response(200, json=list(datasets.values()))
        if request.method == 'POST' and path == '/datasets':
            payload = json.loads(request.content)
            payload.update(id=dataset_id, created_at=timestamp, modified_at=timestamp)
            datasets[dataset_id] = payload
            return httpx.Response(200, json=payload)
        if request.method == 'POST' and path == '/examples':
            payload = json.loads(request.content)
            assert payload['metadata']['synthetic'] is True
            # The live API records the base split in returned metadata. Include
            # it in the reviewed payload; do not weaken all metadata equality.
            payload['metadata']['dataset_split'] = ['base']
            payload.update(created_at=timestamp, modified_at=timestamp)
            examples[payload['id']] = payload
            return httpx.Response(200, json=payload)
        if request.method == 'GET' and path.startswith('/examples/'):
            payload = examples.get(path.rsplit('/', 1)[-1])
            return httpx.Response(200 if payload else 404, json=payload or {})
        raise AssertionError('Unexpected SDK operation')
    mock_sdk_http(monkeypatch, handler)
    args = ['sync', '--workspace-id', WORKSPACE, '--confirm-dataset-sha256', ev.DATASET_SHA256]
    assert ev.main(args) == 0
    first = json.loads(capsys.readouterr().out)
    assert first['created'] == first['verified'] == 7
    assert len(examples) == 7 and len(datasets) == 1
    assert len(requests) == 24  # Info, GET+POST dataset, 7 misses, 7 writes, 7 readbacks.
    before = len(requests)
    assert ev.main(args) == 0
    second = json.loads(capsys.readouterr().out)
    assert second['created'] == 0 and second['already_present'] == second['verified'] == 7
    assert len(requests) - before == 16  # Info, dataset, examples, independent readbacks.
    assert all(method == 'GET' for method, _ in requests[before:])


def test_real_sdk_warning_body_is_suppressed_and_logger_state_restored(monkeypatch, capsys, caplog):
    import httpx
    import logging
    requests = []
    marker = 'SYNTHETIC_PRIVATE_RESPONSE_BODY'
    def handler(request):
        requests.append((request.method, request.url.path))
        return httpx.Response(503, json={'detail': marker})
    mock_sdk_http(monkeypatch, handler)
    async_logger = logging.getLogger('langsmith.async_client')
    sync_logger = logging.getLogger('langsmith.client')
    monkeypatch.setattr(async_logger, 'disabled', False)
    monkeypatch.setattr(sync_logger, 'disabled', True)
    assert ev.main(['sync', '--workspace-id', WORKSPACE,
                    '--confirm-dataset-sha256', ev.DATASET_SHA256]) == 2
    assert requests == [('GET', '/info'), ('GET', '/datasets')]
    output = capsys.readouterr()
    assert marker not in output.out + output.err + caplog.text
    assert async_logger.disabled is False and sync_logger.disabled is True


def test_sdk_log_scope_restores_settings_on_success_and_leaves_root_unchanged(monkeypatch):
    import logging
    async_logger = logging.getLogger('langsmith.async_client')
    sync_logger = logging.getLogger('langsmith.client')
    root_logger = logging.getLogger()
    root_settings = (root_logger.disabled, root_logger.level, list(root_logger.handlers))
    monkeypatch.setattr(async_logger, 'disabled', True)
    monkeypatch.setattr(sync_logger, 'disabled', False)
    with ev._quiet_sdk_logs():
        assert async_logger.disabled is True and sync_logger.disabled is True
    assert async_logger.disabled is True and sync_logger.disabled is False
    assert (root_logger.disabled, root_logger.level, root_logger.handlers) == root_settings
