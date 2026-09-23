"""Scheduled consolidation trigger: stop is checked before any request, the run
goes through the bridge (team.brain) and never retries an unknown outcome."""
import importlib.util
from pathlib import Path
import plistlib
import sqlite3
import types
import urllib.error

import pytest

ROOT = Path(__file__).resolve().parents[2]
spec = importlib.util.spec_from_file_location('consolidate_memory_under_test',
                                              ROOT / 'tools/consolidate_memory.py')
script = importlib.util.module_from_spec(spec)
spec.loader.exec_module(script)
BRIDGE_STOP = script.TEAM_DIR / 'brain_stop.py'


def fake_stop(outcome):
    module = types.SimpleNamespace()

    class BrainStopped(RuntimeError):
        pass

    class StopStateUnavailable(BrainStopped):
        pass

    def require_running():
        if outcome == 'stopped':
            raise BrainStopped()
        if outcome == 'unavailable':
            raise StopStateUnavailable()
        if outcome == 'crash':
            raise OSError()

    module.BrainStopped, module.StopStateUnavailable = BrainStopped, StopStateUnavailable
    module.require_running = require_running
    return module


def forbidden_sender(*args):
    pytest.fail('request sent while stop is active or unverifiable')


@pytest.mark.parametrize('outcome, status', [
    ('stopped', 'stopped'), ('unavailable', 'stop_state_unavailable'),
    ('crash', 'stop_state_unavailable')])
def test_no_request_when_stop_active_or_unverifiable(outcome, status):
    result = script.run(brain_stop=fake_stop(outcome), sender=forbidden_sender,
                        key_reader=lambda: pytest.fail('key read while stopped'))
    assert result == {'status': status, 'sent': False}


def test_dry_run_checks_stop_only():
    result = script.run(dry_run=True, brain_stop=fake_stop('running'), sender=forbidden_sender)
    assert result['status'] == 'dry_run' and result['sent'] is False


def test_running_sends_one_bridge_request_with_fixed_prompt():
    sent = []

    def sender(body, key):
        sent.append((body, key))
        return {'choices': [{'finish_reason': 'stop', 'message': {'content': 'нет новых фактов'}}],
                'usage': {'total_tokens': 10}}

    result = script.run(brain_stop=fake_stop('running'), sender=sender, key_reader=lambda: 'k')
    assert len(sent) == 1
    body, key = sent[0]
    assert key == 'k' and body['model'] == 'team.brain' and body['stream'] is False
    assert body['max_tokens'] == script.MAX_TOKENS and 'tools' not in body
    assert body['messages'] == [{'role': 'user', 'content': script.CONSOLIDATION_PROMPT}]
    assert result['status'] == 'completed' and result['usage'] == {'total_tokens': 10}


@pytest.mark.parametrize('error, status, sent', [
    (urllib.error.HTTPError('u', 500, 'x', {}, None), 'bridge_error', True),
    (TimeoutError(), 'transport_error', 'unknown')])
def test_failures_are_reported_without_retry(error, status, sent):
    calls = []

    def sender(body, key):
        calls.append(1)
        raise error

    result = script.run(brain_stop=fake_stop('running'), sender=sender, key_reader=lambda: 'k')
    assert calls == [1] and result['status'] == status and result['sent'] == sent


@pytest.mark.skipif(not BRIDGE_STOP.exists(), reason='bridge volume not mounted')
@pytest.mark.parametrize('value, status', [('0', None), ('1', 'stopped')])
def test_real_bridge_stop_reader_against_temp_database(tmp_path, monkeypatch, value, status):
    database = tmp_path / 'right-hand.sqlite'
    with sqlite3.connect(database) as connection:
        connection.execute('CREATE TABLE app_settings (key TEXT, value TEXT)')
        connection.execute("INSERT INTO app_settings VALUES ('emergency_stop', ?)", (value,))
    monkeypatch.setenv('ORCH_STOP_DB_PATH', str(database))
    assert script.stop_state(script.load_brain_stop()) == status
    monkeypatch.setenv('ORCH_STOP_DB_PATH', str(tmp_path / 'missing.sqlite'))
    assert script.stop_state(script.load_brain_stop()) == 'stop_state_unavailable'


def test_launchd_template_is_periodic_and_not_run_at_load():
    plist = plistlib.loads((ROOT / 'tools/launchd/com.sanare.brain-consolidation.plist').read_bytes())
    assert plist['StartInterval'] == 21600 and plist['RunAtLoad'] is False
    assert plist['ProgramArguments'][-1].endswith('tools/consolidate_memory.py')
    assert 'KeepAlive' not in plist


def test_recent_conversations_offered_only_in_consolidation_turn():
    from deep_agent import msty_native, consolidator, msty_tool_routing
    normal = [{'role': 'user', 'content': 'Привет'}]
    run = [{'role': 'user', 'content': consolidator.CONSOLIDATION_PROMPT}]
    assert consolidator.CONSOLIDATION_MARKER not in msty_tool_routing.latest_user_text(normal)
    assert consolidator.CONSOLIDATION_MARKER in msty_tool_routing.latest_user_text(run)


def test_calls_this_turn_counts_only_after_last_owner_message():
    from deep_agent import msty_native
    name = msty_native.RECENT_CONVERSATIONS_TOOL
    call = {'role': 'assistant', 'content': '', 'tool_calls': [{'id': 'a', 'name': name, 'args': {}}]}
    msgs = [{'role': 'user', 'content': 'x'}, call, {'role': 'user', 'content': 'y'}, call, call]
    assert msty_native._calls_this_turn(msgs, name) == 2
