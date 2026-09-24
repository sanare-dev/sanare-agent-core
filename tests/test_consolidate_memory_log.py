"""The scheduled job must not print Brain's answer into its launchd log."""
import importlib.util
import io
import json
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location('consolidate_memory_log_under_test',
                                              ROOT / 'tools/consolidate_memory.py')
script = importlib.util.module_from_spec(spec)
spec.loader.exec_module(script)


def test_summary_contains_accounting_metadata_without_answer_text():
    sensitive_answer = 'Private owner conversation content must not reach run.log'
    response = {
        'choices': [{'finish_reason': 'stop', 'message': {'content': sensitive_answer}}],
        'usage': {'prompt_tokens': 12, 'completion_tokens': 5, 'total_tokens': 17},
        'sanare_usage_stages': [{'stage': 'brain', 'tokens': 17}],
    }

    summary = script.summarize(response, 1.25)

    assert summary['status'] == 'completed'
    assert summary['usage'] == response['usage']
    assert summary['usage_stages'] == response['sanare_usage_stages']
    assert summary['finish_reason'] == 'stop'
    assert 'answer' not in summary
    assert sensitive_answer not in json.dumps(summary)


def test_launchd_output_file_is_private(tmp_path):
    log = tmp_path / 'run.log'
    log.touch(mode=0o644)
    log.chmod(0o644)

    with log.open('a') as stream:
        script.protect_log_stream(stream)

    assert log.stat().st_mode & 0o777 == 0o600
    script.protect_log_stream(io.StringIO())


def test_permission_error_aborts_before_logging(tmp_path, monkeypatch):
    log = tmp_path / 'run.log'
    log.touch()

    with log.open('a') as stream:
        def deny_chmod(_fd, _mode):
            raise OSError('permission denied')

        monkeypatch.setattr(script.os, 'fchmod', deny_chmod)
        with pytest.raises(OSError, match='permission denied'):
            script.protect_log_stream(stream)
