"""Scheduled candidate-memory consolidation through the existing Brain bridge.

One run = one ordinary `team.brain` request to the local bridge (127.0.0.1:9099),
the same path as tools/evaluate_bridge.py. Everything paid is enforced by the
bridge, not here:
- emergency stop: brain_stop.require_running() before every paid stage;
- cost: reserve before send, measured settle, unknown outcome stays reserved
  (ledger ~/Library/Application Support/SanareOrchestrator/accounting/ledger.json),
  Brain task cap (MSTY_BRAIN_TASK_CAP_USD) and the team daily cap.
This script additionally reads the same stop flag with the bridge's own
brain_stop.py before connecting, so a stopped system gets no request at all.

Trigger: launchd StartInterval (tools/launchd/com.sanare.brain-consolidation.plist,
installed only by the owner). Requires on the deployment
MSTY_RECENT_CONVERSATIONS=on and the bridge admitting native_recent_conversations.

    uv run python tools/consolidate_memory.py [--dry-run]

Prints one JSON line; never prints the API key or the answer beyond 300 chars.
"""
import argparse
import importlib.util
import json
import os
from pathlib import Path
import sys
import time
import urllib.error
import urllib.request

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from deep_agent.consolidator import CONSOLIDATION_PROMPT  # noqa: E402

BRIDGE = 'http://127.0.0.1:9099/v1/chat/completions'
TEAM_DIR = Path('/Volumes/LLM-Data/50-projects/open-webui/orchestrator/_sanare_team')
SECRET_FILE = Path('/Volumes/LLM-Data/50-projects/open-webui/config/pipelines.env')
STOP_DB = '/Users/vb/Documents/ChatGPT/LLM/right-hand-mvp/data/right-hand.sqlite'
MAX_TOKENS = 2048
TIMEOUT_SECONDS = 900


def load_brain_stop(team_dir: Path = TEAM_DIR):
    """The bridge's own stop reader, loaded by file (no package __init__ side effects)."""
    spec = importlib.util.spec_from_file_location('sanare_brain_stop', team_dir / 'brain_stop.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def stop_state(brain_stop) -> str | None:
    """None when running; otherwise 'stopped' or 'stop_state_unavailable' (fail-closed)."""
    os.environ.setdefault('ORCH_STOP_DB_PATH', STOP_DB)
    try:
        brain_stop.require_running()
    except brain_stop.StopStateUnavailable:
        return 'stop_state_unavailable'
    except brain_stop.BrainStopped:
        return 'stopped'
    except Exception:
        return 'stop_state_unavailable'
    return None


def payload() -> dict:
    return {'model': 'team.brain', 'stream': False, 'max_tokens': MAX_TOKENS,
            'messages': [{'role': 'user', 'content': CONSOLIDATION_PROMPT}]}


def bridge_key(secret_file: Path = SECRET_FILE) -> str:
    return next(line.split('=', 1)[1].strip() for line in secret_file.read_text().splitlines()
                if line.startswith('PIPELINES_API_KEY='))


def send(body: dict, key: str) -> dict:
    request = urllib.request.Request(
        BRIDGE, data=json.dumps(body).encode(), method='POST',
        headers={'Authorization': 'Bearer ' + key, 'Content-Type': 'application/json'})
    with urllib.request.urlopen(request, timeout=TIMEOUT_SECONDS) as response:
        return json.load(response)


def summarize(response: dict, seconds: float) -> dict:
    choice = (response.get('choices') or [{}])[0]
    return {'status': 'completed', 'seconds': round(seconds, 1),
            'finish_reason': choice.get('finish_reason'), 'usage': response.get('usage'),
            'usage_stages': response.get('sanare_usage_stages'),
            'answer': ((choice.get('message') or {}).get('content') or '')[:300]}


def run(dry_run: bool = False, brain_stop=None, sender=send, key_reader=bridge_key) -> dict:
    state = stop_state(brain_stop or load_brain_stop())
    if state is not None:
        return {'status': state, 'sent': False}
    if dry_run:
        return {'status': 'dry_run', 'sent': False, 'payload_bytes': len(json.dumps(payload()))}
    started = time.monotonic()
    try:
        response = sender(payload(), key_reader())
    except urllib.error.HTTPError as exc:
        # The bridge keeps the ledger reservation for an unknown outcome; no retry here.
        return {'status': 'bridge_error', 'sent': True, 'http_status': exc.code}
    except (urllib.error.URLError, TimeoutError, OSError) as exc:
        # Outcome unknown (may have reached the bridge): the ledger decides, no retry here.
        return {'status': 'transport_error', 'sent': 'unknown', 'error': type(exc).__name__}
    return summarize(response, time.monotonic() - started)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument('--dry-run', action='store_true', help='check stop only, send nothing')
    result = run(parser.parse_args().dry_run)
    result['time'] = time.strftime('%Y-%m-%dT%H:%M:%S%z')
    print(json.dumps(result, ensure_ascii=False), flush=True)
    return 0 if result['status'] in ('completed', 'dry_run', 'stopped') else 1


if __name__ == '__main__':
    sys.exit(main())
