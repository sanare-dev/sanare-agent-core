"""Opt-in live synthetic regression against the local Msty endpoint.

Costs up to three small model steps. No private project documents are sent.
Uses the existing orchestrator accounting and provider credentials.
"""
import json
from pathlib import Path
import time
import urllib.request
import uuid

def main():
    secret_file = Path('/Volumes/LLM-Data/50-projects/open-webui/config/pipelines.env')
    key = next(line.split('=', 1)[1].strip() for line in secret_file.read_text().splitlines() if line.startswith('PIPELINES_API_KEY='))
    headers = {'Authorization': 'Bearer ' + key, 'Content-Type': 'application/json'}
    def call(messages, **extra):
        payload = {'model': 'team.brain', 'messages': messages, 'stream': False, 'max_tokens': 200, **extra}
        request = urllib.request.Request('http://127.0.0.1:9099/v1/chat/completions', headers=headers,
                                         data=json.dumps(payload).encode(), method='POST')
        with urllib.request.urlopen(request, timeout=170) as response:
            return json.load(response)['choices'][0]
    started = time.monotonic()
    answer = call([{'role': 'user', 'content': 'Ответь ровно: BRIDGE_OK'}])
    assert answer['message']['content'] == 'BRIDGE_OK', answer
    print(json.dumps({'case': 'direct_answer', 'passed': True, 'seconds': round(time.monotonic()-started, 2)}), flush=True)
    tools = [{'type': 'function', 'function': {'name': 'read_test_marker',
              'description': 'Read the synthetic integration test marker.',
              'parameters': {'type': 'object', 'properties': {}, 'additionalProperties': False}}}]
    history = [{'role': 'user', 'content': 'Вызови read_test_marker. Получив результат, ответь только точным значением marker.'}]
    answer = call(history, tools=tools, tool_choice='required')
    calls = answer['message'].get('tool_calls') or []
    assert answer['finish_reason'] == 'tool_calls' and len(calls) == 1, answer
    assert calls[0]['function']['name'] == 'read_test_marker'
    marker = 'verified-' + uuid.uuid4().hex[:12]
    answer['message']['content'] = answer['message'].get('content') or ''
    history += [answer['message'], {'role': 'tool', 'tool_call_id': calls[0]['id'], 'content': json.dumps({'marker': marker})}]
    final = call(history, tools=tools)
    assert final['message']['content'] == marker, final
    print(json.dumps({'case': 'tool_handoff_and_resume', 'passed': True}), flush=True)

if __name__ == '__main__':
    main()
