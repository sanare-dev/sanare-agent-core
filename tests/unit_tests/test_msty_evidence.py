"""TAU L5: Evidence Gate — негативное утверждение только по доказательству."""
import asyncio

from langchain_core.messages import AIMessage

from deep_agent import msty, msty_evidence


def test_negative_markers_ru_and_en():
    assert msty_evidence.has_negative_claim('Cron-синхронизация не настроена.')
    assert msty_evidence.has_negative_claim('The connector is not configured.')
    assert msty_evidence.has_negative_claim('Похоже, сервис сломан.')
    assert not msty_evidence.has_negative_claim('Синхронизация здорова, свежесть 3 минуты.')


def test_markers_are_configurable_via_env(monkeypatch):
    monkeypatch.setenv('MSTY_NEGATIVE_MARKERS', 'offline-only-marker')
    assert msty_evidence.has_negative_claim('offline-only-marker случился')
    assert not msty_evidence.has_negative_claim('не работает')
    monkeypatch.setenv('MSTY_NEGATIVE_MARKERS', '')
    assert not msty_evidence.has_negative_claim('не работает')  # Gate отключён


STATUS_HISTORY = [
    {'role': 'user', 'content': 'Проверь синхронизацию магазина.'},
    {'role': 'assistant', 'content': '', 'tool_calls': [{'id': 's1', 'type': 'function',
        'function': {'name': 'msty_store_sync_status', 'arguments': '{}'}}]},
    {'role': 'tool', 'tool_call_id': 's1', 'content': '{"status": "ok", "freshness_s": 210}'},
]


def test_successful_status_read_from_history_and_from_evidence_log():
    assert msty_evidence.successful_status_reads({'messages': STATUS_HISTORY}) == [
        'msty_store_sync_status']
    state = {'messages': [], 'tau_evidence': [
        {'kind': 'evidence', 'tool': 'msty_admin_health', 'evidence_class': 'status_read',
         'tool_call_id': 'h1'}]}
    assert msty_evidence.successful_status_reads(state) == ['msty_admin_health']


def test_failed_status_read_is_not_evidence():
    failed = [dict(STATUS_HISTORY[0]), STATUS_HISTORY[1],
              {'role': 'tool', 'tool_call_id': 's1', 'content': 'HTTP 503 Service Unavailable'}]
    assert msty_evidence.successful_status_reads({'messages': failed}) == []
    # Обычный инструмент без evidence_class — тоже не доказательство.
    other = [dict(STATUS_HISTORY[0]),
             {'role': 'assistant', 'content': '', 'tool_calls': [{'id': 'x', 'type': 'function',
                 'function': {'name': 'execute_sql', 'arguments': '{}'}}]},
             {'role': 'tool', 'tool_call_id': 'x', 'content': 'rows: 1'}]
    assert msty_evidence.successful_status_reads({'messages': other}) == []


def _result(content, **meta):
    return AIMessage(content=content, tool_calls=[], usage_metadata={
        'input_tokens': 10, 'output_tokens': 5, 'total_tokens': 15},
        response_metadata=meta)


def test_gate_rewrites_bare_negative_claim_and_preserves_original():
    result, label = msty_evidence.gate_final_answer(
        {'messages': []}, _result('Cron-синхронизация не настроена.'))
    assert label == 'unconfirmed'
    assert result.content.startswith('Не могу подтвердить негативный вывод')
    assert 'Cron-синхронизация не настроена.' in result.content  # исходник сохранён
    assert 'msty_store_sync_status' in result.content  # перечень профильных чтений
    assert result.response_metadata['tau_evidence_gate'] == 'unconfirmed'


def test_gate_passes_negative_claim_after_successful_status_read():
    original = _result('Cron работает, но товары не обновляются: синхронизация не работает.')
    result, label = msty_evidence.gate_final_answer(
        {'messages': STATUS_HISTORY}, original)
    assert label is None and result is original


def test_gate_ignores_tool_steps_and_blocked_or_gated_results():
    with_call = AIMessage(content='', tool_calls=[{'id': 'c', 'name': 'execute_sql',
        'args': {}, 'type': 'tool_call'}], response_metadata={})
    result, _ = msty_evidence.gate_final_answer({'messages': []}, with_call)
    assert result is with_call
    blocked, label = msty_evidence.gate_final_answer(
        {'messages': []}, _result('Ответ не работает.', msty_blocked=True))
    assert label is None and 'не могу подтвердить' not in blocked.content.lower()


# --- Врезка в _respond_step: финальный ответ проходит Gate до публикации --------

def _install_model(monkeypatch, content):
    monkeypatch.setenv('MSTY_MODEL_PROFILE', 'sonnet')
    response = AIMessage(content=content, tool_calls=[],
        response_metadata={'stop_reason': 'end_turn'},
        usage_metadata={'input_tokens': 20, 'output_tokens': 7, 'total_tokens': 27})

    class Model:
        def __init__(self, **kwargs):
            pass

        def bind_tools(self, tools, **kwargs):
            return self

        async def ainvoke(self, messages):
            return response

    monkeypatch.setattr(msty, 'ChatAnthropic', Model)


def test_respond_rewrites_unconfirmed_negative_final(monkeypatch):
    _install_model(monkeypatch, 'Cron-синхронизация не настроена.')
    result = asyncio.run(msty.respond({'messages': [
        {'role': 'user', 'content': 'Как там синхронизация?'}], 'tools': []}))['result']
    assert result['content'].startswith('Не могу подтвердить негативный вывод')
    assert result['response_metadata']['tau_evidence_gate'] == 'unconfirmed'


def test_respond_passes_confirmed_negative_final(monkeypatch):
    _install_model(monkeypatch, 'Синхронизация не работает: freshness растёт.')
    result = asyncio.run(msty.respond({'messages': STATUS_HISTORY, 'tools': []}))['result']
    assert result['content'] == 'Синхронизация не работает: freshness растёт.'
    assert 'tau_evidence_gate' not in result['response_metadata']


def test_respond_positive_final_is_untouched(monkeypatch):
    _install_model(monkeypatch, 'Синхронизация здорова, свежесть 3 минуты.')
    result = asyncio.run(msty.respond({'messages': [
        {'role': 'user', 'content': 'Как там синхронизация?'}], 'tools': []}))['result']
    assert result['content'] == 'Синхронизация здорова, свежесть 3 минуты.'
