"""TAU L5: Evidence Gate — негативное утверждение только по доказательству."""
import asyncio
import pytest

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
    # Журнал tau_evidence — наблюдаемость, не доказательство: он переживает ходы
    # и не должен разрешать негатив на новую тему.
    state = {'messages': [], 'tau_evidence': [
        {'kind': 'evidence', 'tool': 'msty_admin_health', 'evidence_class': 'status_read',
         'tool_call_id': 'h1'}]}
    assert msty_evidence.successful_status_reads(state) == []


def test_status_read_from_previous_turn_is_not_evidence():
    later = [*STATUS_HISTORY, {'role': 'assistant', 'content': 'Синхронизация в порядке.'},
             {'role': 'user', 'content': 'А таблица orders в Supabase есть?'}]
    assert msty_evidence.successful_status_reads({'messages': later}) == []


@pytest.mark.parametrize('text', [
    'Дубликат отсутствует, можно публиковать.',
    'Если кнопка не работает, обновите страницу.',
    'Nothing is missing.',
    'Unbroken chain of backups.',
    'Модуль был сломан в 1.2, исправлен в 1.3.',
    'Бот не настроен на выходные — так вы и просили.',
    'В документе отсутствует раздел о ценах.',
    'Ошибок не обнаружено.',
])
def test_ordinary_answers_are_not_negative_claims(text):
    assert not msty_evidence.has_negative_claim(text)


@pytest.mark.parametrize('text', [
    'Синхронизация магазина не настроена, cron отсутствует.',
    'Сервис оплаты не работает.',
    'The webhook is not configured.',
    'Ключ Supabase отсутствует в окружении.',
])
def test_infrastructure_diagnoses_are_negative_claims(text):
    assert msty_evidence.has_negative_claim(text)


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


STATUS_ASK = [{'role': 'user', 'content': 'Работает ли синхронизация магазина?'}]


def test_gate_rewrites_bare_negative_claim_and_preserves_original():
    result, label = msty_evidence.gate_final_answer(
        {'messages': STATUS_ASK}, _result('Cron-синхронизация не настроена.'))
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


def test_failed_status_read_in_openai_dicts_is_not_evidence():
    """Ревью 2026-09-23: convert_to_openai_messages теряет status='error';
    отказ с префиксом tau_class= длиннее 300 символов засчитывался успехом."""
    failed = [dict(STATUS_HISTORY[0]), STATUS_HISTORY[1],
              {'role': 'tool', 'tool_call_id': 's1',
               'content': 'tau_class=transient. Политика: x\n---\nError: 503 Service Unavailable '
                          + 'x' * 400}]
    assert msty_evidence.successful_status_reads({'messages': failed}) == []


@pytest.mark.parametrize('text', [
    'Интеграция Stripe не работает — проверьте токен.',
    'Cron не работает с тех пор, когда обновили сервер.',
    'Ничего не синхронизируется: cron сломан.',
])
def test_diagnosis_with_advice_is_still_a_claim(text):
    assert msty_evidence.has_negative_claim(text)


@pytest.mark.parametrize('text', [
    # «Когда …» больше не исключается (ревью 3: «Когда я проверил статус,
    # синхронизация не работает» — мнимая проверка).
    'Убедитесь, что вебхук не сломан.',
    'Проверьте, что сервис не отключён и ключ не отсутствует.',
    'Ключ API не отсутствует.',
])
def test_advice_and_conditions_are_not_claims_round2(text):
    assert not msty_evidence.has_negative_claim(text)


def test_short_answer_prefix_is_still_a_claim():
    assert msty_evidence.has_negative_claim('Если коротко: синхронизация не работает.')


def test_status_tool_that_could_not_read_is_not_evidence():
    unread = [dict(STATUS_HISTORY[0]), STATUS_HISTORY[1],
              {'role': 'tool', 'tool_call_id': 's1',
               'content': '{"state": "unavailable", "code": "vercel_logs_failed"}'}]
    assert msty_evidence.successful_status_reads({'messages': unread}) == []


@pytest.mark.parametrize('text', [
    'Когда я проверил статус, синхронизация не работает.',
    'Убедитесь, что токен задан — сейчас интеграция не работает.',
    'If you look at the logs, the cron is not working.',
])
def test_pseudo_checked_diagnoses_are_claims_round3(text):
    assert msty_evidence.has_negative_claim(text)


def test_failed_state_is_evidence_not_unread():
    read = [dict(STATUS_HISTORY[0]), STATUS_HISTORY[1],
            {'role': 'tool', 'tool_call_id': 's1', 'content': '{"state": "failed", "last_run": 1}'}]
    assert msty_evidence.successful_status_reads({'messages': read}) == ['msty_store_sync_status']


# --- Область Gate: только вопросы о состоянии системы (24.09.2026) ----------

DAILY_REVIEW = ('Ежедневный разбор проекта «Улучшение Brain Desk и Brain». Разбери очередь '
                '(queue.md, owner-ideas, открытые issues и PR), выбери следующую задачу и '
                'предложи план: шаги, зона, что искать готовым, проверки, кто лучше подходит.')
BRIEF = ('Составь мой утренний бриф на сегодня по данным ниже. Коротко, по-русски. Начни '
         'сразу с «Главное сегодня» — без вступления, оговорок и предупреждений.')


@pytest.mark.parametrize('question', [DAILY_REVIEW, BRIEF,
                                      'Составь план улучшения Brain на неделю',
                                      'Напиши текст письма поставщику'])
def test_gate_leaves_planning_briefs_and_texts_alone(question):
    """Архитектор 24.09: «очередь начинается со сломанного… [1]» — не диагноз."""
    original = _result('Очередь начинается со сломанного и медленного, что блокирует '
                       'владельца; сервис оплаты не работает — это задача #12.')
    result, label = msty_evidence.gate_final_answer(
        {'messages': [{'role': 'user', 'content': question}]}, original)
    assert label is None and result is original


@pytest.mark.parametrize('question', [
    'Работает ли синхронизация магазина?',
    'Проверь, почему не обновляются товары',
    'Какой статус у cron?',
    'Is the webhook down?',
    'Почему упал деплой?',
])
def test_gate_applies_to_status_questions(question):
    _, label = msty_evidence.gate_final_answer(
        {'messages': [{'role': 'user', 'content': question}]},
        _result('Синхронизация не работает.'))
    assert label == 'unconfirmed'


def test_gate_applies_when_status_tool_was_called_this_turn():
    """Модель сама пошла диагностировать: неудачное статус-чтение — в области Gate."""
    turn = [{'role': 'user', 'content': DAILY_REVIEW},
            {'role': 'assistant', 'content': '', 'tool_calls': [{'id': 's1', 'type': 'function',
                'function': {'name': 'msty_store_sync_status', 'arguments': '{}'}}]},
            {'role': 'tool', 'tool_call_id': 's1', 'content': 'HTTP 503 Service Unavailable'}]
    _, label = msty_evidence.gate_final_answer(
        {'messages': turn}, _result('Синхронизация не работает.'))
    assert label == 'unconfirmed'


def test_status_words_deep_in_long_prompt_are_data_not_the_question():
    long_prompt = BRIEF + ' ' + 'x' * msty_evidence.STATUS_HEAD + ' статус: сбой синхронизации'
    assert not msty_evidence.is_status_question(long_prompt)
    assert msty_evidence.is_status_question('Статус синхронизации? ' + 'x' * 1000)


def test_respond_leaves_daily_review_untouched(monkeypatch):
    _install_model(monkeypatch, 'Следующая задача: #237. Очередь начинается со сломанного '
                                'и медленного; сервис оплаты не работает — см. #12.')
    result = asyncio.run(msty.respond({'messages': [
        {'role': 'user', 'content': DAILY_REVIEW}], 'tools': []}))['result']
    assert result['content'].startswith('Следующая задача: #237.')
    assert 'tau_evidence_gate' not in result['response_metadata']
