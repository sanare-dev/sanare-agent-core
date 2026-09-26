"""Per-task reasoning effort (owner order 2026-09-26): classifier, mapping, graph.

Offline: fake keys, mock construction; no provider calls.
"""
import asyncio

import pytest
from langchain_core.messages import AIMessage

from deep_agent import msty, msty_effort as effort, msty_execution as execution, msty_models

USAGE = {'input_tokens': 10, 'output_tokens': 2, 'total_tokens': 12}


@pytest.fixture(autouse=True)
def offline(monkeypatch):
    for name in ('OPENAI_API_KEY', 'DEEPSEEK_API_KEY', 'ANTHROPIC_API_KEY'):
        monkeypatch.setenv(name, 'synthetic-offline-not-a-real-key')
    monkeypatch.delenv('MSTY_MODEL_PROFILE', raising=False)
    monkeypatch.setenv('LANGSMITH_TRACING', 'false')
    monkeypatch.setenv('LANGCHAIN_TRACING_V2', 'false')


@pytest.mark.parametrize('text,level,reason', [
    ('привет', 'low', 'greeting'),
    ('Привет!', 'low', 'greeting'),
    ('привет, как дела?', 'low', 'greeting'),
    ('Спасибо', 'low', 'greeting'),
    ('hi', 'low', 'greeting'),
    ('сколько сейчас времени в Москве?', 'low', 'short_question'),
    ('напиши письмо поставщику о задержке поставки и попроси подтвердить новые сроки к пятнице',
     'medium', 'ordinary'),
    ('составь план миграции базы', 'high', 'deep_work'),
    ('исправь баг в коде моста', 'high', 'deep_work'),
    ('поручи Архитектору разбор задачи', 'high', 'deep_work'),
    ('подумай, стоит ли переходить на новый склад', 'high', 'deep_work'),
    ('проведи аудит расходов за неделю', 'high', 'deep_work'),
    ('Compare the synthetic fixtures.', 'high', 'deep_work'),
    ('see the fixtures list', 'low', 'short_question'),
    ('!max привет', 'max', 'owner_force'),
    ('!low составь план миграции', 'low', 'owner_force'),
    ('максимально подумай над этим', 'max', 'owner_force'),
    ('', 'medium', 'no_owner_text'),
    ('а' * 1300, 'high', 'long_request'),
])
def test_classifier(text, level, reason):
    assert effort.classify(text) == (level, reason)


def test_architect_persona_raises_ordinary_work_but_not_greeting():
    persona = {'role': 'system', 'content': 'Ты отвечаешь владельцу как его агент «Архитектор» в Brain Desk.'}
    ask = {'role': 'user', 'content': 'что по задаче #12?'}
    assert effort.choose_effort({'messages': [persona, ask]})['level'] == 'high'
    assert effort.choose_effort({'messages': [persona, {'role': 'user', 'content': 'привет'}]})['level'] == 'low'
    other = {'role': 'system', 'content': 'Ты отвечаешь владельцу как его агент «Бухгалтер» в Brain Desk.'}
    assert effort.choose_effort({'messages': [other, ask]})['level'] == 'low'


def test_tool_loop_follow_up_reuses_the_turn_level():
    turn = [{'role': 'user', 'content': 'составь план релиза'}]
    first = effort.choose_effort({'messages': turn})
    follow = turn + [{'role': 'assistant', 'content': '', 'tool_calls': [
        {'id': 'c1', 'type': 'function', 'function': {'name': 'read', 'arguments': '{}'}}]},
        {'role': 'tool', 'tool_call_id': 'c1', 'content': 'ok'}]
    assert effort.choose_effort({'messages': follow}) == first


@pytest.mark.parametrize('level,limit,applied', [
    ('max', 8192, 'max'), ('max', 4096, 'high'), ('high', 4096, 'high'),
    ('high', 2048, 'medium'), ('medium', 512, 'low'), ('low', 16, 'low')])
def test_fit_steps_down_to_the_output_limit(level, limit, applied):
    assert effort.fit(level, limit) == applied


@pytest.mark.parametrize('profile,level,limit,value', [
    ('luna', 'low', 4096, 'low'), ('luna', 'high', 4096, 'high'), ('luna', 'max', 8192, 'max'),
    ('luna', 'max', 4096, 'high'),
    ('sol6', 'max', 8192, 'high'), ('sol6', 'medium', 2048, 'medium'),
    ('deepseek', 'max', 8192, None), ('sonnet', 'high', 8192, None), ('opus5', 'high', 8192, None),
    ('astra', 'max', 8192, None)])
def test_provider_mapping(profile, level, limit, value):
    assert msty_models.effort_value(profile, level, limit) == value


def test_make_model_sends_the_mapped_level():
    assert msty_models.make_model('luna', 4096, 'low').reasoning == {'effort': 'low'}
    assert msty_models.make_model('luna', 8192, 'max').reasoning == {'effort': 'max'}
    assert msty_models.make_model('luna', 4096, 'max').reasoning == {'effort': 'high'}
    assert msty_models.make_model('sol6', 2048, 'high').reasoning_effort == 'medium'
    deepseek = msty_models.make_model('deepseek', 4096, 'max')
    assert deepseek.extra_body == {'thinking': {'type': 'disabled'}, 'max_tokens': 4096}
    with pytest.raises(msty_models.ModelAdapterError):
        msty_models.make_model('luna', 4096, 'extreme')


def test_record_marks_a_stepped_down_level():
    choice = {'version': 1, 'level': 'max', 'reason': 'owner_force'}
    assert msty_models.effort_record('luna', choice, 4096) == {
        **choice, 'profile': 'luna', 'provider_value': 'high',
        'applied_level': 'high', 'output_limit': 4096}
    assert msty_models.effort_record('deepseek', choice, 4096) == {
        **choice, 'profile': 'deepseek', 'provider_value': None}


def _graph(monkeypatch, text, max_tokens=4096):
    created = []

    class Model:
        def bind_tools(self, tools, **kwargs):
            return self

        async def ainvoke(self, messages):
            return AIMessage(content='OK', usage_metadata=dict(USAGE), response_metadata={})

    def construct(profile, limit, level=None):
        created.append((profile, limit, level))
        return Model()

    monkeypatch.setattr(msty_models, 'make_model', construct)
    state = {'messages': [{'role': 'user', 'content': text}], 'tools': [], 'max_tokens': max_tokens,
             'tool_choice': 'auto', 'result': {}, 'context_budget': None, 'context_budget_check': None,
             'execution_protocol': execution.PROTOCOL, 'execution': {}}
    result = asyncio.run(msty.graph.ainvoke(state))
    return created, result['result']['response_metadata'][effort.METADATA_KEY]


def test_graph_greeting_runs_low_and_records_it(monkeypatch):
    created, record = _graph(monkeypatch, 'привет')
    assert created == [('luna', 4096, 'low')]
    assert record == {'version': 1, 'level': 'low', 'reason': 'greeting',
                      'profile': 'luna', 'provider_value': 'low'}


def test_graph_planning_runs_high(monkeypatch):
    created, record = _graph(monkeypatch, 'Спланируй запуск нового продукта по шагам')
    assert created == [('luna', 4096, 'high')]
    assert record['provider_value'] == 'high'


def test_graph_forced_max_steps_down_below_8192(monkeypatch):
    _, record = _graph(monkeypatch, '!max разбери стратегию', 4096)
    assert record['level'] == 'max' and record['provider_value'] == 'high'
    assert record['applied_level'] == 'high'
    _, record = _graph(monkeypatch, '!max разбери стратегию', 8192)
    assert record['provider_value'] == 'max' and 'applied_level' not in record
