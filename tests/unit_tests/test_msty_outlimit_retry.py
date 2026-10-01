"""Out-of-limit recovery: incomplete → one retry in the same stage → answer.

Live 28.09 (Brain Desk, tax documents): the lead stage ended with
reasoning_tokens == output_tokens == 8192 and no text, so ~7 minutes of tool
work ended in a refusal. Offline real graph; only provider construction,
counting and inference are replaced. Not a quality benchmark.
"""
import asyncio
from copy import deepcopy
import uuid

import pytest
from langchain_core.messages import AIMessage, HumanMessage

from deep_agent import msty, msty_effort as effort, msty_execution as execution, msty_models

TOOLS = [{'type': 'function', 'function': {'name': 'read_fixture', 'parameters': {
    'type': 'object', 'properties': {'name': {'type': 'string'}},
    'required': ['name'], 'additionalProperties': False}}}]
OWNER = 'Проанализируй налоговые документы ИП: InvoiceXpress, досье и календарь AT.'
INCOMPLETE = {'status': 'incomplete', 'incomplete_details': {'reason': 'max_output_tokens'},
              'model_name': 'gpt-6-luna'}
DONE = {'status': 'completed', 'model_name': 'gpt-6-luna'}


@pytest.fixture(autouse=True)
def offline(monkeypatch):
    monkeypatch.delenv('MSTY_MODEL_PROFILE', raising=False)
    monkeypatch.setenv('LANGSMITH_TRACING', 'false')
    monkeypatch.setenv('LANGCHAIN_TRACING_V2', 'false')


def usage(output, *, input_tokens=134938, cached=133764):
    return {'input_tokens': input_tokens, 'output_tokens': output, 'total_tokens': input_tokens + output,
            'input_token_details': {'cache_read': cached}, 'output_token_details': {'reasoning': output}}


def reasoning_only(output):
    return AIMessage(content=[{'type': 'reasoning', 'id': 'rs_synthetic', 'summary': []}],
                     response_metadata=dict(INCOMPLETE), usage_metadata=usage(output))


def answer(text='Итог: досье сверено, срок AT — 20.10; не проверено: InvoiceXpress за сентябрь.',
           output=900, calls=None):
    return AIMessage(content=text, tool_calls=calls or [], response_metadata=dict(DONE),
                     usage_metadata={**usage(output), 'output_token_details': {'reasoning': 300}})


def initial(max_tokens=8192, text=OWNER, **changes):
    return {'messages': [{'role': 'user', 'content': text},
                         {'role': 'assistant', 'content': '', 'tool_calls': [{
                             'id': 'call_1', 'type': 'function',
                             'function': {'name': 'read_fixture', 'arguments': '{"name":"dossier"}'}}]},
                         {'role': 'tool', 'tool_call_id': 'call_1', 'content': 'Synthetic dossier.'}],
            'tools': deepcopy(TOOLS), 'max_tokens': max_tokens, 'tool_choice': 'auto',
            'result': {}, 'context_budget': None, 'context_budget_check': None,
            'execution_protocol': execution.PROTOCOL, 'execution': {}, **changes}


def bound(max_tokens=8192, input_limit=180000, **changes):
    return initial(max_tokens, execution_task_id=str(uuid.uuid4()), task_budget_binding={
        'version': 1, 'pricing_version': execution.PRICING_VERSION, 'profile': 'luna',
        'input_limit': input_limit, 'output_limit': max_tokens}, **changes)


def install(monkeypatch, responses, counts=None):
    seen = {'created': [], 'bound': [], 'invocations': [], 'counts': []}
    queue = list(responses)

    class Model:
        def bind_tools(self, tools, **kwargs):
            seen['bound'].append(deepcopy(kwargs))
            return self

        async def ainvoke(self, messages):
            seen['invocations'].append(deepcopy(messages))
            assert queue, 'Unexpected additional paid-model attempt'
            item = queue.pop(0)
            if isinstance(item, Exception):
                raise item
            return item

    def construct(profile, max_tokens, level=None):
        seen['created'].append((profile, max_tokens, level))
        return Model()

    monkeypatch.setattr(msty_models, 'make_model', construct)
    if counts is not None:
        remaining = list(counts)

        async def count(profile, model, messages, tools):
            seen['counts'].append(len(messages))
            assert remaining, 'Unexpected count'
            return remaining.pop(0)
        monkeypatch.setattr(msty_models, 'count_input', count)
    return seen


def run(state):
    return asyncio.run(msty.graph.ainvoke(state))


def test_incomplete_then_retry_answers_within_the_stage_limit(monkeypatch):
    seen = install(monkeypatch, [reasoning_only(6144), answer()])
    result = run(initial())
    # First call keeps the recovery reserve; the retry gets exactly the rest.
    assert seen['created'] == [('luna', 6144, 'high'), ('luna', 2048, 'low')]
    assert seen['bound'][-1] == {'tool_choice': 'none'}
    first, retry = seen['invocations']
    assert retry[:-1] == first  # same messages: the collected tool results
    assert isinstance(retry[-1], HumanMessage) and retry[-1].content == msty.RETRY_NOTE
    message = result['result']
    assert message['content'].startswith('Итог: досье сверено')
    assert message['tool_calls'] == []
    assert message['usage_metadata'] == {
        'input_tokens': 2 * 134938, 'output_tokens': 7044, 'total_tokens': 2 * 134938 + 7044,
        'input_token_details': {'cache_read': 2 * 133764}, 'output_token_details': {'reasoning': 6444}}
    assert message['usage_metadata']['output_tokens'] <= 8192
    record = message['response_metadata'][msty.RETRY_KEY]
    assert record == {'version': 1, 'stage_output_limit': 8192, 'tools_reexecuted': 0,
                      'status': 'answered',
                      'first': {'level': 'high', 'output_limit': 6144, 'output_tokens': 6144,
                                'input_tokens': 134938},
                      'retry': {'level': 'low', 'output_limit': 2048, 'output_tokens': 900,
                                'input_tokens': 134938}}
    assert message['response_metadata'][effort.METADATA_KEY]['reason'] == 'outlimit_retry'
    assert result['execution']['status'] == 'answered'
    assert result['execution']['actions_issued'] == 0


def test_larger_bound_limit_retries_one_level_lower(monkeypatch):
    seen = install(monkeypatch, [reasoning_only(12288), answer()])
    run(initial(16384))
    assert seen['created'] == [('luna', 12288, 'high'), ('luna', 4096, 'medium')]


def test_empty_retry_is_one_honest_refusal_without_a_third_call(monkeypatch):
    seen = install(monkeypatch, [reasoning_only(6144), reasoning_only(2048)])
    result = run(initial())
    assert len(seen['invocations']) == 2
    message = result['result']
    assert message['response_metadata']['finish_reason'] == 'length'
    assert 'повтор без инструментов на уровне low' in message['content']
    assert '6144 из 6144' in message['content'] and '2048 из 2048' in message['content']
    assert 'инструменты повторно не вызывались' in message['content']
    assert '«Продолжить»' in message['content']
    assert message['usage_metadata']['output_tokens'] == 8192
    assert result['execution']['status'] == 'incomplete'


def test_retry_never_issues_a_tool_call(monkeypatch):
    call = {'id': 'call_new', 'name': 'read_fixture', 'args': {'name': 'again'}}
    seen = install(monkeypatch, [reasoning_only(6144), answer('Читаю ещё раз.', calls=[call])])
    result = run(initial())
    assert len(seen['invocations']) == 2
    assert result['result']['tool_calls'] == []
    assert result['execution']['actions_issued'] == 0
    assert result['execution']['status'] != 'waiting_tools'


def test_bound_budget_counts_both_inputs_and_keeps_the_binding(monkeypatch):
    seen = install(monkeypatch, [reasoning_only(6144), answer()], counts=[70000, 70010])
    result = run(bound())
    assert len(seen['invocations']) == 2
    assert seen['counts'] == [len(seen['invocations'][0]), len(seen['invocations'][1])]
    check = result['context_budget_check']
    assert check['status'] == 'accepted' and check['input_tokens'] == 140010 <= check['limit']
    assert result['task_budget_binding']['output_limit'] == 8192
    assert result['result']['usage_metadata']['output_tokens'] <= 8192


def test_no_retry_when_both_inputs_exceed_the_stage_input_limit(monkeypatch):
    seen = install(monkeypatch, [reasoning_only(6144)], counts=[100000, 100010])
    result = run(bound())
    assert len(seen['invocations']) == 1
    assert 'исчерпан лимит вывода (6144 токенов)' in result['result']['content']
    assert result['context_budget_check']['input_tokens'] == 100000
    assert result['execution']['status'] == 'incomplete'


def test_no_retry_without_output_headroom(monkeypatch):
    seen = install(monkeypatch, [reasoning_only(6500)])
    run(initial())
    assert len(seen['invocations']) == 1


@pytest.mark.parametrize('state', [
    initial(4096),                                   # below the split threshold
    initial(text='!max разбери стратегию налогов'),  # owner force keeps its limit
])
def test_no_split_no_retry_cases(monkeypatch, state):
    seen = install(monkeypatch, [reasoning_only(min(state['max_tokens'], 2048))])
    run(state)
    assert len(seen['invocations']) == 1


def test_consultation_and_policy_fallback_keep_no_reserve():
    choice = {'version': 1, 'level': 'high', 'reason': 'deep_work'}
    assert msty.retry_reserve({}, 'luna', choice, 8192) == 2048
    assert msty.retry_reserve({'brain_task_role': 'analyst'}, 'luna', choice, 8192) == 0
    assert msty.retry_reserve({msty.POLICY_FALLBACK_KEY: {'version': 1}}, 'deepseek', choice, 8192) == 0
    assert msty.retry_reserve({}, 'sonnet', choice, 8192) == 0


def test_deepseek_has_no_reasoning_reserve(monkeypatch):
    monkeypatch.setenv('MSTY_MODEL_PROFILE', 'deepseek')
    seen = install(monkeypatch, [AIMessage(content='', response_metadata={'finish_reason': 'length'},
                                           usage_metadata={'input_tokens': 1, 'output_tokens': 8192,
                                                           'total_tokens': 8193})])
    run(initial())
    assert seen['created'] == [('deepseek', 8192, 'high')]
    assert len(seen['invocations']) == 1


def test_slow_first_call_is_not_retried(monkeypatch):
    class Clock:
        calls = 0

        @classmethod
        def monotonic(cls):
            cls.calls += 1
            return 0.0 if cls.calls == 1 else msty.RETRY_ADMISSION_SECONDS + 1
    monkeypatch.setattr(msty, 'time', Clock)
    seen = install(monkeypatch, [reasoning_only(6144)])
    run(initial())
    assert len(seen['invocations']) == 1


def test_failed_retry_keeps_cost_unknown_and_explains(monkeypatch):
    seen = install(monkeypatch, [reasoning_only(6144), TimeoutError('synthetic')])
    result = run(initial())
    assert len(seen['invocations']) == 2
    message = result['result']
    assert message['usage_metadata'] is None  # never a false complete total
    assert message['response_metadata'][msty.RETRY_KEY]['status'] == 'failed'
    assert 'повтор без инструментов не завершён' in message['content']
    assert result['execution']['status'] == 'incomplete'


def test_partial_text_is_kept_and_marked(monkeypatch):
    partial = AIMessage(content='Промежуточный вывод: досье сверено.', response_metadata=dict(INCOMPLETE),
                        usage_metadata=usage(6144))
    seen = install(monkeypatch, [partial])
    result = run(initial())
    assert len(seen['invocations']) == 1  # visible text is delivered, not replaced
    content = result['result']['content']
    assert content.startswith('Промежуточный вывод: досье сверено.')
    assert 'Ответ обрезан лимитом вывода (6144 токенов)' in content


def test_streamed_partial_text_is_not_rewritten():
    partial = AIMessage(content='Показанный текст', response_metadata=dict(INCOMPLETE))
    result = msty._explain_unfinished(partial, 'length', 6144, streamed=True)
    assert result.content == 'Показанный текст'
    assert result.response_metadata['finish_reason'] == 'length'


@pytest.mark.parametrize('applied,headroom,expected', [
    ('high', 2048, 'low'), ('high', 4096, 'medium'), ('max', 8192, 'medium'),
    ('max', 16384, 'high'), ('medium', 8192, 'low'), ('low', 2048, 'low')])
def test_retry_level_steps_down(applied, headroom, expected):
    assert effort.retry_level(applied, headroom) == expected


def test_sum_usage_is_never_cheaper_than_the_truth():
    a = {'input_tokens': 10, 'output_tokens': 5, 'total_tokens': 15,
         'input_token_details': {'cache_read': 8, 'cache_creation': 2}}
    b = {'input_tokens': 10, 'output_tokens': 5, 'total_tokens': 15,
         'input_token_details': {'cache_read': 8}}
    assert msty._sum_usage(a, b) == {'input_tokens': 20, 'output_tokens': 10, 'total_tokens': 30,
                                     'input_token_details': {'cache_read': 16}}
    assert msty._sum_usage(a, None) is None
    assert msty._sum_usage(a, {**b, 'output_tokens': -1}) is None


class PolicyRejected(Exception):
    def __init__(self):
        super().__init__('Error code: 400 - flagged')
        self.status_code, self.code = 400, 'invalid_prompt'
        self.body = {'code': 'invalid_prompt', 'type': 'invalid_request_error'}


def test_policy_fallback_runs_below_a_larger_luna_limit(monkeypatch):
    reply = AIMessage(content='Ответ.', usage_metadata={'input_tokens': 1, 'output_tokens': 1, 'total_tokens': 2})
    seen = install(monkeypatch, [PolicyRejected(), reply], counts=[1000, 1000])
    # DeepSeek's step ceiling is 32768 since #1195; Luna's limit is above it.
    result = run(bound(65536))
    assert [c[:2] for c in seen['created']] == [('luna', 49152), ('deepseek', 32768)]
    assert result['result']['response_metadata']['msty_model_profile'] == 'deepseek'
    assert result['task_budget_binding']['output_limit'] == 65536  # the reserve is unchanged
