"""One model step; the client executes tools and returns their real results.

State is replaced with the client's canonical conversation on each turn, avoiding
duplicate messages when a persisted thread is resumed.
"""
import asyncio
import json
import os
import time
from typing import TypedDict
from jsonschema import Draft202012Validator, FormatChecker
from jsonschema.validators import validator_for
from langchain_anthropic import ChatAnthropic
from langchain_anthropic.chat_models import _format_messages
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage, convert_to_messages
from langgraph.config import get_stream_writer
from langgraph.graph import StateGraph, START, END
from referencing import Registry
from . import msty_breaker, msty_effort, msty_evidence, msty_taxonomy
from . import msty_execution, msty_models, msty_compaction, msty_task, msty_stream, msty_memory
from .msty_prompts import ANALYST_POLICY, POLICY

COUNT_TRIGGER_BYTES = 200000
# Default/legacy admission; a bound task uses msty_execution.input_limit(state).
INPUT_TOKEN_LIMIT = msty_execution.LEGACY_INPUT_LIMIT
CONTEXT_BUDGET_PROTOCOL = 'anthropic-count-v1'
MODEL_BUDGET_PROTOCOL = 'msty-model-count-v1'
COUNT_TIMEOUT_SECONDS = 20.0




def tool_names(tools: list[dict]) -> set[str]:
    """Names come exclusively from the current client's function schemas."""
    names = set()
    for tool in tools:
        if not isinstance(tool, dict):
            continue
        function = tool.get('function')
        if tool.get('type') == 'function' and isinstance(function, dict):
            name = function.get('name')
            if isinstance(name, str) and name:
                names.add(name)
    return names


def tool_availability_context(tools: list[dict]) -> str:
    names = tool_names(tools)
    if not names:
        return """
MSTY_TOOLS_UNAVAILABLE: в этом запросе инструменты не подключены.
У тебя нет выполнения команд, чтения файлов, браузера или запуска других агентов.
Для обычного вопроса ответь по имеющемуся контексту. Если задача требует этих
возможностей, прямо скажи, что инструменты не подключены и действие не выполнено.
Не обещай проверить систему и не изображай tool_call/tool_response/tool_result,
XML/JSON-вызовы или результаты чтения в тексте. Не придумывай содержимое файлов.
Исторические результаты можно обсуждать как историю, но не как свежую проверку.
"""
    return """
MSTY_TOOLS_AVAILABLE: только следующие имена имеют схемы в текущем запросе:
""" + json.dumps(sorted(names), ensure_ascii=False) + """
Используй только установленным способом переданные схемы; отсутствующий сервис
считай отключённым для этого запроса. Вызывай инструмент через штатный механизм tool_calls, не печатай имитацию вызова
или ответа инструмента в XML/JSON. Дождись настоящего tool-сообщения от клиента.
Наличие схемы не доказывает работоспособность сервиса: учитывай результат вызова.
Ошибка одного вызова — это результат шага, а не доказательство отказа подсистемы.
Если вызов вернул «Unknown tool» или предлагает discover_tools, это имя относится
к другому коннектору: вызови его напрямую по его собственной схеме, не через
execute_tool. После ошибки один раз измени способ вызова и перечитай свежие
данные; одинаковый неуспешный вызов не повторяй. Вывод «не работает/не настроено»
делай только после успешного профильного статусного чтения, а не из косвенных
признаков (например списка cron на WordPress-хостинге) или неудачного вызова.
"""


def policy_for_tools(tools: list[dict]) -> str:
    return POLICY + tool_availability_context(tools)


def _has_cache_control(value) -> bool:
    if isinstance(value, dict):
        return 'cache_control' in value or any(_has_cache_control(v) for v in value.values())
    if isinstance(value, list):
        return any(_has_cache_control(v) for v in value)
    return False


def cache_system_prefix(messages: list[BaseMessage], tools: list[dict]) -> list[BaseMessage]:
    """Mark only our first, built-in policy, never client project/history blocks.

    The provider prefix also includes preceding tools. Never add a breakpoint
    when the client already controls caching: avoid slot/TTL conflicts or a
    silent retention change. No project system, user message or tool result gets a
    new marker; no warmup request or local content cache is created.
    """
    if os.getenv('MSTY_STATIC_CACHE', '1').strip().lower() in ('0', 'false', 'off'):
        return messages
    if _has_cache_control(tools) or any(_has_cache_control(m.content) for m in messages):
        return messages
    if not messages or not isinstance(messages[0], SystemMessage):
        return messages
    content = messages[0].content
    blocks = [{'type': 'text', 'text': content}] if isinstance(content, str) else content
    target = None
    for block_index, block in enumerate(blocks):
        if isinstance(block, str):
            text = block
        elif isinstance(block, dict) and block.get('type') == 'text':
            text = block.get('text')
        else:
            text = None
        if isinstance(text, str) and text.strip():
            target = block_index
    if target is None:
        return messages
    content = list(blocks)
    block = content[target]
    content[target] = {**({'type': 'text', 'text': block} if isinstance(block, str) else block),
                       'cache_control': {'type': 'ephemeral', 'ttl': '5m'}}
    prepared = list(messages)
    prepared[0] = messages[0].model_copy(update={'content': content})
    return prepared


class State(TypedDict):
    messages: list[dict]
    tools: list[dict]
    tool_choice: str | dict | None
    max_tokens: int
    result: dict
    context_budget: str | None
    context_budget_check: dict | None
    execution_protocol: str | None
    execution: dict
    brain_task_role: str
    execution_task_id: str
    task_budget_binding: dict
    compaction_protocol: str
    context_memory: dict
    compaction_stage: dict | None
    # How many compaction stages have run back to back since the last real
    # (non-compaction) generation. Internal only: never part of the bridge's
    # msty-compaction-v1 interrupt/resume wire shape. Capped at
    # msty_compaction.MAX_COMPACTIONS_PER_TURN by _respond_step.
    compaction_round: int
    task_contract: dict | None
    text_stream_protocol: str | None
    reasoning_stream_protocol: str | None
    project_memory_delivery: dict
    consult_profile: str | None
    lead_profile: str | None


def selected_profile(state: State) -> str:
    role = state.get('brain_task_role', 'lead')
    if role not in ('lead', 'analyst'):
        raise msty_models.ModelAdapterError('Недопустимая роль Brain.')
    if role == 'analyst':
        history = state.get('messages') or []
        if (state.get('tools') or any(not isinstance(m.get('content'), str) or m.get('tool_calls') for m in history)
                or len(json.dumps(history, ensure_ascii=False).encode()) > 96000):
            raise msty_models.ModelAdapterError('Аналитик принимает только ограниченный текст без инструментов.')
        consult = state.get('consult_profile')
        if consult is None:
            return 'deepseek'
        if consult not in msty_models.CONSULT_PROFILES:
            raise msty_models.ModelAdapterError('Недопустимый профиль консультанта.')
        return consult
    routed = state.get('lead_profile')
    if routed is not None:
        if routed not in msty_models.LEAD_PROFILES:
            raise msty_models.ModelAdapterError('Недопустимый маршрут основной модели Brain.')
        return routed
    # Compatibility for checkpoints created before server-side Jev routing.
    profile = os.getenv('MSTY_MODEL_PROFILE', msty_models.DEFAULT_PROFILE)
    if profile not in msty_models.PROFILES:
        raise msty_models.ModelAdapterError('Недопустимый серверный профиль Brain.')
    return profile


def consult_name(name: str) -> bool:
    return isinstance(name, str) and name.endswith('msty_brain_consult')


def consultation_limit() -> int:
    """Server-owned cap for optional paid analyst calls; disabled by default."""
    raw = os.getenv('MSTY_CONSULT_LIMIT', '0')
    try:
        limit = int(raw)
    except ValueError as error:
        raise msty_models.ModelAdapterError('Недопустимый лимит консультаций.') from error
    if not 0 <= limit <= 2:
        raise msty_models.ModelAdapterError('Лимит консультаций должен быть от 0 до 2.')
    return limit


def consultation_count(state: State) -> int:
    if msty_execution.enabled(state):
        count = (state.get('execution') or {}).get('consultations', 0)
        if type(count) is not int or not 0 <= count <= 2:
            raise msty_models.ModelAdapterError('Счётчик консультаций не подтверждён.')
        return count
    current = []
    for message in reversed(state.get('messages') or []):
        if message.get('role', message.get('type')) in ('user', 'human'):
            break
        current.extend(message.get('tool_calls') or [])
    return sum(consult_name(c.get('name') or c.get('function', {}).get('name', '')) for c in current)


def valid_tool_calls(result: AIMessage, tools: list[dict]) -> bool:
    """Validate the entire batch against exactly the client's current schemas.

    No repair generation or remote/file reference retrieval is permitted here.
    Duplicate names are ambiguous, so even a valid-looking call fails closed.
    """
    if result.invalid_tool_calls:
        return False
    definitions: dict[str, list[dict]] = {}
    for tool in tools:
        if not isinstance(tool, dict) or tool.get('type') != 'function':
            continue
        function = tool.get('function')
        if isinstance(function, dict) and isinstance(function.get('name'), str):
            definitions.setdefault(function['name'], []).append(function)
    for call in result.tool_calls:
        candidates = definitions.get(call.get('name'), [])
        if len(candidates) != 1:
            return False
        schema = candidates[0].get('parameters', {})
        try:
            # No dialect means current JSON Schema; an unknown explicit dialect
            # is rejected, not silently treated as another schema version.
            if isinstance(schema, dict) and '$schema' in schema:
                validator_class = validator_for(schema, default=None)
                if validator_class is None:
                    return False
            else:
                validator_class = Draft202012Validator
            validator_class.check_schema(schema)
            # Explicit empty registry disables the library's legacy network
            # retrieval. In-document $defs / anchors still resolve normally.
            validator = validator_class(schema, registry=Registry(),
                                        format_checker=FormatChecker())
            if not validator.is_valid(call.get('args')):
                return False
        except Exception:
            # Validation errors include instances/schema text; do not expose
            # them to logs, tracing or the client. Broken refs also fail closed.
            return False
    return True


def publish_result(result: AIMessage, budget_check: dict | None):
    """Authoritative complete guarded message; optional prior text is provisional."""
    message = result.model_dump()
    try:
        writer = get_stream_writer()
    except RuntimeError as error:
        # Direct offline callers of respond() retain the pre-stream API. Never
        # suppress errors from an actual runtime writer or other graph failures.
        if str(error) != 'Called get_config outside of a runnable context':
            raise
    else:
        writer({'type': 'validated_result', 'message': message})
    return {'result': message, 'context_budget_check': budget_check}


def rejected_context_budget(explanation: str, input_tokens: int | None = None, *, state=None):
    """A local blocker is not generated inference and does not consume its tokens."""
    result = AIMessage(content=explanation, response_metadata={'msty_generation': 'not_started'}, usage_metadata={
        'input_tokens': 0, 'output_tokens': 0, 'total_tokens': 0})
    return publish_result(result, {
        'version': 1, 'status': 'rejected', 'input_tokens': input_tokens,
        'limit': msty_execution.input_limit(state), 'window_admission': True})


# Отказ политики провайдера (OpenAI `invalid_prompt`) приходит ДО генерации:
# тот же вход той же модели повторять бесполезно. Один переход на другую семью
# за шаг; вложенный резерв и резерв со сжатием не допускаются. Мост принимает
# фактическую модель только по этой же таблице и помечает ответ владельцу.
POLICY_FALLBACK = {'luna': 'deepseek'}
POLICY_FALLBACK_KEY = 'msty_policy_fallback'
_IMAGE_OMITTED = ('[Изображение не передано резервной модели: основная модель отклонила запрос '
                  'фильтром провайдера, а резервная не принимает изображения.]')


def _without_images(messages):
    """Резервная DeepSeek не принимает изображения: явная текстовая замена, не пропуск."""
    result = []
    for message in messages:
        content = message.content
        if isinstance(content, list) and any(
                isinstance(b, dict) and b.get('type') in ('image_url', 'image') for b in content):
            content = [{'type': 'text', 'text': _IMAGE_OMITTED}
                       if isinstance(b, dict) and b.get('type') in ('image_url', 'image') else b
                       for b in content]
            message = message.model_copy(update={'content': content})
        result.append(message)
    return result


def policy_rejected_result(profile: str, code: str, budget_check, reason: str):
    """Честный отказ без генерации: причина видна владельцу, расход нулевой."""
    return publish_result(AIMessage(content=(
        f'Провайдер основной модели отклонил запрос фильтром своей политики (`{code}`), '
        f'генерации не было, действия не выполнены, расхода нет. {reason} '
        'Помогает новый чат (короче история) или сообщение без спорного вложения.'),
        usage_metadata={'input_tokens': 0, 'output_tokens': 0, 'total_tokens': 0},
        response_metadata={'msty_generation': 'not_started', 'msty_blocked': True,
                           'msty_policy_rejection': {'version': 1, 'profile': profile, 'code': code}}),
        budget_check)


async def _policy_fallback_step(state, profile, code, budget_check, *,
                                native_system_prompt, native_result_filter):
    fallback = POLICY_FALLBACK.get(profile)
    if state.get(POLICY_FALLBACK_KEY) is not None or fallback is None:
        return policy_rejected_result(profile, code, budget_check,
                                      'Резервная модель для этого профиля не предусмотрена либо уже отказала.')
    if state.get('brain_task_role') == 'analyst':
        return policy_rejected_result(profile, code, budget_check, 'Консультация не переключается на другую модель.')
    marker = {'version': 1, 'from': profile, 'to': fallback, 'reason': code}
    binding = state.get('task_budget_binding')
    # This nested one-off retry never compacts on its own and must never
    # decline for a cap it never approached: None disables the compaction
    # check outright for this attempt (distinct from an exhausted int count).
    fallback_state = {**state, 'lead_profile': fallback, 'compaction_round': None,
                      POLICY_FALLBACK_KEY: marker}
    if isinstance(binding, dict):
        # Локальная копия только для validate_binding этого шага; состояние
        # графа и мост сохраняют исходный допуск, мост сверяет его с marker.
        # A fallback with a lower output ceiling runs below the reserved limit.
        output = binding.get('output_limit')
        fallback_state['task_budget_binding'] = {**binding, 'profile': fallback, 'output_limit': (
            min(output, msty_models.max_output(fallback)) if type(output) is int else output)}
    return await _respond_step(fallback_state, native_system_prompt=native_system_prompt,
                               native_result_filter=native_result_filter)


#: Out-of-limit recovery (owner order 2026-09-28). Live 28.09, Brain Desk:
#: ~7 minutes of tax-document work ended with «исчерпан лимит вывода (8192
#: токенов)»; the ledger row of that stage had reasoning_tokens == output_tokens
#: == 8192 and no text. The bridge reserves ONE stage for its bound
#: output_limit and marks the task budget breached if the settled output of a
#: stage exceeds it, so a retry has to fit inside the same limit: a reasoning
#: lead call runs with output_limit minus a retry reserve, and only an empty
#: truncated answer spends that reserve on exactly one retry one level lower,
#: with the same messages (the tool results already collected) and tools off.
RETRY_SPLIT_MIN_OUTPUT = 8192
RETRY_MIN_OUTPUT = 2048
#: The bridge ends a request after 150 s; a first call slower than this leaves
#: no room for a retry to finish (Luna ≈ 100 output tokens/s, ledger 28.09).
RETRY_ADMISSION_SECONDS = 100.0
RETRY_KEY = 'msty_outlimit_retry'
RETRY_NOTE = ('[Служебно, не от владельца] Предыдущая попытка этого шага израсходовала весь лимит '
              'вывода на рассуждение и не дала текста. Инструменты в этом шаге недоступны и повторно '
              'не вызываются. Сразу дай итоговый ответ владельцу по уже полученным выше результатам '
              'инструментов: что установлено, что не проверено и следующий шаг. Кратко, без '
              'повторного разбора.')


def retry_reserve(state, profile: str, effort: dict, output_limit: int) -> int:
    """Output tokens held back from the first call for one recovery retry."""
    if (state.get('brain_task_role') == 'analyst' or state.get(POLICY_FALLBACK_KEY) is not None or
            msty_models.EFFORT_VALUES.get(profile) is None or output_limit < RETRY_SPLIT_MIN_OUTPUT):
        return 0
    reserve = max(RETRY_MIN_OUTPUT, output_limit // 4)
    level = effort['level']
    if (effort.get('reason') == 'owner_force' and
            msty_effort.fit(level, output_limit - reserve) != msty_effort.fit(level, output_limit)):
        return 0  # an explicit «!max» keeps its whole limit instead of a retry
    return reserve


def _visible_text(result: AIMessage) -> str:
    try:
        return msty_stream.text_content(result.content).strip()
    except msty_stream.StreamFailure:
        return ''


def _sum_usage(first, second):
    """Measured usage of both calls of one stage; unknown stays unknown.

    A detail known for only one call is dropped for cache_creation (the bridge
    then prices every uncached token as a write: an upper estimate) and read
    as 0 elsewhere (fewer cached tokens: never cheaper than the truth).
    """
    if not isinstance(first, dict) or not isinstance(second, dict):
        return None
    counts = ('input_tokens', 'output_tokens')
    if any(type(u.get(k)) is not int or u[k] < 0 for u in (first, second) for k in counts):
        return None
    total = {k: first[k] + second[k] for k in counts}
    total['total_tokens'] = total['input_tokens'] + total['output_tokens']
    for name in ('input_token_details', 'output_token_details'):
        a, b = first.get(name) or {}, second.get(name) or {}
        if not isinstance(a, dict) or not isinstance(b, dict):
            return None
        merged = {}
        for key in sorted(set(a) | set(b)):
            values = [d[key] for d in (a, b) if key in d]
            if any(type(v) is not int or v < 0 for v in values):
                return None
            if len(values) == 2 or key != 'cache_creation':
                merged[key] = sum(values)
        if merged:
            total[name] = merged
    return total


def _append_text(content, text):
    if isinstance(content, str):
        return content + text
    return [*content, {'type': 'text', 'text': text}]


def _explain_unfinished(result: AIMessage, stop_reason: str, output_limit: int, *,
                        streamed: bool = False) -> AIMessage:
    """Make a provider-terminated reply explicit to every consumer.

    Live 2026-09-26 (Brain Desk run e9b6a175, ledger output_tokens == 4096 ==
    output_limit): under the Responses API with reasoning.effort='max' Luna can
    spend the whole max_output_tokens on reasoning and return status=incomplete
    with no text item. The Responses wire carries no finish_reason, so the
    bridge (which reads only finish_reason/stop_reason) saw a clean 'stop' with
    empty content and reported «Модель вернула пустой ответ». Stamp the
    translated reason and, when no visible text exists, say what happened.
    Partial text is kept and marked; streamed text is never rewritten (the
    bridge rejects a final text that differs from the shown one), there the
    finish_reason 'length' is the mark and Brain Desk offers «Продолжить».
    """
    metadata = result.response_metadata or {}
    update = {}
    if metadata.get('finish_reason') is None and metadata.get('stop_reason') is None:
        update['response_metadata'] = {**metadata, 'finish_reason': stop_reason}
    visible = _visible_text(result)
    retry = metadata.get(RETRY_KEY) if isinstance(metadata.get(RETRY_KEY), dict) else None
    cut = stop_reason in ('max_tokens', 'length')
    if not visible and cut and retry is not None:
        first, second = retry['first'], retry['retry']
        if retry.get('status') == 'answered':
            repeat = (f'повтор без инструментов на уровне {second["level"]} тоже израсходовал '
                      f'{second["output_tokens"]} из {second["output_limit"]} токенов')
        else:
            repeat = ('повтор без инструментов не завершён (сбой провайдера или модели), '
                      'его расход не подтверждён')
        update['content'] = (
            f'Ответ модели не получен: попытка на уровне рассуждения {first["level"]} израсходовала '
            f'весь лимит вывода ({first["output_tokens"]} из {first["output_limit"]} токенов) на '
            f'рассуждение, {repeat}. Результаты инструментов этого хода сохранены в истории; '
            'инструменты повторно не вызывались. Нажмите «Продолжить» или сузьте задачу.')
    elif not visible:
        update['content'] = (
            f'Ответ модели не получен: исчерпан лимит вывода ({output_limit} токенов) до '
            'текстового ответа (у Luna его расходует рассуждение). Действия не выполнены; '
            'результаты инструментов этого хода сохранены. Нажмите «Продолжить» или сузьте задачу.'
            if cut else
            'Ответ модели не получен: провайдер прервал генерацию '
            f'({stop_reason}). Действия не выполнены.')
    elif cut and not streamed:
        update['content'] = _append_text(result.content, (
            f'\n\n— Ответ обрезан лимитом вывода ({output_limit} токенов): это неполный текст. '
            'Нажмите «Продолжить», чтобы Brain дописал его.'))
    return result.model_copy(update=update) if update else result


async def _retry_after_outlimit(state, profile, tools, full_messages, first, effort, first_limit,
                                output_limit, budget_check, stream):
    """One recovery call inside the same stage, or None to keep ``first``.

    Returns (result, budget_check). Admission: measured first output, headroom
    >= RETRY_MIN_OUTPUT, both counted inputs within the stage input limit and
    a closed circuit. Tools are bound with choice 'none': no tool call of this
    step is repeated and none is issued, so paid tool work is never doubled.
    """
    usage = first.usage_metadata
    used = usage.get('output_tokens') if isinstance(usage, dict) else None
    if type(used) is not int or used < 0:
        return None
    headroom = output_limit - used
    if headroom < RETRY_MIN_OUTPUT:
        return None
    applied = msty_effort.fit(effort['level'], first_limit)
    level = msty_effort.retry_level(applied, headroom)
    record = {'version': 1, 'stage_output_limit': output_limit, 'tools_reexecuted': 0,
              'first': {'level': applied, 'output_limit': first_limit, 'output_tokens': used,
                        'input_tokens': usage.get('input_tokens')},
              'retry': {'level': level, 'output_limit': headroom}}
    messages = [*full_messages, HumanMessage(content=RETRY_NOTE)]
    try:
        # Same stream object: the reasoning relay keeps its seq across calls.
        model = (msty_models.make_model(profile, headroom, level, True)
                 if stream is not None and getattr(stream, 'reasoning', None) is not None
                 else msty_models.make_model(profile, headroom, level))
        if budget_check is not None:
            tokens = await msty_models.count_input(profile, model, messages, tools)
            if (type(tokens) is not int or tokens < 0 or
                    budget_check['input_tokens'] + tokens > budget_check['limit']):
                return None
            budget_check = {**budget_check, 'input_tokens': budget_check['input_tokens'] + tokens}
        model = msty_models.bind_tools(profile, model, tools, 'none')
    except Exception:
        return None  # no call was made: keep the honest first result
    connection = 'model:' + profile
    remaining, probe_token = msty_breaker.admit(connection)
    if remaining is not None:
        return None
    try:
        raw = await (stream.invoke(model, messages) if stream else model.ainvoke(messages))
    except Exception as error:
        transient = (getattr(error, 'transient', False) if isinstance(error, msty_stream.StreamFailure)
                     else msty_taxonomy.is_transient_exception(error))
        (msty_breaker.record_transient_failure if transient else msty_breaker.record_success)(connection)
        failed = {**record, 'status': 'failed'}
        # The retry's expense is unknown: the stage usage must not look complete.
        return first.model_copy(update={'usage_metadata': None, 'response_metadata': {
            **first.response_metadata, RETRY_KEY: failed}}), budget_check
    finally:
        msty_breaker.release_probe(connection, probe_token)
    msty_breaker.record_success(connection)
    try:
        second = msty_models.stamp_usage(profile, raw)
    except msty_models.ModelAdapterError:
        if stream:
            stream.invalidate()
        return AIMessage(content='Ответ повтора пришёл от неподтверждённой модели; действия не выполнены.',
            usage_metadata=_sum_usage(usage, msty_models.checked_usage(profile, raw)),
            response_metadata={'msty_generation': 'rejected_model', 'msty_blocked': True,
                               RETRY_KEY: {**record, 'status': 'rejected_model'}}), budget_check
    second_usage = second.usage_metadata or {}
    record['retry'].update(output_tokens=second_usage.get('output_tokens'),
                           input_tokens=second_usage.get('input_tokens'))
    return second.model_copy(update={
        'usage_metadata': _sum_usage(usage, second.usage_metadata),
        'response_metadata': {
            **second.response_metadata,
            msty_effort.METADATA_KEY: msty_models.effort_record(
                profile, {'version': 1, 'level': level, 'reason': 'outlimit_retry'}, headroom),
            RETRY_KEY: {**record, 'status': 'answered'}}}), budget_check


SYNTHESIS_KEY = 'msty_step_synthesis'


async def _synthesis_step(profile, tools, choice, full_messages, low, effort, first_limit,
                          output_limit, budget_check, stream):
    """Re-run a lowered chain step that turned out to be the final answer.

    Owner order 2026-09-28: chain steps run at msty_effort.STEP_LEVEL, the
    answer keeps the turn level. Whether a step is the answer is known only
    after it: a low step without tool calls and with text that was not shown
    yet (native harness withholds text) is repeated once at the turn level
    with the same messages and tools. Returns (result, budget_check) or None to
    keep ``low``. The low answer is the fallback whenever the re-run cannot be
    admitted or does not produce a usable answer; its usage is summed.
    """
    usage = low.usage_metadata
    used = usage.get('output_tokens') if isinstance(usage, dict) else None
    if type(used) is not int or used < 0:
        return None
    limit = min(first_limit, output_limit - used)
    level = msty_effort.fit(effort['task_level'], limit) if limit > 0 else msty_effort.STEP_LEVEL
    if msty_effort.LEVELS.index(level) <= msty_effort.LEVELS.index(msty_effort.STEP_LEVEL):
        return None
    record = {'version': 1, 'status': 'answered', 'tools_reexecuted': 0,
              'first': {'level': effort['level'], 'output_tokens': used,
                        'input_tokens': usage.get('input_tokens')},
              'synthesis': {'level': level, 'output_limit': limit}}
    try:
        model = (msty_models.make_model(profile, limit, level, True)
                 if stream is not None and getattr(stream, 'reasoning', None) is not None
                 else msty_models.make_model(profile, limit, level))
        if budget_check is not None:
            tokens = await msty_models.count_input(profile, model, full_messages, tools)
            if (type(tokens) is not int or tokens < 0 or
                    budget_check['input_tokens'] + tokens > budget_check['limit']):
                return None
            budget_check = {**budget_check, 'input_tokens': budget_check['input_tokens'] + tokens}
        model = msty_models.bind_tools(profile, model, tools, choice)
    except Exception:
        return None  # no call was made: keep the low answer
    connection = 'model:' + profile
    remaining, probe_token = msty_breaker.admit(connection)
    if remaining is not None:
        return None
    try:
        raw = await (stream.invoke(model, full_messages) if stream else model.ainvoke(full_messages))
    except Exception as error:
        transient = (getattr(error, 'transient', False) if isinstance(error, msty_stream.StreamFailure)
                     else msty_taxonomy.is_transient_exception(error))
        (msty_breaker.record_transient_failure if transient else msty_breaker.record_success)(connection)
        # The re-run's expense is unknown: the stage usage must not look complete.
        return low.model_copy(update={'usage_metadata': None, 'response_metadata': {
            **low.response_metadata, SYNTHESIS_KEY: {**record, 'status': 'failed'}}}), budget_check
    finally:
        msty_breaker.release_probe(connection, probe_token)
    msty_breaker.record_success(connection)
    try:
        second = msty_models.stamp_usage(profile, raw)
    except msty_models.ModelAdapterError:
        return low.model_copy(update={'usage_metadata': None, 'response_metadata': {
            **low.response_metadata, SYNTHESIS_KEY: {**record, 'status': 'rejected_model'}}}), budget_check
    summed = _sum_usage(usage, second.usage_metadata)
    second_usage = second.usage_metadata or {}
    record['synthesis'].update(output_tokens=second_usage.get('output_tokens'),
                               input_tokens=second_usage.get('input_tokens'))
    stop = msty_models.finish_reason(second.response_metadata)
    if (stop in ('max_tokens', 'length', 'model_context_window_exceeded', 'refusal', 'content_filter')
            or (not _visible_text(second) and not second.tool_calls)):
        return low.model_copy(update={'usage_metadata': summed, 'response_metadata': {
            **low.response_metadata, SYNTHESIS_KEY: {**record, 'status': 'kept_low'}}}), budget_check
    choice_record = {'version': 1, 'level': effort['task_level'], 'reason': 'synthesis',
                     'step': 'synthesis', 'task_level': effort['task_level'],
                     'task_reason': effort['task_reason'], 'chain_steps': effort['chain_steps']}
    return second.model_copy(update={'usage_metadata': summed, 'response_metadata': {
        **second.response_metadata,
        msty_effort.METADATA_KEY: msty_models.effort_record(profile, choice_record, limit),
        SYNTHESIS_KEY: record}}), budget_check


async def _respond_step(state: State, *, native_system_prompt: str | None = None,
                        native_result_filter=None):
    tools = state.get("tools") or []
    policy_fallback = state.get(POLICY_FALLBACK_KEY)
    try:
        incremental = msty_stream.enabled(state)
        # Live reasoning summary (owner 28.09.2026): streams the model call even
        # where answer text stays withheld (native harness).
        reasoning_live = msty_stream.reasoning_enabled(state)
    except ValueError as error:
        return rejected_context_budget(str(error), state=state)
    try:
        compaction_enabled = msty_compaction.enabled(state)
        if compaction_enabled and not msty_execution.enabled(state):
            raise msty_execution.ExecutionProtocolError('Сжатие требует checkpoint-протокола Msty.')
        profile = selected_profile(state)
        consultations = consultation_count(state)
        choice = state.get("tool_choice") or "auto"
        tools_disabled = choice == 'none' or (isinstance(choice, dict) and choice.get('type') == 'none')
        has_consult_tool = any(consult_name(name) for name in tool_names(tools))
        consult_limit = (consultation_limit()
                         if has_consult_tool and state.get('brain_task_role') != 'analyst' else 2)
        # Do not offer paid consultations to the lead when the server disables
        # them or when this task has already used its explicit allowance.
        model_tools = (tools if tools_disabled or consultations < consult_limit else [
            tool for tool in tools
            if not (isinstance(tool, dict) and isinstance(tool.get('function'), dict)
                    and consult_name(tool['function'].get('name')))
        ])
        cap = 2048 if state.get('brain_task_role') == 'analyst' else msty_models.max_output(profile)
        output_limit = min(max(int(state.get('max_tokens') or 4096), 1), cap)
        msty_execution.validate_binding(state, profile, output_limit)
        # Per-task reasoning level (owner order 2026-09-26): deterministic, from
        # the latest owner message. Per step (owner order 2026-09-28): a step
        # that only continues a simple tool chain runs low; the plan, the
        # synthesis and rethinking after failures keep or raise it.
        effort = msty_effort.choose_effort(state)
        if msty_models.EFFORT_VALUES.get(profile) is not None:
            effort = msty_effort.step_effort(state, effort)
        # output_limit stays the stage's bound total; the first call leaves the
        # recovery reserve unused unless it ends truncated without text.
        reserve = retry_reserve(state, profile, effort, output_limit)
        first_limit = output_limit - reserve
        model = (ChatAnthropic(model='claude-sonnet-4-6', max_tokens=output_limit,
                               base_url='https://api.anthropic.com', timeout=120, max_retries=0)
                 if profile == 'sonnet' and not msty_models.msty_gateway.enabled()
                 else msty_models.make_model(profile, first_limit, effort['level'], reasoning_live)
                 if reasoning_live else msty_models.make_model(profile, first_limit, effort['level']))
        policy = (ANALYST_POLICY if state.get('brain_task_role') == 'analyst' else
                  native_system_prompt if native_system_prompt is not None else
                  policy_for_tools(model_tools) + '\n\n' + msty_memory.system_context())
        if (has_consult_tool and state.get('brain_task_role') != 'analyst'
                and consultations >= consult_limit):
            policy += ('\nКонсультации аналитика отключены настройкой сервера; продолжай без них.'
                       if consult_limit == 0 else
                       '\nЛимит консультаций исчерпан. Продолжай без новых консультаций.')
        if any(msty_task._named(name, msty_task.PLAN_SUFFIX) for name in tool_names(model_tools)):
            policy += ('\nДля поручения с изменениями проверяемых локальных артефактов используй msty_task_plan '
                'с требованиями и конкретными read-only проверками, затем реальные рабочие инструменты. '
                'Для простого ответа, обсуждения или просьбы только составить план запуск проверок не нужен. '
                'Критерии, предложенные тобой, не доказывают полноту требований владельца. '
                'Результат msty_task_verify подтверждает лишь указанные наблюдения, не всю бизнес-задачу.')
        intervention = msty_task.progress_intervention(state)
        if intervention is not None:
            policy += '\n\n' + intervention
        # The step after an open plan gate (brain-desk #443): why the turn goes on.
        continuation = msty_task.open_plan_intervention(state)
        if continuation is not None:
            policy += '\n\n' + continuation
        if policy_fallback is not None:
            policy += ('\n\nОсновная модель отклонила этот запрос фильтром политики провайдера; '
                       'отвечаешь ты как резервная модель. Изображения из переписки тебе не переданы: '
                       'если вопрос о картинке, прямо скажи, что её не видно, и попроси описать текстом. '
                       'Если отвечаешь текстом, начни с одной строки: «↪ Ответ резервной модели DeepSeek: '
                       'основная модель OpenAI отклонила запрос фильтром своей политики.»')
        def assemble(projected):
            messages = convert_to_messages(msty_compaction.clip_stale_browser_results(projected))
            if policy_fallback is not None:
                messages = _without_images(messages)
            full = [SystemMessage(content=policy), *messages]
            return (cache_system_prefix(full, model_tools) if profile == 'sonnet'
                    else msty_models.prepare_messages(profile, full, model_tools))
        full_messages = assemble(msty_compaction.project_messages(state))
    except (msty_models.ModelAdapterError, msty_models.msty_gateway.GatewayConfigurationError,
            msty_execution.ExecutionProtocolError) as error:
        return rejected_context_budget(str(error), state=state)
    # Byte size is only the preflight trigger, never a tokenizer estimate.
    # Count the exact complete messages and schemas used for generation below.
    def wire_bytes(full):
        return len(json.dumps({'messages': [m.model_dump(mode='json') for m in full],
                               'tools': model_tools}, ensure_ascii=False).encode('utf-8'))

    async def count_tokens(full):
        """Exact count, or a rejection dict (never raises provider details)."""
        try:
            if profile == 'sonnet':
                count_options = {'timeout': COUNT_TIMEOUT_SECONDS}
                formatted_system, _ = _format_messages(full)
                if isinstance(formatted_system, list):
                    count_options['system'] = formatted_system
                counted = await asyncio.to_thread(
                    model.get_num_tokens_from_messages, full, tools=model_tools, **count_options)
            else:
                counted = await msty_models.count_input(profile, model, full, model_tools)
            if isinstance(counted, bool) or not isinstance(counted, int) or counted < 0:
                raise ValueError('Invalid token count')
            return counted
        except msty_models.ModelAdapterError as error:
            return rejected_context_budget(str(error), state=state)
        except Exception:
            # Provider exceptions may contain prompts, headers or credentials;
            # keep raw details out of both the user result and our own logs.
            return rejected_context_budget(
                'Генерация не запущена: проверка размера контекста не завершилась. '
                'Контекст сохранён без обрезки; требуется восстановить проверку его размера.', state=state)

    input_bytes = wire_bytes(full_messages)
    budget_check = None
    memory_update = None
    protocol = state.get('context_budget')
    if protocol not in (None, CONTEXT_BUDGET_PROTOCOL, MODEL_BUDGET_PROTOCOL):
        return rejected_context_budget(
            'Генерация не запущена: неподдерживаемая версия проверки контекста. '
            'Сообщения и инструкции не сокращались.', state=state)
    if protocol is not None or state.get('task_budget_binding') is not None or input_bytes > COUNT_TRIGGER_BYTES:
        tokens = await count_tokens(full_messages)
        if isinstance(tokens, dict):
            return tokens
        limit = msty_execution.input_limit(state)
        compaction_round = state.get('compaction_round', 0)
        if (compaction_enabled and compaction_round is not None and
                tokens >= msty_compaction.trigger_tokens(state)):
            plan = msty_compaction.make_plan(state)
            if plan is not None and compaction_round < msty_compaction.MAX_COMPACTIONS_PER_TURN:
                return await _compact_step(state, profile, output_limit, policy, plan, compaction_round + 1)
            # brain-desk #899: no second paid summary in this request and no
            # refusal. Fit deterministically (no model call), then generate
            # while the input is within the admitted limit.
            fitted = await _fit_without_model(state, tokens, input_bytes, limit, assemble,
                                              wire_bytes, count_tokens)
            if isinstance(fitted, dict) and 'result' in fitted:
                return fitted
            full_messages, tokens, memory_update = fitted
        if tokens > limit:
            return rejected_context_budget(
                f'Генерация не запущена: входной контекст превышает безопасный лимит '
                f'{limit} токенов. Сообщения, инструкции и результаты '
                'инструментов не сокращались; нужно уменьшить выбранные вложения '
                'или разделить задачу.', tokens, state=state)
        budget_check = {'version': 1, 'status': 'accepted', 'input_tokens': tokens,
                        'limit': limit, 'window_admission': True,
                        'method': msty_models.count_method(profile, full_messages),
                        'model_profile': profile}
    if profile != 'sonnet' and model_tools:
        try:
            model = msty_models.bind_tools(profile, model, model_tools, choice)
        except msty_models.ModelAdapterError as error:
            return rejected_context_budget(str(error), state=state)
    elif model_tools:
        if tools_disabled:
            # Keep the schemas required by historical tool_use/tool_result
            # blocks. This SDK interprets the string "none" as a tool name.
            choice = {'type': 'none'}
        elif choice == "required":
            choice = "any"
        elif isinstance(choice, dict) and choice.get("type") == "function":
            choice = choice["function"]["name"]
        model = model.bind_tools(model_tools, tool_choice=choice)
    # TAU L4 circuit breaker: единственные удалённые вызовы графа — генерация
    # модели по серверным профилям. Открытый контур даёт детерминированный отказ
    # без обращения к провайдеру (без расхода и без нагрузки на лежащий сервис).
    connection = 'model:' + profile
    remaining, probe_token = msty_breaker.admit(connection)
    if remaining is not None:
        return publish_result(AIMessage(content=(
            f'Контур модели недоступен: circuit breaker открыт после '
            f'{msty_breaker.FAILURE_THRESHOLD} transient-отказов подряд, cooldown ещё '
            f'{int(remaining) + 1} с. Провайдер не вызывался; действия не выполнены, расхода нет.'),
            usage_metadata=None,
            response_metadata={'msty_generation': 'not_started', 'msty_blocked': True,
                               'tau_circuit_open': connection}), budget_check)
    stream = None
    policy_rejection = None
    started = time.monotonic()
    try:
        stream = (msty_stream.TextStream(state) if incremental else
                  msty_stream.TextStream(state, text=False) if reasoning_live else None)
        try:
            raw_result = (await stream.invoke(model, full_messages) if stream else
                          await model.ainvoke(full_messages))
        except msty_stream.StreamFailure as error:
            if getattr(error, 'policy_rejection', None) and not getattr(error, 'emitted', True):
                # Провайдер ответил отказом до первого фрагмента: контур жив.
                msty_breaker.record_success(connection)
                policy_rejection = error.policy_rejection
            elif getattr(error, 'transient', False):
                msty_breaker.record_transient_failure(connection)
            else:
                msty_breaker.record_success(connection)  # провайдер ответил: контур жив
            if policy_rejection is None:
                return publish_result(AIMessage(content=str(error), usage_metadata=None,
                    response_metadata={'msty_generation': 'stream_failed', 'msty_blocked': True}), budget_check)
        except Exception as error:
            policy_rejection = msty_taxonomy.policy_rejection_code(error)
            if policy_rejection is not None:
                msty_breaker.record_success(connection)  # провайдер ответил отказом: контур жив
            elif not msty_taxonomy.is_transient_exception(error):
                msty_breaker.record_success(connection)  # не транспорт: не держать пробу
                raise
            else:
                # Transient-отказ транспорта: классифицированный честный отказ вместо
                # падения рана; повтор поколения не выполняем — оно платное.
                opened = msty_breaker.record_transient_failure(connection)
                if stream:
                    stream.invalidate()
                return publish_result(AIMessage(content=(
                    'Вызов модели не завершён из-за временного сбоя контура'
                    + ('; circuit breaker открыт, контур охлаждается.' if opened else
                       '; допустим один повтор позже.')
                    + ' Действия не выполнены, расход не подтверждён.'), usage_metadata=None,
                    response_metadata={'msty_generation': 'transient_failure', 'msty_blocked': True,
                                       'tau_circuit_open': connection if opened else None}), budget_check)
    finally:
        # Исход пробы полуоткрытого контура записан выше (success/transient);
        # отмена или непредвиденный выход не должны держать пробу 150 с.
        msty_breaker.release_probe(connection, probe_token)
    if policy_rejection is not None:
        # Отказ до генерации: расход не возник, повтор той же модели бесполезен.
        return await _policy_fallback_step(state, profile, policy_rejection, budget_check,
                                           native_system_prompt=native_system_prompt,
                                           native_result_filter=native_result_filter)
    msty_breaker.record_success(connection)
    try:
        result = msty_models.stamp_usage(profile, raw_result)
    except msty_models.ModelAdapterError:
        # Generation already happened. Keep checked current usage but omit an
        # unverified identity; gateway retains unknown cost rather than zero.
        if stream:
            stream.invalidate()
        return publish_result(AIMessage(content='Ответ пришёл от неподтверждённой модели; '
            'действия не выполнены. Требуется проверить модель провайдера.',
            usage_metadata=msty_models.checked_usage(profile, raw_result),
            response_metadata={'msty_generation': 'rejected_model', 'msty_blocked': True}), budget_check)
    result = result.model_copy(update={'response_metadata': {
        **result.response_metadata,
        msty_effort.METADATA_KEY: msty_models.effort_record(profile, effort, first_limit)}})
    stop_reason = msty_models.finish_reason(result.response_metadata)
    if (effort.get('step') == 'tool_chain' and not tools_disabled and not result.tool_calls and
            stop_reason not in ('max_tokens', 'length', 'model_context_window_exceeded', 'refusal',
                                'content_filter') and _visible_text(result) and
            not (stream and stream.parts) and time.monotonic() - started <= RETRY_ADMISSION_SECONDS):
        synthesized = await _synthesis_step(profile, tools, choice, full_messages, result, effort,
                                            first_limit, output_limit, budget_check, stream)
        if synthesized is not None:
            result, budget_check = synthesized
            stop_reason = msty_models.finish_reason(result.response_metadata)
    if (reserve and stop_reason in ('max_tokens', 'length') and not _visible_text(result) and
            not (stream and stream.parts) and time.monotonic() - started <= RETRY_ADMISSION_SECONDS):
        retried = await _retry_after_outlimit(state, profile, tools, full_messages, result, effort,
                                              first_limit, output_limit, budget_check, stream)
        if retried is not None:
            result, budget_check = retried
            tools_disabled = True  # the retry is bound with tool_choice 'none'
            stop_reason = msty_models.finish_reason(result.response_metadata)
    if stop_reason in ('max_tokens', 'length', 'model_context_window_exceeded', 'refusal', 'content_filter'):
        # A parsed prefix is not a completed instruction. Preserve provider
        # termination and measured usage, but never suspend for partial actions.
        result = result.model_copy(update={'tool_calls': [], 'invalid_tool_calls': [],
                                          'additional_kwargs': {}})
        result = _explain_unfinished(result, stop_reason, first_limit,
                                     streamed=bool(stream and stream.parts))
    allowed = tool_names(tools)
    if (tools_disabled and result.tool_calls) or not valid_tool_calls(result, tools):
        # Fail closed before the client can execute an invented operation. Keep
        # measured usage: rejecting output does not undo the provider expense.
        if tools_disabled:
            explanation = 'Исполнение инструментов отключено для этого шага; вызов не выполнен.'
        else:
            explanation = ('В этом чате инструменты не подключены; действие не выполнено.'
                           if not allowed else
                           'Модель запросила неподключённый или некорректный инструмент; вызов не выполнен.')
        result = result.model_copy(update={'content': explanation, 'tool_calls': [],
                                          'invalid_tool_calls': [], 'additional_kwargs': {}})
    requested_consults = sum(consult_name(c['name']) for c in result.tool_calls)
    if requested_consults and consultations + requested_consults > consult_limit:
        reason = ('Консультации аналитика отключены настройкой сервера; '
                  if consult_limit == 0 else
                  f'Лимит консультаций этого хода ({consult_limit}) исчерпан; ')
        result = result.model_copy(update={'content': reason +
            'новые вызовы не выполнены. Нужна работа основного Brain с имеющимися доказательствами.',
            'tool_calls': [call for call in result.tool_calls if not consult_name(call['name'])],
            'invalid_tool_calls': [], 'additional_kwargs': {}})
    result = msty_task.gate_final(state, result, tools, tools_disabled)
    if not valid_tool_calls(result, tools):
        result = result.model_copy(update={'content': 'Схема проверки плана не подтверждена; действие не выдано.',
            'tool_calls': [], 'invalid_tool_calls': [], 'additional_kwargs': {},
            'response_metadata': {**result.response_metadata, 'msty_blocked': True}})
    if native_result_filter is not None:
        # Trusted Python integration only; never selected by request/state data.
        result = native_result_filter(result)
    # TAU L5 Evidence Gate: негативный вывод допускается только после успешного
    # профильного статус-чтения. До stream.finish: переписанный текст
    # инвалидирует уже показанный provisional-поток.
    result, _ = msty_evidence.gate_final_answer(state, result)
    # Explicit None clears any check left in a persisted LangGraph thread;
    # an earlier accepted count must never attest a different request.
    if policy_fallback is not None:
        # Мост принимает фактическую модель резерва только по этой метке.
        result = result.model_copy(update={'response_metadata': {
            **result.response_metadata, POLICY_FALLBACK_KEY: dict(policy_fallback)}})
    if stream:
        stream.finish(result)
    published = publish_result(result, budget_check)
    if memory_update is not None:
        # Deterministic segments of this step persist, so the next step or
        # request does not redo (or pay for) the same compaction.
        published['context_memory'] = memory_update
    return published


FIT_ATTEMPTS = 3


async def _fit_without_model(state, tokens, input_bytes, limit, assemble, wire_bytes, count_tokens):
    """Deterministic fit after the one paid pass (brain-desk #899).

    1. mechanical_fit: old tool runs → verbatim extracts, old summaries →
       shorter ones, aiming below TARGET_SHARE of the trigger (persisted).
    2. Only while the input still exceeds the admitted limit: fit_projection
       drops the oldest turns / clips the largest tool results for this
       generation only (never persisted, never owner text).
    Each attempt re-counts exactly; counting is not generation. Returns
    (full_messages, tokens, context_memory|None) or a rejection dict.
    """
    ratio = input_bytes / max(tokens, 1)
    work = state
    memory_update = None
    full_messages = None
    target = msty_compaction.mechanical_target_tokens(state)
    try:
        memory, _ = msty_compaction.mechanical_fit(work, int((tokens - target) * ratio * 1.15) + 1)
    except msty_execution.ExecutionProtocolError as error:
        return rejected_context_budget(str(error), state=state)
    if memory is not None:
        memory_update = memory
        work = {**state, 'context_memory': memory}
        full_messages = assemble(msty_compaction.project_messages(work))
        tokens = await count_tokens(full_messages)
        if isinstance(tokens, dict):
            return tokens
    projected = msty_compaction.project_messages(work)
    for attempt in range(FIT_ATTEMPTS):
        if tokens <= limit:
            break
        target = int(limit * msty_compaction.TARGET_SHARE)
        need = int((tokens - target) * ratio * 1.15 * (attempt + 1)) + 1
        projected, saved = msty_compaction.fit_projection(projected, need)
        if saved <= 0:
            break  # nothing left to drop or clip: an honest limit refusal follows
        full_messages = assemble(projected)
        tokens = await count_tokens(full_messages)
        if isinstance(tokens, dict):
            return tokens
        ratio = wire_bytes(full_messages) / max(tokens, 1)
    if full_messages is None:
        full_messages = assemble(projected)
    return full_messages, tokens, memory_update


async def _compact_step(state, profile, output_limit, policy, plan, round_number):
    """Exactly one model call, published/charged before the internal resume.

    round_number (1..MAX_COMPACTIONS_PER_TURN) is this turn's compaction count
    so far, including this stage; it is recorded on the state (compaction_round)
    so the next respond can cap or continue rounds. It is never part of the
    compaction_stage dict or the interrupt/resume wire payload themselves.
    """
    cap = min(msty_compaction.SUMMARY_OUTPUT_CAP, output_limit)
    try:
        model = msty_models.make_model(profile, cap, msty_effort.COMPACTION_LEVEL)
        messages = [SystemMessage(content=policy), *convert_to_messages(
            msty_compaction.summary_messages(state, plan))]
        messages = msty_models.prepare_messages(profile, messages, [])
        tokens = await msty_models.count_input(profile, model, messages, [])
        if type(tokens) is not int or not 0 <= tokens <= msty_execution.input_limit(state):
            return rejected_context_budget('Сводка не помещается в безопасный контекст; исходники сохранены.',
                                           tokens, state=state)
        model = msty_models.bind_tools(profile, model, [], 'none')
    except Exception:
        return rejected_context_budget('Не удалось проверить вход сводки; исходники сохранены без обрезки.',
                                       state=state)
    budget_check = {'version': 1, 'status': 'accepted', 'input_tokens': tokens,
                    'limit': msty_execution.input_limit(state), 'window_admission': True,
                    'method': msty_models.count_method(profile, messages),
                    'model_profile': profile}
    raw = await model.ainvoke(messages)
    try:
        result = msty_models.stamp_usage(profile, raw)
    except msty_models.ModelAdapterError:
        return publish_result(AIMessage(content='Модель сводки не подтверждена; исходники сохранены.',
            usage_metadata=msty_models.checked_usage(profile, raw),
            response_metadata={'msty_generation': 'rejected_model', 'msty_blocked': True}), budget_check)
    try:
        reason = msty_models.finish_reason(result.response_metadata)
        if reason in ('max_tokens', 'length', 'model_context_window_exceeded', 'refusal', 'content_filter'):
            raise msty_execution.ExecutionProtocolError('Сводка не завершена; исходники сохранены.')
        try:
            update = msty_compaction.accept_summary(state, plan, result)
        except msty_execution.ExecutionProtocolError as rejected:
            # The call completed normally but its text is unusable (not JSON,
            # wrong sources, too long, tool calls). Publishing the rejection
            # ended the owner's turn with no action (live 28.09): keep the
            # paid stage and continue on a deterministic extract instead.
            update = msty_compaction.accept_mechanical(state, plan, str(rejected))
    except msty_execution.ExecutionProtocolError as error:
        blocked = result.model_copy(update={'content': str(error), 'tool_calls': [],
            'invalid_tool_calls': [], 'additional_kwargs': {},
            'response_metadata': {**result.response_metadata, 'msty_blocked': True}})
        return publish_result(blocked, budget_check)
    result = result.model_copy(update={'content': '', 'tool_calls': [], 'invalid_tool_calls': [],
        'additional_kwargs': {}, 'response_metadata': {**result.response_metadata, 'msty_stage': 'compaction'}})
    return {**publish_result(result, budget_check), **update, 'compaction_round': round_number}


async def respond(state: State):
    durable = msty_execution.enabled(state)
    result = await _respond_step(state)
    if durable:
        compacted = (result.get('compaction_stage') or {}).get('status') == 'ready'
        result['execution'] = (msty_compaction.execution_after(state) if compacted else
                               msty_execution.execution_after(state, result))
        result['execution']['consultations'] = consultation_count(state) + sum(
            consult_name(c['name']) for c in result['result'].get('tool_calls', []))
        result['task_contract'] = msty_task.after_result(state, result['result'])
        result['execution']['status'] = msty_task.final_status(
            {**state, 'task_contract': result['task_contract']}, result['execution']['status'])
    if not result.get('compaction_stage'):
        result.update(compaction_round=0, compaction_stage=None)
    return result


builder = StateGraph(State)
builder.add_node("load_project_memory", msty_memory.load_context)
builder.add_node("respond", respond)
builder.add_node("wait_external", msty_execution.wait_external)
builder.add_node("wait_compaction", msty_compaction.wait_compaction)
builder.add_edge(START, "load_project_memory")
builder.add_edge("load_project_memory", "respond")
builder.add_conditional_edges("respond", msty_execution.next_node,
                              {"wait_external": "wait_external", "wait_compaction": "wait_compaction", "__end__": END})
builder.add_edge("wait_external", "respond")
builder.add_edge("wait_compaction", "respond")
graph = builder.compile()
