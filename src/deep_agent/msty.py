"""One model step; the client executes tools and returns their real results.

State is replaced with the client's canonical conversation on each turn, avoiding
duplicate messages when a persisted thread is resumed.
"""
import asyncio
import json
import os
from typing import TypedDict
from jsonschema import Draft202012Validator, FormatChecker
from jsonschema.validators import validator_for
from langchain_anthropic import ChatAnthropic
from langchain_anthropic.chat_models import _format_messages
from langchain_core.messages import AIMessage, BaseMessage, SystemMessage, convert_to_messages
from langgraph.config import get_stream_writer
from langgraph.graph import StateGraph, START, END
from referencing import Registry
from . import msty_execution, msty_models, msty_compaction, msty_task, msty_stream, msty_memory
from .msty_prompts import ANALYST_POLICY, POLICY

COUNT_TRIGGER_BYTES = 200000
INPUT_TOKEN_LIMIT = 180000
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
или ответа инструмента в XML/JSON. Дождись настоящего tool-сообщения от Msty.
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
    compaction_skip_once: bool
    task_contract: dict | None
    text_stream_protocol: str | None
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


def rejected_context_budget(explanation: str, input_tokens: int | None = None):
    """A local blocker is not generated inference and does not consume its tokens."""
    result = AIMessage(content=explanation, response_metadata={'msty_generation': 'not_started'}, usage_metadata={
        'input_tokens': 0, 'output_tokens': 0, 'total_tokens': 0})
    return publish_result(result, {
        'version': 1, 'status': 'rejected', 'input_tokens': input_tokens,
        'limit': INPUT_TOKEN_LIMIT})


async def _respond_step(state: State, *, native_system_prompt: str | None = None,
                        native_result_filter=None):
    tools = state.get("tools") or []
    try:
        incremental = msty_stream.enabled(state)
    except ValueError as error:
        return rejected_context_budget(str(error))
    try:
        compaction_enabled = msty_compaction.enabled(state)
        if compaction_enabled and not msty_execution.enabled(state):
            raise msty_execution.ExecutionProtocolError('Сжатие требует checkpoint-протокола Msty.')
        messages = convert_to_messages(msty_compaction.project_messages(state))
        profile = selected_profile(state)
        consultations = consultation_count(state)
        cap = 2048 if state.get('brain_task_role') == 'analyst' else 8192
        output_limit = min(max(int(state.get('max_tokens') or 4096), 1), cap)
        msty_execution.validate_binding(state, profile, output_limit)
        model = (ChatAnthropic(model='claude-sonnet-4-6', max_tokens=output_limit,
                               base_url='https://api.anthropic.com', timeout=120, max_retries=0)
                 if profile == 'sonnet' and not msty_models.msty_gateway.enabled()
                 else msty_models.make_model(profile, output_limit))
        policy = (ANALYST_POLICY if state.get('brain_task_role') == 'analyst' else
                  native_system_prompt if native_system_prompt is not None else
                  policy_for_tools(tools) + '\n\n' + msty_memory.system_context())
        if consultations >= 2:
            policy += '\nЛимит консультаций исчерпан. Продолжай своими инструментами; не вызывай консультанта снова.'
        if any(msty_task._named(name, msty_task.PLAN_SUFFIX) for name in tool_names(tools)):
            policy += ('\nДля поручения с изменениями проверяемых локальных артефактов используй msty_task_plan '
                'с требованиями и конкретными read-only проверками, затем реальные рабочие инструменты. '
                'Для простого ответа, обсуждения или просьбы только составить план запуск проверок не нужен. '
                'Критерии, предложенные тобой, не доказывают полноту требований владельца. '
                'Результат msty_task_verify подтверждает лишь указанные наблюдения, не всю бизнес-задачу.')
        intervention = msty_task.progress_intervention(state)
        if intervention is not None:
            policy += '\n\n' + intervention
        full_messages = [SystemMessage(content=policy), *messages]
        full_messages = (cache_system_prefix(full_messages, tools) if profile == 'sonnet'
                         else msty_models.prepare_messages(profile, full_messages, tools))
    except (msty_models.ModelAdapterError, msty_models.msty_gateway.GatewayConfigurationError,
            msty_execution.ExecutionProtocolError) as error:
        return rejected_context_budget(str(error))
    # Byte size is only the preflight trigger, never a tokenizer estimate.
    # Count the exact complete messages and schemas used for generation below.
    input_bytes = len(json.dumps({'messages': [m.model_dump(mode='json') for m in full_messages],
                                 'tools': tools}, ensure_ascii=False).encode('utf-8'))
    budget_check = None
    protocol = state.get('context_budget')
    if protocol not in (None, CONTEXT_BUDGET_PROTOCOL, MODEL_BUDGET_PROTOCOL):
        return rejected_context_budget(
            'Генерация не запущена: неподдерживаемая версия проверки контекста. '
            'Сообщения и инструкции не сокращались.')
    if protocol is not None or state.get('task_budget_binding') is not None or input_bytes > COUNT_TRIGGER_BYTES:
        try:
            if profile == 'sonnet':
                count_options = {'timeout': COUNT_TIMEOUT_SECONDS}
                formatted_system, _ = _format_messages(full_messages)
                if isinstance(formatted_system, list):
                    count_options['system'] = formatted_system
                tokens = await asyncio.to_thread(
                    model.get_num_tokens_from_messages, full_messages, tools=tools, **count_options)
            else:
                tokens = await msty_models.count_input(profile, model, full_messages, tools)
            if isinstance(tokens, bool) or not isinstance(tokens, int) or tokens < 0:
                raise ValueError('Invalid token count')
        except msty_models.ModelAdapterError as error:
            return rejected_context_budget(str(error))
        except Exception:
            # Provider exceptions may contain prompts, headers or credentials;
            # keep raw details out of both the user result and our own logs.
            return rejected_context_budget(
                'Генерация не запущена: проверка размера контекста не завершилась. '
                'Контекст сохранён без обрезки; требуется восстановить проверку его размера.')
        if (compaction_enabled and not state.get('compaction_skip_once') and
                tokens >= msty_compaction.TRIGGER_TOKENS):
            plan = msty_compaction.make_plan(state)
            if plan is not None:
                return await _compact_step(state, profile, output_limit, policy, plan)
        if tokens > INPUT_TOKEN_LIMIT:
            return rejected_context_budget(
                f'Генерация не запущена: входной контекст превышает безопасный лимит '
                f'{INPUT_TOKEN_LIMIT} токенов. Сообщения, инструкции и результаты '
                'инструментов не сокращались; нужно уменьшить выбранные вложения '
                'или разделить задачу.', tokens)
        budget_check = {'version': 1, 'status': 'accepted', 'input_tokens': tokens,
                        'limit': INPUT_TOKEN_LIMIT, 'method': msty_models.count_method(profile, full_messages),
                        'model_profile': profile}
    choice = state.get("tool_choice") or "auto"
    tools_disabled = choice == 'none' or (isinstance(choice, dict) and choice.get('type') == 'none')
    if profile != 'sonnet':
        try:
            model = msty_models.bind_tools(profile, model, tools, choice)
        except msty_models.ModelAdapterError as error:
            return rejected_context_budget(str(error))
    elif tools:
        if tools_disabled:
            # Keep the schemas required by historical tool_use/tool_result
            # blocks. This SDK interprets the string "none" as a tool name.
            choice = {'type': 'none'}
        elif choice == "required":
            choice = "any"
        elif isinstance(choice, dict) and choice.get("type") == "function":
            choice = choice["function"]["name"]
        model = model.bind_tools(tools, tool_choice=choice)
    stream = msty_stream.TextStream(state) if incremental else None
    try:
        raw_result = (await stream.invoke(model, full_messages) if stream else
                      await model.ainvoke(full_messages))
    except msty_stream.StreamFailure as error:
        return publish_result(AIMessage(content=str(error), usage_metadata=None,
            response_metadata={'msty_generation': 'stream_failed', 'msty_blocked': True}), budget_check)
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
    stop_reason = result.response_metadata.get('stop_reason',
                                              result.response_metadata.get('finish_reason'))
    if stop_reason in ('max_tokens', 'length', 'model_context_window_exceeded', 'refusal', 'content_filter'):
        # A parsed prefix is not a completed instruction. Preserve provider
        # termination and measured usage, but never suspend for partial actions.
        result = result.model_copy(update={'tool_calls': [], 'invalid_tool_calls': [],
                                          'additional_kwargs': {}})
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
    if consultations + sum(consult_name(c['name']) for c in result.tool_calls) > 2:
        result = result.model_copy(update={'content': 'Лимит двух консультаций этого хода исчерпан; '
            'новые вызовы не выполнены. Нужна работа основного Brain с имеющимися доказательствами.',
            'tool_calls': [], 'invalid_tool_calls': [], 'additional_kwargs': {}})
    result = msty_task.gate_final(state, result, tools, tools_disabled)
    if not valid_tool_calls(result, tools):
        result = result.model_copy(update={'content': 'Схема проверки плана не подтверждена; действие не выдано.',
            'tool_calls': [], 'invalid_tool_calls': [], 'additional_kwargs': {},
            'response_metadata': {**result.response_metadata, 'msty_blocked': True}})
    if native_result_filter is not None:
        # Trusted Python integration only; never selected by request/state data.
        result = native_result_filter(result)
    # Explicit None clears any check left in a persisted LangGraph thread;
    # an earlier accepted count must never attest a different request.
    if stream:
        stream.finish(result)
    return publish_result(result, budget_check)


async def _compact_step(state, profile, output_limit, policy, plan):
    """Exactly one model call, published/charged before the internal resume."""
    cap = min(msty_compaction.SUMMARY_OUTPUT_CAP, output_limit)
    try:
        model = msty_models.make_model(profile, cap)
        messages = [SystemMessage(content=policy), *convert_to_messages(
            msty_compaction.summary_messages(state, plan))]
        messages = msty_models.prepare_messages(profile, messages, [])
        tokens = await msty_models.count_input(profile, model, messages, [])
        if type(tokens) is not int or not 0 <= tokens <= INPUT_TOKEN_LIMIT:
            return rejected_context_budget('Сводка не помещается в безопасный контекст; исходники сохранены.', tokens)
        model = msty_models.bind_tools(profile, model, [], 'none')
    except Exception:
        return rejected_context_budget('Не удалось проверить вход сводки; исходники сохранены без обрезки.')
    budget_check = {'version': 1, 'status': 'accepted', 'input_tokens': tokens,
                    'limit': INPUT_TOKEN_LIMIT, 'method': msty_models.count_method(profile, messages),
                    'model_profile': profile}
    raw = await model.ainvoke(messages)
    try:
        result = msty_models.stamp_usage(profile, raw)
    except msty_models.ModelAdapterError:
        return publish_result(AIMessage(content='Модель сводки не подтверждена; исходники сохранены.',
            usage_metadata=msty_models.checked_usage(profile, raw),
            response_metadata={'msty_generation': 'rejected_model', 'msty_blocked': True}), budget_check)
    try:
        reason = result.response_metadata.get('stop_reason', result.response_metadata.get('finish_reason'))
        if reason in ('max_tokens', 'length', 'model_context_window_exceeded', 'refusal', 'content_filter'):
            raise msty_execution.ExecutionProtocolError('Сводка не завершена; исходники сохранены.')
        update = msty_compaction.accept_summary(state, plan, result)
    except msty_execution.ExecutionProtocolError as error:
        blocked = result.model_copy(update={'content': str(error), 'tool_calls': [],
            'invalid_tool_calls': [], 'additional_kwargs': {},
            'response_metadata': {**result.response_metadata, 'msty_blocked': True}})
        return publish_result(blocked, budget_check)
    result = result.model_copy(update={'content': '', 'tool_calls': [], 'invalid_tool_calls': [],
        'additional_kwargs': {}, 'response_metadata': {**result.response_metadata, 'msty_stage': 'compaction'}})
    return {**publish_result(result, budget_check), **update}


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
        result.update(compaction_skip_once=False, compaction_stage=None)
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
