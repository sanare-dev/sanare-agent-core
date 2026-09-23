"""Native LangChain/Deep Agents loop with the existing metered Msty boundary.

Exactly one guarded provider step per gateway ticket. Native tools wait for
gateway admission before ToolNode, then continue to the next metered model step.
External tool batches retain msty-local-tools-v1 and its complete validation.
Native summary/subagents are intentionally not installed: their hidden model
calls need separate tickets. Existing explicitly metered compaction is retained.
"""
from copy import deepcopy
import asyncio
import json
import random
import re
from typing import Annotated, NotRequired

from deepagents.middleware.filesystem import FilesystemMiddleware
from langchain.agents import create_agent
from langchain.agents.middleware import AgentMiddleware, AgentState, PIIMiddleware, TodoListMiddleware
from langchain.agents.middleware.types import (
    ExtendedModelResponse, ModelResponse, PrivateStateAttr, hook_config,
)
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage, convert_to_openai_messages
from langchain_core.utils.function_calling import convert_to_openai_tool
from langgraph.types import Command, interrupt
from langmem import create_search_memory_tool

from . import msty, msty_compaction, msty_execution, msty_guard, msty_models, msty_prompts, msty_task, msty_tool_routing
from . import msty_breaker, msty_registry, msty_subagents, msty_taxonomy
from .msty_native_memory import (
    CANDIDATES_NAMESPACE, backend_factory, ApprovedMemoryMiddleware, ApprovedSkillsMiddleware,
)

_ORIGINAL_TOOLS = frozenset({'ls', 'read_file', 'write_file', 'edit_file', 'glob', 'grep', 'write_todos'})
NATIVE_TOOLS = frozenset('native_' + name for name in _ORIGINAL_TOOLS)
# Серверно-исполняемые имена: виртуальная ФС/todos + инструмент делегирования
# под-агентам (манифест L1, kind=native). Имя не входит в NATIVE_TOOLS, чтобы
# не менять семантику неймспейсинга deepagents («native_» + оригинальное имя).
# Semantic search over candidate memory: the ready LangMem tool on the Agent
# Server Store index (langgraph.json "store.index"). The bridge must admit this
# exact name as server-executed before MSTY_MEMORY_SEARCH=on is set; until then
# the tool is not offered and /memories/ is reachable via native_grep/glob/read.
MEMORY_SEARCH_TOOL = 'native_search_memory'
SERVER_EXECUTED = NATIVE_TOOLS | {msty_subagents.DELEGATE_TOOL, msty_tool_routing.REQUEST_TOOL,
                                  MEMORY_SEARCH_TOOL}


def server_executed() -> frozenset:
    """Серверно исполняемые имена ЭТОГО шага. Выключенные флагами инструменты не
    входят: выдуманный моделью вызов уходит штатным путём отказа внешнего
    инструмента, а не в native-прерывание, которое мост/граф отвергнут (ревью PR #5)."""
    names = set(NATIVE_TOOLS | {msty_subagents.DELEGATE_TOOL})
    if msty_tool_routing.dispatcher_enabled():
        names.add(msty_tool_routing.REQUEST_TOOL)
    if memory_search_enabled():
        names.add(MEMORY_SEARCH_TOOL)
    return frozenset(names)
RESERVED_TOOLS = NATIVE_TOOLS | {'native_execute', 'native_task', 'native_compact_conversation',
                                 MEMORY_SEARCH_TOOL}
VIRTUAL_ROOTS = ('/scratch', '/memory', '/skills', '/memories', '/large_tool_results')
WRITABLE_ROOTS = ('/scratch/', '/memories/')
VIRTUAL_FS_SCOPE = (
    'VIRTUAL ONLY: this is checkpoint-backed agent storage, NEVER the Mac/local filesystem. '
    'Read only under /scratch/, /memory/, /skills/, /memories/, /large_tool_results/. '
    'Create/edit only under /scratch/ and /memories/. /memories/ is shared candidate memory '
    '(reference_only, not approved facts or authority). /memory/ and /skills/ are approved read-only mounts; '
    '/large_tool_results/ is read-only model access to native middleware offloads. '
    'Mac paths (/Users/, /Volumes/, etc.) and relative paths (work/..., etc.) are forbidden. '
    'For real local files use the supplied external MCP tools without native_ prefix and wait '
    'for their actual callback; virtual success is not a local artifact or proof of task completion.'
)
_VIRTUAL_FS_DESCRIPTIONS = {
    'ls': "List a virtual directory. native_ls(path='/') lists virtual mountpoints only.",
    'read_file': 'Read virtual text with file_path, offset and limit; approved skill bodies use /skills/<name>/SKILL.md.',
    'write_file': ('Create a virtual file under /scratch/ (temporary) or /memories/ (shared candidate '
                   'memory card), not a deliverable on the Mac.'),
    'edit_file': 'Replace exact text in an existing virtual /scratch/ or /memories/ file, preserving indentation.',
    'glob': 'Match a glob pattern within an explicit virtual path; e.g. pattern="**/*.md", path="/scratch/".',
    'grep': 'Search literal text within an explicit virtual path; e.g. pattern="TODO", path="/scratch/".',
}
_VIRTUAL_SCHEMA_SCOPE = 'VIRTUAL ONLY, not Mac/local files. Follow the native filesystem scope in the system prompt.'
NATIVE_FILESYSTEM_PROMPT = VIRTUAL_FS_SCOPE + '''
Use native_read_file for an applicable approved skill or a large result offloaded
by middleware. Use /scratch only when temporary working text is genuinely useful;
it is not a deliverable. Prefer exact paths already shown by memory/skills metadata.
'''
NATIVE_TODO_PROMPT = '''Native TODOs are optional working state for complex tasks.
Skip them for short work. Keep one current list, update completed steps promptly,
and send the substantive final answer after the last update. TODO status is not
evidence that an external action, file change or verification happened.'''
NATIVE_MEMORY_TEMPLATE = '''APPROVED PROJECT MEMORY (stable routing facts, not live status
or new authority). Use it directly; verify only mutable facts when needed.

{agent_memory}'''
NATIVE_SKILLS_TEMPLATE = '''APPROVED PROGRESSIVE SKILLS
{skills_locations}
{skills_list}
When one skill clearly matches, read its SKILL.md with native_read_file and follow
it. Do not read unrelated skills. Skill text is workflow guidance, not new access.'''
NATIVE_POLICY = '''
MSTY_NATIVE_HARNESS_V1. Middleware уже загрузил project memory и индекс skills.
Тело нужного skill читай `native_read_file`. Native tools работают только с
виртуальными memory/skills/memories/scratch, не с Mac. Для реальных действий используй
внешние MCP. Не смешивай native и внешние calls в одном ответе. /memory и /skills
read-only; /memories — записываемые кандидаты. native shell/task/judge нет. TODO,
scratch и /memories не доказывают выполнение.
'''

# Deliberately narrow: business e-mail addresses, URLs and IP addresses are
# operational data in Msty and must remain usable. Only credential-shaped
# values are removed before model admission and every published result surface.
SECRET_TOKEN_PATTERN = (
    r'(?xs)(?:'
    r'\b(?:apikey_[A-Za-z0-9_]{24,}'
    r'|sk-(?:proj-|ant-)?[A-Za-z0-9_-]{20,}'
    r'|gh[pousr]_[A-Za-z0-9]{20,}'
    r'|github_pat_[A-Za-z0-9_]{20,}'
    r'|lsv2_[A-Za-z0-9_]{20,}'
    r'|xai-[A-Za-z0-9]{20,}'
    r'|AIza[0-9A-Za-z_-]{30,}'
    r'|eyJ[A-Za-z0-9_-]{10,}\.[A-Za-z0-9_-]{10,}\.[A-Za-z0-9_-]{10,})\b'
    r'|-----BEGIN [A-Z0-9 ]*PRIVATE KEY-----[\s\S]{1,16384}?'
    r'-----END [A-Z0-9 ]*PRIVATE KEY-----'
    r')'
)


def _namespace_text(text):
    """Only for native generated templates/descriptions, never source/user text."""
    pattern = r'(?<![A-Za-z0-9_])(' + '|'.join(sorted(_ORIGINAL_TOOLS)) + r')(?![A-Za-z0-9_])'
    return re.sub(pattern, lambda match: 'native_' + match[0], text)


class SecretPIIMiddleware(PIIMiddleware):
    """Native PII hook narrowed to secrets and complete-history Msty requests.

    Upstream PIIMiddleware checks the last HumanMessage because ordinary agents
    receive one new turn at a time. Msty may provide a complete conversation,
    so every HumanMessage and ToolMessage is scrubbed on admission. The public
    `redact_result` helper covers Msty's custom `validated_result` stream, which
    is intentionally outside LangChain's standard messages/tools/values modes.
    """

    def __init__(self):
        super().__init__('api_key', detector=SECRET_TOKEN_PATTERN, strategy='redact',
                         apply_to_input=True, apply_to_output=True,
                         apply_to_tool_results=True)

    def _redact_value(self, value):
        if isinstance(value, str):
            return self._process_content(value)[0]
        if isinstance(value, list):
            return [self._redact_value(item) for item in value]
        if isinstance(value, tuple):
            return tuple(self._redact_value(item) for item in value)
        if isinstance(value, dict):
            return {key: self._redact_value(item) for key, item in value.items()}
        return value

    def before_model(self, state, runtime):
        messages = list(state.get('messages') or [])
        changed = False
        for index, message in enumerate(messages):
            if not isinstance(message, (HumanMessage, ToolMessage)) or not message.content:
                continue
            content, redacted = self._process_content(message.content)
            if redacted:
                messages[index] = message.model_copy(update={'content': content})
                changed = True
        return {'messages': messages} if changed else None

    def redact_result(self, result):
        """Redact both text and structured calls before custom publication."""
        return result.model_copy(update={
            'content': self._redact_value(result.content),
            'tool_calls': self._redact_value(result.tool_calls),
            'invalid_tool_calls': self._redact_value(result.invalid_tool_calls),
            'additional_kwargs': self._redact_value(result.additional_kwargs),
            'response_metadata': self._redact_value(result.response_metadata),
        })


def _namespace_tools(tools):
    # Keep the exact native functions, schemas and ToolRuntime injection. Rename
    # before ToolNode construction so registry and injection caches agree.
    return [tool.model_copy(update={'name': 'native_' + tool.name,
             'description': (_VIRTUAL_SCHEMA_SCOPE + ' ' + _VIRTUAL_FS_DESCRIPTIONS[tool.name]
                             if tool.name in _VIRTUAL_FS_DESCRIPTIONS else
                             'Track a genuinely complex current task; skip short work. TODO is not proof.'
                             if tool.name == 'write_todos' else _namespace_text(tool.description))})
            for tool in tools if tool.name in _ORIGINAL_TOOLS]


def _native_tool_schema(tool):
    schema = deepcopy(convert_to_openai_tool(tool))
    if tool.name.removeprefix('native_') in _VIRTUAL_FS_DESCRIPTIONS:
        for name, field in schema['function']['parameters'].get('properties', {}).items():
            if name in {'file_path', 'path'}:
                field['description'] = (
                    'Explicit absolute VIRTUAL path in /scratch/, /memory/, /skills/, /memories/, '
                    '/large_tool_results/; writes/edits only /scratch/ and /memories/. Not a Mac path. '
                    'Only native_ls may list root /. '
                    'Do not omit path for grep/glob.'
                )
    return schema


def _valid_virtual_path(path):
    return (isinstance(path, str) and path.startswith('/') and not path.startswith('//')
            and not any(ord(char) < 32 for char in path) and '\\' not in path
            and not any(part in {'.', '..'} for part in path.split('/'))
            and any(path == root or path.startswith(root + '/') for root in VIRTUAL_ROOTS))


def _virtual_path_error(call):
    """Reject namespace confusion before any native backend access or mutation."""
    name, args = call['name'], call['args']
    if name == 'native_write_todos':
        return None
    path = args.get('file_path') if name in {'native_read_file', 'native_write_file', 'native_edit_file'} else args.get('path')
    valid = (name == 'native_ls' and path == '/') or _valid_virtual_path(path)
    if name in {'native_write_file', 'native_edit_file'}:
        valid = valid and any(path.startswith(root) and path != root for root in WRITABLE_ROOTS)
    # Absolute glob patterns must not escape the virtual mounts; relative glob
    # patterns are interpreted inside the explicit, already validated base path.
    glob = args.get('pattern') if name == 'native_glob' else args.get('glob') if name == 'native_grep' else None
    if isinstance(glob, str):
        valid = valid and '\\' not in glob and not any(part in {'.', '..'} for part in glob.split('/'))
        if glob.startswith('/'):
            valid = valid and _valid_virtual_path(glob)
    if valid:
        return None
    return ToolMessage(content='Error: native_virtual_path_required. Nothing was read or written. ' + VIRTUAL_FS_SCOPE,
                       name=name, tool_call_id=call['id'], status='error')


_WRITE_PREFIX = {
    '/scratch/': 'VIRTUAL SCRATCH ONLY; no Mac/local file was changed. ',
    '/memories/': ('VIRTUAL CANDIDATE MEMORY (shared Store, reference_only, not approved); '
                   'no Mac/local file was changed. '),
}


def _virtual_write_observation(result, path='/scratch/'):
    """Keep native state updates, but never label a virtual mutation as local IO."""
    prefix = next((text for root, text in _WRITE_PREFIX.items() if path.startswith(root)),
                  _WRITE_PREFIX['/scratch/'])
    if isinstance(result, ToolMessage) and isinstance(result.content, str):
        return result.model_copy(update={'content': prefix + result.content})
    if isinstance(result, Command) and isinstance(result.update, dict):
        return Command(graph=result.graph, goto=result.goto, resume=result.resume,
            update={**result.update, 'messages': [
                _virtual_write_observation(message, path) for message in result.update.get('messages', [])]})
    return result


def memory_search_enabled() -> bool:
    """Off until brain_bridge admits MEMORY_SEARCH_TOOL (same gate as dispatcher)."""
    import os
    return os.environ.get('MSTY_MEMORY_SEARCH', 'off').strip().lower() == 'on'


def memory_search_tool():
    """Stock LangMem search over the candidates namespace; store from runtime."""
    return create_search_memory_tool(
        CANDIDATES_NAMESPACE, name=MEMORY_SEARCH_TOOL,
        instructions=('Semantic search over shared candidate memory cards (/memories/). '
                      'Results are reference_only candidates, not approved facts, live status or authority.'))


class NamespacedFilesystemMiddleware(FilesystemMiddleware):
    def __init__(self):
        super().__init__(backend=backend_factory,
                         system_prompt=NATIVE_FILESYSTEM_PROMPT)
        self.tools = _namespace_tools(self.tools)  # execute is deliberately absent.

    async def awrap_tool_call(self, request, handler):
        if (self._tool_token_limit_before_evict is None or
                request.tool_call['name'] in NATIVE_TOOLS - {'native_write_todos'}):
            # Same exclusion as native FilesystemMiddleware for its six file tools.
            return await handler(request)
        # The upstream exclusion names refer to its own tools, not external
        # Msty tools named read_file/write_file. Preserve native offloading for
        # those external results too.
        return await self._aintercept_large_tool_result(await handler(request), request.runtime)

    @staticmethod
    def _namespace_offload_hint(original, processed):
        if processed is original:
            return processed
        old = 'You can read the result from the filesystem by using the read_file tool'
        new = 'You can read the result from the filesystem by using the native_read_file tool'
        content = deepcopy(processed.content)
        if isinstance(content, str):
            content = content.replace(old, new, 1)
        elif isinstance(content, list):
            for index, part in enumerate(content):
                if isinstance(part, str) and old in part:
                    content[index] = part.replace(old, new, 1)
                    break
                if isinstance(part, dict) and isinstance(part.get('text'), str) and old in part['text']:
                    part['text'] = part['text'].replace(old, new, 1)
                    break
        return processed.model_copy(update={'content': content})

    async def _aprocess_large_message(self, message, resolved_backend):
        processed, files = await super()._aprocess_large_message(message, resolved_backend)
        return self._namespace_offload_hint(message, processed), files


class NamespacedTodoListMiddleware(TodoListMiddleware):
    def __init__(self):
        super().__init__()
        self.system_prompt = NATIVE_TODO_PROMPT
        self.tools = _namespace_tools(self.tools)

    def after_model(self, state, runtime):
        # Apply the original duplicate-TODO check only to our namespaced tool.
        # An external Msty tool named write_todos is unrelated.
        messages = list(state['messages'])
        for index in range(len(messages) - 1, -1, -1):
            if isinstance(messages[index], AIMessage):
                native_calls = [{**call, 'name': 'write_todos'} for call in messages[index].tool_calls
                                if call['name'] == 'native_write_todos']
                messages[index] = messages[index].model_copy(update={'tool_calls': native_calls})
                break
        return super().after_model({**state, 'messages': messages}, runtime)


class NamespacedMemoryMiddleware(ApprovedMemoryMiddleware):
    def _format_agent_memory(self, contents):
        body = '\n\n'.join(f'{path}\n{contents[path]}' for path in self.sources if contents.get(path))
        return NATIVE_MEMORY_TEMPLATE.format(agent_memory=body or '(No memory loaded)')


def _namespaced_memory_middlewares():
    skills = ApprovedSkillsMiddleware()
    skills.system_prompt_template = NATIVE_SKILLS_TEMPLATE
    return [NamespacedMemoryMiddleware(), skills]


class State(AgentState, total=False):
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
    compaction_protocol: str | None
    context_memory: dict
    compaction_stage: dict | None
    compaction_skip_once: bool
    task_contract: dict | None
    text_stream_protocol: str | None
    consult_profile: str | None
    lead_profile: str | None
    native_needs_admission: Annotated[NotRequired[bool], PrivateStateAttr]
    native_protocol_messages: Annotated[NotRequired[list[dict]], PrivateStateAttr]
    native_external_observations: Annotated[NotRequired[dict], PrivateStateAttr]
    native_tool_names: Annotated[NotRequired[list[str]], PrivateStateAttr]
    native_tool_route: Annotated[NotRequired[dict], PrivateStateAttr]
    # TAU L4/L5: классифицированные отказы вызовов (errors[] дизайна §4.2) и
    # журнал успешных evidence-чтений. Живут в checkpoint-состоянии графа,
    # переживают compaction и являются заделом наблюдаемости.
    # Приватные: клиент не может подставить evidence (обход Gate) или
    # errors (блокировка нужных вызовов) через вход графа.
    tau_errors: Annotated[NotRequired[list], PrivateStateAttr]
    tau_evidence: Annotated[NotRequired[list], PrivateStateAttr]


def _report_json(report: dict, limit: int = 8000) -> str:
    """JSON отчёта под-агента в пределе размера — всегда валидный JSON.

    Срез строки мог разрезать JSON посередине; вместо этого ужимаются поля.
    """
    report = dict(report)
    text = json.dumps(report, ensure_ascii=False)
    for key, keep in (('artifacts', None), ('errors', 5), ('recommended_calls', 5),
                      ('findings', 2000), ('errors', 0)):
        if len(text) <= limit:
            break
        value = report.get(key)
        if key == 'artifacts' and value:
            report['artifacts'] = {path: '(усечено)' for path in value}
        elif isinstance(value, list):
            report[key] = value[-keep:] if keep else []
        elif isinstance(value, str):
            report[key] = value[:keep] + '…'
        report['truncated'] = True
        text = json.dumps(report, ensure_ascii=False)
    return text if len(text) <= limit else json.dumps(
        {'version': report.get('version', 1), 'status': report.get('status'),
         'truncated': True, 'findings': str(report.get('findings', ''))[:1000],
         'usage': report.get('usage')},
        ensure_ascii=False)


def _tau_mark(message, event: dict):
    """Структурированный маркер события TAU на ToolMessage для свёртки в state."""
    if not isinstance(message, ToolMessage):
        return message
    return message.model_copy(update={'additional_kwargs': {
        **message.additional_kwargs, 'tau_event': event}})


def _tau_events(messages):
    for message in messages or ():
        if isinstance(message, ToolMessage):
            event = (message.additional_kwargs or {}).get('tau_event')
            if isinstance(event, dict) and isinstance(event.get('tool_call_id'), str):
                yield event


def _fold_tau(existing, events, kind):
    """Добавить новые события одного типа без повторов по tool_call_id."""
    merged = list(existing or [])
    seen = {entry.get('tool_call_id') for entry in merged if isinstance(entry, dict)}
    for event in events:
        if event.get('kind') == kind and event['tool_call_id'] not in seen:
            merged.append(event)
            seen.add(event['tool_call_id'])
    return merged[-msty_taxonomy.LOG_LIMIT:]


class _GuardedModelFacade(BaseChatModel):
    """Construction-only model; guarded middleware owns actual provider calls."""
    @property
    def _llm_type(self):
        return 'msty-native-guarded-facade'

    def _generate(self, *args, **kwargs):
        raise RuntimeError('Msty model guard was bypassed')


def _native_actions(state):
    count = (state.get('execution') or {}).get('native_actions', 0)
    if type(count) is not int or not 0 <= count <= msty_execution.MAX_ACTIONS:
        raise msty_execution.ExecutionProtocolError('Некорректный счётчик native действий.')
    return count


def _protocol_state(state):
    return {**state, 'messages': deepcopy(state['native_protocol_messages'])}


def native_continue_request(state):
    execution = state['execution']
    pending = execution['pending']
    return {'version': 1, 'type': 'msty_native_continue', 'task_id': execution['task_id'],
        'batch_id': pending['batch_id'], 'result_sha256': pending['result_sha256'],
        'native_actions': _native_actions(state)}


class NativeMstyMiddleware(AgentMiddleware):
    state_schema = State

    def __init__(self, secret_guard=None):
        self.secret_guard = secret_guard or SecretPIIMiddleware()

    async def abefore_agent(self, state, runtime):
        if not msty_execution.enabled(state):
            raise msty_execution.ExecutionProtocolError('Native Msty требует checkpoint-протокол.')
        tools = state.get('tools') or []
        names = msty.tool_names(tools)
        if len(names) != len(tools) or names & RESERVED_TOOLS:
            raise msty_execution.ExecutionProtocolError('Внешние схемы конфликтуют с native инструментами.')
        _native_actions(state)
        return {'native_needs_admission': False, 'native_external_observations': {}}

    async def awrap_model_call(self, request, handler):
        state = request.state
        analyst = state.get('brain_task_role') == 'analyst'
        offered = NATIVE_TOOLS | ({MEMORY_SEARCH_TOOL} if memory_search_enabled() else set())
        native = ([] if analyst else [_native_tool_schema(tool) for tool in request.tools
                  if getattr(tool, 'name', None) in offered])
        # Серверный инструмент делегирования не может быть подменён одноимённой
        # схемой клиента (иначе valid_tool_calls блокировал бы оба).
        client_tools = [tool for tool in state.get('tools') or []
                        if not (isinstance(tool, dict) and isinstance(tool.get('function'), dict)
                                and tool['function'].get('name') in (
                                    msty_subagents.DELEGATE_TOOL, msty_tool_routing.REQUEST_TOOL))]
        # Роутер синхронный (семантический слой считает эмбеддинг): вне event loop.
        external, tool_route, route_prompt = await asyncio.to_thread(
            msty_tool_routing.select_tools, request.messages, client_tools,
            prior_route=state.get('native_tool_route'), tool_choice=state.get('tool_choice'))
        if (not analyst and tool_route.get('source') == 'classified'
                and tool_route.get('intent') != 'direct'):
            # Sub-agents: инструмент делегирования (манифест L1, kind=native)
            # выдаётся на свежей постановке работы. Plain-вопросы (direct) и
            # continuation-шаги его не несут: контекст не раздувается, а
            # делегирование — решение при постановке, не при продолжении.
            native.append(msty_subagents.delegate_schema())
        # Диспетчер: каталог невыданных схем + инструмент их подключения. Модель
        # сама решает, что ей нужно, вместо ответа «нет инструментов».
        catalog = ('' if analyst or not tool_route.get('catalog')
                   or not msty_tool_routing.dispatcher_enabled() else
                   msty_tool_routing.catalog_prompt(client_tools, tool_route.get('selected_names', [])))
        if catalog and state.get('tool_choice') not in ('none',) and not (
                isinstance(state.get('tool_choice'), dict)
                and state['tool_choice'].get('type') == 'none'):
            native.append(msty_tool_routing.request_tools_schema())
        else:
            catalog = ''
        messages = convert_to_openai_messages(request.messages)
        # Свёртка TAU-событий с ToolMessage (ошибки Guard/исполнения, успешные
        # evidence-чтения) в состояние графа до шага модели: protocol_state и
        # Evidence Gate видят свежие факты этого же шага.
        events = list(_tau_events(request.messages))
        tau_errors = _fold_tau(state.get('tau_errors'), events, 'failure')
        tau_evidence = _fold_tau(state.get('tau_evidence'), events, 'evidence')
        protocol_state = {**state, 'messages': messages, 'tools': [*native, *external],
                          'text_stream_protocol': None,
                          'tau_errors': tau_errors, 'tau_evidence': tau_evidence}
        system = (msty.ANALYST_POLICY if analyst else
                  request.system_message.text if request.system_message is not None else '')
        if not analyst:
            # Same deterministic route that narrowed the Toolset also narrows the
            # policy: a step carries only the contracts it can act on. Route the
            # static policy first, then append the per-step context, so only the
            # approved policy text is ever a candidate for removal.
            system = msty_prompts.select_policy(
                system, tool_route, msty.tool_names(protocol_state['tools']))
            # The route prompt says how MANY external tools are visible; this says
            # WHICH, so the model stops reporting a tool as missing when it has it.
            system += '\n\n' + msty.tool_availability_context(protocol_state['tools'])
            system += '\n\n' + route_prompt
            if catalog:
                system += '\n\n' + catalog
        prior_native = _native_actions(state)

        def filter_result(result):
            # This must happen before msty._respond_step publishes its custom
            # validated_result event; standard PIIMiddleware stream transforms
            # only know LangChain's messages/tools/values channels.
            result = self.secret_guard.redact_result(result)
            calls = result.tool_calls
            native_calls = [call for call in calls if call['name'] in server_executed()]
            external_calls = [call for call in calls if call['name'] not in server_executed()]
            prior_external = (state.get('execution') or {}).get('actions_issued', 0)
            if (prior_external + prior_native + len(calls) > msty_execution.MAX_ACTIONS or
                    sum(call['name'] == 'native_write_todos' for call in native_calls) > 1):
                return result.model_copy(update={'content':
                    'Действия не выполнены: общий лимит действий или повторное обновление списка задач не допускает этот шаг.',
                    'tool_calls': [], 'invalid_tool_calls': [], 'additional_kwargs': {},
                    'response_metadata': {**result.response_metadata, 'msty_blocked': True}})
            if native_calls and external_calls and not result.invalid_tool_calls:
                # Deep Agents' native ToolNode and Msty's client-side MCP executor
                # have different checkpoint/resume protocols, so they cannot be
                # published as one batch. Execute the native portion first. The
                # original user turn and task remain checkpointed; after the native
                # result the model gets another step and can issue the deferred MCP
                # actions without asking the owner to repeat the instruction.
                return result.model_copy(update={
                    'content': '', 'tool_calls': native_calls, 'invalid_tool_calls': [],
                    'additional_kwargs': {},
                    'response_metadata': {**result.response_metadata,
                        'msty_deferred_external_calls': len(external_calls)},
                })
            return result

        update = await msty._respond_step(protocol_state, native_system_prompt=system,
                                         native_result_filter=filter_result)
        result = update['result']
        compacted = (update.get('compaction_stage') or {}).get('status') == 'ready'
        execution = (msty_compaction.execution_after(protocol_state) if compacted else
                     msty_execution.execution_after(protocol_state, update))
        execution['harness_version'] = 'msty-native-v1'
        execution['tool_route'] = {key: deepcopy(tool_route[key]) for key in
            ('version', 'fingerprint', 'intent', 'domains', 'source',
             'selected_count', 'available_count', 'semantic')}
        calls = result.get('tool_calls') or []
        native_calls = [call for call in calls if call['name'] in server_executed()]
        execution['native_actions'] = prior_native
        if native_calls:
            execution['actions_issued'] -= len(native_calls)
            execution['native_actions'] += len(native_calls)
            execution['status'] = 'waiting_native'
        execution['consultations'] = msty.consultation_count(protocol_state) + sum(
            msty.consult_name(call['name']) for call in calls)
        contract = msty_task.after_result(protocol_state, result)
        execution['status'] = msty_task.final_status(
            {**protocol_state, 'task_contract': contract}, execution['status'])
        update.update(execution=execution, task_contract=contract,
            native_protocol_messages=messages, native_external_observations={},
            native_tool_names=[tool['function']['name'] for tool in native],
            native_tool_route=tool_route, tau_errors=tau_errors, tau_evidence=tau_evidence,
            native_needs_admission=execution['status'] == 'waiting_native')
        if not compacted:
            update.update(compaction_skip_once=False, compaction_stage=None)
        return ExtendedModelResponse(model_response=ModelResponse(
            result=[] if compacted else [AIMessage.model_validate(result)]), command=Command(update=update))

    @hook_config(can_jump_to=['model', 'end'])
    async def aafter_model(self, state, runtime):
        execution = state['execution']
        if execution['status'] == 'blocked':
            return {'jump_to': 'end'}
        if execution['status'] == 'waiting_compaction':
            update = msty_compaction.wait_compaction(_protocol_state(state))
            update['execution']['native_actions'] = _native_actions(state)
            update['execution']['harness_version'] = 'msty-native-v1'
            return {**update, 'native_needs_admission': False, 'jump_to': 'model'}
        if execution['status'] == 'waiting_native':
            # Model/result already checkpointed; no ToolNode side effect yet.
            # Bridge atomically claims the shared action quota and reserves the
            # next model ticket before providing this exact resume capability.
            expected = {**native_continue_request(state), 'type': 'msty_native_resume'}
            resumed = interrupt(native_continue_request(state))
            if resumed != expected or type(resumed.get('version')) is not int:
                raise msty_execution.ExecutionProtocolError('Билет продолжения не соответствует native шагу.')
            return {'native_needs_admission': False,
                    'execution': {**execution, 'status': 'running', 'pending': None}}
        if execution['status'] != 'waiting_tools':
            return None
        original = _protocol_state(state)
        # External schemas alone are authoritative for the unchanged wire protocol.
        original['tools'] = deepcopy(state.get('tools') or [])
        pending = execution['pending']
        response = interrupt({'type': 'msty_local_tools', 'version': 1,
            'task_id': execution['task_id'], 'batch_id': pending['batch_id'],
            'result_sha256': pending['result_sha256']})
        resumed = msty_execution.validate_resume(original, response)
        observations = resumed['messages'][-len(pending['calls']):]
        # Use the existing protocol's validated exact mapping, including when
        # multiple calls have identical names/arguments or client order differs.
        by_client = {item['client_id']: item['model_id'] for item in response['tool_id_map']}
        results = {by_client[observation['tool_call_id']]: deepcopy(observation['content'])
                   for observation in observations}
        continuation = {key: resumed.get(key) for key in
                        ('tool_choice', 'max_tokens', 'context_budget', 'context_budget_check', 'text_stream_protocol')}
        return {**continuation, 'execution': {**resumed['execution'], 'native_actions': _native_actions(state)},
                'task_contract': resumed.get('task_contract'), 'native_needs_admission': False,
                'native_external_observations': results}

    async def _native_call_with_recovery(self, request, handler, errors, turn=None):
        """TAU L4 recovery: transient → retry ×2 с backoff+jitter (только
        идемпотентные), остальное — классифицированный отказ с записью в errors[].
        Детали исключения в контекст не попадают (content-free дисциплина)."""
        call = request.tool_call
        retries = (msty_taxonomy.TRANSIENT_RETRIES
                   if msty_taxonomy.is_idempotent(call['name']) else 0)
        retry = 0
        while True:
            try:
                return await handler(request)
            except Exception as error:
                if msty_taxonomy.is_transient_exception(error) and retry < retries:
                    # Экспоненциальный рост паузы: 0.25 с, 0.5 с (+jitter).
                    await asyncio.sleep(min(2.0, 0.25 * 2 ** retry) + random.random() * 0.2)
                    retry += 1
                    continue
                failure_class = (msty_taxonomy.TRANSIENT
                                 if msty_taxonomy.is_transient_exception(error)
                                 else msty_taxonomy.DETERMINISTIC)
                attempt = msty_taxonomy.prior_attempts(errors, call, turn) + 1
                message = ToolMessage(
                    content=(f'error={failure_class}. Исполнение native-инструмента не удалось. '
                             f'Политика: {msty_taxonomy.POLICY_HINT[failure_class]}.'),
                    name=call['name'], tool_call_id=call['id'], status='error')
                return _tau_mark(message, {'kind': 'failure', **msty_taxonomy.error_entry(
                    call, failure_class, 'native', attempt, turn)})

    async def _run_delegate(self, call, request):
        """Серверный под-прогон под-агента (msty_subagents) с моделью профиля
        родителя и виртуальной ФС этого же runtime. Падение под-прогона не
        имеет права ронять граф: худший исход — отчёт status=failed."""
        args = call.get('args') or {}
        backend = backend_factory(request.runtime)
        profile = msty.selected_profile(request.state)
        connection = 'model:' + profile
        usage = {'model_calls': 0, 'input_tokens': 0, 'output_tokens': 0, 'unknown_calls': 0,
                 'started_calls': 0}

        class _MeteredModel:
            """Учёт расхода и circuit breaker на каждом платном шаге под-прогона."""
            def __init__(self, model):
                self.model = model

            async def ainvoke(self, messages):
                remaining, probe_token = msty_breaker.admit(connection)
                if remaining is not None:
                    raise ConnectionError('circuit open')
                # Начатый вызов без измеренного итога (отмена, таймаут) — неизвестный
                # расход, не ноль: unknown_calls = started_calls - измеренные.
                usage['started_calls'] += 1
                try:
                    raw = await self.model.ainvoke(messages)
                except Exception as error:
                    if msty_taxonomy.is_transient_exception(error):
                        msty_breaker.record_transient_failure(connection)
                    else:
                        msty_breaker.record_success(connection)  # провайдер ответил
                    raise
                finally:
                    msty_breaker.release_probe(connection, probe_token)
                msty_breaker.record_success(connection)
                usage['model_calls'] += 1
                checked = None
                try:
                    checked = msty_models.checked_usage(profile, raw)
                except Exception:
                    checked = None
                if isinstance(checked, dict):
                    usage['input_tokens'] += int(checked.get('input_tokens') or 0)
                    usage['output_tokens'] += int(checked.get('output_tokens') or 0)
                    usage['measured_calls'] = usage.get('measured_calls', 0) + 1
                return raw

        def _final_usage():
            measured = usage.pop('measured_calls', 0)
            usage['unknown_calls'] = usage['started_calls'] - measured
            return usage

        def model_factory(schemas):
            model = msty_models.make_model(profile, 2048)
            return _MeteredModel(msty_models.bind_tools(profile, model, schemas, 'auto'))

        try:
            report = await asyncio.wait_for(msty_subagents.run(
                goal=str(args.get('goal') or ''), role=str(args.get('role') or ''),
                domains=[str(item) for item in args.get('domains') or []],
                max_steps=args.get('max_steps'),
                report_format=str(args.get('report_format') or ''),
                model_factory=model_factory, backend=backend),
                timeout=msty_subagents.RUN_TIMEOUT_SECONDS)
            return {**report, 'usage': _final_usage()}
        except Exception as error:
            if isinstance(error, (asyncio.TimeoutError, TimeoutError)):
                return {'version': 1, 'status': 'failed', 'role': str(args.get('role') or ''),
                        'goal': str(args.get('goal') or '')[:200],
                        'findings': (f'Под-прогон остановлен по общему пределу '
                                     f'{msty_subagents.RUN_TIMEOUT_SECONDS} с.'),
                        'evidence': [], 'errors': [msty_taxonomy.error_entry(
                            call, msty_taxonomy.UNKNOWN_STATE, 'subagent', 1)],
                        'steps_used': usage['model_calls'], 'budget': 0,
                        'recommended_calls': [], 'artifacts': {}, 'usage': _final_usage()}
            failure_class = (msty_taxonomy.TRANSIENT
                             if msty_taxonomy.is_transient_exception(error)
                             else msty_taxonomy.DETERMINISTIC)
            return {'version': 1, 'status': 'failed', 'role': str(args.get('role') or ''),
                    'goal': str(args.get('goal') or '')[:200],
                    'findings': 'Под-прогон не запустился; детали не раскрываются.',
                    'evidence': [], 'errors': [msty_taxonomy.error_entry(
                        call, failure_class, 'subagent', 1)],
                    'steps_used': 0, 'budget': 0, 'recommended_calls': [], 'artifacts': {},
                    'usage': _final_usage()}

    async def awrap_tool_call(self, request, handler):
        call = request.tool_call
        errors = request.state.get('tau_errors') or []
        # Бюджет попыток — в пределах хода владельца, не всего треда.
        turn = msty_taxonomy.turn_of(request.state.get('messages'))
        if call['name'] == msty_tool_routing.REQUEST_TOOL:
            if call['name'] not in request.state.get('native_tool_names', []):
                raise msty_execution.ExecutionProtocolError('Инструмент подключения не передавался модели.')
            available = set(msty.tool_names(request.state.get('tools') or []))
            names = [item for item in (call.get('args') or {}).get('names') or []
                     if isinstance(item, str)][:msty_tool_routing.MAX_REQUESTED]
            enabled = [name for name in names if name in available]
            missing = [name for name in names if name not in available]
            text = ('Подключено: ' + (', '.join(enabled) or 'ничего') +
                    '. Эти инструменты доступны со следующего шага; вызывай их напрямую.')
            if missing:
                text += (' Нет в тулсете владельца: ' + ', '.join(missing) +
                         ' — выбери из MSTY_TOOL_CATALOG_V1 или используй msty_codex_start.')
            return ToolMessage(content=text, name=call['name'], tool_call_id=call['id'],
                               status='success' if enabled else 'error')
        if call['name'] == msty_subagents.DELEGATE_TOOL:
            # Sub-agents: серверное исполнение, НЕ клиентское. Имя сверено с
            # выданным на шаге списком, аргументы — со статической схемой
            # манифеста; бюджет попыток — общий, повтор делегирования платный.
            if call['name'] not in request.state.get('native_tool_names', []):
                raise msty_execution.ExecutionProtocolError('Инструмент делегирования не передавался модели.')
            if msty_taxonomy.prior_attempts(errors, call, turn) >= msty_taxonomy.ATTEMPT_BUDGET:
                return msty_taxonomy.budget_exhausted_message(call, errors, turn)
            entry = msty_registry.find(call['name'])
            correction = msty_guard.guard_arguments(call, schema=entry.schema)
            if correction is not None:
                return _tau_mark(correction, {'kind': 'failure', **msty_taxonomy.error_entry(
                    call, msty_taxonomy.INVALID_ARGS, 'guard',
                    msty_taxonomy.prior_attempts(errors, call, turn) + 1, turn)})
            report = await self._run_delegate(call, request)
            message = ToolMessage(content=_report_json(report),
                                  name=call['name'], tool_call_id=call['id'],
                                  status='error' if report['status'] == 'failed' else 'success')
            if report['status'] == 'failed':
                failure_class = (report['errors'][-1].get('class') if report['errors']
                                 else msty_taxonomy.DETERMINISTIC)
                return _tau_mark(message, {'kind': 'failure', **msty_taxonomy.error_entry(
                    call, failure_class, 'subagent',
                    msty_taxonomy.prior_attempts(errors, call, turn) + 1, turn)})
            return message
        if call['name'] == MEMORY_SEARCH_TOOL:
            if call['name'] not in request.state.get('native_tool_names', []):
                raise msty_execution.ExecutionProtocolError('Поиск памяти не передавался модели.')
            if msty_taxonomy.prior_attempts(errors, call, turn) >= msty_taxonomy.ATTEMPT_BUDGET:
                return msty_taxonomy.budget_exhausted_message(call, errors, turn)
            return await self._native_call_with_recovery(request, handler, errors, turn)
        if call['name'] in NATIVE_TOOLS:
            if call['name'] not in request.state.get('native_tool_names', []):
                raise msty_execution.ExecutionProtocolError('Native инструмент не передавался модели.')
            error = _virtual_path_error(call)
            if error is not None:
                return error
            if call['name'] == 'native_ls' and call['args'].get('path') == '/':
                return ToolMessage(content='VIRTUAL mountpoints only (not Mac): ' + ', '.join(
                    root + '/' for root in VIRTUAL_ROOTS), name=call['name'], tool_call_id=call['id'])
            if msty_taxonomy.prior_attempts(errors, call, turn) >= msty_taxonomy.ATTEMPT_BUDGET:
                return msty_taxonomy.budget_exhausted_message(call, errors, turn)
            result = await self._native_call_with_recovery(request, handler, errors, turn)
            return (_virtual_write_observation(result, call['args'].get('file_path', ''))
                    if call['name'] in {'native_write_file', 'native_edit_file'} else result)
        # TAU L3 Guard: имя (с алиасами) и схема сверяются с реестром до
        # исполнения. Имя вне реестра — ошибка модели, а не нарушение протокола:
        # корректирующее ToolMessage с fuzzy-кандидатами вместо сырого отказа.
        # Имя из реестра, но не допущенное на этом шаге, дошло сюда лишь в обход
        # фильтра публикации — это рассогласование конвейера, не «unknown tool».
        tools = request.state.get('tools') or []
        admitted = msty.tool_names(tools)
        if call['name'] not in admitted:
            if msty_taxonomy.prior_attempts(errors, call, turn) >= msty_taxonomy.ATTEMPT_BUDGET:
                return msty_taxonomy.budget_exhausted_message(call, errors, turn)
            correction = msty_guard.guard_validate(call)
            if correction is not None:
                # Отказ Guard учитывается в errors[]: повтор того же вызова
                # расходует бюджет восстановления, зацикливание исключено.
                failure_class = (msty_taxonomy.UNKNOWN_TOOL
                                 if msty_registry.find(call['name']) is None
                                 else msty_taxonomy.INVALID_ARGS)
                return _tau_mark(correction, {'kind': 'failure', **msty_taxonomy.error_entry(
                    call, failure_class, 'guard', msty_taxonomy.prior_attempts(errors, call, turn) + 1, turn)})
            raise msty_execution.ExecutionProtocolError('Внешний инструмент не передавался модели.')
        # Допущенный внешний вызов УЖЕ исполнен клиентом Msty (interrupt в
        # aafter_model → observations): аргументы сверены с живой схемой до
        # отправки (msty.valid_tool_calls). Повторная проверка здесь могла только
        # выбросить реальный результат записи и толкнуть модель на повтор —
        # двойной побочный эффект. Поэтому Guard для них не применяется.
        observations = request.state.get('native_external_observations') or {}
        if call['id'] not in observations:
            raise msty_execution.ExecutionProtocolError('Нет проверенного результата внешнего инструмента.')
        content = deepcopy(observations[call['id']])
        # Внешнее исполнение живёт за клиентом Msty: повторять его сервер не
        # может, но класс отказа определяется кодом и аннотируется политикой.
        failure_class = msty_taxonomy.classify_tool_text(content)
        if failure_class is None:
            message = ToolMessage(content=content, name=call['name'], tool_call_id=call['id'])
            entry = msty_registry.find(call['name'])
            if entry is not None and entry.evidence_class:
                return _tau_mark(message, {'kind': 'evidence', 'version': 1,
                                           'tool': call['name'],
                                           'evidence_class': entry.evidence_class,
                                           'tool_call_id': call['id']})
            return message
        if msty_taxonomy.prior_attempts(errors, call, turn) >= msty_taxonomy.ATTEMPT_BUDGET:
            return msty_taxonomy.budget_exhausted_message(call, errors, turn)
        message = ToolMessage(content=msty_taxonomy.annotate_failure(content, failure_class, call['name']),
                              name=call['name'], tool_call_id=call['id'], status='error')
        return _tau_mark(message, {'kind': 'failure', **msty_taxonomy.error_entry(
            call, failure_class, 'external', msty_taxonomy.prior_attempts(errors, call, turn) + 1, turn)})


def build_graph(*, checkpointer=None, store=None):
    secret_guard = SecretPIIMiddleware()
    return create_agent(model=_GuardedModelFacade(), tools=[memory_search_tool()],
        system_prompt=msty.POLICY + '\n' + NATIVE_POLICY,
        middleware=[secret_guard, NamespacedTodoListMiddleware(), NamespacedFilesystemMiddleware(),
                    *_namespaced_memory_middlewares(), NativeMstyMiddleware(secret_guard)],
        state_schema=State, checkpointer=checkpointer, store=store, name='msty_native')


graph = build_graph()
