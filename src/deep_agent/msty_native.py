"""Native LangChain/Deep Agents loop with the existing metered Msty boundary.

Exactly one guarded provider step per gateway ticket. Native tools wait for
gateway admission before ToolNode, then continue to the next metered model step.
External tool batches retain msty-local-tools-v1 and its complete validation.
Native summary/subagents are intentionally not installed: their hidden model
calls need separate tickets. Existing explicitly metered compaction is retained.
"""
from copy import deepcopy
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

from . import msty, msty_compaction, msty_execution, msty_prompts, msty_task, msty_tool_routing
from .msty_native_memory import backend_factory, ApprovedMemoryMiddleware, ApprovedSkillsMiddleware

_ORIGINAL_TOOLS = frozenset({'ls', 'read_file', 'write_file', 'edit_file', 'glob', 'grep', 'write_todos'})
NATIVE_TOOLS = frozenset('native_' + name for name in _ORIGINAL_TOOLS)
RESERVED_TOOLS = NATIVE_TOOLS | {'native_execute', 'native_task', 'native_compact_conversation'}
VIRTUAL_ROOTS = ('/scratch', '/memory', '/skills', '/large_tool_results')
VIRTUAL_FS_SCOPE = (
    'VIRTUAL ONLY: this is checkpoint-backed agent storage, NEVER the Mac/local filesystem. '
    'Read only under /scratch/, /memory/, /skills/, /large_tool_results/. '
    'Create/edit only under /scratch/. /memory/ and /skills/ are approved read-only mounts; '
    '/large_tool_results/ is read-only model access to native middleware offloads. '
    'Mac paths (/Users/, /Volumes/, etc.) and relative paths (work/..., etc.) are forbidden. '
    'For real local files use the supplied external MCP tools without native_ prefix and wait '
    'for their actual callback; virtual success is not a local artifact or proof of task completion.'
)
_VIRTUAL_FS_DESCRIPTIONS = {
    'ls': "List a virtual directory. native_ls(path='/') lists virtual mountpoints only.",
    'read_file': 'Read virtual text with file_path, offset and limit; approved skill bodies use /skills/<name>/SKILL.md.',
    'write_file': 'Create a virtual scratch file under /scratch/, not a deliverable on the Mac.',
    'edit_file': 'Replace exact text in an existing virtual /scratch/ file, preserving indentation.',
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
виртуальными memory/skills/scratch, не с Mac. Для реальных действий используй
внешние MCP. Не смешивай native и внешние calls в одном ответе. Memory/skills
read-only; native shell/task/judge нет. TODO и scratch не доказывают выполнение.
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
                    'Explicit absolute VIRTUAL path in /scratch/, /memory/, /skills/, /large_tool_results/; '
                    'writes/edits only /scratch/. Not a Mac path. Only native_ls may list root /. '
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
        valid = valid and path.startswith('/scratch/') and path != '/scratch/'
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


def _virtual_write_observation(result):
    """Keep native state updates, but never label a scratch mutation as local IO."""
    if isinstance(result, ToolMessage) and isinstance(result.content, str):
        return result.model_copy(update={'content':
            'VIRTUAL SCRATCH ONLY; no Mac/local file was changed. ' + result.content})
    if isinstance(result, Command) and isinstance(result.update, dict):
        return Command(graph=result.graph, goto=result.goto, resume=result.resume,
            update={**result.update, 'messages': [
                _virtual_write_observation(message) for message in result.update.get('messages', [])]})
    return result


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
        native = ([] if analyst else [_native_tool_schema(tool) for tool in request.tools
                  if getattr(tool, 'name', None) in NATIVE_TOOLS])
        external, tool_route, route_prompt = msty_tool_routing.select_tools(
            request.messages, state.get('tools') or [],
            prior_route=state.get('native_tool_route'), tool_choice=state.get('tool_choice'))
        messages = convert_to_openai_messages(request.messages)
        protocol_state = {**state, 'messages': messages, 'tools': [*native, *external],
                          'text_stream_protocol': None}
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
        prior_native = _native_actions(state)

        def filter_result(result):
            # This must happen before msty._respond_step publishes its custom
            # validated_result event; standard PIIMiddleware stream transforms
            # only know LangChain's messages/tools/values channels.
            result = self.secret_guard.redact_result(result)
            calls = result.tool_calls
            native_calls = [call for call in calls if call['name'] in NATIVE_TOOLS]
            external_calls = [call for call in calls if call['name'] not in NATIVE_TOOLS]
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
             'selected_count', 'available_count')}
        calls = result.get('tool_calls') or []
        native_calls = [call for call in calls if call['name'] in NATIVE_TOOLS]
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
            native_tool_route=tool_route,
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

    async def awrap_tool_call(self, request, handler):
        call = request.tool_call
        if call['name'] in NATIVE_TOOLS:
            if call['name'] not in request.state.get('native_tool_names', []):
                raise msty_execution.ExecutionProtocolError('Native инструмент не передавался модели.')
            error = _virtual_path_error(call)
            if error is not None:
                return error
            if call['name'] == 'native_ls' and call['args'].get('path') == '/':
                return ToolMessage(content='VIRTUAL mountpoints only (not Mac): ' + ', '.join(
                    root + '/' for root in VIRTUAL_ROOTS), name=call['name'], tool_call_id=call['id'])
            result = await handler(request)
            return (_virtual_write_observation(result) if call['name'] in {
                'native_write_file', 'native_edit_file'} else result)
        if call['name'] not in msty.tool_names(request.state.get('tools') or []):
            raise msty_execution.ExecutionProtocolError('Внешний инструмент не передавался модели.')
        observations = request.state.get('native_external_observations') or {}
        if call['id'] not in observations:
            raise msty_execution.ExecutionProtocolError('Нет проверенного результата внешнего инструмента.')
        return ToolMessage(content=deepcopy(observations[call['id']]), name=call['name'], tool_call_id=call['id'])


def build_graph(*, checkpointer=None, store=None):
    secret_guard = SecretPIIMiddleware()
    return create_agent(model=_GuardedModelFacade(),
        system_prompt=msty.POLICY + '\n' + NATIVE_POLICY,
        middleware=[secret_guard, NamespacedTodoListMiddleware(), NamespacedFilesystemMiddleware(),
                    *_namespaced_memory_middlewares(), NativeMstyMiddleware(secret_guard)],
        state_schema=State, checkpointer=checkpointer, store=store, name='msty_native')


graph = build_graph()
