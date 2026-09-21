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

from deepagents.middleware.filesystem import FilesystemMiddleware, FILESYSTEM_SYSTEM_PROMPT
from deepagents.middleware.memory import MEMORY_SYSTEM_PROMPT
from langchain.agents import create_agent
from langchain.agents.middleware import AgentMiddleware, AgentState, TodoListMiddleware
from langchain.agents.middleware.types import (
    ExtendedModelResponse, ModelResponse, PrivateStateAttr, hook_config,
)
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, ToolMessage, convert_to_openai_messages
from langchain_core.utils.function_calling import convert_to_openai_tool
from langgraph.types import Command, interrupt

from . import msty, msty_compaction, msty_execution, msty_task
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
NATIVE_POLICY = '''
MSTY_NATIVE_HARNESS_V1: память и индекс навыков загружает штатный middleware.
Для тела относящегося навыка используй native_read_file по показанному пути.
Native файловые tools работают с виртуальными scratch/approved memory/skills,
а не с диском Mac. Для действий на Mac нужны переданные внешние инструменты.
Не смешивай native и внешние инструменты в одном ответе. Сначала заверши один
необходимый шаг. Один native_write_todos на ответ; TODO не доказывает выполнение задачи.
Память/skills защищены от записи моделью. Нет native task, shell или скрытого judge.
''' + '\n' + VIRTUAL_FS_SCOPE


def _namespace_text(text):
    """Only for native generated templates/descriptions, never source/user text."""
    pattern = r'(?<![A-Za-z0-9_])(' + '|'.join(sorted(_ORIGINAL_TOOLS)) + r')(?![A-Za-z0-9_])'
    return re.sub(pattern, lambda match: 'native_' + match[0], text)


def _namespace_tools(tools):
    # Keep the exact native functions, schemas and ToolRuntime injection. Rename
    # before ToolNode construction so registry and injection caches agree.
    return [tool.model_copy(update={'name': 'native_' + tool.name,
             'description': (VIRTUAL_FS_SCOPE + '\n' + _VIRTUAL_FS_DESCRIPTIONS[tool.name]
                             if tool.name in _VIRTUAL_FS_DESCRIPTIONS else _namespace_text(tool.description))})
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
                         system_prompt=VIRTUAL_FS_SCOPE + '\n' + _namespace_text(FILESYSTEM_SYSTEM_PROMPT))
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
        self.system_prompt = _namespace_text(self.system_prompt)
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
        return _namespace_text(MEMORY_SYSTEM_PROMPT).format(agent_memory=body or '(No memory loaded)')


def _namespaced_memory_middlewares():
    skills = ApprovedSkillsMiddleware()
    skills.system_prompt_template = _namespace_text(skills.system_prompt_template)
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
    native_needs_admission: Annotated[NotRequired[bool], PrivateStateAttr]
    native_protocol_messages: Annotated[NotRequired[list[dict]], PrivateStateAttr]
    native_external_observations: Annotated[NotRequired[dict], PrivateStateAttr]
    native_tool_names: Annotated[NotRequired[list[str]], PrivateStateAttr]


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


def _blocked_update(state, error):
    update = msty.rejected_context_budget(str(error))
    return {**update, 'execution': {**(state.get('execution') or {}), 'status': 'blocked',
                                    'pending': None}, 'jump_to': 'end'}


def native_continue_request(state):
    execution = state['execution']
    pending = execution['pending']
    return {'version': 1, 'type': 'msty_native_continue', 'task_id': execution['task_id'],
        'batch_id': pending['batch_id'], 'result_sha256': pending['result_sha256'],
        'native_actions': _native_actions(state)}


class NativeMstyMiddleware(AgentMiddleware):
    state_schema = State

    @hook_config(can_jump_to=['end'])
    async def abefore_agent(self, state, runtime):
        try:
            if not msty_execution.enabled(state):
                raise msty_execution.ExecutionProtocolError('Native Msty требует checkpoint-протокол.')
            tools = state.get('tools') or []
            names = msty.tool_names(tools)
            parsed_names = []
            for tool in tools:
                if not isinstance(tool, dict):
                    continue
                function = tool.get('function')
                if tool.get('type') == 'function' and isinstance(function, dict):
                    name = function.get('name')
                    if isinstance(name, str) and name:
                        parsed_names.append(name)
                elif tool.get('type') != 'function' and isinstance(tool.get('name'), str) and tool.get('name') and (
                        'inputSchema' in tool or 'input_schema' in tool):
                    parsed_names.append(tool['name'])
            if names & RESERVED_TOOLS:
                raise msty_execution.ExecutionProtocolError('Внешняя схема использует зарезервированное имя.')
            if len(parsed_names) != len(names):
                raise msty_execution.ExecutionProtocolError('Внешние схемы содержат повторяющиеся имена.')
            if len(parsed_names) != len(tools):
                raise msty_execution.ExecutionProtocolError('Имя одной или нескольких внешних схем не распознано.')
            _native_actions(state)
            return {'native_needs_admission': False, 'native_external_observations': {}}
        except msty_execution.ExecutionProtocolError as error:
            return _blocked_update(state, error)

    async def awrap_model_call(self, request, handler):
        state = request.state
        analyst = state.get('brain_task_role') == 'analyst'
        native = ([] if analyst else [_native_tool_schema(tool) for tool in request.tools
                  if getattr(tool, 'name', None) in NATIVE_TOOLS])
        external = deepcopy(state.get('tools') or [])
        messages = convert_to_openai_messages(request.messages)
        protocol_state = {**state, 'messages': messages, 'tools': [*native, *external],
                          'text_stream_protocol': None}
        system = (msty.ANALYST_POLICY if analyst else
                  request.system_message.text if request.system_message is not None else '')
        prior_native = _native_actions(state)

        def filter_result(result):
            calls = result.tool_calls
            native_calls = [call for call in calls if call['name'] in NATIVE_TOOLS]
            external_calls = [call for call in calls if call['name'] not in NATIVE_TOOLS]
            prior_external = (state.get('execution') or {}).get('actions_issued', 0)
            if (native_calls and external_calls or
                    prior_external + prior_native + len(calls) > msty_execution.MAX_ACTIONS or
                    sum(call['name'] == 'native_write_todos' for call in native_calls) > 1):
                return result.model_copy(update={'content':
                    'Действия не выполнены: смешанная batch или общий лимит действий не допускает этот шаг.',
                    'tool_calls': [], 'invalid_tool_calls': [], 'additional_kwargs': {},
                    'response_metadata': {**result.response_metadata, 'msty_blocked': True}})
            return result

        update = await msty._respond_step(protocol_state, native_system_prompt=system,
                                         native_result_filter=filter_result)
        result = update['result']
        compacted = (update.get('compaction_stage') or {}).get('status') == 'ready'
        execution = (msty_compaction.execution_after(protocol_state) if compacted else
                     msty_execution.execution_after(protocol_state, update))
        execution['harness_version'] = 'msty-native-v1'
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
            native_needs_admission=execution['status'] == 'waiting_native')
        if not compacted:
            update.update(compaction_skip_once=False, compaction_stage=None)
        return ExtendedModelResponse(model_response=ModelResponse(
            result=[] if compacted else [AIMessage.model_validate(result)]), command=Command(update=update))

    @hook_config(can_jump_to=['model', 'end'])
    async def aafter_model(self, state, runtime):
        try:
            return await self._aafter_model(state, runtime)
        except msty_execution.ExecutionProtocolError as error:
            return _blocked_update(state, error)

    async def _aafter_model(self, state, runtime):
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
    return create_agent(model=_GuardedModelFacade(),
        system_prompt=msty.POLICY + '\n' + NATIVE_POLICY,
        middleware=[NamespacedTodoListMiddleware(), NamespacedFilesystemMiddleware(),
                    *_namespaced_memory_middlewares(), NativeMstyMiddleware()],
        state_schema=State, checkpointer=checkpointer, store=store, name='msty_native')


graph = build_graph()
