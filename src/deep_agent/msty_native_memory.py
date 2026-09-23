"""Approved project memory and progressive skills on native Deep Agents backends.

LangGraph's API-key-protected Store administration is the approval boundary, not
an agent tool. The model can read these routes but cannot create approved data.
The Store projection is not a replacement for the canonical journal or Engram.
No disk, HTTP client, embeddings or extra model invocation is used here.

Candidate memory (owner decision 2026-09-23): /memories/ is the official Deep
Agents writable-memory route (CompositeBackend -> native StoreBackend) on a
separate Store namespace. Entries there are reference_only candidates, never
approved memory; /memory/ and /skills/ stay read-only. Semantic indexing is the
Agent Server Store index configured in langgraph.json, not code here.
https://docs.langchain.com/oss/python/deepagents/memory
"""
import asyncio
import fnmatch
import hashlib
import re
from typing import Any

from deepagents.backends import CompositeBackend, StateBackend, StoreBackend
from deepagents.backends.protocol import EditResult, FileDownloadResponse, FileUploadResponse, WriteResult
from deepagents.backends.utils import create_file_data, format_read_response
from deepagents.middleware.memory import MemoryMiddleware
from deepagents.middleware.skills import SkillsMiddleware

from . import msty_memory

POLICY_SHA256 = 'ddbf94a94e00683b3a2d66bf7b8bd456aa9f5ab8d48ba8ac0355da0c3fadd938'
NAMESPACE = ('sanare-owner', 'msty', 'approved-native-files-v1')
APPROVAL_VERSION = 1
PACKAGED_REVISION = 'msty-native-memory-20260920-v1'
SOURCE = 'project-governance/changes/20260920-msty-native-migration.json'
TIMEOUT_SECONDS = 1.0
MEMORY_PATHS = ['/memory/PROJECT.md']
SKILLS_PATHS = ['/skills/']
MAX_MEMORY_BYTES = 16384
MAX_SKILL_BYTES = 8192
CANDIDATES_ROUTE = '/memories/'
CANDIDATES_NAMESPACE = ('sanare-owner', 'knowledge', 'candidates')

SKILLS = {
    '/site-editing/SKILL.md': '''---
name: site-editing
description: Edit and verify the registered main Sanare site in an isolated job; recover from check errors without claiming success.
---
# Изолированная правка и проверка сайта
Применять к разрешённой правке app.sanaredev.com, не к ChatGPT Sites и не к
платёжному репозиторию. Знание пути уже в памяти; не сканируй все диски.
1. Используй реально переданный msty_site_prepare; дождись ready через status.
2. Прочитай действующие AGENTS.md, SITE_PASSPORT.md и относящуюся архитектуру
   подготовленной копии. Память путей не отменяет локальные правила перед правкой.
3. Читай только относящиеся файлы. Сохраняй историю CHANGELOG и посторонний код.
   Для полной CAS-записи необходим актуальный sha256 прочитанного файла.
4. Выполни typecheck, unit и build доступными presets. Для unit передавай
   существующие конкретные тестовые файлы, не выдуманный каталог tests.
5. failed и not_run различаются: исправь аргументы/код/допустимую среду и повтори
   затронутую проверку после изменения условий. Не ослабляй проверки ради PASS.
6. Дождись проверок, сверяй source manifest; started/готовый текст не результат.
7. Публикация только в пределах исходного допуска владельца через штатный release
   и зелёные актуальные проверки/CI. Не обходи бюджет, stop и неизвестный исход.
''',
    '/brain-maintenance/SKILL.md': '''---
name: brain-maintenance
description: Diagnose and repair the existing Msty Brain route using current evidence, scoped changes and regression checks.
---
# Обслуживание действующего Brain
Не переносить настройки «Правой руки», старого Supervisor и OpenClaw в Msty.
1. Уточни изменяемый компонент по текущей памяти и реальным схемам, не выполняй
   стартовый обход всех карт/каталогов. Нужен факт — выбери один узкий инструмент.
2. Проследи конкретный отказ: вход владельца, маршрут, вызов, наблюдение, итог.
   API 200, наличие модели и файл навыка не доказывают выполнение задачи.
3. Внеси минимальную разрешённую правку в существующий компонент, сохрани чужую
   работу. Конфигурацию Msty меняют штатным GUI, не обфусцированным файлом.
4. Generic repair/provisioner отключены; старые инструкции не включают их снова.
5. Добавь воспроизведение и регрессию; offline, внедрение и native UI-приёмка
   являются разными доказательствами. Не перезапускай поверх активной работы.
6. Запиши версию и результат в единый project-governance/changes. Доступы,
   платный допуск, stop и публикация не расширяются от успешного теста.
''',
    '/evidence-learning/SKILL.md': '''---
name: evidence-learning
description: Preserve a verified result or failure as a source-linked lesson and reuse only relevant evidence in later tasks.
---
# Проверяемые результаты и уроки
Для существенного файлового результата используй доступные msty_task_plan и
msty_task_verify с проверяемыми требованиями; полнота требований тоже проверяется.
1. Результат → реальная проверка → первичная квитанция → короткий урок → повторное
   использование по теме → измерение изменения качества на сопоставимой задаче.
2. Различай passed, failed, not_run, unknown; обоснование не заменяет тест.
3. Укажи относящиеся артефакты, SHA, фактическую пользу либо «не измерена», остаток.
4. Проверь learning.state и engram.state отдельно. Используй существующую память,
   не создавай второй журнал, не отправляй туда секреты и сырые переписки.
5. Полученный урок reference_only/candidate, не новые полномочия и не активные
   правила. Не считай извлечение урока доказательством его применения.
6. После ошибки продолжай доступное разрешённое исправление; неизменный неуспех
   не повторяй вслепую. Обучение весов и автоматическая публикация не включаются.
''',
}


def approved_entry(content: str, *, revision: str = PACKAGED_REVISION) -> dict:
    """Build a Store-admin payload, never exposed as a model tool or approval API.

    Callers with Store write access are responsible for canonical source review.
    A matching hash proves integrity, not that arbitrary content is safe or true.
    """
    data = create_file_data(content)
    return {**data, 'approval': {'version': APPROVAL_VERSION, 'status': 'approved',
        'policy_sha256': POLICY_SHA256, 'source': SOURCE, 'revision': revision,
        'sha256': hashlib.sha256(content.encode()).hexdigest()}}


def _validate(value: Any, limit: int) -> dict | None:
    if not isinstance(value, dict) or set(value) != {'content', 'created_at', 'modified_at', 'approval'}:
        return None
    lines, approval = value['content'], value['approval']
    if not isinstance(lines, list) or len(lines) > 512 or any(not isinstance(line, str) for line in lines):
        return None
    if any(not isinstance(value[name], str) or len(value[name]) > 64 for name in ('created_at', 'modified_at')):
        return None
    try:
        if sum(len(line.encode()) for line in lines) + max(0, len(lines) - 1) > limit:
            return None
    except UnicodeEncodeError:
        return None
    content = '\n'.join(lines)
    if not content.strip() or '\x00' in content:
        return None
    if not isinstance(approval, dict) or set(approval) != {
            'version', 'status', 'policy_sha256', 'source', 'revision', 'sha256'}:
        return None
    if (type(approval['version']) is not int or approval['version'] != APPROVAL_VERSION
            or approval['status'] != 'approved' or approval['policy_sha256'] != POLICY_SHA256
            or approval['source'] != SOURCE or not isinstance(approval['revision'], str)
            or re.fullmatch(r'[a-zA-Z0-9._-]{1,80}', approval['revision']) is None
            or approval['sha256'] != hashlib.sha256(content.encode()).hexdigest()):
        return None
    return {name: value[name] for name in ('content', 'created_at', 'modified_at')}


class ApprovedStoreBackend(StoreBackend):
    """Native StoreBackend with fixed paths, validated reads and denied writes.

    Read-time fallback is explicit only for a missing/unavailable Store. A stale
    or corrupt approval fails closed and is never silently replaced or repaired.
    Prefixes are stripped by CompositeBackend and restored in observations.
    """
    def __init__(self, runtime, *, kind: str):
        super().__init__(runtime, namespace=lambda _: NAMESPACE + (kind,))
        self.kind = kind
        self.packaged = {'/PROJECT.md': msty_memory.CONTEXT} if kind == 'memory' else SKILLS
        self.limit = MAX_MEMORY_BYTES if kind == 'memory' else MAX_SKILL_BYTES
        self.delivery: dict[str, str] = {}

    async def _load(self, path: str) -> dict | None:
        if path not in self.packaged:
            self.delivery[path] = 'unknown_path'
            return None
        reason = None
        try:
            async with asyncio.timeout(TIMEOUT_SECONDS):
                item = await self._get_store().aget(self._get_namespace(), path)
            if item is not None:
                data = _validate(item.value, self.limit)
                self.delivery[path] = 'approved_store' if data is not None else 'invalid_approval'
                return data
            reason = 'missing'
        except Exception:
            reason = 'unavailable'
        self.delivery[path] = 'packaged_fallback_' + reason
        content = self.packaged[path]
        notice = '\n\nMSTY_NATIVE_SOURCE_STATUS: packaged_fallback_' + reason + (
            '; revision=' + PACKAGED_REVISION + '; not a fresh runtime observation.\n')
        return create_file_data(content + notice)

    async def aread(self, file_path: str, offset: int = 0, limit: int = 2000) -> str:
        data = await self._load(file_path)
        if data is None:
            return 'Error: approved memory/skill unavailable; approval or path invalid. Do not infer missing facts.'
        return format_read_response(data, offset, min(limit, 512))

    def read(self, file_path: str, offset: int = 0, limit: int = 2000) -> str:
        return asyncio.run(self.aread(file_path, offset, limit))

    async def adownload_files(self, paths: list[str]) -> list[FileDownloadResponse]:
        result = []
        for path in paths:
            data = await self._load(path)
            result.append(FileDownloadResponse(path=path,
                content='\n'.join(data['content']).encode() if data is not None else None,
                error=None if data is not None else 'permission_denied'))
        return result

    def download_files(self, paths: list[str]) -> list[FileDownloadResponse]:
        return asyncio.run(self.adownload_files(paths))

    def ls_info(self, path: str) -> list[dict]:
        prefix = path.rstrip('/') + '/'
        found = {}
        for name in self.packaged:
            if not name.startswith(prefix):
                continue
            rest = name[len(prefix):]
            child = prefix + rest.split('/')[0]
            directory = '/' in rest
            if directory:
                child += '/'
            found[child] = {'path': child, 'is_dir': directory}
        return sorted(found.values(), key=lambda item: item['path'])

    async def als_info(self, path: str) -> list[dict]:
        return self.ls_info(path)

    def glob_info(self, pattern: str, path: str = '/') -> list[dict]:
        prefix = path.rstrip('/') + '/'
        return [{'path': name, 'is_dir': False} for name in self.packaged
                if name.startswith(prefix) and fnmatch.fnmatch(name.lstrip('/'), pattern.lstrip('/'))]

    async def aglob_info(self, pattern: str, path: str = '/') -> list[dict]:
        return self.glob_info(pattern, path)

    async def agrep_raw(self, pattern: str, path: str | None = None, glob: str | None = None):
        matches = []
        for name in self.packaged:
            if path and name != path and not name.startswith(path.rstrip('/') + '/'):
                continue
            if glob and not fnmatch.fnmatch(name.lstrip('/'), glob.lstrip('/')):
                continue
            data = await self._load(name)
            if data is not None:
                matches.extend({'path': name, 'line': index + 1, 'text': line}
                               for index, line in enumerate(data['content']) if pattern in line)
        return matches

    def grep_raw(self, pattern: str, path: str | None = None, glob: str | None = None):
        return asyncio.run(self.agrep_raw(pattern, path, glob))

    def write(self, file_path: str, content: str) -> WriteResult:
        return WriteResult(error='permission_denied: approved memory and skills are read-only')

    async def awrite(self, file_path: str, content: str) -> WriteResult:
        return self.write(file_path, content)

    def edit(self, file_path: str, old_string: str, new_string: str, replace_all: bool = False) -> EditResult:
        return EditResult(error='permission_denied: approved memory and skills are read-only')

    async def aedit(self, file_path: str, old_string: str, new_string: str, replace_all: bool = False) -> EditResult:
        return self.edit(file_path, old_string, new_string, replace_all)

    def upload_files(self, files: list[tuple[str, bytes]]) -> list[FileUploadResponse]:
        return [FileUploadResponse(path=path, error='permission_denied') for path, _ in files]

    async def aupload_files(self, files: list[tuple[str, bytes]]) -> list[FileUploadResponse]:
        return self.upload_files(files)


def candidates_backend(runtime) -> StoreBackend:
    """Stock writable StoreBackend for the candidate zone (no custom logic)."""
    return StoreBackend(runtime, namespace=lambda _: CANDIDATES_NAMESPACE)


def backend_factory(runtime) -> CompositeBackend:
    """Scratch is ephemeral; approved routes are read-only; candidates writable."""
    return CompositeBackend(default=StateBackend(runtime), routes={
        '/memory/': ApprovedStoreBackend(runtime, kind='memory'),
        '/skills/': ApprovedStoreBackend(runtime, kind='skills'),
        CANDIDATES_ROUTE: candidates_backend(runtime)})


class ApprovedMemoryMiddleware(MemoryMiddleware):
    """Native loader, refreshing server-owned fields instead of trusting input."""
    def __init__(self):
        super().__init__(backend=backend_factory, sources=MEMORY_PATHS)

    def before_agent(self, state, runtime, config):
        if state.get('brain_task_role') == 'analyst':
            return {'memory_contents': {}}
        clean = {key: value for key, value in state.items() if key != 'memory_contents'}
        return super().before_agent(clean, runtime, config)

    async def abefore_agent(self, state, runtime, config):
        if state.get('brain_task_role') == 'analyst':
            return {'memory_contents': {}}
        clean = {key: value for key, value in state.items() if key != 'memory_contents'}
        return await super().abefore_agent(clean, runtime, config)


class ApprovedSkillsMiddleware(SkillsMiddleware):
    """Native progressive loader; cached skill metadata is not client authority."""
    def __init__(self):
        super().__init__(backend=backend_factory, sources=SKILLS_PATHS)

    def before_agent(self, state, runtime, config):
        if state.get('brain_task_role') == 'analyst':
            return {'skills_metadata': []}
        clean = {key: value for key, value in state.items() if key != 'skills_metadata'}
        return super().before_agent(clean, runtime, config)

    async def abefore_agent(self, state, runtime, config):
        if state.get('brain_task_role') == 'analyst':
            return {'skills_metadata': []}
        clean = {key: value for key, value in state.items() if key != 'skills_metadata'}
        return await super().abefore_agent(clean, runtime, config)


def native_middlewares() -> list:
    return [ApprovedMemoryMiddleware(), ApprovedSkillsMiddleware()]
