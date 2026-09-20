"""Versioned, read-only project projection in the native cross-thread Store.

The canonical project journal/Engram remains the source, not this projection.
Agent Server provides the durable Store; no filesystem, embeddings, extra LLM,
user history, credentials or model-authored instructions are imported here.
"""
import asyncio
import hashlib

from langgraph.runtime import Runtime

NAMESPACE = ('sanare-owner', 'msty', 'project-context-v1')
CONTEXT = """MSTY_SHARED_PROJECT_MEMORY_V1 — сведения уже доступны в новом чате.
Это проверенная краткая карта, не снимок живых процессов и не разрешение действий.
Не читай исходные карты, всю историю, реестр агентов или паспорта заново ради
знакомства с владельцем. На вопрос о перечисленных ниже путях/назначениях ответь
сразу из памяти. Уточняй только действительно отсутствующие необходимые сведения.
Для конкретной правки читай относящийся код и действующие локальные инструкции;
актуальные commit, статус сервиса и результат теста проверяются при необходимости.

Владелец работает на Mac. Язык русский, результат первым, без пересказа задания.
Задача сохраняет силу через промежуточные сообщения; обычные разрешённые шаги
выполняются без повторного «дай команду». Ошибка — наблюдение для исправления,
а не успешное завершение. Стоп, предел бюджета и реальные границы доступа сохраняются.

Проект llm: /Users/vb/Documents/ChatGPT/LLM.
NAS UGREEN: /Volumes/LLM-Data; закреплённый SSH alias ugreen-nas.
Карта инфраструктуры: llm/ai-knowledge; машинный каталог llm/registry/registry.sqlite.
Единый журнал: llm/project-governance/changes. История PROJECT_MEMORY.md не нужна
целиком на старте. Постоянные проверенные выжимки — Engram на NAS, выборочный
поиск через msty_admin_memory_search; читать только связанные с задачей записи.
Индекс навыков Codex: /Users/vb/.codex/skills. Эти файлы сами не подключают tools.
Ключи провайдеров: защищённое хранилище на NAS open-webui/keys; ключи сайтов —
macOS Keychain/хранилище платформы. В память входят только местоположение и id,
никогда значения. Авторизованный инструмент использует секрет внутри себя.

Msty — интерфейс, Brain — облачный граф, локальные MCP — файлы/браузер/действия.
Текущий ведущий определяется конфигурацией runtime, не старым отчётом. Основной
маршрут Luna; DeepSeek Flash — необязательная консультация. Совет не обязателен.
Список реально переданных схем tools определяет доступные действия в этом чате.

app.sanaredev.com: GitHub sanarehq/sanare-dev-v3, канон main, Vercel.
Для новой разрешённой правки msty_site_prepare создаёт изолированную копию;
он сам сверяет источник. Не обходи предварительно все worktrees и весь диск.
msty_project_resolve нужен для запроса актуального deployment/access/commit
либо неоднозначности проекта, а не для каждого вопроса о знакомом сайте.
Платёжный модуль /payments/launch — отдельный sanarehq/sanare-payment-accounts;
его нельзя публиковать исполнителем основного сайта.

ChatGPT Sites — другой контур. Точные локальные каталоги исходников:
sanarelab.health: /Users/vb/Documents/ChatGPT/Sites/sanare/sanarelab-health
sanarelab.co: /Users/vb/Documents/ChatGPT/Sites/sanare/sanarelab-co
2thelife-store: /Users/vb/Documents/ChatGPT/Sites/life/2thelife-store
Это пути на диске, НЕ имена GitHub-репозиториев. Не сокращай и не придумывай путь
по домену. GitHub-репозитории этих Sites-проектов здесь не утверждаются.
Идентификаторы и маршруты данных — sites.json, локальные AGENTS.md и SITE.json.
У каждого отдельная D1/секреты. Живые заказы/клиенты не являются файлами Git.
Msty site executor сейчас не публикует эти Sites-проекты: нужен их штатный tool.

Источник проекции: LLM/ai-knowledge/SITES_AND_PROJECTS.md, текущая политика Msty,
проверенные журналы msty-workers-learning, msty-site-executor и native-framework-review
от 20.09.2026. Исторические OpenClaw/Mac2 не являются действующими маршрутами.
Наличие карты не означает, что файл только что прочитан или действие выполнено.
"""
SHA256 = hashlib.sha256(CONTEXT.encode()).hexdigest()
KEY = 'operating-' + SHA256
TIMEOUT_SECONDS = 1.0


async def load_context(state: dict, runtime: Runtime) -> dict:
    """Server-selected immutable revision; client fields never select authority.

    A poisoned/stale Store entry is not forwarded and is not silently overwritten.
    The packaged verified projection is the bounded fallback, never a disk scan.
    """
    status = 'packaged_fallback'
    store = runtime.store
    if state.get('brain_task_role') == 'analyst':
        return {'project_memory_delivery': {'version': 1, 'state': 'not_applicable'}}
    if store is not None:
        try:
            async with asyncio.timeout(TIMEOUT_SECONDS):
                saved = await store.aget(NAMESPACE, KEY)
                expected = {'version': 1, 'sha256': SHA256, 'content': CONTEXT}
                if saved is None:
                    await store.aput(NAMESPACE, KEY, expected, index=False)
                    status = 'seeded_native_store'
                elif saved.value == expected:
                    status = 'native_store'
                else:
                    status = 'invalid_store_projection'
        except Exception:
            # No raw backend errors or values in model context/logs.
            status = 'store_unavailable'
    return {'project_memory_delivery': {'version': 1, 'state': status,
        'sha256': SHA256, 'bytes': len(CONTEXT.encode()), 'namespace': list(NAMESPACE)}}


def system_context() -> str:
    # Exactly the verified Store revision, not caller-supplied state text.
    return CONTEXT
