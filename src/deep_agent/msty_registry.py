"""TAU, слой L1: единый машиночитаемый реестр инструментов Brain.

Единый источник истины о том, какими инструментами управляет роутер
(`msty_tool_routing`) и что валидирует Guard (`msty_guard`): native + MCP +
passthrough-доступ. До TAU эти сведения были размазаны по литералам роутера
и prompt-текстам; теперь роутер ВЫВОДИТ свои наборы из этого манифеста, а не
перечисляет имена сам.

Формат — Python-модуль с типизированными записями, а не YAML: манифест импортируется
как код, не читает файловую систему в рантайме веб-сервера (требование AGENTS.md),
не добавляет зависимостей и проверяется линтером вместе с остальным пакетом.

Поля записи — по эталонной архитектуре TAU (§4.4): name, aliases, domains,
critical_path, lexical_triggers, kind (native|mcp|passthrough), evidence_class,
schema/description. Дополнительно:
- router_groups — членство в наборах роутера (core_read, supabase_read, ...);
  это перенос прежних литералов роутера без изменения его внешнего поведения;
- access — read/write, задел под таксономию отказов недели 2 (retry только
  для идемпотентных read).

Схемы аргументов MCP-инструментов поставляет клиент Msty в каждом запросе, поэтому
`schema` здесь — None («схема принадлежит коннектору»); Guard сверяет аргументы
с живой схемой шага. Статическая схема фиксируется только там, где мы сами владеем
контрактом.
"""
from __future__ import annotations

from dataclasses import dataclass, field
import difflib
import functools
import re


# Детерминированный гард проекции статус-инструментов при commerce-лексике.
# Перенесён из роутера (коммит 5c63fae) без изменения семантики: запрос о
# синхронизации магазина должен видеть нативные статус-чтения Brain, а не только
# passthrough execute_tool (дефект 2026-09-22, ложный диагноз «cron не настроен»).
TRIGGER_STORE_SYNC = (
    r"(?is)(?:синхронизац|sync[-_ ]?status|store[-_ ]?sync|"
    r"не\s+приходят?\s+заказы|не\s+обновляются?\s+товары|свежесть\s+данных)"
)

KIND_NATIVE = 'native'          # исполняется внутри графа Brain / его штатным контуром
KIND_MCP = 'mcp'                # операция внешнего коннектора клиента, прямое имя
KIND_PASSTHROUGH = 'passthrough'  # мета-операция, проксирующая вызов другому коннектору

EVIDENCE_STATUS_READ = 'status_read'  # успешное чтение = доказательство для Evidence Gate


@dataclass(frozen=True)
class ToolEntry:
    """Запись реестра об одном инструменте (§4.4 эталонной архитектуры TAU)."""
    name: str
    description: str
    kind: str
    domains: tuple[str, ...] = ()
    aliases: tuple[str, ...] = ()
    router_groups: tuple[str, ...] = ()
    access: str = 'read'
    critical_path: bool = False
    lexical_triggers: tuple[str, ...] = ()
    evidence_class: str | None = None
    schema: dict | None = field(default=None, compare=False)


def _entry(name, description, kind=KIND_MCP, domains=(), aliases=(), groups=(),
           access='read', critical=False, triggers=(), evidence=None, schema=None):
    return ToolEntry(name=name, description=description, kind=kind, domains=domains,
                     aliases=aliases, router_groups=groups, access=access,
                     critical_path=critical, lexical_triggers=triggers,
                     evidence_class=evidence, schema=schema)


# Домены статусной тройки: 'brain' — их штатный домен, 'commerce' — домен, в котором
# срабатывает их лексический гард. Перенос семантики 5c63fae: роутер проецирует
# запись, если её lexical_trigger совпал с текстом и её домены пересекаются с
# доменами маршрута; для brain-запросов эти инструменты и так входит в brain_read.
_STATUS_DOMAINS = ('brain', 'commerce')

TOOLS: tuple[ToolEntry, ...] = (
    # --- Резолверы ядра: открывают доступ ко всему остальному -----------------
    _entry('msty_admin_memory_search', 'Точечный поиск по одобренной памяти проекта Brain.',
           domains=('brain',), groups=('core_read',)),
    _entry('msty_admin_route_request', 'Маршрутизатор запроса владельца к контуру Brain.',
           domains=('brain',), groups=('core_read',)),
    _entry('msty_project_resolve', 'Резолвер проекта: найти project_slug по имени или контексту.',
           domains=('brain',), aliases=('project_resolve',), groups=('core_read',)),

    # --- Универсальный контракт задачи с проверками ---------------------------
    _entry('msty_task_plan', 'План задачи с требованиями и read-only проверками результата.',
           domains=('files',), groups=('task',)),
    _entry('msty_task_verify', 'Проверка выполнения требований плана по фактическим наблюдениям.',
           domains=('files',), groups=('task',)),
    _entry('msty_project_verify_result', 'Итоговая сверка результата по контракту проекта.',
           domains=('files',), groups=('task',)),

    # --- Ограниченный исполнитель зарегистрированных сайтов -------------------
    _entry('msty_site_prepare', 'Подготовить изолированную рабочую копию зарегистрированного сайта.',
           domains=('sites',), groups=('site',), access='write'),
    _entry('msty_site_status', 'Статус site job по её идентификатору.',
           domains=('sites',), groups=('site',)),
    _entry('msty_site_file', 'Прочитать файл из рабочей копии site job.',
           domains=('sites',), groups=('site',)),
    _entry('msty_site_patch', 'Внести правку в файл рабочей копии site job.',
           domains=('sites',), groups=('site',), access='write'),
    _entry('msty_site_check', 'Прогнать проверки рабочей копии site job (readback).',
           domains=('sites',), groups=('site',)),
    _entry('msty_site_release', 'Публикация site job: push_pr, merge, verify.',
           domains=('sites',), groups=('site',), access='write'),
    _entry('msty_site_cancel', 'Отменить site job.',
           domains=('sites',), groups=('site',), access='write'),
    _entry('msty_vercel_runtime_logs', 'Runtime-логи деплоя Vercel для диагностики сайта.',
           domains=('sites',), groups=('site',)),

    # --- Pressable: ленивое обнаружение операций и passthrough-исполнение -----
    _entry('discover_tools', 'Поиск операции ленивого MCP-коннектора по короткой фразе.',
           domains=('pressable',), groups=('pressable',)),
    _entry('describe_tool', 'Описание и схема одной операции ленивого MCP-коннектора.',
           domains=('pressable',), groups=('pressable',)),
    _entry('execute_tool', 'Passthrough: выполнить операцию коннектора по её имени и аргументам.',
           kind=KIND_PASSTHROUGH, domains=('pressable',), groups=('pressable',), access='write'),

    # --- Supabase: чтение ------------------------------------------------------
    _entry('search_docs', 'Поиск по документации Supabase.',
           domains=('supabase',), groups=('supabase_read',)),
    _entry('list_projects', 'Список проектов Supabase организации.',
           domains=('supabase',), groups=('supabase_read',)),
    _entry('get_project', 'Сведения о проекте Supabase.',
           domains=('supabase',), groups=('supabase_read',)),
    _entry('list_tables', 'Список таблиц базы проекта Supabase.',
           domains=('supabase',), groups=('supabase_read',)),
    _entry('list_migrations', 'Список миграций проекта Supabase.',
           domains=('supabase',), groups=('supabase_read',)),
    _entry('get_advisors', 'Рекомендации advisors по проекту Supabase.',
           domains=('supabase',), groups=('supabase_read',)),
    _entry('query_logs', 'Логи сервисов проекта Supabase.',
           domains=('supabase',), groups=('supabase_read',)),
    _entry('get_project_url', 'URL проекта Supabase.',
           domains=('supabase',), groups=('supabase_read',)),
    # access=write: произвольный SQL может менять данные; группа роутера прежняя.
    _entry('execute_sql', 'Выполнить SQL в базе проекта Supabase.',
           domains=('supabase',), groups=('supabase_read',), access='write'),

    # --- Supabase: запись ------------------------------------------------------
    _entry('apply_migration', 'Применить миграцию базы проекта Supabase.',
           domains=('supabase',), groups=('supabase_write',), access='write'),
    _entry('deploy_edge_function', 'Деплой edge function Supabase.',
           domains=('supabase',), groups=('supabase_write',), access='write'),
    _entry('create_branch', 'Создать ветку базы Supabase.',
           domains=('supabase',), groups=('supabase_write',), access='write'),
    _entry('delete_branch', 'Удалить ветку базы Supabase.',
           domains=('supabase',), groups=('supabase_write',), access='write'),
    _entry('merge_branch', 'Слить ветку базы Supabase.',
           domains=('supabase',), groups=('supabase_write',), access='write'),
    _entry('rebase_branch', 'Перебазировать ветку базы Supabase.',
           domains=('supabase',), groups=('supabase_write',), access='write'),
    _entry('reset_branch', 'Сбросить ветку базы Supabase.',
           domains=('supabase',), groups=('supabase_write',), access='write'),
    _entry('create_project', 'Создать проект Supabase.',
           domains=('supabase',), groups=('supabase_write',), access='write'),
    _entry('pause_project', 'Приостановить проект Supabase.',
           domains=('supabase',), groups=('supabase_write',), access='write'),
    _entry('restore_project', 'Восстановить проект Supabase.',
           domains=('supabase',), groups=('supabase_write',), access='write'),

    # --- Браузер: чтение -------------------------------------------------------
    # access=write: переход по URL может вызвать действие на сайте (GET-ссылки).
    _entry('browser_navigate', 'Открыть URL в управляемом браузере.',
           domains=('browser',), groups=('browser_read',), access='write'),
    _entry('browser_snapshot', 'Снимок дерева доступности текущей страницы.',
           domains=('browser',), groups=('browser_read',)),
    _entry('browser_console_messages', 'Сообщения консоли страницы.',
           domains=('browser',), groups=('browser_read',)),
    _entry('browser_network_requests', 'Сетевые запросы страницы.',
           domains=('browser',), groups=('browser_read',)),
    _entry('browser_take_screenshot', 'Скриншот страницы или элемента.',
           domains=('browser',), groups=('browser_read',)),
    _entry('browser_wait_for', 'Ожидание условия на странице.',
           domains=('browser',), groups=('browser_read',)),

    # --- Браузер: запись -------------------------------------------------------
    _entry('browser_click', 'Клик по элементу страницы.',
           domains=('browser',), groups=('browser_write',), access='write'),
    _entry('browser_file_upload', 'Загрузка файла в элемент страницы.',
           domains=('browser',), groups=('browser_write',), access='write'),
    _entry('browser_fill_form', 'Заполнение полей формы на странице.',
           domains=('browser',), groups=('browser_write',), access='write'),
    _entry('browser_press_key', 'Нажатие клавиши на странице.',
           domains=('browser',), groups=('browser_write',), access='write'),
    _entry('browser_select_option', 'Выбор значения в выпадающем списке.',
           domains=('browser',), groups=('browser_write',), access='write'),
    _entry('browser_type', 'Ввод текста в элемент страницы.',
           domains=('browser',), groups=('browser_write',), access='write'),

    # --- Файлы: чтение ---------------------------------------------------------
    _entry('get_file_info', 'Метаданные файла или каталога на машине владельца.',
           domains=('files',), groups=('files_read',)),
    _entry('list_directory', 'Список содержимого каталога.',
           domains=('files',), groups=('files_read',)),
    _entry('read_file', 'Чтение файла целиком или по строкам.',
           domains=('files',), groups=('files_read',)),
    _entry('read_multiple_files', 'Чтение нескольких файлов за один вызов.',
           domains=('files',), groups=('files_read',)),
    _entry('read_text_file', 'Чтение текстового файла.',
           domains=('files',), groups=('files_read',)),
    _entry('search_files', 'Поиск файлов по имени или содержимому.',
           domains=('files',), groups=('files_read',)),

    # --- Файлы: запись ---------------------------------------------------------
    _entry('create_directory', 'Создать каталог.',
           domains=('files',), groups=('files_write',), access='write'),
    _entry('edit_file', 'Точечная правка текстового файла.',
           domains=('files',), groups=('files_write',), access='write'),
    _entry('move_file', 'Переместить или переименовать файл.',
           domains=('files',), groups=('files_write',), access='write'),
    _entry('write_file', 'Создать или перезаписать файл.',
           domains=('files',), groups=('files_write',), access='write'),

    # --- Веб -------------------------------------------------------------------
    _entry('fetch', 'Скачать содержимое веб-страницы по URL.',
           domains=('web',), groups=('web',)),
    _entry('msty_web_fetch', 'Веб-чтение через штатный коннектор Msty.',
           domains=('web',), groups=('web',)),

    # --- Brain: статусные чтения (critical_path, доказательства для Gate) ------
    _entry('msty_admin_health', 'Сводное здоровье контура Brain: сервисы, ключи, очереди.',
           kind=KIND_NATIVE, domains=_STATUS_DOMAINS, aliases=('admin_health', 'msty_health'),
           groups=('brain_read',), critical=True, triggers=(TRIGGER_STORE_SYNC,),
           evidence=EVIDENCE_STATUS_READ),
    _entry('msty_admin_keys_health', 'Состояние ключей и секретов, доступных Brain.',
           kind=KIND_NATIVE, domains=('brain',), groups=('brain_read',),
           evidence=EVIDENCE_STATUS_READ),
    _entry('msty_admin_last_repair', 'Последнее исправление Brain: версия и итог проверок.',
           kind=KIND_NATIVE, domains=('brain',), groups=('brain_read',)),
    _entry('msty_admin_system_map', 'Карта системы: сервисы, зависимости, точки наблюдения.',
           kind=KIND_NATIVE, domains=('brain',), groups=('brain_read',)),
    _entry('msty_system_overview', 'Обзор состояния системы владельца целиком.',
           kind=KIND_NATIVE, domains=_STATUS_DOMAINS, aliases=('system_overview',),
           groups=('brain_read',), critical=True, triggers=(TRIGGER_STORE_SYNC,),
           evidence=EVIDENCE_STATUS_READ),
    _entry('msty_store_sync_status', 'Статус синхронизации магазина: свежесть данных, cron.',
           kind=KIND_NATIVE, domains=_STATUS_DOMAINS,
           aliases=('store_sync_status', 'msty_store_sync'),
           groups=('brain_read',), critical=True, triggers=(TRIGGER_STORE_SYNC,),
           evidence=EVIDENCE_STATUS_READ),
    _entry('msty_brain_lessons', 'Подтверждённые уроки и практики Brain.',
           kind=KIND_NATIVE, domains=('brain',), groups=('brain_read',)),
    _entry('msty_self_skills', 'Список установленных skills Brain.',
           kind=KIND_NATIVE, domains=('brain',), groups=('brain_read',)),
    _entry('msty_selfimprove_status', 'Статус job самоизменения Brain.',
           kind=KIND_NATIVE, domains=('brain',), groups=('brain_read', 'selfimprove')),

    # --- Brain: защищённый контур самоизменения и исполнители ------------------
    _entry('msty_admin_plan_repair', 'Подготовить план исправления контура Brain.',
           kind=KIND_NATIVE, domains=('brain',), groups=('brain_write',), access='write'),
    _entry('msty_admin_apply_repair', 'Применить одобренное исправление контура Brain.',
           kind=KIND_NATIVE, domains=('brain',), groups=('brain_write',), access='write'),
    _entry('msty_selfimprove_prepare', 'Подготовить изолированную копию для самоизменения.',
           kind=KIND_NATIVE, domains=('brain',), groups=('brain_write',), access='write'),
    _entry('msty_selfimprove_file', 'Прочитать файл в контуре самоизменения.',
           kind=KIND_NATIVE, domains=('brain',), groups=('brain_write', 'selfimprove')),
    _entry('msty_selfimprove_patch', 'Правка файла в контуре самоизменения.',
           kind=KIND_NATIVE, domains=('brain',), groups=('brain_write', 'selfimprove'),
           access='write'),
    _entry('msty_selfimprove_check', 'Проверки контура самоизменения перед релизом.',
           kind=KIND_NATIVE, domains=('brain',), groups=('brain_write', 'selfimprove')),
    _entry('msty_selfimprove_release', 'Релиз изменений контура самоизменения.',
           kind=KIND_NATIVE, domains=('brain',), groups=('brain_write', 'selfimprove'),
           access='write'),
    _entry('msty_worker_start', 'Запустить worker-job для отдельного исполняемого артефакта.',
           kind=KIND_NATIVE, domains=('brain',), groups=('brain_write',), access='write'),
    _entry('msty_worker_status', 'Статус worker job.',
           kind=KIND_NATIVE, domains=('brain',), groups=('brain_write', 'worker')),
    _entry('msty_worker_cancel', 'Отменить worker job.',
           kind=KIND_NATIVE, domains=('brain',), groups=('brain_write', 'worker'),
           access='write'),
    _entry('msty_brain_job', 'Статус или продолжение долгой job Brain по её id.',
           kind=KIND_NATIVE, domains=('brain',), groups=('brain_write', 'brain_job')),
    _entry('msty_brain_verify', 'Квитанция проверки результата job Brain.',
           kind=KIND_NATIVE, domains=('brain',),
           groups=('brain_write', 'brain_job', 'known_only')),
    _entry('msty_brain_consult', 'Консультация второй модели для сложного противоречия.',
           kind=KIND_NATIVE, domains=('brain',), groups=('brain_write',), access='write'),

    # --- Codex: широкая автономная локальная задача ----------------------------
    _entry('msty_codex_start', 'Запустить одну автономную Codex job с исходной целью.',
           kind=KIND_NATIVE, domains=('brain', 'files'), groups=('codex',), access='write'),
    _entry('msty_codex_status', 'Статус Codex job до terminal state.',
           kind=KIND_NATIVE, domains=('brain', 'files'), groups=('codex',)),
    _entry('msty_codex_cancel', 'Отменить Codex job.',
           kind=KIND_NATIVE, domains=('brain', 'files'), groups=('codex',), access='write'),

    # --- Известные операции коннекторов вне маршрута по умолчанию --------------
    # Выбираются только точным именем; широкое лексическое совпадение («project»,
    # «file») не должно подтягивать их в контекст.
    _entry('list_organizations', 'Список организаций Supabase.',
           domains=('supabase',), groups=('known_only',)),
    _entry('get_organization', 'Сведения об организации Supabase.',
           domains=('supabase',), groups=('known_only',)),
    _entry('list_extensions', 'Расширения базы Supabase.',
           domains=('supabase',), groups=('known_only',)),
    _entry('get_publishable_keys', 'Публичные ключи проекта Supabase.',
           domains=('supabase',), groups=('known_only',)),
    _entry('get_edge_function', 'Сведения об edge function Supabase.',
           domains=('supabase',), groups=('known_only',)),
    _entry('list_edge_functions', 'Список edge functions Supabase.',
           domains=('supabase',), groups=('known_only',)),
    _entry('list_branches', 'Список веток базы Supabase.',
           domains=('supabase',), groups=('known_only',)),
    _entry('generate_typescript_types', 'Генерация TypeScript-типов по схеме Supabase.',
           domains=('supabase',), groups=('known_only',)),
    _entry('get_cost', 'Оценка стоимости платной операции Supabase.',
           domains=('supabase',), groups=('known_only',)),
    _entry('confirm_cost', 'Подтверждение платной операции Supabase.',
           domains=('supabase',), groups=('known_only',), access='write'),
    _entry('browser_close', 'Закрыть управляемый браузер.',
           domains=('browser',), groups=('known_only',), access='write'),
    _entry('browser_drag', 'Перетаскивание элемента страницы.',
           domains=('browser',), groups=('known_only',), access='write'),
    _entry('browser_drop', 'Завершение перетаскивания элемента.',
           domains=('browser',), groups=('known_only',), access='write'),
    _entry('browser_emulate_media', 'Эмуляция медиа-условий страницы.',
           domains=('browser',), groups=('known_only',)),
    _entry('browser_evaluate', 'Выполнить JavaScript на странице.',
           domains=('browser',), groups=('known_only',), access='write'),
    _entry('browser_find', 'Поиск элемента на странице.',
           domains=('browser',), groups=('known_only',)),
    _entry('browser_handle_dialog', 'Обработка диалога страницы.',
           domains=('browser',), groups=('known_only',), access='write'),
    _entry('browser_hover', 'Наведение курсора на элемент.',
           domains=('browser',), groups=('known_only',)),
    _entry('browser_navigate_back', 'Назад по истории браузера.',
           domains=('browser',), groups=('known_only',)),
    _entry('browser_resize', 'Изменить размер окна браузера.',
           domains=('browser',), groups=('known_only',), access='write'),
    _entry('browser_run_code_unsafe', 'Произвольный код в контексте браузера (небезопасно).',
           domains=('browser',), groups=('known_only',), access='write'),
    _entry('browser_tabs', 'Список и управление вкладками браузера.',
           domains=('browser',), groups=('known_only',)),
    _entry('directory_tree', 'Дерево каталога рекурсивно.',
           domains=('files',), groups=('known_only',)),
    _entry('list_allowed_directories', 'Каталоги, разрешённые файловому коннектору.',
           domains=('files',), groups=('known_only',)),
    _entry('list_directory_with_sizes', 'Содержимое каталога с размерами файлов.',
           domains=('files',), groups=('known_only',)),
    _entry('read_media_file', 'Прочитать медиафайл как изображение.',
           domains=('files',), groups=('known_only',)),
    _entry('msty_image_read', 'Чтение изображения штатным коннектором Msty.',
           domains=('files',), groups=('known_only',)),
    _entry('msty_project_create', 'Создать проект в реестре Msty.',
           domains=('brain',), groups=('known_only',), access='write'),
    _entry('msty_project_read', 'Карточка проекта из реестра Msty.',
           domains=('brain',), groups=('known_only',)),
    _entry('msty_projects_list', 'Список проектов реестра Msty.',
           domains=('brain',), groups=('known_only',)),

    # --- Native-инструменты виртуальной ФС и TODO графа (msty_native) ----------
    # Роутер ими не управляет (они всегда в native harness); реестр покрывает их,
    # чтобы Guard на границе исполнения видел полное меню имён.
    _entry('native_ls', 'Список виртуальных mountpoint/каталогов checkpoint-хранилища.',
           kind=KIND_NATIVE, domains=('brain',)),
    _entry('native_read_file', 'Чтение виртуального файла (/skills, /memory, /scratch).',
           kind=KIND_NATIVE, domains=('brain',)),
    _entry('native_write_file', 'Создание виртуального scratch-файла (не Mac).',
           kind=KIND_NATIVE, domains=('brain',), access='write'),
    _entry('native_edit_file', 'Точечная правка виртуального scratch-файла.',
           kind=KIND_NATIVE, domains=('brain',), access='write'),
    _entry('native_glob', 'Поиск по glob-паттерну внутри виртуального пути.',
           kind=KIND_NATIVE, domains=('brain',)),
    _entry('native_grep', 'Поиск литерального текста внутри виртуального пути.',
           kind=KIND_NATIVE, domains=('brain',)),
    _entry('native_write_todos', 'Рабочий список шагов сложной задачи (не доказательство).',
           kind=KIND_NATIVE, domains=('brain',), access='write'),

    # Sub-agents (фундамент, задание владельца «создавать ботов»): Brain поручает
    # ограниченный под-прогон роли с детерминированным loadout по этому манифесту.
    # kind=native: исполняется серверным контуром (msty_subagents), НЕ клиентом.
    # access=write: делегирование не идемпотентно — повтор порождает новый прогон.
    _entry('msty_delegate_task',
           'Purpose: поручить под-агенту ограниченную задачу и получить '
           'структурированный отчёт (status/findings/evidence/errors/steps_used). '
           'Guidelines: короткие атомарные задачи — operator, длинные read-only '
           'исследования — researcher, перепроверка критичных утверждений — auditor. '
           'Опирайся на evidence отчёта, а не на уверенность текста. '
           'Limitations: под-агент исполняет только серверные инструменты '
           '(виртуальная ФС); внешние msty_* он не вызывает — возвращает '
           'recommended_calls для исполнения родителем. Глубина 1: под-агент '
           'не может делегировать дальше. Бюджет шагов ограничен. '
           'Parameters: goal (цель), role (researcher|operator|auditor), '
           'domains (домены инструментов), max_steps, report_format. '
           'Examples: исследовать содержимое /memory по теме; проверить '
           'утверждение по файлам /scratch перед ответом владельцу.',
           kind=KIND_NATIVE, domains=('brain',), access='write',
           schema={'type': 'object',
                   'properties': {
                       'goal': {'type': 'string',
                                'description': 'Цель задачи под-агента, одна конкретная.'},
                       'role': {'type': 'string',
                                'enum': ['researcher', 'operator', 'auditor']},
                       'domains': {'type': 'array', 'items': {'type': 'string'},
                                   'description': 'Домены манифеста, чьи инструменты нужны.'},
                       'max_steps': {'type': 'integer', 'minimum': 1, 'maximum': 12},
                       'report_format': {'type': 'string',
                                         'description': 'Ожидаемая форма findings.'}},
                   'required': ['goal', 'role'],
                   'additionalProperties': False}),
)


# --- Индексы поверх манифеста --------------------------------------------------

_BY_NAME: dict[str, ToolEntry] = {entry.name: entry for entry in TOOLS}
_BY_ALIAS: dict[str, ToolEntry] = {
    alias: entry for entry in TOOLS for alias in entry.aliases}
# Сначала длинные имена: суффиксная проверка не должна резать «worker_status»
# до более короткого совпадения.
_NAMES_BY_LENGTH: tuple[str, ...] = tuple(sorted(_BY_NAME, key=len, reverse=True))


def find(name: str) -> ToolEntry | None:
    """Каноническая запись по имени вызова: точное имя, алиас или суффиксная
    конвенция неймспейсинга клиента (например sanare_admin_msty_task_plan)."""
    if not isinstance(name, str) or not name:
        return None
    entry = _BY_NAME.get(name) or _BY_ALIAS.get(name)
    if entry is not None:
        return entry
    # Суффикс признаётся для явного MCP-неймспейса «сервер__имя» и для
    # составных имён (execute_sql, msty_store_sync_status). Однословные общие
    # имена (fetch) по суффиксу не резолвятся: сторонний shop_fetch не должен
    # получать доступ/evidence-класс чужой записи.
    for candidate in _NAMES_BY_LENGTH:
        if (name.endswith('__' + candidate) or
                '_' in candidate and name.endswith('_' + candidate)):
            return _BY_NAME[candidate]
    return None


def fuzzy(name: str, top: int = 3) -> list[str]:
    """До `top` ближайших канонических имён реестра для опечатанного имени."""
    if not isinstance(name, str) or not name:
        return []
    choices = {entry.name: entry.name for entry in TOOLS}
    choices.update({alias: entry.name for entry in TOOLS for alias in entry.aliases})
    matches = difflib.get_close_matches(name, list(choices), n=top, cutoff=0.5)
    canonical = []
    for match in matches:
        if choices[match] not in canonical:
            canonical.append(choices[match])
    return canonical


def group(group_name: str) -> frozenset[str]:
    """Имена одного набора роутера (core_read, supabase_read, ...)."""
    return frozenset(entry.name for entry in TOOLS if group_name in entry.router_groups)


def routed_names() -> frozenset[str]:
    """Все имена, которыми управляет роутер (бывший _KNOWN)."""
    return frozenset(entry.name for entry in TOOLS if entry.router_groups)


@functools.lru_cache(maxsize=32)
def _compiled(trigger: str) -> re.Pattern[str]:
    return re.compile(trigger)


def lexical_projection(domains: set[str], text: str) -> set[str]:
    """Проекция по лексическим гардам манифеста (семантика гардов 5c63fae).

    Запись добавляется, если хотя бы один её lexical_trigger совпал с текстом
    и её домены пересекаются с доменами маршрута. Домен может только добавлять
    инструменты, никогда — исключать.
    """
    if not domains or not text:
        return set()
    return {entry.name for entry in TOOLS
            if entry.lexical_triggers and set(entry.domains) & domains
            and any(_compiled(trigger).search(text) for trigger in entry.lexical_triggers)}
