"""TAU L1: манифест загружается и покрывает все инструменты роутера.

Литералы ниже — зафиксированный эталон наборов роутера до миграции на манифест
(коммит 5c63fae). Реестр обязан воспроизводить их точно: поведение роутера
снаружи не меняется.
"""
import re

from deep_agent import msty_registry
from deep_agent import msty_tool_routing as routing


EXPECTED_GROUPS = {
    'core_read': {"msty_admin_memory_search", "msty_admin_route_request", "msty_project_resolve"},
    'task': {"msty_task_plan", "msty_task_verify", "msty_project_verify_result"},
    'site': {"msty_site_prepare", "msty_site_status", "msty_site_file", "msty_site_patch",
             "msty_site_check", "msty_site_release", "msty_site_cancel", "msty_vercel_runtime_logs"},
    'pressable': {"discover_tools", "describe_tool", "execute_tool"},
    'supabase_read': {"search_docs", "list_projects", "get_project", "list_tables",
                      "list_migrations", "get_advisors", "query_logs", "get_project_url",
                      "execute_sql"},
    'supabase_write': {"apply_migration", "deploy_edge_function", "create_branch", "delete_branch",
                       "merge_branch", "rebase_branch", "reset_branch", "create_project",
                       "pause_project", "restore_project"},
    'browser_read': {"browser_navigate", "browser_snapshot", "browser_console_messages",
                     "browser_network_requests", "browser_take_screenshot", "browser_wait_for"},
    'browser_write': {"browser_click", "browser_file_upload", "browser_fill_form",
                      "browser_press_key", "browser_select_option", "browser_type"},
    'files_read': {"get_file_info", "list_directory", "read_file", "read_multiple_files",
                   "read_text_file", "search_files"},
    'files_write': {"create_directory", "edit_file", "move_file", "write_file"},
    'web': {"fetch", "msty_web_fetch"},
    'brain_read': {"msty_admin_health", "msty_admin_keys_health", "msty_admin_last_repair",
                   "msty_admin_system_map", "msty_system_overview", "msty_store_sync_status",
                   "msty_brain_lessons", "msty_self_skills", "msty_selfimprove_status"},
    'brain_write': {"msty_admin_plan_repair", "msty_admin_apply_repair", "msty_selfimprove_prepare",
                    "msty_selfimprove_file", "msty_selfimprove_patch", "msty_selfimprove_check",
                    "msty_selfimprove_release", "msty_worker_start", "msty_worker_status",
                    "msty_worker_cancel", "msty_brain_job", "msty_brain_verify",
                    "msty_brain_consult"},
    'codex': {"msty_codex_start", "msty_codex_status", "msty_codex_cancel"},
    'selfimprove': {"msty_selfimprove_status", "msty_selfimprove_file", "msty_selfimprove_patch",
                    "msty_selfimprove_check", "msty_selfimprove_release"},
    'worker': {"msty_worker_status", "msty_worker_cancel"},
    'brain_job': {"msty_brain_job", "msty_brain_verify"},
    'known_only': {"list_organizations", "get_organization", "list_extensions",
                   "get_publishable_keys", "get_edge_function", "list_edge_functions",
                   "list_branches", "generate_typescript_types", "get_cost", "confirm_cost",
                   "browser_close", "browser_drag", "browser_drop", "browser_emulate_media",
                   "browser_evaluate", "browser_find", "browser_handle_dialog", "browser_hover",
                   "browser_navigate_back", "browser_resize", "browser_run_code_unsafe",
                   "browser_tabs", "directory_tree", "list_allowed_directories",
                   "list_directory_with_sizes", "read_media_file", "msty_image_read",
                   "msty_project_create", "msty_project_read", "msty_projects_list",
                   "msty_brain_verify"},
}


def test_manifest_loads_with_unique_names_and_valid_fields():
    entries = msty_registry.TOOLS
    assert entries, 'манифест пуст'
    names = [entry.name for entry in entries]
    assert len(names) == len(set(names)), 'имена записей должны быть уникальны'
    aliases = [alias for entry in entries for alias in entry.aliases]
    assert len(aliases) == len(set(aliases)), 'алиасы должны быть уникальны'
    assert not (set(aliases) & set(names)), 'алиас не должен совпадать с чужим именем'
    for entry in entries:
        assert entry.kind in {'native', 'mcp', 'passthrough'}, entry.name
        assert entry.access in {'read', 'write'}, entry.name
        assert entry.description.strip(), entry.name
        for trigger in entry.lexical_triggers:
            re.compile(trigger)  # триггеры обязаны компилироваться


def test_manifest_covers_every_tool_the_router_manages():
    for group_name, expected in EXPECTED_GROUPS.items():
        assert msty_registry.group(group_name) == expected, group_name
    union = set().union(*EXPECTED_GROUPS.values())
    assert msty_registry.routed_names() == union
    assert set(routing._KNOWN) == union


def test_native_status_and_passthrough_tools_are_registered():
    for name in ('msty_store_sync_status', 'msty_system_overview', 'msty_admin_health'):
        entry = msty_registry.find(name)
        assert entry is not None and entry.kind == 'native', name
        assert entry.critical_path and entry.evidence_class == 'status_read', name
        assert msty_registry.TRIGGER_STORE_SYNC in entry.lexical_triggers, name
    execute = msty_registry.find('execute_tool')
    assert execute is not None and execute.kind == 'passthrough'


def test_find_resolves_exact_alias_and_client_namespace_suffix():
    assert msty_registry.find('execute_sql').name == 'execute_sql'
    # Алиас на языке оператора резолвится в каноническое имя.
    assert msty_registry.find('store_sync_status').name == 'msty_store_sync_status'
    # Суффиксная конвенция неймспейсинга клиента (как у роутера и гейта исполнения).
    assert msty_registry.find('sanare_admin_msty_task_plan').name == 'msty_task_plan'
    assert msty_registry.find('no_such_tool_anywhere') is None
    assert msty_registry.find('') is None


def test_fuzzy_returns_closest_canonical_names():
    candidates = msty_registry.fuzzy('msty_store_sync_stats', top=3)
    assert 'msty_store_sync_status' in candidates
    assert len(candidates) <= 3
    assert msty_registry.fuzzy('', top=3) == []


def test_lexical_projection_keeps_the_5c63fae_guard_semantics():
    trio = {'msty_store_sync_status', 'msty_system_overview', 'msty_admin_health'}
    sync_text = 'Проверь статус синхронизации магазина и свежесть данных'
    assert trio <= msty_registry.lexical_projection({'commerce'}, sync_text)
    # Pressable-only cron без commerce не расширяется (негатив 5c63fae).
    assert msty_registry.lexical_projection({'pressable'}, sync_text) == set()
    # Commerce без синхронизационной лексики — без статусных инструментов.
    assert msty_registry.lexical_projection(
        {'commerce'}, 'Составь отчёт по заказам магазина за сентябрь.') == set()
    assert msty_registry.lexical_projection(set(), sync_text) == set()
