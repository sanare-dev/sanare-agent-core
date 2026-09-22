"""Deterministic native middleware routing; no providers or network."""
from deep_agent import msty_tool_routing as routing


NAMES = [
    "msty_admin_memory_search", "msty_admin_route_request", "msty_project_resolve",
    "msty_task_plan", "msty_task_verify", "msty_project_verify_result",
    "msty_site_prepare", "msty_site_status", "msty_site_file", "msty_site_patch",
    "msty_site_check", "msty_site_release", "msty_site_cancel", "msty_vercel_runtime_logs",
    "discover_tools", "describe_tool", "execute_tool",
    "search_docs", "list_projects", "get_project", "list_tables", "list_migrations",
    "get_advisors", "query_logs", "get_project_url", "execute_sql", "apply_migration",
    "deploy_edge_function", "create_branch", "delete_branch", "merge_branch", "rebase_branch",
    "reset_branch", "create_project", "pause_project", "restore_project",
    "browser_navigate", "browser_snapshot", "browser_console_messages", "browser_network_requests",
    "browser_take_screenshot", "browser_wait_for", "browser_click", "browser_file_upload",
    "browser_fill_form", "browser_press_key", "browser_select_option", "browser_type",
    "get_file_info", "list_directory", "read_file", "read_multiple_files", "read_text_file",
    "search_files", "create_directory", "edit_file", "move_file", "write_file",
    "fetch", "msty_web_fetch", "msty_admin_health", "msty_admin_keys_health",
    "msty_admin_last_repair", "msty_admin_system_map", "msty_system_overview",
    "msty_store_sync_status", "msty_brain_lessons", "msty_self_skills",
    "msty_selfimprove_status", "msty_admin_plan_repair", "msty_admin_apply_repair",
    "msty_selfimprove_prepare", "msty_selfimprove_file", "msty_selfimprove_patch",
    "msty_selfimprove_check", "msty_selfimprove_release", "msty_worker_start",
    "msty_worker_status", "msty_worker_cancel", "msty_brain_job", "msty_brain_verify",
    "msty_brain_consult", "future_inventory_lookup",
]


def schema(name):
    description = ("Look up future inventory records" if name == "future_inventory_lookup"
                   else f"Official {name} operation")
    return {"type": "function", "function": {"name": name, "description": description,
        "parameters": {"type": "object", "properties": {}}}}


TOOLS = [schema(name) for name in NAMES]


def route(text, **kwargs):
    selected, value, prompt = routing.select_tools(
        [{"role": "user", "content": text}], TOOLS, **kwargs)
    return {tool["function"]["name"] for tool in selected}, value, prompt


def test_plain_question_has_no_external_tool_overhead():
    names, value, prompt = route("Объясни кратко, чем LangGraph отличается от LangSmith.")
    assert names == set()
    assert value["intent"] == "direct"
    assert value["domains"] == ["brain"]
    assert "выбраны автоматически" in prompt


def test_site_mutation_gets_only_bounded_site_bundle():
    names, value, _ = route("Исправь страницу app.sanaredev.com и проверь её в браузере.")
    assert {"msty_site_prepare", "msty_site_patch", "msty_site_check",
            "browser_navigate", "browser_snapshot", "msty_task_verify"} <= names
    assert "execute_sql" not in names and "discover_tools" not in names
    assert "edit_file" not in names and "browser_fill_form" not in names
    assert value["intent"] == "mutate"
    assert {"sites", "browser"} <= set(value["domains"])
    assert len(names) <= routing.MAX_SELECTED_TOOLS


def test_pressable_incident_uses_lazy_meta_tools_not_every_connector():
    names, value, _ = route("На sanarelab.club сломан WooCommerce cron, почини и проверь.")
    assert {"discover_tools", "describe_tool", "execute_tool", "msty_task_verify"} <= names
    assert "apply_migration" not in names and "msty_site_patch" not in names
    assert value["domains"][0] == "pressable"
    assert len(names) <= 8


def test_supabase_read_and_write_are_separated():
    read, read_route, _ = route("Покажи текущие таблицы Supabase проекта налогов.")
    assert {"list_tables", "get_project", "execute_sql"} <= read
    assert "apply_migration" not in read
    assert "msty_task_plan" not in read and "browser_navigate" not in read
    assert read_route["intent"] == "read"

    write, write_route, _ = route("Создай миграцию Supabase и добавь таблицу налогов.")
    assert {"apply_migration", "execute_sql", "msty_task_plan", "msty_task_verify"} <= write
    assert write_route["intent"] == "mutate"
    assert len(write) <= routing.MAX_SELECTED_TOOLS


def test_short_continuation_reuses_previous_route_without_reclassification():
    _, prior, _ = route("Почини cron sanarelab.club через Pressable.")
    names, value, _ = route("Делай", prior_route=prior)
    assert value["source"] == "continued"
    assert value["domains"] == prior["domains"]
    assert names == set(prior["selected_names"])


def test_exact_future_tool_and_historical_schema_are_preserved():
    exact, _, _ = route("Вызови future_inventory_lookup для остатков.")
    assert "future_inventory_lookup" in exact

    messages = [
        {"role": "assistant", "content": "", "tool_calls": [{"id": "x", "type": "function",
            "function": {"name": "future_inventory_lookup", "arguments": "{}"}}]},
        {"role": "tool", "tool_call_id": "x", "content": "done"},
        {"role": "user", "content": "Объясни результат."},
    ]
    selected, value, _ = routing.select_tools(messages, TOOLS)
    assert {tool["function"]["name"] for tool in selected} == {"future_inventory_lookup"}
    assert value["intent"] == "direct"


def test_tool_choice_none_keeps_history_but_adds_nothing_new():
    messages = [
        {"role": "assistant", "content": "", "tool_calls": [{"name": "execute_sql", "args": {}, "id": "x"}]},
        {"role": "tool", "tool_call_id": "x", "content": "done"},
        {"role": "user", "content": "Создай миграцию Supabase."},
    ]
    selected, _, _ = routing.select_tools(messages, TOOLS, tool_choice="none")
    assert [tool["function"]["name"] for tool in selected] == ["execute_sql"]
