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
    "msty_brain_consult", "msty_codex_start", "msty_codex_status",
    "msty_codex_cancel", "future_inventory_lookup",
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
            "browser_navigate", "browser_snapshot"} <= names
    assert not ({"msty_task_plan", "msty_task_verify", "msty_project_verify_result"} & names)
    assert "execute_sql" not in names and "discover_tools" not in names
    assert "edit_file" not in names and "browser_fill_form" not in names
    assert value["intent"] == "mutate"
    assert {"sites", "browser"} <= set(value["domains"])
    assert len(names) <= routing.MAX_SELECTED_TOOLS


def test_pressable_incident_uses_lazy_meta_tools_not_every_connector():
    names, value, _ = route("На sanarelab.club сломан WooCommerce cron, почини и проверь.")
    assert {"discover_tools", "describe_tool", "execute_tool"} <= names
    assert not ({"msty_task_plan", "msty_task_verify", "msty_project_verify_result"} & names)
    assert "apply_migration" not in names and "msty_site_patch" not in names
    assert value["domains"][0] == "pressable"
    assert len(names) <= 8


def test_store_sync_question_reaches_native_status_tool():
    names, value, _ = route(
        "Проверь статус синхронизации магазина и доложи: здоров ли store и работает ли cron-синхронизация?")
    assert "msty_store_sync_status" in names
    assert "msty_system_overview" in names
    assert "execute_tool" in names
    assert "msty_site_patch" not in names


def test_pressable_only_cron_without_commerce_gets_no_store_status_tool():
    names, _, _ = route("На sanarelab.club сломан WooCommerce cron, почини и проверь.")
    assert "msty_store_sync_status" not in names


def test_commerce_without_sync_vocabulary_gets_no_store_status_tool():
    names, _, _ = route("Составь отчёт по заказам магазина за сентябрь.")
    assert "msty_store_sync_status" not in names


def test_supabase_read_and_write_are_separated():
    read, read_route, _ = route("Покажи текущие таблицы Supabase проекта налогов.")
    assert {"list_tables", "get_project", "execute_sql"} <= read
    assert "apply_migration" not in read
    assert "msty_task_plan" not in read and "browser_navigate" not in read
    assert read_route["intent"] == "read"

    write, write_route, _ = route("Создай миграцию Supabase и добавь таблицу налогов.")
    assert {"apply_migration", "execute_sql"} <= write
    assert not ({"msty_task_plan", "msty_task_verify", "msty_project_verify_result"} & write)
    assert write_route["intent"] == "mutate"
    assert len(write) <= routing.MAX_SELECTED_TOOLS


def test_generic_multi_step_code_work_uses_one_autonomous_codex_job():
    names, value, _ = route("Исправь код Python в файле репозитория и проверь результат.")
    assert {"msty_codex_start", "msty_codex_status", "msty_codex_cancel"} <= names
    assert not ({"msty_task_plan", "msty_task_verify", "read_file", "edit_file"} & names)
    assert value["intent"] == "mutate"


def test_narrow_file_read_does_not_start_autonomous_executor():
    names, value, _ = route("Прочитай файл README.md и покажи заголовок.")
    assert "read_file" in names
    assert "msty_codex_start" not in names
    assert value["intent"] == "read"


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


def test_domain_request_with_an_unknown_verb_still_gets_read_tools():
    """A turn naming a business domain is never treated as small talk.

    "выгрузи"/"посчитай"/"что у нас с" are not in the action lexicons, so these
    used to fall through to intent=direct and reach the model with no schemas at
    all — the model was told to act and given nothing to act with.
    """
    for text in ("выгрузи список таблиц",
                 "что у нас с базой supabase",
                 "налоги за квартал посчитай"):
        names, value, _ = route(text)
        assert value["intent"] == "read", text
        assert value["domains"], text
        assert names, f"{text}: маршрут без инструментов"
        # Promotion reaches read only; mutation stays behind an explicit verb.
        assert not (names & routing._SUPABASE_WRITE), text


def test_explain_only_stays_direct_even_for_a_known_domain():
    """The explain-only guard is checked before the domain promotion."""
    for text in ("объясни, как работает supabase",
                 "расскажи, что такое msty toolset"):
        names, value, _ = route(text)
        assert value["intent"] == "direct" and names == set(), text


def test_small_talk_without_a_domain_is_not_promoted():
    """"да" is not here: it is a continuation verb, covered by the tests below."""
    for text in ("привет", "спасибо", "как дела"):
        names, value, _ = route(text)
        assert value["intent"] == "direct" and names == set(), text


def test_site_job_id_reaches_the_bounded_site_executor():
    """The id the site executor returns is how the owner names the job next turn.

    "опубликуй job site-<id>" used to miss _REGISTERED_SITE_EXECUTOR and fall to
    the generic branch: browser and filesystem tools, no msty_site_release, so
    the job could not be published at all.
    """
    for text in ("опубликуй job site-c7228bab8ae54e8cbf0c0b5a8f2573c3",
                 "site-c7228bab8ae54e8cbf0c0b5a8f2573c3 — какой статус",
                 "проверь job site-3b36138917414f028e7674705d31b3ca"):
        names, value, _ = route(text)
        assert "sites" in value["domains"], text
        assert {"msty_site_release", "msty_site_status", "msty_site_check"} <= names, text
        # The bounded executor replaces the generic filesystem branch.
        assert not (names & routing._FILES_WRITE), text


def test_a_site_word_without_a_registered_target_stays_generic():
    """Only a named registered target unlocks the bounded executor."""
    names, value, _ = route("сделай лендинг на каком-нибудь сайте")
    assert "msty_site_release" not in names


def test_every_executor_job_id_reaches_its_own_continuation_tools():
    """Policy tells the model to resume long work by job_id; routing must allow it.

    A job reference carries no action verb and usually no domain word, so these
    turns classified as small talk and arrived with no way to reach the job.
    """
    cases = [
        ("опубликуй job site-" + "c" * 32, "msty_site_release"),
        ("проверь job self-" + "3" * 32, "msty_selfimprove_status"),
        ("что со статусом codex-" + "9" * 32, "msty_codex_status"),
        ("отмени worker-" + "1" * 32, "msty_worker_cancel"),
        ("проверь brain-" + "4" * 32, "msty_brain_job"),
    ]
    for text, expected in cases:
        names, value, _ = route(text)
        assert value["intent"] != "direct", text
        assert expected in names, text
        # Asking about a running job must never offer to start another one.
        assert not (names & {"msty_codex_start", "msty_worker_start"}), text


def test_removed_session_id_gets_core_resolvers_and_a_catalog_escape_hatch():
    """A removed window session ("rs_<hex>") is a job reference like any other.

    Live defect 2026-09-25: new window tools for removed sessions shipped with
    no bundle here. "что с rs_<hex>" carried no verb this router knows and no
    domain word, so it classified as intent=direct with zero tools selected
    AND catalog=False — a dead end with no way for the model to even ask for
    the right tool by name.
    """
    for text in ("что с rs_67d8f2a9b1c04e77", "восстанови rs_67d8f2a9b1c04e77",
                 "статус rs_a1b2c3d4e5f6"):
        names, value, _ = route(text)
        assert value["intent"] != "direct", text
        assert value["catalog"], text
        assert names & routing._CORE_READ, text


def test_a_bare_word_without_a_job_id_gets_no_executor_bundle():
    """Only an actual id unlocks a job bundle; the word alone must not.

    msty_selfimprove_status is deliberately excluded: it belongs to _BRAIN_READ
    as well, so the brain domain may legitimately deliver it without any job.
    """
    job_only = (routing._WORKER | routing._BRAIN_JOB
                | (routing._SELFIMPROVE - routing._BRAIN_READ))
    for text in ("расскажи про worker", "что такое codex", "как устроен brain"):
        names, _, _ = route(text)
        assert not (names & job_only), text


BROAD = "опубликуй job site-" + "c" * 32 + " и проверь базу и файлы и браузер и блог"


def _selected(order):
    selected, _, _ = routing.select_tools(
        [{"role": "user", "content": BROAD}], [schema(name) for name in order])
    return frozenset(tool["function"]["name"] for tool in selected)


def test_truncation_does_not_depend_on_the_order_msty_sends_schemas():
    """The cap used to keep whatever came first, so the client decided the route."""
    orders = (NAMES, NAMES[::-1], sorted(NAMES), sorted(NAMES, key=len))
    results = {_selected(order) for order in orders}
    assert len(results) == 1, "набор зависит от порядка подачи схем"
    assert len(next(iter(results))) == routing.MAX_SELECTED_TOOLS


def test_truncation_keeps_the_bounded_executor_and_drops_writes_first():
    names = _selected(NAMES[::-1])
    assert routing._SITE <= names, "ограниченный исполнитель сайта должен выживать целиком"
    assert "msty_project_resolve" in names
    assert not (names & routing._BROWSER_WRITE)
    assert not (names & routing._FILES_WRITE)


PRIOR = {"version": routing.ROUTE_VERSION, "intent": "mutate", "domains": ["sites"],
         "selected_names": ["msty_site_release", "msty_site_status", "msty_site_check"]}


def test_a_continuation_is_recognised_however_it_is_phrased():
    """fullmatch on a fixed list broke on any extra word, and those turns got
    no tools at all — the owner said carry on and nothing could."""
    for text in ("Делай", "Делай, не спрашивай", "И фикси", "бери и делай",
                 "перенастраивай", "доведи до конца", "сделал"):
        names, value, _ = route(text, prior_route=PRIOR)
        assert value["source"] == "continued", text
        assert names == set(PRIOR["selected_names"]), text


def test_a_continuation_that_names_a_domain_is_a_fresh_request():
    """A new target must re-route, not inherit the previous one."""
    for text in ("почини WooCommerce на sanarelab.club",
                 "сделай лендинг на app.sanaredev.com"):
        names, value, _ = route(text, prior_route=PRIOR)
        assert value["source"] == "classified", text
        assert names != set(PRIOR["selected_names"]), text


def test_a_continuation_without_a_prior_route_can_still_orient():
    """After a compaction or restart there is nothing to continue; answering
    with no schemas at all is the worst of the options."""
    for text in ("Делай", "Фикси", "продолжай"):
        names, value, _ = route(text)
        assert value["source"] == "continuation-recovered", text
        assert names == routing._CORE_READ, text
        # Recovery orients, it never acts on its own.
        assert value["intent"] == "read", text


import pytest as _pytest  # noqa: E402,I001


@_pytest.mark.parametrize('question', [
    'Дай фактический обзор состояния всей системы: что живо, что требует внимания.',
    'Что сейчас с системой?',
    'Всё ли работает?',
    'Какой статус всего контура?',
    'Is everything ok?',
    'Всё ли в порядке?',
    'Всё ли в порядке с системой?',
    'Всё ли у нас работает сегодня?',
    'Всё ли работает? Проверь.',
    'Что упало?',
    'Что требует внимания?',
])
def test_general_system_status_question_gets_status_tools(question):
    """Живой дефект 2026-09-23: общий вопрос о состоянии без доменного слова
    получал 0 инструментов при переданных msty_system_overview/msty_admin_health."""
    tools = [{'type': 'function', 'function': {'name': name, 'description': 'x',
              'parameters': {'type': 'object', 'properties': {}}}}
             for name in ('msty_store_sync_status', 'msty_system_overview',
                          'msty_admin_health', 'execute_sql', 'list_tables')]
    _, route, _ = routing.select_tools([{"role": "user", "content": question}], tools)
    assert {'msty_system_overview', 'msty_admin_health'} <= set(route['selected_names'])


@_pytest.mark.parametrize('question', [
    'Что лежит в папке Downloads?', 'Всё ли в порядке с текстом письма?', 'что упало в цене'])
def test_non_system_questions_do_not_get_status_tools(question):
    tools = [{'type': 'function', 'function': {'name': name, 'description': 'x',
              'parameters': {'type': 'object', 'properties': {}}}}
             for name in ('msty_system_overview', 'msty_admin_health')]
    _, route, _ = routing.select_tools([{"role": "user", "content": question}], tools)
    assert not {'msty_system_overview', 'msty_admin_health'} & set(route['selected_names'])


ORG = ["org_structure", "delegate", "delegate_many", "review"]


def test_brain_desk_org_tools_are_brains_own_for_task_turns():
    """brain-desk #297: Brain delegates to departments; never cut away."""
    tools = TOOLS + [schema(name) for name in ORG]
    selected, value, _ = routing.select_tools(
        [{"role": "user", "content": "Разбери задачу #237 и предложи план исправления"}], tools)
    names = [tool["function"]["name"] for tool in selected]
    assert set(ORG) <= set(names)
    assert len(names) <= routing.MAX_SELECTED_TOOLS
    # A broad mutation that fills the limit still keeps them.
    selected, _, _ = routing.select_tools(
        [{"role": "user", "content": "Исправь сайт app.sanaredev.com, базу supabase, файлы репозитория и проверь в браузере"}],
        tools)
    assert set(ORG) <= {tool["function"]["name"] for tool in selected}
    # Small talk gets no tools at all, as before.
    selected, value, _ = routing.select_tools(
        [{"role": "user", "content": "Объясни кратко, что такое LangGraph."}], tools)
    assert selected == [] and value["intent"] == "direct"
    # Without the client's schemas nothing is invented.
    names, _, _ = route("Разбери задачу #237 и предложи план исправления")
    assert not set(ORG) & names


def test_nas_structure_question_gets_file_reads():
    """brain-desk #296: NAS is read through the file server."""
    names, value, _ = route("Проверь структуру хранилища NAS, как в библиотеке: какие разделы?")
    assert "files" in value["domains"]
    assert {"list_directory", "read_text_file"} <= names
    assert not ({"write_file", "edit_file", "move_file"} & names)


# Live 24.09 (Brain Desk, Amazon project chat): «сходи на GitHub и поставь»
# got no fetch/browser, and the Msty resolver was offered for a window project.
DESK_SYSTEM = {"role": "system",
               "content": "Проект.\n\n[Brain Desk · правая рука владельца] Клиент — окно Brain Desk."}


def test_verified_sources_route_to_web_tools():
    for text in ("Найди на GitHub RDP-клиент и поставь его",
                 "Поищи модель на Hugging Face",
                 "Что пишут в Discord проекта про этот MCP?"):
        names, value, _ = route(text)
        assert "web" in value["domains"], text
        assert {"fetch", "msty_web_fetch"} & names, text
        assert "browser_navigate" in names, text


def test_window_project_is_not_resolved_in_msty_registry():
    messages = [DESK_SYSTEM, {"role": "user", "content": "Собери товары и цены Amazon для Sanare Lab UK"}]
    selected, _, _ = routing.select_tools(messages, TOOLS)
    names = {tool["function"]["name"] for tool in selected}
    assert "msty_project_resolve" not in names
    # Without the window mark (Msty itself) the resolver stays.
    names_msty, _, _ = route("Собери товары и цены Amazon для Sanare Lab UK")
    assert "msty_project_resolve" in names_msty


def test_window_working_turn_always_has_web_and_connector_finder():
    tools = TOOLS + [schema("connector_search"), schema("connector_propose")]
    for text in ("решай проблему", "подключись к серверу Amazon и разверни бота"):
        selected, value, _ = routing.select_tools(
            [DESK_SYSTEM, {"role": "user", "content": text}], tools)
        names = {tool["function"]["name"] for tool in selected}
        assert {"connector_search", "connector_propose"} <= names, text
        assert {"fetch", "msty_web_fetch"} & names, text
        assert "msty_project_resolve" not in names, text
    # Msty (no window mark) keeps its narrow routing.
    selected, _, _ = routing.select_tools([{"role": "user", "content": "решай проблему"}], tools)
    assert "connector_search" not in {tool["function"]["name"] for tool in selected}


SSH = ["list-connections", "read-command", "run-command", "sftp-list", "open-session"]


def test_window_server_thread_gets_ssh_tools_even_on_short_follow_up():
    tools = TOOLS + [schema(n) for n in SSH] + [schema("connector_search"), schema("connector_propose")]
    messages = [DESK_SYSTEM,
                {"role": "user", "content": "Сервер Crin-Barbu 188.227.57.24, порт 2222, Administrator — проверь hostname"},
                {"role": "assistant", "content": "SSH-инструментов нет."},
                {"role": "user", "content": "подклбчи и сдеай реши вопрос"}]
    selected, _, _ = routing.select_tools(messages, tools)
    names = {tool["function"]["name"] for tool in selected}
    assert {"run-command", "read-command", "list-connections"} <= names
    assert len(names) <= routing.MAX_SELECTED_TOOLS
    # Msty without the window mark: no SSH projection.
    selected, _, _ = routing.select_tools(messages[1:], tools)
    assert "run-command" not in {tool["function"]["name"] for tool in selected}


def test_window_skill_request_gets_skill_and_source_tools():
    extra = ["skills_list", "skills_get", "skills_find_ready", "skills_save",
             "search_repositories", "search_code", "connector_search", "connector_propose"]
    tools = TOOLS + [schema(n) for n in extra]
    text = ("налоговое обложение, бухгалтерский учёт, юридические моменты по United Kingdom, "
            "подача декларации. Ищи и загрузи себе все актуальные скилы по этим вопросам")
    selected, _, _ = routing.select_tools([DESK_SYSTEM, {"role": "user", "content": text}], tools)
    names = {tool["function"]["name"] for tool in selected}
    assert {"skills_find_ready", "skills_save", "search_repositories", "fetch"} <= names
    assert len(names) <= routing.MAX_SELECTED_TOOLS


def test_window_skill_catalog_gets_skills_get_on_a_plain_task():
    # brain-desk #367: the window lists skills (id, name, «когда применять»)
    # in its system message; a plain task without the word «навык» must still
    # let the model open the fitting skill with skills_get.
    extra = ["skills_list", "skills_get", "skills_find_ready", "skills_save"]
    tools = TOOLS + [schema(n) for n in extra]
    catalog = {"role": "system", "content": DESK_SYSTEM["content"] + "\n\n[Brain Desk · каталог навыков] …\n"
               "- msty:taxes-compliance — taxes-compliance: Налоги и сроки."}
    text = "У моей UK Ltd прибыль £120 000. Посчитай корпоративный налог и сроки CT600."
    selected, _, _ = routing.select_tools([catalog, {"role": "user", "content": text}], tools)
    names = {tool["function"]["name"] for tool in selected}
    assert "skills_get" in names
    assert "skills_save" not in names  # writes stay on the intent routes
    # No catalog in the window's system message — no extra schema.
    selected, _, _ = routing.select_tools([DESK_SYSTEM, {"role": "user", "content": text}], tools)
    assert "skills_get" not in {tool["function"]["name"] for tool in selected}
    # Msty without the window mark: the catalog mark alone does nothing.
    foreign = {"role": "system", "content": "[Brain Desk · каталог навыков] …"}
    selected, _, _ = routing.select_tools([foreign, {"role": "user", "content": text}], tools)
    assert "skills_get" not in {tool["function"]["name"] for tool in selected}
