"""Zero-model per-turn tool and prompt routing for the native Msty agent.

The Msty client may attach the owner's complete Toolset so the user never has to
choose a connector manually.  Only a small request-relevant projection is sent
to the provider.  Selection is deterministic: it adds no classifier request,
does not grant a tool that the client did not supply, and keeps schemas needed
by historical tool-use messages for provider compatibility.
"""
from __future__ import annotations

from copy import deepcopy
import hashlib
import re
from typing import Any


ROUTE_VERSION = 2
MAX_SELECTED_TOOLS = 28

_CONTINUATION = re.compile(
    r"(?is)^\s*(?:да+|ok|ок(?:ей)?|продолжай(?:те)?|делай(?:те)?|доделывай(?:те)?|"
    r"исправляй(?:те)?|впер[её]д|дальше)\s*[.!?]*\s*$"
)
_INCIDENT = re.compile(
    r"(?is)(?:не\s+работ|сломал|сломано|ошибк|сбой|пропал|не\s+приход|не\s+синхрон|"
    r"не\s+отправ|не\s+откры|failed|failure|error|broken|outage|incident|missing)"
)
_MUTATION = re.compile(
    r"(?is)(?:исправ|почин|сдела|созда|добав|обнов|настро|подключ|перенес|перенос|"
    r"удал|убер|замен|измени|опубли|запуст|останов|отправ|запиш|внес|включ|выключ|"
    r"fix|repair|create|add|update|configure|connect|delete|remove|replace|change|"
    r"deploy|publish|run|stop|send|write|enable|disable)"
)
_READ_ACTION = re.compile(
    r"(?is)(?:проверь|провер|найди|покажи|прочитай|открой|посмотри|проанализ|"
    r"перечисл|сравни|диагност|аудит|статус|доступ|последн|сколько|есть\s+ли|"
    r"check|find|show|read|open|inspect|analy[sz]e|list|compare|diagnos|audit|"
    r"status|access|latest|how\s+many)"
)
_EXPLAIN_ONLY = re.compile(
    r"(?is)^\s*(?:объясни|расскажи|что\s+такое|для\s+чего|как\s+работает|"
    r"explain|tell\s+me|what\s+is|how\s+does)"
)
_BROWSER_INTERACTION = re.compile(
    r"(?is)(?:клик|нажм|заполни|введи|выбери|загрузи\s+файл|click|fill|type|select|upload)"
)
_REGISTERED_SITE_EXECUTOR = re.compile(
    # The job id the site executor itself hands back ("site-<32 hex>") is how the
    # owner names a job in the next turn ("опубликуй job site-c7228..."). Without
    # it such a turn missed the bounded site bundle and got the generic browser
    # and filesystem tools instead — no msty_site_release, no way to publish.
    r"(?is)(?:app\.sanaredev\.com|msty_site_|sanare-dev-v3|\bsite-[0-9a-f]{8,}\b)"
)
_AUTONOMOUS_EXECUTION = re.compile(
    r"(?is)(?:автоном|полностью|под\s+ключ|до\s+(?:конца|результат)|"
    r"end[- ]?to[- ]?end|(?:разработ|реализ|внедр|исправ|почин|настро|подключ)\w*"
    r".{0,180}(?:провер|тест|запуст|собер|опубли)|"
    r"(?:сложн|масштаб|архитектур|вся\s+систем|всю\s+систем).{0,180}"
    r"(?:сдела|исправ|провер|реализ|анализ))"
)

_DOMAIN_RULES: tuple[tuple[str, re.Pattern[str]], ...] = (
    ("pressable", re.compile(
        r"(?is)(?:pressable|wordpress|woocommerce|wp[-_ ]?cli|wp-admin|sanarelab\.club|cron)")),
    ("supabase", re.compile(
        r"(?is)(?:supabase|postgres|sql\b|database|баз[аеуы]\s+данных|таблиц|schema|"
        r"project[_ -]?id|edge\s+function|migration|миграц)")),
    ("sites", re.compile(
        r"(?is)(?:https?://|\b(?:site|website|vercel|domain|dns|frontend|page|deploy)\b|"
        r"сайт|страниц|домен|витрин|app\.sanaredev\.com|sanarelab\.|2the\.life|2thelife)")),
    ("browser", re.compile(
        r"(?is)(?:браузер|browser|клик|нажм|форма|поле|вкладк|скриншот|screenshot|ui\b)")),
    ("files", re.compile(
        r"(?is)(?:\b(?:file|folder|path|repo|repository|code|git|github|python|typescript|"
        r"javascript|json|yaml|markdown)\b|файл|папк|путь|репозитор|код|коммит|ветк)")),
    ("communications", re.compile(
        r"(?is)(?:почт|email|e-mail|gmail|outlook|письм|сообщени|переписк|slack|teams)")),
    ("commerce", re.compile(
        r"(?is)(?:amazon|shopify|магазин|товар|заказ|каталог|листинг|commerce|paypal|"
        r"плат[её]ж|checkout|vk\s*commerce)")),
    ("tax", re.compile(
        r"(?is)(?:налог|tax(?:es)?\b|vat\b|ндс|комплаенс|compliance|декларац|обязательств)")),
    ("content", re.compile(
        r"(?is)(?:блог|контент|стать|публикац|ghost|content|blog|post\b|seo\b)")),
    ("brain", re.compile(
        r"(?is)(?:\bmsty\b|langgraph|langsmith|\bbrain\b|мозг|агент|оркестрац|toolset|"
        r"тулсет|prompt|промпт|skill|скилл|памят|middleware|маршрут)")),
    ("web", re.compile(
        r"(?is)(?:интернет|web\b|веб|url\b|ссылк|онлайн|search\s+web|browse)")),
)

_CORE_READ = {
    "msty_admin_memory_search", "msty_admin_route_request", "msty_project_resolve",
}
_TASK = {"msty_task_plan", "msty_task_verify", "msty_project_verify_result"}
_SITE = {
    "msty_site_prepare", "msty_site_status", "msty_site_file", "msty_site_patch",
    "msty_site_check", "msty_site_release", "msty_site_cancel", "msty_vercel_runtime_logs",
}
_PRESSABLE = {"discover_tools", "describe_tool", "execute_tool"}
_SUPABASE_READ = {
    "search_docs", "list_projects", "get_project", "list_tables", "list_migrations",
    "get_advisors", "query_logs", "get_project_url", "execute_sql",
}
_SUPABASE_WRITE = {
    "apply_migration", "deploy_edge_function", "create_branch", "delete_branch", "merge_branch",
    "rebase_branch", "reset_branch", "create_project", "pause_project", "restore_project",
}
_BROWSER_READ = {
    "browser_navigate", "browser_snapshot", "browser_console_messages",
    "browser_network_requests", "browser_take_screenshot", "browser_wait_for",
}
_BROWSER_WRITE = {
    "browser_click", "browser_file_upload", "browser_fill_form", "browser_press_key",
    "browser_select_option", "browser_type",
}
_FILES_READ = {
    "get_file_info", "list_directory", "read_file", "read_multiple_files",
    "read_text_file", "search_files",
}
_FILES_WRITE = {"create_directory", "edit_file", "move_file", "write_file"}
_WEB = {"fetch", "msty_web_fetch"}
_BRAIN_READ = {
    "msty_admin_health", "msty_admin_keys_health", "msty_admin_last_repair",
    "msty_admin_system_map", "msty_system_overview", "msty_store_sync_status",
    "msty_brain_lessons", "msty_self_skills", "msty_selfimprove_status",
}
_BRAIN_WRITE = {
    "msty_admin_plan_repair", "msty_admin_apply_repair", "msty_selfimprove_prepare",
    "msty_selfimprove_file", "msty_selfimprove_patch", "msty_selfimprove_check",
    "msty_selfimprove_release", "msty_worker_start", "msty_worker_status",
    "msty_worker_cancel", "msty_brain_job", "msty_brain_verify", "msty_brain_consult",
}
_CODEX = {"msty_codex_start", "msty_codex_status", "msty_codex_cancel"}
_SELFIMPROVE = {
    "msty_selfimprove_status", "msty_selfimprove_file", "msty_selfimprove_patch",
    "msty_selfimprove_check", "msty_selfimprove_release",
}
_WORKER = {"msty_worker_status", "msty_worker_cancel"}
_BRAIN_JOB = {"msty_brain_job", "msty_brain_verify"}

# Every executor hands back an id of the form "<prefix>-<32 hex>" and the policy
# tells the model to resume long work by that id. Such a turn usually carries no
# action verb and no domain word ("статус job codex-9f2a...", "отмени
# worker-11aa..."), so it used to classify as small talk and reach the model
# with no way to reach the running job at all. Naming a job is itself the
# request: give that executor's continuation tools, never its start tool.
_JOB_BUNDLES: tuple[tuple[re.Pattern[str], frozenset[str]], ...] = (
    (re.compile(r"(?i)\bsite-[0-9a-f]{8,32}\b"), frozenset(_SITE)),
    (re.compile(r"(?i)\bself-[0-9a-f]{8,32}\b"), frozenset(_SELFIMPROVE)),
    (re.compile(r"(?i)\bcodex-[0-9a-f]{8,32}\b"), frozenset({"msty_codex_status", "msty_codex_cancel"})),
    (re.compile(r"(?i)\bworker-[0-9a-f]{8,32}\b"), frozenset(_WORKER)),
    (re.compile(r"(?i)\bbrain-[0-9a-f]{8,32}\b"), frozenset(_BRAIN_JOB)),
)
_KNOWN_ONLY = {
    # Installed connector operations that are intentionally not in a default
    # route. They remain selectable by exact name but never leak in through a
    # generic lexical match such as "project" or "file".
    "list_organizations", "get_organization", "list_extensions", "get_publishable_keys",
    "get_edge_function", "list_edge_functions", "list_branches",
    "generate_typescript_types", "get_cost", "confirm_cost",
    "browser_close", "browser_drag", "browser_drop", "browser_emulate_media",
    "browser_evaluate", "browser_find", "browser_handle_dialog", "browser_hover",
    "browser_navigate_back", "browser_resize", "browser_run_code_unsafe", "browser_tabs",
    "directory_tree", "list_allowed_directories", "list_directory_with_sizes",
    "read_media_file", "msty_image_read", "msty_project_create", "msty_project_read",
    "msty_projects_list", "msty_brain_verify",
}
_KNOWN = set().union(
    _CORE_READ, _TASK, _SITE, _PRESSABLE, _SUPABASE_READ, _SUPABASE_WRITE,
    _BROWSER_READ, _BROWSER_WRITE, _FILES_READ, _FILES_WRITE, _WEB,
    _BRAIN_READ, _BRAIN_WRITE, _CODEX, _KNOWN_ONLY,
)
# Which tools survive the MAX_SELECTED_TOOLS cap must not depend on the order
# Msty happens to send its schemas in: the same request would otherwise get a
# working toolset or a crippled one at random. Rank by what the step needs
# first — resolvers that unlock everything else, then the bounded executors,
# then reads, then writes — and break ties by name so the client never decides.
_TRUNCATION_TIERS: tuple[frozenset[str], ...] = (
    frozenset(_CORE_READ),
    frozenset(_SITE | _SELFIMPROVE | _WORKER | _BRAIN_JOB | _CODEX | _PRESSABLE),
    frozenset(_TASK | _SUPABASE_READ | _BRAIN_READ),
    frozenset(_FILES_READ | _BROWSER_READ | _WEB),
    frozenset(_SUPABASE_WRITE | _BRAIN_WRITE | _FILES_WRITE | _BROWSER_WRITE),
)


def _truncation_rank(name: str, lowered: str) -> tuple[int, str]:
    if name.lower() in lowered:
        return (0, name)
    for index, tier in enumerate(_TRUNCATION_TIERS, start=1):
        if name in tier or any(name.endswith("_" + member) for member in tier):
            return (index, name)
    return (len(_TRUNCATION_TIERS) + 1, name)


_TOKEN_STOP = {
    "this", "that", "with", "from", "have", "your", "tool", "tools", "project",
    "используй", "нужно", "надо", "этот", "этого", "чтобы", "который", "через",
}


def _content_text(content: Any) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for item in content:
            if isinstance(item, str):
                parts.append(item)
            elif isinstance(item, dict) and isinstance(item.get("text"), str):
                parts.append(item["text"])
        return "\n".join(parts)
    return ""


def latest_user_text(messages: list[Any]) -> str:
    for message in reversed(messages):
        role = message.get("role") if isinstance(message, dict) else getattr(message, "type", None)
        if role in {"user", "human"}:
            content = message.get("content") if isinstance(message, dict) else message.content
            return _content_text(content).strip()
    return ""


def _historical_tool_names(messages: list[Any]) -> set[str]:
    names: set[str] = set()
    for message in messages:
        calls = message.get("tool_calls") if isinstance(message, dict) else getattr(message, "tool_calls", None)
        for call in calls or []:
            if not isinstance(call, dict):
                continue
            name = call.get("name") or (call.get("function") or {}).get("name")
            if isinstance(name, str) and name:
                names.add(name)
    return names


def _tool_name(tool: dict) -> str:
    function = tool.get("function") if isinstance(tool, dict) else None
    return function.get("name", "") if tool.get("type") == "function" and isinstance(function, dict) else ""


def _tool_text(tool: dict) -> str:
    function = tool.get("function") if isinstance(tool, dict) else {}
    return " ".join(str(function.get(field, "")) for field in ("name", "description")).lower()


def _tokens(text: str) -> set[str]:
    return {token for token in re.findall(r"[a-zа-яё0-9_]{4,}", text.lower()) if token not in _TOKEN_STOP}


def _explicit_choice(tool_choice: Any) -> str | None:
    if not isinstance(tool_choice, dict) or tool_choice.get("type") != "function":
        return None
    function = tool_choice.get("function")
    return function.get("name") if isinstance(function, dict) else None


def _intent(text: str, domains: tuple[str, ...] | list[str] = ()) -> str:
    if _INCIDENT.search(text) or _MUTATION.search(text):
        return "mutate"
    if _READ_ACTION.search(text):
        return "read"
    if _EXPLAIN_ONLY.search(text):
        return "direct"
    # A turn that names a business domain is not small talk, even when its verb
    # is not one this router knows ("выгрузи таблицы", "налоги посчитай", "что у
    # нас с базой"). Falling through to "direct" left the whole tool selection
    # behind `intent != "direct"`, so the model was asked to act with no schemas
    # at all. Read is the safe promotion: write tools stay gated on "mutate".
    if domains:
        return "read"
    return "direct"


def _domains(text: str) -> list[str]:
    return [name for name, pattern in _DOMAIN_RULES if pattern.search(text)]


def _route_prompt(route: dict) -> str:
    domains = "+".join(route["domains"]) or "general"
    prompt = (
        f"MSTY_DYNAMIC_ROUTE_V1: intent={route['intent']}; domains={domains}; "
        f"visible_external_tools={route['selected_count']}. Маршрут, предметный контекст и "
        "инструменты этого шага выбраны автоматически. Не проси владельца выбирать Project, "
        "Toolset, skill или специальный prompt. Для обычного вопроса отвечай прямо из уже "
        "загруженной памяти; для действия используй видимый узкий инструмент и продолжай до "
        "проверенного результата."
    )
    hints = []
    if "pressable" in route["domains"]:
        hints.append("Pressable: discover→describe→execute только для нужной операции.")
    if "supabase" in route["domains"]:
        hints.append("Supabase: используй известный project_id и один узкий live-запрос.")
    if "sites" in route["domains"]:
        hints.append(
            "Сайт: используй только штатный site executor и его status/check/readback; "
            "общий msty_task_plan для site job не применим."
        )
    if "brain" in route["domains"]:
        hints.append("Brain: меняй себя только через штатный self-improve контур и проверки.")
    if "msty_codex_start" in route.get("selected_names", []):
        hints.append(
            "Широкая локальная задача: запусти ровно один msty_codex_start с исходной целью, "
            "затем жди тот же job через msty_codex_status до terminal state; не дублируй job "
            "и не заменяй его промежуточным диагнозом."
        )
    return prompt + (" " + " ".join(hints) if hints else "")


def select_tools(messages: list[Any], tools: list[dict], *, prior_route: dict | None = None,
                 tool_choice: Any = None) -> tuple[list[dict], dict, str]:
    """Return provider-visible schemas, a serializable route and its short prompt."""
    available = {_tool_name(tool): tool for tool in tools if _tool_name(tool)}
    text = latest_user_text(messages)
    fingerprint = hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]
    continuation = bool(_CONTINUATION.fullmatch(text))
    if continuation and isinstance(prior_route, dict) and prior_route.get("version") == ROUTE_VERSION:
        intent = prior_route.get("intent", "mutate")
        domains = [item for item in prior_route.get("domains", []) if isinstance(item, str)]
        chosen = {item for item in prior_route.get("selected_names", []) if item in available}
        source = "continued"
    else:
        domains = _domains(text)
        intent = _intent(text, domains)
        chosen: set[str] = set()
        source = "classified"

        # Exact tool names in the user turn always win; this also supports newly
        # installed connectors without changing this router.
        lowered = text.lower()
        chosen.update(name for name in available if name.lower() in lowered)

        # A named running job is an instruction on its own, whatever the verb.
        job_tools = {name for pattern, bundle in _JOB_BUNDLES if pattern.search(text)
                     for name in bundle}
        if job_tools:
            chosen.update(job_tools)
            if intent == "direct":
                intent = "read"

        if intent != "direct":
            # The generic task contract verifies explicit local artifact files.
            # Service-specific executors (site, Pressable, Supabase and Brain
            # self-improvement) have their own receipts and verification. Giving
            # them msty_task_plan caused the model to submit directory names such
            # as ``app``/``src/app`` to a file-only verifier and then terminate on
            # the resulting blocked contract.
            specialized = bool(set(domains) & {"sites", "pressable", "supabase", "brain"})
            if intent == "mutate" and "files" in domains and not specialized:
                chosen.update(_TASK)
            if not domains:
                chosen.update({"msty_admin_route_request", "msty_admin_memory_search"})
            if set(domains) & {"sites", "pressable", "supabase", "commerce", "tax", "content"}:
                chosen.add("msty_project_resolve")
            if re.search(r"(?is)(?:раньше|истори|памят|где\s+леж|вспомни|previous|memory)", text):
                chosen.add("msty_admin_memory_search")

            if "sites" in domains and "pressable" not in domains:
                if _REGISTERED_SITE_EXECUTOR.search(text):
                    chosen.update(_SITE | _WEB | _BROWSER_READ)
                else:
                    chosen.update(_WEB | _BROWSER_READ | _FILES_READ)
                    if intent == "mutate":
                        chosen.update(_FILES_WRITE)
            if "pressable" in domains:
                chosen.update(_PRESSABLE | {"msty_project_resolve"})
            if "supabase" in domains:
                chosen.update(_SUPABASE_READ)
                if intent == "mutate":
                    chosen.update(_SUPABASE_WRITE)
            if "files" in domains:
                chosen.update(_FILES_READ)
                if intent == "mutate":
                    chosen.update(_FILES_WRITE)
            if "browser" in domains:
                chosen.update(_BROWSER_READ)
                if intent == "mutate" and _BROWSER_INTERACTION.search(text):
                    chosen.update(_BROWSER_WRITE)
            if "web" in domains:
                chosen.update(_WEB | _BROWSER_READ)
            if set(domains) & {"communications", "commerce", "tax", "content"} and "pressable" not in domains:
                if "communications" in domains:
                    chosen.add("msty_admin_route_request")
                if set(domains) & {"commerce", "tax", "content"} and "supabase" not in domains:
                    chosen.update(_SUPABASE_READ)
                if set(domains) & {"commerce", "content"}:
                    chosen.update(_WEB)
            if "brain" in domains:
                chosen.update(_BRAIN_READ)
                if intent == "mutate":
                    chosen.update(_BRAIN_WRITE)

            # Broad outcome work belongs to the mature Codex harness.  The
            # Brain launches one persistent job and observes it; it must not
            # combine that run with the old file-by-file planner or another
            # hand-written agent loop.  Domain-specific bounded executors keep
            # priority for the registered site, Pressable, Supabase and Brain's
            # own protected self-improvement lane.
            bounded_domain = bool(set(domains) & {"pressable", "supabase", "brain"})
            registered_site = bool(_REGISTERED_SITE_EXECUTOR.search(text))
            use_codex = (
                intent == "mutate"
                and not bounded_domain
                and not registered_site
                and (
                    bool(_AUTONOMOUS_EXECUTION.search(text))
                    or ("files" in domains and len(text) >= 160)
                )
            )
            if use_codex:
                explicit_names = {name for name in chosen if name.lower() in lowered}
                chosen = explicit_names | _CODEX

            # Generic future connectors get a small lexical projection. Known
            # connectors stay policy-routed above, avoiding broad "project" hits.
            query_tokens = _tokens(text)
            scored = []
            for name, tool in available.items():
                if name in _KNOWN or name in chosen:
                    continue
                score = len(query_tokens & _tokens(_tool_text(tool)))
                if score:
                    scored.append((score, name))
            chosen.update(name for _, name in sorted(scored, reverse=True)[:4])

    historical_calls = _historical_tool_names(messages)
    historical = historical_calls & available.keys()
    # If a native virtual-file attempt was rejected, surface the corresponding
    # real external MCP operation on the next model step without broadening to
    # the complete filesystem Toolset.
    historical.update(
        name.removeprefix("native_") for name in historical_calls
        if name.startswith("native_") and name.removeprefix("native_") in available
    )
    explicit = _explicit_choice(tool_choice)
    required = {explicit} if explicit in available else set()
    if tool_choice == "required" and not chosen:
        chosen.update(_CORE_READ & available.keys())
    # Msty may namespace a tool schema by server (for example
    # sanare_admin_msty_task_plan). Preserve the same suffix convention used by
    # the execution gate instead of depending on one client-side display name.
    aliases = {
        name for name in available
        if any(name == canonical or name.endswith("_" + canonical) for canonical in chosen)
    }
    chosen.update(aliases)
    chosen = (chosen | historical | required) & available.keys()

    # `none` blocks new actions, but historical schemas remain for providers
    # that require definitions alongside earlier tool_use/tool_result blocks.
    if tool_choice == "none" or (isinstance(tool_choice, dict) and tool_choice.get("type") == "none"):
        chosen = historical

    protected = historical | required
    ordered = [name for name in available if name in chosen]
    if len(ordered) > MAX_SELECTED_TOOLS:
        keep = [name for name in ordered if name in protected]
        lowered_turn = text.lower()
        candidates = sorted((name for name in ordered if name not in protected),
                            key=lambda name: _truncation_rank(name, lowered_turn))
        keep.extend(candidates[:max(0, MAX_SELECTED_TOOLS - len(keep))])
        chosen = set(keep)
        ordered = [name for name in available if name in chosen]

    selected = [deepcopy(available[name]) for name in ordered]
    route = {
        "version": ROUTE_VERSION,
        "fingerprint": fingerprint,
        "intent": intent,
        "domains": domains,
        "source": source,
        "selected_names": ordered,
        "selected_count": len(ordered),
        "available_count": len(available),
    }
    return selected, route, _route_prompt(route)
