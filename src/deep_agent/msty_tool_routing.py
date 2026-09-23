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

from . import msty_registry, msty_semantic


ROUTE_VERSION = 2
MAX_SELECTED_TOOLS = 28

# fullmatch against a fixed list meant that any extra word broke it: "Делай, не
# спрашивай", "И фикси", "бери и делай" and "перенастраивай" all failed to be
# recognised, fell through to intent=direct and reached the model with no tools
# at all — told to carry on and handed nothing to carry on with. Detect instead
# by shape: a short turn built around a continuation verb that names no new
# domain is a continuation, however it is phrased.
_CONTINUATION_VERB = re.compile(
    r"(?is)\b(?:да+|ok|ок(?:ей)?|продолж\w*|сдела\w*|делай(?:те)?|доделыв\w*|доделай|"
    r"исправ\w*|почин\w*|чини|фикс\w*|правь|перенастра\w*|довед\w*|доводи|заверш\w*|"
    r"впер[её]д|дальше)\b"
)
MAX_CONTINUATION_CHARS = 64
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
# Общий вопрос о состоянии системы без доменного слова («Всё ли работает?»,
# «Дай обзор состояния системы», «Что сейчас с системой?») раньше не получал ни
# одного инструмента: доменов нет, intent=direct. Модель отвечала «инструментов
# нет» при переданных msty_system_overview/msty_admin_health.
# Только формулировки о системе в целом: «что лежит в папке», «всё ли в
# порядке с письмом», «что упало в цене» маршрут не получают.
_SYSTEM_STATUS = re.compile(
    r"(?is)(?:вс[её]\s+ли\s+(?:у\s+нас\s+)?(?:работает|живо|в\s+порядке|ок)"
    r"(?:\s+(?:сегодня|сейчас|с\s+систем\w*|в\s+систем\w*))?\s*[?.!]|"
    r"^\W*что\s+(?:сейчас\s+|у\s+нас\s+)?(?:упало|лежит|требует\s+внимания)\s*[?.!]?\s*$|"
    r"что\s+(?:сейчас\s+|у\s+нас\s+)?(?:с|со)\s+(?:систем|контур|сервис|инфраструктур)|"
    r"(?:обзор|состояни|статус|здоров\w*|health|overview)\W+(?:\w+\W+){0,3}?"
    r"(?:систем|контур|сервис|инфраструктур|всего\s+контур|всей\s+систем)|"
    r"is\s+everything\s+(?:ok|okay|fine|working|up)\s*[?.!]|system\s+(?:status|health|overview))"
)
_SYSTEM_STATUS_TOOLS = frozenset({"msty_system_overview", "msty_admin_health",
                                  "msty_admin_system_map"})
# «Проверь систему», «посмотри всю систему», «check the system» — тоже обзор.
_CHECK_SYSTEM = re.compile(
    r"(?is)(?:провер\w*|посмотр\w*|глянь|check|inspect)\s+(?:вс[юе]\s+|мою\s+|нашу\s+|the\s+)?"
    r"(?:систем|контур|инфраструктур|system)")

# Диспетчер инструментов (живой дефект 2026-09-23: 116 переданных схем, модели
# выдано 4, ответ «нет инструментов»). Модель получает каталог всех переданных
# схем и серверный инструмент подключения; запрошенное становится видимым со
# следующего шага этого же хода.
REQUEST_TOOL = "native_request_tools"
# «Установи / разверни / запусти бота» на Mac владельца: узкого коннектора нет,
# исполнитель — одна Codex job (терминал, файлы, браузер), не отказ.
_INSTALL = re.compile(
    r"(?is)(?:установ(?!лен|к)\w*|инсталл\w*|разверн\w*|install\w*|set\s*up|запусти|подключи|настрой)"
    r"[^.!?\n]{0,60}?(?:\bбот|\bbot|приложени|\bapp\b|пакет|package|\bmcp\b|коннектор|connector|"
    r"сервер|server|локальн\w*\s+модел|\bcli\b|утилит|программ|на\s+mac|на\s+маке)")
# Ограниченные домены со своими исполнителями: Codex там не выдаётся.
_INSTALL_EXCLUDED_DOMAINS = frozenset({"supabase", "pressable", "sites"})
# Brain меняет себя только через self-improve; «подключи бота к Msty» — не это.
_SELF_CHANGE = re.compile(r"(?is)\bbrain\b|мозг|себя")
# Вопрос об установке («Можно ли установить…?», «Какой сервер поставить?») —
# не поручение: Codex и маршрут изменения не выдаются.
_QUESTION = re.compile(
    r"(?is)\?|можно\s+ли|стоит\s+ли|как(?:ой|ую|ие|ое)\b|\bчто\s+(?:лучше\s+)?установ|"
    r"\bhow\s+(?:to|do|can)\b|\bshould\s+i\b|\bwhich\b|\bcan\s+i\b")
_INSTALL_TOOLS = frozenset({"msty_codex_start", "msty_codex_status"})


def dispatcher_enabled() -> bool:
    """Каталог + native_request_tools. Требует моста, допускающего это имя в
    серверном native-исполнении (brain_bridge NATIVE_TOOLS); до его выкладки
    выключено, иначе батч с этим вызовом падал бы на чеке моста."""
    import os
    return os.environ.get("MSTY_TOOL_DISPATCHER", "off").strip().lower() == "on"
MAX_REQUESTED = 20
_PASSTHROUGH_SUFFIX = "execute_tool"
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

# Наборы инструментов ниже ВЫВОДЯТСЯ из манифеста TAU L1 (msty_registry), который
# является единым источником истины об именах, доменах, алиасах и лексических
# гардах. Здесь остаётся только политика выбора, не перечни имён.
_CORE_READ = msty_registry.group('core_read')
_TASK = msty_registry.group('task')
_SITE = msty_registry.group('site')
_PRESSABLE = msty_registry.group('pressable')
# Статусные чтения Brain и их лексический гард живут в манифесте (поля
# lexical_triggers записей). Проекция при commerce-лексике ниже вызывает
# msty_registry.lexical_projection — это перенос семантики 5c63fae как есть
# (дефект 2026-09-22: только execute_tool → «Unknown tool» → ложный отказ).
_STORE_SYNC_STATUS = frozenset(
    entry.name for entry in msty_registry.TOOLS
    if msty_registry.TRIGGER_STORE_SYNC in entry.lexical_triggers)
_STORE_SYNC = re.compile(msty_registry.TRIGGER_STORE_SYNC)
_SUPABASE_READ = msty_registry.group('supabase_read')
_SUPABASE_WRITE = msty_registry.group('supabase_write')
_BROWSER_READ = msty_registry.group('browser_read')
_BROWSER_WRITE = msty_registry.group('browser_write')
_FILES_READ = msty_registry.group('files_read')
_FILES_WRITE = msty_registry.group('files_write')
_WEB = msty_registry.group('web')
_BRAIN_READ = msty_registry.group('brain_read')
_BRAIN_WRITE = msty_registry.group('brain_write')
_CODEX = msty_registry.group('codex')
_SELFIMPROVE = msty_registry.group('selfimprove')
_WORKER = msty_registry.group('worker')
_BRAIN_JOB = msty_registry.group('brain_job')

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
# Установленные операции коннекторов, сознательно не входящие в маршрут по
# умолчанию (группа known_only манифеста): выбираются точным именем, но не
# протекают в выбор через широкое лексическое совпадение вроде "project"/"file".
_KNOWN_ONLY = msty_registry.group('known_only')
_KNOWN = msty_registry.routed_names()
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


def _access(name: str) -> str:
    """Класс доступа по манифесту; неизвестный инструмент считается write."""
    entry = msty_registry.find(name)
    return entry.access if entry is not None else "write"


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


def _current_turn(messages: list[Any]) -> list[Any]:
    for index in range(len(messages) - 1, -1, -1):
        message = messages[index]
        role = message.get("role") if isinstance(message, dict) else getattr(message, "type", None)
        if role in {"user", "human"}:
            return messages[index + 1:]
    return list(messages)


def _call_args(call: dict) -> dict:
    args = call.get("args")
    if args is None and isinstance(call.get("function"), dict):
        raw = call["function"].get("arguments")
        try:
            import json
            args = json.loads(raw) if isinstance(raw, str) else raw
        except ValueError:
            args = None
    return args if isinstance(args, dict) else {}


def requested_names(messages: list[Any], available: set[str]) -> set[str]:
    """Имена, которые модель сама подключила в текущем ходе.

    Источники: вызовы native_request_tools (names) и passthrough execute_tool,
    чей tool_name — прямой инструмент этого тулсета (исходный дефект
    2026-09-22: нативное имя через Pressable execute_tool → «Unknown tool»).
    """
    found: set[str] = set()
    for message in _current_turn(messages):
        calls = message.get("tool_calls") if isinstance(message, dict) else getattr(message, "tool_calls", None)
        for call in calls or []:
            if not isinstance(call, dict):
                continue
            name = call.get("name") or (call.get("function") or {}).get("name") or ""
            args = _call_args(call)
            if name == REQUEST_TOOL:
                # Только при включённом диспетчере: иначе история (или клиент)
                # не может выдать схемы в обход маршрута.
                names = args.get("names")
                if dispatcher_enabled() and isinstance(names, list):
                    found.update(item for item in names[:MAX_REQUESTED] if isinstance(item, str))
            elif name.endswith(_PASSTHROUGH_SUFFIX) and isinstance(args.get("tool_name"), str):
                # Точное имя (или неймспейс клиента поверх канонического имени
                # реестра); короткие «file»/«status»/«start» ничего не открывают.
                target = args["tool_name"]
                entry = msty_registry.find(target)
                canonical = entry.name if entry is not None and entry.name == target else None
                found.update(item for item in available
                             if item == target or canonical and item.endswith("_" + canonical))
    return found & available


def _explicit_requests(messages: list[Any]) -> set[str]:
    """Имена из native_request_tools текущего хода (только при диспетчере)."""
    if not dispatcher_enabled():
        return set()
    names: set[str] = set()
    for message in _current_turn(messages):
        calls = message.get("tool_calls") if isinstance(message, dict) else getattr(message, "tool_calls", None)
        for call in calls or []:
            if isinstance(call, dict) and (call.get("name") or (call.get("function") or {}).get("name")) == REQUEST_TOOL:
                values = _call_args(call).get("names")
                if isinstance(values, list):
                    names.update(item for item in values if isinstance(item, str))
    return names


def catalog_prompt(tools: list[dict], selected: list[str], limit: int = 160) -> str:
    """Каталог переданных, но не выданных на шаге схем: имя — короткое назначение."""
    lines = []
    for tool in tools:
        name = _tool_name(tool)
        if not name or name in selected:
            continue
        entry = msty_registry.find(name)
        description = (entry.description if entry is not None else
                       " ".join(str((tool.get("function") or {}).get("description") or "").split()))
        lines.append(f"- {name}: {description[:90]}")
        if len(lines) >= limit:
            break
    if not lines:
        return ""
    return ("MSTY_TOOL_CATALOG_V1: у тебя ЕСТЬ и другие инструменты владельца, не показанные "
            "на этом шаге. Если для задачи нужен любой из них — вызови native_request_tools "
            "с их точными именами (до 20), и они станут доступны на следующем шаге. "
            "Никогда не отвечай «нет инструментов», не проверив этот каталог. "
            "Для широкой работы на Mac (найти, установить, запустить) есть msty_codex_start, "
            "карта системы — msty_admin_system_map.\n" + "\n".join(lines))


def request_tools_schema() -> dict:
    return {"type": "function", "function": {
        "name": REQUEST_TOOL,
        "description": ("Подключить инструменты владельца из MSTY_TOOL_CATALOG_V1 по точным "
                        "именам; они станут видимы со следующего шага этого хода."),
        "parameters": {"type": "object", "properties": {
            "names": {"type": "array", "items": {"type": "string"},
                      "minItems": 1, "maxItems": MAX_REQUESTED},
            "reason": {"type": "string", "description": "Зачем нужны эти инструменты."}},
            "required": ["names"], "additionalProperties": False}}}


def select_tools(messages: list[Any], tools: list[dict], *, prior_route: dict | None = None,
                 tool_choice: Any = None) -> tuple[list[dict], dict, str]:
    """Return provider-visible schemas, a serializable route and its short prompt."""
    available = {_tool_name(tool): tool for tool in tools if _tool_name(tool)}
    requested = requested_names(messages, set(available))
    # Запросы не выходят за общий лимит шага.
    requested = set(sorted(requested)[:MAX_SELECTED_TOOLS])
    text = latest_user_text(messages)
    fingerprint = hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]
    semantic: dict | None = None  # диагностика слоя L2; только свежая классификация
    semantic_names: set[str] = set()
    continuation = (len(text) <= MAX_CONTINUATION_CHARS
                    and bool(_CONTINUATION_VERB.search(text))
                    and not _domains(text))
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

        # "Делай" with no route to continue — the thread was compacted, or this
        # is a fresh turn after a restart. Answering "не понял" is useless and
        # sitting mute with no schemas is worse, so hand over the resolver and
        # memory search: enough to find out what was being done and resume.
        if continuation:
            intent = "read"
            chosen.update(_CORE_READ & available.keys())
            source = "continuation-recovered"

        # Exact tool names in the user turn always win; this also supports newly
        # installed connectors without changing this router.
        lowered = text.lower()
        chosen.update(name for name in available if name.lower() in lowered)

        if (_INSTALL.search(text) and not set(domains) & _INSTALL_EXCLUDED_DOMAINS
                and not _SELF_CHANGE.search(text) and not _QUESTION.search(text)):
            chosen.update(_INSTALL_TOOLS)
            intent = "mutate"  # установка — изменение; гейты записи применяются
        if _SYSTEM_STATUS.search(text) or _CHECK_SYSTEM.search(text):
            chosen.update(_SYSTEM_STATUS_TOOLS)
            if intent == "direct":
                intent = "read"

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
            # Лексические гарды манифеста (перенос 5c63fae без изменения
            # поведения): запись проецируется, когда её lexical_trigger совпал с
            # текстом и её домены пересекаются с маршрутом. Для статусной тройки
            # это ровно прежнее правило «commerce в доменах + sync-лексика»:
            # в brain-маршруте эти инструменты и так входят в _BRAIN_READ.
            chosen.update(msty_registry.lexical_projection(set(domains), text))
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

            # TAU L2 (неделя 4): семантический top-K по эмбеддингам описаний
            # реестра — строгое ДОПОЛНЕНИЕ к детерминированной проекции выше.
            # Уже выбранное не дублируется; known_only не протекает через
            # широкое (теперь и эмбеддинговое) совпадение — только точное имя.
            # Слой отключён или недоступен → маршрут идентичен lexical-only.
            # Кандидаты: только read (запись выдаёт лишь детерминированный
            # маршрут по намерению; живая проверка показала msty_site_cancel/
            # patch от семантики на инцидентном вопросе), без добавок к
            # изолированному Codex-маршруту. Добавки занимают только свободное
            # место под лимитом и при усечении идут последними: семантика не
            # вытесняет детерминированный выбор.
            if not use_codex:
                candidates = {name for name in (available.keys() - chosen) - _KNOWN_ONLY
                              if _access(name) == "read"}
                room = max(0, MAX_SELECTED_TOOLS - len(chosen & available.keys()))
                semantic = msty_semantic.select(text, candidates=candidates)
                semantic = {**semantic, "hits": semantic["hits"][:room]}
                semantic_names = {hit['name'] for hit in semantic['hits']}
                chosen.update(semantic_names)

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
    # Инструмент записи, раскрытый через execute_tool, видим только при намерении
    # изменить (native_request_tools при диспетчере — явный выбор модели).
    # Через execute_tool раскрывается только чтение: запись выдаёт маршрут по
    # доменам и намерению (ревью PR #3: «установи бота» + execute_tool
    # apply_migration открывало запись Supabase вне маршрута).
    explicit = _explicit_requests(messages)
    requested = {name for name in requested if _access(name) == "read" or name in explicit}
    chosen = (chosen | historical | required | requested) & available.keys()

    # `none` blocks new actions, but historical schemas remain for providers
    # that require definitions alongside earlier tool_use/tool_result blocks.
    if tool_choice == "none" or (isinstance(tool_choice, dict) and tool_choice.get("type") == "none"):
        chosen = historical

    protected = historical | required | requested
    ordered = [name for name in available if name in chosen]
    if len(ordered) > MAX_SELECTED_TOOLS:
        keep = [name for name in ordered if name in protected]
        lowered_turn = text.lower()
        candidates = sorted((name for name in ordered if name not in protected),
                            key=lambda name: (name in semantic_names,
                                              _truncation_rank(name, lowered_turn)))
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
        "requested": sorted(requested),
        # Каталог не нужен объяснению («что такое vault») и короткой реплике
        # («привет», «спасибо»); любой вопрос о системе/данных его получает.
        "catalog": bool(intent != "direct" or requested or (
            not _EXPLAIN_ONLY.search(text) and ("?" in text or len(text) > 25))),
        # Наблюдаемость слоя L2: какие инструменты добавлены семантикой, с какими
        # скорами; 'skipped' — continuation/direct маршрут без вызова слоя.
        "semantic": semantic if semantic is not None else {"status": "skipped"},
    }
    return selected, route, _route_prompt(route)
