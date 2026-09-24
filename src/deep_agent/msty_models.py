"""Small server-profile adapter; no routing, tools execution or credential storage.

Uses the official MIT LangChain adapters, not another agent framework:
https://github.com/langchain-ai/langchain/tree/master/libs/partners/openai
https://github.com/langchain-ai/langchain/blob/master/libs/partners/openai/LICENSE
https://docs.langchain.com/oss/python/integrations/chat/openai
https://developers.openai.com/api/docs/models/gpt-6-luna
https://api-docs.deepseek.com/guides/thinking_mode/
https://github.com/openai/tiktoken/blob/main/tiktoken/model.py
https://github.com/openai/tiktoken/blob/main/LICENSE

Chat Completions is explicit. Luna reasoning=none and DeepSeek thinking=disabled
are deliberate non-reasoning profiles, not configurable client overrides.

For Sonnet, count_input uses the provider's exact counter. For text Luna, the
official tiktoken mapping for the fixed model ID tokenizes the complete wire JSON;
we add 25%, 4096 fixed, 128/message and 512/tool safety allowances. This is a
conservative admission ESTIMATE, not a mathematical upper-bound proof, exact
provider count, or billable usage. No unknown-model tokenizer fallback is used.
DeepSeek has no verified installed tokenizer: four units per UTF-8 wire byte,
plus 4096 fixed, 1024/message and 2048/tool framing allowances are charged.
Luna image admission adds the official server-enforced per-image patch envelope,
not a guessed size conversion: low<=256 patches, high<=2500, auto/original<=30000;
multiply by1.2, round up, add one rounding token and128 framing allowance/image.
https://developers.openai.com/api/docs/guides/images-vision (verified 2026-09-20).
Only the image component is a documented upper bound; combined text/schema
admission remains the above estimate, NOT an exact count or provider bill.
Image bytes/URLs are replaced only in the local counting copy, never generation.
No download, model change, Responses migration, extra API call or chars/4 occurs.
The exact /responses/input_tokens endpoint counts Responses payloads, not our
different Chat Completions wire format. DeepSeek images and Chat Completions
tool-role image blocks fail closed. The caller enforces the existing input cap.
"""
from __future__ import annotations

import asyncio
from copy import deepcopy
from dataclasses import dataclass
import json
import os
import re
from types import MappingProxyType
import tiktoken
import httpx

from langchain_anthropic import ChatAnthropic
from langchain_anthropic.chat_models import _format_messages
from langchain_core.messages import (
    AIMessage, BaseMessage, HumanMessage, ToolMessage, convert_to_messages,
    convert_to_openai_messages,
)
from langchain_openai import ChatOpenAI as _ChatOpenAI
from deep_agent import msty_gateway


class ModelAdapterError(ValueError):
    """Content-free error: never include SDK exceptions, messages or credentials."""


class ChatOpenAI(_ChatOpenAI):
    """Pinned langchain-openai1.1.11 stream converter, raw accounting only.

    The native adapter keeps normalized usage (which can default missing fields
    to zero) but not the raw stream usage. Preserve the latter for our existing
    fail-unknown validator, without changing inference payloads or nonstreaming.
    """
    def _convert_chunk_to_generation_chunk(self, chunk, default_chunk_class, base_generation_info):
        result = super()._convert_chunk_to_generation_chunk(chunk, default_chunk_class, base_generation_info)
        if result is not None and chunk.get('usage') is not None:
            result.message.response_metadata['token_usage'] = deepcopy(chunk['usage'])
        return result


@dataclass(frozen=True)
class Profile:
    provider: str
    model: str
    endpoint: str
    key_variable: str


DEFAULT_PROFILE = 'luna'
PROFILES = MappingProxyType({
    # gpt-6-luna с 2026-09-23 (решение владельца): вдвое дешевле 5.6 при том же
    # контексте 1.05M; цена сверена со страницей модели и учётом моста (v4-luna6).
    'luna': Profile('openai', 'gpt-6-luna', 'https://api.openai.com/v1', 'OPENAI_API_KEY'),
    'deepseek': Profile('deepseek', 'deepseek-flash', 'https://api.deepseek.com/v1', 'DEEPSEEK_API_KEY'),
    'sonnet': Profile('anthropic', 'claude-sonnet-4-6', 'https://api.anthropic.com', 'ANTHROPIC_API_KEY'),
    # Gateway profiles use provider-prefixed BYOK ids. They are retained for
    # explicit operator/direct use, but premium profiles are not admitted as
    # autonomous Brain leads or consultants.
    'astra': Profile('openai', 'gpt-6-astra', 'https://api.openai.com/v1', 'OPENAI_API_KEY'),
    'sol': Profile('openai', 'gpt-5.6-sol', 'https://api.openai.com/v1', 'OPENAI_API_KEY'),
    'opus': Profile('anthropic', 'claude-opus-4-8', 'https://api.anthropic.com', 'ANTHROPIC_API_KEY'),
    'fable': Profile('anthropic', 'claude-fable-5-1', 'https://api.anthropic.com', 'ANTHROPIC_API_KEY'),
})
# Server-owned allowlist for the optional analyst consultation profile. The
# model never supplies this directly: the bridge validates the parent-issued
# request field before it reaches the graph.
CONSULT_PROFILES = frozenset(('deepseek', 'astra', 'opus', 'fable'))
LEAD_PROFILES = frozenset(('luna', 'deepseek'))
COUNT_METHODS = MappingProxyType({
    'luna': 'tiktoken-admission-v1', 'deepseek': 'conservative-text-v1',
    'sonnet': 'anthropic-exact-v1',
    'astra': 'tiktoken-admission-v1', 'sol': 'tiktoken-admission-v1',
    'opus': 'anthropic-exact-v1', 'fable': 'anthropic-exact-v1',
})
COUNT_TIMEOUT_SECONDS = 20.0
LUNA_IMAGE_PATCH_LIMITS = MappingProxyType({'low': 256, 'high': 2500, 'original': 30000, 'auto': 30000})
LUNA_IMAGE_COUNT_METHOD = 'tiktoken-image-envelope-v1'
MAX_ADMISSION_IMAGES = 32
MAX_MESSAGES = 512
#: Input: client schemas one request may carry (brain-desk #183: the window
#: sends <=120 today; connectors may need more). Every one is validated here;
#: the native harness routes <=msty_tool_routing.MAX_SELECTED_TOOLS to the model.
MAX_TOOLS = 256
#: Provider cap on functions bound to ONE generation (OpenAI and DeepSeek Chat
#: Completions: 128). A legacy path that binds every client schema is refused
#: above it before any paid call instead of receiving a provider 400.
MAX_MODEL_TOOLS = 128


def _profile(profile: str) -> Profile:
    if not isinstance(profile, str) or profile not in PROFILES:
        raise ModelAdapterError('Неизвестный серверный профиль модели.')
    return PROFILES[profile]


def make_model(profile: str = DEFAULT_PROFILE, max_tokens: int = 4096):
    config = _profile(profile)
    if type(max_tokens) is not int or not 1 <= max_tokens <= 8192:
        raise ModelAdapterError('Недопустимый предел ответа модели.')
    # Explicit key + endpoint prevent generic SDK base-url environment overrides
    # from accidentally sending this provider's credential elsewhere.
    try:
        gateway = msty_gateway.overrides(profile)
    except msty_gateway.GatewayConfigurationError as error:
        raise ModelAdapterError(str(error)) from None
    key = gateway.get('api_key') or os.getenv(config.key_variable)
    if not key or not key.strip():
        raise ModelAdapterError('Ключ выбранного провайдера не настроен на сервере.')
    common = dict(model=config.model, api_key=key, base_url=config.endpoint,
                  max_tokens=max_tokens, timeout=120, max_retries=0)
    common.update(gateway)
    if gateway:
        # Keep gateway credentials and routing metadata on the fixed host.
        common.update(http_client=httpx.Client(follow_redirects=False),
                      http_async_client=httpx.AsyncClient(follow_redirects=False))
    if config.provider == 'anthropic' and not gateway:
        try:
            return ChatAnthropic(**common)
        except Exception:
            raise ModelAdapterError('Клиент выбранного провайдера не создан.') from None
    options = dict(use_responses_api=False, stream_usage=False)
    if profile == 'luna':
        options.update(reasoning_effort='none', store=False)
    elif profile == 'sol':
        # Kept for an explicit operator/direct profile only. Brain's autonomous
        # lead and consultation allowlists deliberately exclude it.
        options.update(reasoning_effort='medium', store=False)
    elif profile == 'astra':
        # gpt-6-astra has no 'none' tier; 'low' is its minimal reasoning effort.
        options.update(reasoning_effort='low', store=False)
    elif config.provider == 'anthropic':
        # Anthropic-family consult profiles ride the gateway's OpenAI-compatible
        # wire (provider-prefixed BYOK id); keep the payload minimal there.
        pass
    else:
        # ChatOpenAI 1.1.11 renames max_tokens to OpenAI's
        # max_completion_tokens; DeepSeek documents max_tokens instead.
        common.pop('max_tokens')
        options.update(extra_body={'thinking': {'type': 'disabled'}, 'max_tokens': max_tokens})
    try:
        return ChatOpenAI(**common, **options)
    except Exception:
        raise ModelAdapterError('Клиент выбранного провайдера не создан.') from None


def _canonical(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(',', ':'),
                      allow_nan=False)


def check_tools(tools: list[dict]) -> None:
    """Input contract for client schemas: <=MAX_TOOLS, unique names, finite JSON."""
    _checked_tools(tools, MAX_TOOLS)


def _checked_tools(tools: list[dict], limit: int) -> list[dict]:
    if not isinstance(tools, list) or len(tools) > limit:
        raise ModelAdapterError(f'Недопустимый список инструментов: больше {limit} схем.'
                                if isinstance(tools, list) else 'Недопустимый список инструментов.')
    result, names = deepcopy(tools), set()
    for tool in result:
        function = tool.get('function') if isinstance(tool, dict) else None
        if (not isinstance(function, dict) or tool.get('type') != 'function'
                or not isinstance(function.get('name'), str) or not function['name']
                or function['name'] in names or not isinstance(function.get('parameters', {}), dict)):
            raise ModelAdapterError('Недопустимая или неоднозначная схема инструмента.')
        names.add(function['name'])
    try:
        _canonical(result)
    except (TypeError, ValueError):
        raise ModelAdapterError('Схемы инструментов не являются конечным JSON.') from None
    return result


def _tools(profile: str, tools: list[dict]) -> list[dict]:
    """Schemas of one generation: the input contract plus the provider cap."""
    _profile(profile)
    result = _checked_tools(tools, MAX_MODEL_TOOLS)
    if profile != 'sonnet':
        for tool in result:
            # Strip provider metadata only, never a property named cache_control
            # inside the actual tool's argument schema.
            tool.pop('cache_control', None)
            tool['function'].pop('cache_control', None)
    return result


def _openai_content(message: BaseMessage):
    blocks, ids = [], {}
    if isinstance(message, AIMessage):
        if message.invalid_tool_calls:
            raise ModelAdapterError('В истории есть некорректный вызов инструмента.')
        for call in message.tool_calls:
            if not call.get('id') or call['id'] in ids or not isinstance(call.get('args'), dict):
                raise ModelAdapterError('В истории есть неоднозначный вызов инструмента.')
            ids[call['id']] = call
    if isinstance(message, ToolMessage) and message.status == 'error':
        message = message.model_copy(update={'content': _canonical({
            'is_error': True, 'content': message.content}), 'status': 'success'}, deep=True)
    if isinstance(message.content, str):
        return message
    result_blocks = 0
    for block in message.content:
        if isinstance(block, str):
            blocks.append(block)
            continue
        if not isinstance(block, dict):
            raise ModelAdapterError('Неподдерживаемый блок истории.')
        kind = block.get('type')
        copy = deepcopy(block)
        copy.pop('cache_control', None)
        if kind == 'text' and isinstance(block.get('text'), str):
            blocks.append(copy)
        elif kind == 'thinking' and isinstance(message, AIMessage) and isinstance(block.get('thinking'), str):
            # Preserve historical text as assistant content; provider signatures
            # are not portable to Chat Completions. Never treat it as a new user.
            blocks.append({'type': 'text', 'text': block['thinking']})
        elif kind == 'tool_use' and isinstance(message, AIMessage):
            if (not isinstance(block.get('id'), str) or not block['id']
                    or not isinstance(block.get('name'), str) or not block['name']
                    or not isinstance(block.get('input'), dict)):
                raise ModelAdapterError('Некорректный исторический вызов инструмента.')
            old = ids.get(block['id'])
            if old and (old['name'] != block['name'] or _canonical(old['args']) != _canonical(block['input'])):
                raise ModelAdapterError('Конфликт исторических вызовов инструмента.')
            if any(isinstance(b, dict) and b.get('type') == 'tool_use' and b.get('id') == block['id'] for b in blocks):
                raise ModelAdapterError('Повторён исторический вызов инструмента.')
            blocks.append(copy)
        elif kind == 'tool_result' and isinstance(message, HumanMessage):
            if (not isinstance(block.get('tool_use_id'), str) or not block['tool_use_id']
                    or not isinstance(block.get('content'), (str, list))):
                raise ModelAdapterError('Некорректный исторический результат инструмента.')
            result_blocks += 1
            nested = ToolMessage(content=block['content'], tool_call_id=block['tool_use_id'],
                                 status='error' if block.get('is_error') is True else 'success')
            copy['content'] = _openai_content(nested).content
            blocks.append(copy)
        elif kind in ('image', 'image_url') and isinstance(message, (HumanMessage, ToolMessage)):
            # Official converter handles Anthropic sources and common OpenAI
            # image_url blocks. Counting never downloads these resources.
            if kind == 'image_url':
                url = block.get('image_url')
                url = url.get('url') if isinstance(url, dict) else url
                if not isinstance(url, str) or not url:
                    raise ModelAdapterError('Некорректный исторический блок изображения.')
            blocks.append(copy)
        else:
            # Redacted reasoning, compaction, files/audio and unknown blocks must
            # not disappear in a profile change. Keep canonical history intact.
            raise ModelAdapterError('Этот блок истории нельзя безопасно перенести в выбранный профиль.')
    if result_blocks and result_blocks != len(blocks):
        # The stock converter reorders mixed human text/tool results. Refuse
        # rather than alter instruction ordering or tool protocol silently.
        raise ModelAdapterError('Смешанные текст и результаты инструментов требуют отдельного переноса.')
    return message.model_copy(update={'content': blocks}, deep=True)


def prepare_messages(profile: str, messages, tools: list[dict]) -> list[BaseMessage]:
    _profile(profile)
    _tools(profile, tools)
    if not isinstance(messages, (list, tuple)) or len(messages) > MAX_MESSAGES:
        raise ModelAdapterError('Недопустимая история сообщений.')
    try:
        canonical = deepcopy(convert_to_messages(messages))
        if profile == 'sonnet':
            return canonical
        prepared = [_openai_content(m) for m in canonical]
        wire = convert_to_openai_messages(prepared, text_format='string', pass_through_unknown_blocks=False)
        # The converter supports more than this adapter. Reject unknown roles,
        # residual provider-specific blocks and malformed calls, never drop them.
        for row in wire:
            if row.get('role') not in ('system', 'developer', 'user', 'assistant', 'tool'):
                raise ModelAdapterError('Неподдерживаемая роль истории.')
            content = row.get('content')
            if isinstance(content, list) and any(not isinstance(b, dict) or b.get('type') not in ('text', 'image_url') for b in content):
                raise ModelAdapterError('Неподдерживаемый формат содержимого.')
        _canonical(wire)
        return convert_to_messages(wire)
    except ModelAdapterError:
        raise
    except (ValueError, TypeError, KeyError, AttributeError):
        raise ModelAdapterError('История не перенесена: неподдерживаемый или повреждённый формат.') from None


def bind_tools(profile: str, model, tools: list[dict], choice):
    schemas = _tools(profile, tools)
    names = {t['function']['name'] for t in schemas}
    if choice is None:
        choice = 'auto'
    if isinstance(choice, dict):
        if choice == {'type': 'none'}:
            choice = 'none'
        elif choice.get('type') == 'function' and isinstance(choice.get('function'), dict):
            choice = choice['function'].get('name')
        elif choice.get('type') == 'tool' and isinstance(choice.get('name'), str):
            choice = choice['name']
        else:
            raise ModelAdapterError('Неподдерживаемый выбор инструмента.')
    if not isinstance(choice, str) or choice not in names | {'auto', 'none', 'any', 'required'}:
        raise ModelAdapterError('Выбранный инструмент не предоставлен.')
    if not schemas:
        if choice not in ('none', 'auto'):
            raise ModelAdapterError('Нельзя требовать отсутствующий инструмент.')
        return model
    if profile == 'sonnet':
        choice = {'type': 'none'} if choice == 'none' else 'any' if choice == 'required' else choice
    else:
        choice = 'required' if choice == 'any' else choice
    return model.bind_tools(schemas, tool_choice=choice)


async def count_input(profile: str, model, messages, tools: list[dict]) -> int:
    _profile(profile)
    schemas = _tools(profile, tools)
    if profile == 'sonnet':
        try:
            options = {'timeout': COUNT_TIMEOUT_SECONDS}
            system, _ = _format_messages(messages)
            if isinstance(system, list):
                options['system'] = system
            async with asyncio.timeout(COUNT_TIMEOUT_SECONDS + 1):
                count = await asyncio.to_thread(model.get_num_tokens_from_messages, messages,
                                                tools=schemas, **options)
            if type(count) is not int or count < 0:
                raise ModelAdapterError('Провайдер не вернул допустимый подсчёт контекста.')
            return count
        except ModelAdapterError:
            raise
        except Exception:
            raise ModelAdapterError('Точный подсчёт контекста не завершился; запрос не допущен.') from None
    prepared = prepare_messages(profile, messages, tools)
    wire = convert_to_openai_messages(prepared, pass_through_unknown_blocks=False)
    counting_wire, image_envelope = _image_counting_projection(profile, wire)
    payload = _canonical({'messages': counting_wire, 'tools': schemas})
    if profile == 'luna':
        def admission():
            # gpt-6-luna tiktoken ещё не сопоставляет; o200k_base сверен с API
            # 2026-09-23 (2383 локально против 2389 prompt_tokens, разница —
            # служебная обёртка сообщения), поверх — запас ×1.25 ниже.
            # Tokenizer assets use tiktoken's hash-verified public cache; prompt
            # content is tokenized locally and never sent to a counting model.
            encoding = tiktoken.get_encoding('o200k_base')
            tokens = len(encoding.encode(payload, disallowed_special=()))
            return (tokens * 5 + 3) // 4 + 4096 + 128 * len(wire) + 512 * len(schemas) + image_envelope
        try:
            async with asyncio.timeout(COUNT_TIMEOUT_SECONDS):
                return await asyncio.to_thread(admission)
        except Exception:
            raise ModelAdapterError('Локальный подсчёт контекста не завершился; запрос не допущен.') from None
    payload_bytes = len(payload.encode('utf-8'))
    return 4 * payload_bytes + 4096 + 1024 * len(wire) + 2048 * len(schemas)


def _image_counting_projection(profile, wire):
    """Return a counting-only copy and a per-image envelope, never fetch images.

    Server limits reject oversized patch inputs; URLs may change but every
    accepted image still obeys the same cap. Unsupported forms fail closed.
    """
    projected, envelope, images = [deepcopy(message) for message in wire], 0, 0
    for message in projected:
        content = message.get('content')
        if not isinstance(content, list):
            continue
        for index, block in enumerate(content):
            if not isinstance(block, dict) or block.get('type') != 'image_url':
                continue
            if profile != 'luna':
                raise ModelAdapterError('Запрос с изображениями не допущен: для этого профиля нет проверенного учёта.')
            if message.get('role') != 'user':
                raise ModelAdapterError('Изображение в результате инструмента не поддержано текущим Chat Completions '
                                        'маршрутом. Используйте текстовый результат или пользовательское вложение.')
            spec = block.get('image_url')
            if (set(block) != {'type', 'image_url'} or not isinstance(spec, dict) or
                    set(spec) - {'url', 'detail'} or not isinstance(spec.get('url'), str) or not spec['url']):
                raise ModelAdapterError('Некорректный формат изображения; бюджетный допуск не выполнен.')
            detail = spec.get('detail', 'auto')
            if not isinstance(detail, str) or detail not in LUNA_IMAGE_PATCH_LIMITS:
                raise ModelAdapterError('Неподдерживаемый уровень детализации изображения; генерация не запущена.')
            url = spec['url']
            if not (url.startswith(('https://', 'http://')) or
                    re.match(r'^data:image/(?:png|jpeg|webp|gif);base64,', url)):
                raise ModelAdapterError('Неподдерживаемый адрес или тип изображения; генерация не запущена.')
            images += 1
            if images > MAX_ADMISSION_IMAGES:
                raise ModelAdapterError('Для одного запроса разрешено не более 32 изображений; генерация не запущена.')
            patches = LUNA_IMAGE_PATCH_LIMITS[detail]
            # ceil(patches * 1.2) + documented +/-1 rounding + framing safety.
            envelope += (patches * 6 + 4) // 5 + 1 + 128
            content[index] = {'type': 'image_url', 'image_url': {
                **spec, 'url': '[image content counted separately]'}}
    return projected, envelope


def count_method(profile, messages):
    if profile == 'luna':
        for message in messages:
            content = message.content if isinstance(message, BaseMessage) else message.get('content')
            if isinstance(content, list) and any(isinstance(b, dict) and b.get('type') == 'image_url' for b in content):
                return LUNA_IMAGE_COUNT_METHOD
    return COUNT_METHODS[profile]


def checked_usage(profile: str, result: AIMessage):
    """Preserve measured tokens even if subsequent identity validation fails."""
    _profile(profile)
    if not isinstance(result, AIMessage):
        return None
    if profile == 'sonnet':
        return deepcopy(result.usage_metadata)
    metadata = result.response_metadata
    return _checked_openai_usage(metadata.get('token_usage'), result.usage_metadata,
                                 raw_present='token_usage' in metadata)


def stamp_usage(profile: str, result: AIMessage) -> AIMessage:
    """Record the selected identity; preserve all usage, including None/unknown.

    Capture provider usage before calling this function: a wrong provider model
    is an error after an actual call, not permission to settle its cost as zero.
    """
    config = _profile(profile)
    if not isinstance(result, AIMessage):
        raise ModelAdapterError('Провайдер вернул неподдерживаемое сообщение.')
    metadata = deepcopy(result.response_metadata)
    reported = metadata.get('model_name') or metadata.get('model')
    if msty_gateway.enabled() and reported is None:
        raise ModelAdapterError('LLM Gateway не подтвердил модель ответа; результат не принят.')
    if reported is not None:
        accepted = [re.escape(config.model) + r'(?:-\d{4}-\d{2}-\d{2})?']
        if profile in msty_gateway.CONSULT_WIRE and msty_gateway.enabled():
            # The gateway echoes the provider-prefixed BYOK selector verbatim
            # (live-proven 2026-09-21); accept exactly that id as well.
            accepted.append(re.escape(msty_gateway.CONSULT_WIRE[profile]) + r'(?:-\d{4}-\d{2}-\d{2})?')
        if not isinstance(reported, str) or not re.fullmatch('(?:' + '|'.join(accepted) + ')', reported):
            raise ModelAdapterError('Ответ получен от неожиданной модели; результат не принят.')
        metadata['provider_model_name'] = reported
    usage = checked_usage(profile, result)
    metadata.update(model_name=config.model, msty_model_name=config.model,
                    msty_model_profile=profile, msty_model_provider=config.provider)
    return result.model_copy(update={'response_metadata': metadata, 'usage_metadata': usage}, deep=True)


def _checked_openai_usage(raw, supplied, *, raw_present):
    def number(value):
        if type(value) is not int or value < 0:
            raise ValueError
        return value

    try:
        if raw_present:
            if not isinstance(raw, dict):
                return None
            incoming, outgoing, total = (number(raw.get(k)) for k in
                                        ('prompt_tokens', 'completion_tokens', 'total_tokens'))
            detail = raw.get('prompt_tokens_details')
            if detail is None:
                detail = {}
            output = raw.get('completion_tokens_details')
            if output is None:
                output = {}
            if not isinstance(detail, dict) or not isinstance(output, dict):
                return None
            input_detail, output_detail = {}, {}
            for source, target in (('cached_tokens', 'cache_read'), ('cache_write_tokens', 'cache_creation'), ('audio_tokens', 'audio')):
                # OpenAI SDK materializes optional known fields as None even
                # when the provider omitted them; this is not a malformed count.
                if source in detail and detail[source] is not None:
                    input_detail[target] = number(detail[source])
            if 'prompt_cache_hit_tokens' in raw:
                hit = number(raw['prompt_cache_hit_tokens'])
                if 'cache_read' in input_detail and input_detail['cache_read'] != hit:
                    return None
                input_detail['cache_read'] = hit
            if 'prompt_cache_miss_tokens' in raw:
                miss = number(raw['prompt_cache_miss_tokens'])
                if 'cache_read' not in input_detail or miss + input_detail['cache_read'] != incoming:
                    return None
            for source, target in (('reasoning_tokens', 'reasoning'), ('audio_tokens', 'audio')):
                if source in output and output[source] is not None:
                    output_detail[target] = number(output[source])
            value = {'input_tokens': incoming, 'output_tokens': outgoing, 'total_tokens': total,
                     'input_token_details': input_detail, 'output_token_details': output_detail}
        else:
            if not isinstance(supplied, dict):
                return None
            value = deepcopy(supplied)
            incoming, outgoing, total = (number(value.get(k)) for k in
                                        ('input_tokens', 'output_tokens', 'total_tokens'))
            input_detail = value.get('input_token_details') or {}
            output_detail = value.get('output_token_details') or {}
            if not isinstance(input_detail, dict) or not isinstance(output_detail, dict):
                return None
            for detail in (input_detail, output_detail):
                for count in detail.values():
                    number(count)
        if total != incoming + outgoing:
            return None
        if sum(input_detail.get(k, 0) for k in ('cache_read', 'cache_creation')) > incoming:
            return None
        if any(n > incoming for n in input_detail.values()) or any(n > outgoing for n in output_detail.values()):
            return None
        return value
    except (ValueError, TypeError):
        return None
