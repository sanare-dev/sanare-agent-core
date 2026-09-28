"""Opt-in provisional text, never tool execution or a second model call.

Uses native MIT LangChain AIMessageChunk aggregation and LangGraph custom events:
https://reference.langchain.com/python/langgraph/config/get_stream_writer
https://github.com/langchain-ai/langchain/blob/master/libs/core/langchain_core/messages/utils.py

The existing guarded final message remains authoritative. Consumers must withhold
actions until final/checkpoint validation, and treat invalidation as failed output:
already displayed provisional text cannot be erased by an OpenAI SSE append.
"""
import asyncio

from langchain_core.messages import AIMessageChunk, message_chunk_to_message
from langgraph.config import get_stream_writer

from . import msty_models


PROTOCOL = 'msty-text-delta-v1'
MAX_TEXT_BYTES = 256 * 1024
#: Live «ход рассуждений» for the owner's window (brain-desk, owner 28.09.2026).
#: Opt-in by the bridge, independent of provisional answer text: the native
#: harness keeps answer text withheld until the guarded result, yet the model's
#: own reasoning summary is shown while it thinks. Display only: never part of
#: the answer, tool calls, history replay or acceptance.
REASONING_PROTOCOL = 'msty-reasoning-delta-v1'
MAX_REASONING_BYTES = 64 * 1024
#: A piece is published only up to its last whitespace (or when it grows past
#: this size), so a credential-shaped token is never split across two events
#: and always reaches the secret filter whole.
REASONING_FLUSH_BYTES = 2048
KNOWN_FINISH = frozenset(('stop', 'tool_calls', 'function_call', 'end_turn', 'tool_use', 'stop_sequence',
    'max_tokens', 'length', 'model_context_window_exceeded', 'refusal', 'content_filter'))


class StreamFailure(RuntimeError):
    """Content-free failure; partial provider totals are not final usage.

    `transient` — сбой транспорта провайдера (timeout/5xx/429): его учитывает
    circuit breaker. Детали исходного исключения не сохраняются.
    """
    transient = False


def enabled(state):
    protocol = state.get('text_stream_protocol')
    if protocol not in (None, PROTOCOL):
        raise ValueError('Неподдерживаемая версия текстового потока; генерация не запущена.')
    return protocol == PROTOCOL


def reasoning_enabled(state):
    protocol = state.get('reasoning_stream_protocol')
    if protocol not in (None, REASONING_PROTOCOL):
        raise ValueError('Неподдерживаемая версия потока рассуждений; генерация не запущена.')
    return protocol == REASONING_PROTOCOL


def _summary_text(value):
    parts = []
    for item in value if isinstance(value, list) else []:
        if isinstance(item, dict) and isinstance(item.get('text'), str):
            parts.append(item['text'])
    return ''.join(parts)


def reasoning_text(chunk):
    """The model's own thoughts in one stream chunk — one rule for every provider.

    OpenAI Responses (Luna): ``reasoning`` blocks with ``summary[].text``
    (``response.reasoning_summary_text.delta``) or the v0 ``additional_kwargs``
    reasoning item; LangChain v1 ``reasoning`` strings; Anthropic ``thinking``
    blocks; DeepSeek / Qwen / vLLM ``reasoning_content``; Gemini-compatible
    ``thought``. Encrypted or redacted reasoning is never text and is skipped.
    """
    parts = []
    content = chunk.content
    for block in content if isinstance(content, list) else []:
        if not isinstance(block, dict):
            continue
        kind = block.get('type')
        if kind == 'reasoning':
            parts.append(_summary_text(block.get('summary')))
            if isinstance(block.get('reasoning'), str):
                parts.append(block['reasoning'])
        elif kind == 'thinking' and isinstance(block.get('thinking'), str):
            parts.append(block['thinking'])
    extra = chunk.additional_kwargs or {}
    for key in ('reasoning_content', 'reasoning', 'thought'):
        value = extra.get(key)
        if isinstance(value, str):
            parts.append(value)
        elif key == 'reasoning' and isinstance(value, dict):
            parts.append(_summary_text(value.get('summary')))
    return ''.join(parts)


def redact_secrets(text):
    """Same credential filter as every other published surface of the graph."""
    import re
    from .msty_native import SECRET_TOKEN_PATTERN  # local: msty_native imports msty
    return re.sub(SECRET_TOKEN_PATTERN, '[REDACTED_API_KEY]', text)


class ReasoningRelay:
    """Bounded, secret-filtered ``reasoning_delta`` custom events.

    Lossy by design: a display surface, so an oversized or failed relay stops
    relaying and never fails the generation or changes its result.
    """

    def __init__(self, writer):
        self.writer = writer
        self.pending = ''
        self.seq = 0
        self.bytes = 0
        self.stopped = False

    def feed(self, text):
        if self.stopped or not text:
            return
        self.pending += text
        cut = max(self.pending.rfind(' '), self.pending.rfind('\n'))
        if cut < 0 and len(self.pending.encode('utf-8')) < REASONING_FLUSH_BYTES:
            return
        head, self.pending = ((self.pending[:cut + 1], self.pending[cut + 1:]) if cut >= 0
                              else (self.pending, ''))
        self._emit(head)

    def flush(self):
        if not self.stopped and self.pending:
            head, self.pending = self.pending, ''
            self._emit(head)

    def _emit(self, text):
        text = redact_secrets(text)
        self.bytes += len(text.encode('utf-8'))
        if self.bytes > MAX_REASONING_BYTES:
            self.stopped = True
            return
        try:
            self.writer({'type': 'reasoning_delta', 'version': 1, 'seq': self.seq, 'text': text})
        except Exception:  # noqa: BLE001 — display only; CancelledError still propagates
            self.stopped = True
            return
        self.seq += 1


def text_content(content):
    if isinstance(content, str):
        return content
    if not isinstance(content, list):
        raise StreamFailure('Неподдерживаемый текстовый поток.')
    return ''.join(block if isinstance(block, str) else block.get('text', '')
                   for block in content if isinstance(block, str) or
                   isinstance(block, dict) and block.get('type') == 'text' and isinstance(block.get('text'), str))


class TextStream:
    def __init__(self, state, *, text=True):
        # text=False: only the reasoning relay (native harness withholds text).
        self.buffered = bool(state.get('task_contract')) or not text
        self.parts = []
        self.bytes = 0
        self.invalidated = False
        try:
            self.writer = get_stream_writer()
        except RuntimeError as error:
            if str(error) != 'Called get_config outside of a runnable context':
                raise
            self.writer = lambda event: None
        self.reasoning = ReasoningRelay(self.writer) if reasoning_enabled(state) else None

    def invalidate(self):
        if self.parts and not self.invalidated:
            self.writer({'type': 'text_invalidated', 'version': 1})
            self.invalidated = True

    def finish(self, result):
        if self.parts and (text_content(result.content) != ''.join(self.parts) or
                           result.response_metadata.get('msty_blocked') is True):
            self.invalidate()

    async def invoke(self, model, messages):
        aggregate = None
        raw_usage_events = 0
        try:
            # Explicit usage is required with custom Gateway base URLs, where
            # ChatOpenAI does not enable it by default. Sonnet accepts this too.
            async for chunk in model.astream(messages, stream_usage=True):
                if not isinstance(chunk, AIMessageChunk):
                    raise StreamFailure('Провайдер вернул некорректный фрагмент потока.')
                if 'token_usage' in chunk.response_metadata:
                    raw_usage_events += 1
                    if raw_usage_events > 1:
                        raise StreamFailure('Повторён итоговый учёт потока; расход не подтверждён.')
                aggregate = chunk if aggregate is None else aggregate + chunk
                if self.reasoning is not None:
                    self.reasoning.feed(reasoning_text(chunk))
                text = text_content(chunk.content)
                if text:
                    self.bytes += len(text.encode('utf-8'))
                    if self.bytes > MAX_TEXT_BYTES:
                        raise StreamFailure('Текстовый поток превысил безопасный предел.')
                    if not self.buffered:
                        self.writer({'type': 'text_delta', 'version': 1,
                                     'seq': len(self.parts), 'text': text})
                        self.parts.append(text)
            if self.reasoning is not None:
                self.reasoning.flush()
            if aggregate is None:
                raise StreamFailure('Провайдер не вернул текстовый поток.')
            reason = msty_models.finish_reason(aggregate.response_metadata)
            if not isinstance(reason, str) or reason not in KNOWN_FINISH:
                raise StreamFailure('Поток не содержит подтверждения завершения; действия не выданы.')
            return message_chunk_to_message(aggregate)
        except asyncio.CancelledError:
            self.invalidate()
            raise
        except StreamFailure:
            self.invalidate()
            raise
        except Exception as error:
            self.invalidate()
            from . import msty_taxonomy  # локально: без цикла импорта
            failure = StreamFailure('Поток модели не завершён; действия не выданы, расход не подтверждён.')
            failure.transient = msty_taxonomy.is_transient_exception(error)
            failure.policy_rejection = msty_taxonomy.policy_rejection_code(error)
            failure.emitted = bool(self.parts)
            raise failure from None
