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


PROTOCOL = 'msty-text-delta-v1'
MAX_TEXT_BYTES = 256 * 1024
KNOWN_FINISH = frozenset(('stop', 'tool_calls', 'function_call', 'end_turn', 'tool_use', 'stop_sequence',
    'max_tokens', 'length', 'model_context_window_exceeded', 'refusal', 'content_filter'))


class StreamFailure(RuntimeError):
    """Content-free failure; partial provider totals are not final usage."""


def enabled(state):
    protocol = state.get('text_stream_protocol')
    if protocol not in (None, PROTOCOL):
        raise ValueError('Неподдерживаемая версия текстового потока; генерация не запущена.')
    return protocol == PROTOCOL


def text_content(content):
    if isinstance(content, str):
        return content
    if not isinstance(content, list):
        raise StreamFailure('Неподдерживаемый текстовый поток.')
    return ''.join(block if isinstance(block, str) else block.get('text', '')
                   for block in content if isinstance(block, str) or
                   isinstance(block, dict) and block.get('type') == 'text' and isinstance(block.get('text'), str))


class TextStream:
    def __init__(self, state):
        self.buffered = bool(state.get('task_contract'))
        self.parts = []
        self.bytes = 0
        self.invalidated = False
        try:
            self.writer = get_stream_writer()
        except RuntimeError as error:
            if str(error) != 'Called get_config outside of a runnable context':
                raise
            self.writer = lambda event: None

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
                text = text_content(chunk.content)
                if text:
                    self.bytes += len(text.encode('utf-8'))
                    if self.bytes > MAX_TEXT_BYTES:
                        raise StreamFailure('Текстовый поток превысил безопасный предел.')
                    if not self.buffered:
                        self.writer({'type': 'text_delta', 'version': 1,
                                     'seq': len(self.parts), 'text': text})
                        self.parts.append(text)
            if aggregate is None:
                raise StreamFailure('Провайдер не вернул текстовый поток.')
            reason = aggregate.response_metadata.get('stop_reason', aggregate.response_metadata.get('finish_reason'))
            if not isinstance(reason, str) or reason not in KNOWN_FINISH:
                raise StreamFailure('Поток не содержит подтверждения завершения; действия не выданы.')
            return message_chunk_to_message(aggregate)
        except asyncio.CancelledError:
            self.invalidate()
            raise
        except Exception:
            self.invalidate()
            raise StreamFailure('Поток модели не завершён; действия не выданы, расход не подтверждён.') from None
