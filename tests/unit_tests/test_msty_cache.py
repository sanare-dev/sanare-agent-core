"""Synthetic/offline tests of provider prefix metadata and SDK usage semantics."""
from copy import deepcopy

import pytest
from anthropic.types import Usage
from langchain_anthropic.chat_models import _create_usage_metadata, _format_messages
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage

from deep_agent import msty


@pytest.fixture(autouse=True)
def isolated_cache_setting(monkeypatch):
    monkeypatch.delenv('MSTY_STATIC_CACHE', raising=False)


def test_only_builtin_policy_gets_5m_marker_project_and_history_are_unchanged():
    messages = [SystemMessage(content='POLICY'), SystemMessage(content=[
        {'type': 'text', 'text': 'PROJECT RULES'}, {'type': 'text', 'text': ''}]),
        HumanMessage(content='USER'), AIMessage(content=''),
        ToolMessage(content='TOOL RESULT', tool_call_id='call-1')]
    before = deepcopy(messages)
    prepared = msty.cache_system_prefix(messages, [])
    assert messages == before
    assert prepared[0].content == [{'type': 'text', 'text': 'POLICY',
        'cache_control': {'type': 'ephemeral', 'ttl': '5m'}}]
    assert prepared[1:] == messages[1:]


def test_real_sdk_format_preserves_entire_prompt_except_one_cache_marker():
    messages = [SystemMessage(content='POLICY'), SystemMessage(content='PROJECT'),
                HumanMessage(content='QUESTION')]
    raw_system, raw_conversation = _format_messages(messages)
    system, conversation = _format_messages(msty.cache_system_prefix(messages, []))
    assert conversation == raw_conversation
    assert system[0].pop('cache_control') == {'type': 'ephemeral', 'ttl': '5m'}
    assert system == raw_system


@pytest.mark.parametrize('content', ['SYSTEM', ['PART 1', 'PART 2']])
def test_string_system_formats_supported_by_sdk_are_cacheable(content):
    messages = [SystemMessage(content=content), HumanMessage(content='QUESTION')]
    system, _ = _format_messages(msty.cache_system_prefix(messages, []))
    assert system[-1]['cache_control'] == {'type': 'ephemeral', 'ttl': '5m'}
    assert system[-1]['text'] == (content if isinstance(content, str) else 'PART 2')


def test_existing_client_cache_controls_are_never_overwritten_or_extended():
    cached = {'type': 'text', 'text': 'EXISTING', 'cache_control': {'type': 'ephemeral', 'ttl': '1h'}}
    messages = [SystemMessage(content='POLICY'), HumanMessage(content=[cached])]
    assert msty.cache_system_prefix(messages, []) == messages
    tools = [{'type': 'function', 'function': {'name': 'inspect'},
              'cache_control': {'type': 'ephemeral'}}]
    messages = [SystemMessage(content='POLICY'), HumanMessage(content='USER')]
    assert msty.cache_system_prefix(messages, tools) == messages


def test_no_empty_marker_or_late_system_caching():
    messages = [SystemMessage(content=''), HumanMessage(content='USER'),
                SystemMessage(content='NOT A PREFIX')]
    assert msty.cache_system_prefix(messages, []) == messages


@pytest.mark.parametrize('setting', ['0', 'false', 'off'])
def test_cache_opt_out_preserves_original_messages(monkeypatch, setting):
    monkeypatch.setenv('MSTY_STATIC_CACHE', setting)
    messages = [SystemMessage(content='BUILTIN POLICY'), HumanMessage(content='USER')]
    assert msty.cache_system_prefix(messages, []) is messages


def test_variable_user_suffix_does_not_change_cached_prefix():
    system1, _ = _format_messages(msty.cache_system_prefix([
        SystemMessage(content='STABLE RULES'), HumanMessage(content='first')], []))
    system2, _ = _format_messages(msty.cache_system_prefix([
        SystemMessage(content='STABLE RULES'), HumanMessage(content='second')], []))
    assert system1 == system2


@pytest.mark.parametrize('raw,expected', [
    ({'input_tokens': 198, 'output_tokens': 6, 'cache_read_input_tokens': 16000,
      'cache_creation_input_tokens': 0},
     {'input_tokens': 16198, 'output_tokens': 6, 'total_tokens': 16204,
      'input_token_details': {'cache_read': 16000, 'cache_creation': 0}}),
    ({'input_tokens': 198, 'output_tokens': 6, 'cache_read_input_tokens': 0,
      'cache_creation_input_tokens': 16000,
      'cache_creation': {'ephemeral_5m_input_tokens': 16000, 'ephemeral_1h_input_tokens': 0}},
     {'input_tokens': 16198, 'output_tokens': 6, 'total_tokens': 16204,
      'input_token_details': {'cache_read': 0, 'cache_creation': 0,
         'ephemeral_5m_input_tokens': 16000, 'ephemeral_1h_input_tokens': 0}}),
    ({'input_tokens': 10, 'output_tokens': 2, 'cache_read_input_tokens': 50,
      'cache_creation_input_tokens': 40},
     {'input_tokens': 100, 'output_tokens': 2, 'total_tokens': 102,
      'input_token_details': {'cache_read': 50, 'cache_creation': 40}}),
])
def test_installed_sdk_usage_includes_reads_and_writes_in_total_input(raw, expected):
    assert _create_usage_metadata(Usage(**raw)) == expected
