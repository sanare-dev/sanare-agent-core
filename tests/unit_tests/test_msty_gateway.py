"""Offline tests only: no credentials, real network or model calls."""
import json
import asyncio
import httpx
import pytest
from langchain_core.messages import HumanMessage
from langchain_core.messages import AIMessage
from deep_agent import msty_gateway as gateway, msty_models as models


@pytest.fixture(autouse=True)
def settings(monkeypatch):
    monkeypatch.setenv('MSTY_LLM_GATEWAY_ENABLED', '1')
    monkeypatch.setenv('LANGSMITH_GATEWAY_API_KEY', 'synthetic-gateway-only')
    monkeypatch.setenv('MSTY_LLM_GATEWAY_DEEPSEEK_CONFIG_ID', gateway.DEEPSEEK_CONFIG_ID)
    monkeypatch.setenv('LANGSMITH_TRACING', 'false')
    for name in ('OPENAI_API_KEY', 'DEEPSEEK_API_KEY', 'ANTHROPIC_API_KEY'):
        monkeypatch.delenv(name, raising=False)


@pytest.mark.parametrize('profile,path,selector', [
    ('luna', '/openai/v1', 'gpt-6-luna'),
    ('deepseek', '/v1', gateway.DEEPSEEK_SELECTOR),
])
def test_fixed_endpoint_key_headers_and_wire(profile, path, selector, monkeypatch):
    monkeypatch.setenv('OPENAI_BASE_URL', 'https://untrusted.invalid')
    monkeypatch.setenv('LANGSMITH_GATEWAY_BASE_URL', 'https://untrusted.invalid')
    model = models.make_model(profile, 100)
    seen = []
    async def respond(request):
        seen.append(request)
        return httpx.Response(200, json={'id':'synthetic', 'object':'chat.completion',
            'created':1, 'model':models.PROFILES[profile].model,
            'choices':[{'index':0,'message':{'role':'assistant','content':'OK'},'finish_reason':'stop'}],
            'usage':{'prompt_tokens':3,'completion_tokens':1,'total_tokens':4}})
    async def run():
        async with httpx.AsyncClient(transport=httpx.MockTransport(respond)) as client:
            model.root_async_client._client = client
            return await model.ainvoke([HumanMessage(content='Synthetic probe')])
    reply = asyncio.run(run())
    assert len(seen) == 1
    request = seen[0]
    assert str(request.url) == gateway.HOST + path + '/chat/completions'
    assert request.headers['authorization'] == 'Bearer synthetic-gateway-only'
    assert request.headers['x-tenant-id'] == gateway.WORKSPACE
    assert request.headers['x-gateway-app'] == 'sanare-msty'
    wire = json.loads(request.content)
    assert wire['model'] == selector
    assert model.max_retries == 0
    assert model.use_responses_api is False
    assert model.http_client.follow_redirects is False
    assert model.http_async_client.follow_redirects is False
    if profile == 'luna':
        assert wire['reasoning_effort'] == 'none' and wire['store'] is False
    else:
        assert wire['thinking'] == {'type':'disabled'}
        assert wire['max_tokens'] == 100
    assert models.stamp_usage(profile, reply).usage_metadata['total_tokens'] == 4


def test_missing_gateway_key_never_uses_available_provider_key(monkeypatch):
    monkeypatch.delenv('LANGSMITH_GATEWAY_API_KEY')
    monkeypatch.setenv('OPENAI_API_KEY', 'synthetic-direct-must-not-be-used')
    with pytest.raises(models.ModelAdapterError, match='обход запрещён'):
        models.make_model('luna')


@pytest.mark.parametrize('value', ['', 'true', 'yes', 'https://untrusted.invalid'])
def test_bad_switch_fails_closed(value, monkeypatch):
    monkeypatch.setenv('MSTY_LLM_GATEWAY_ENABLED', value)
    with pytest.raises(models.ModelAdapterError):
        models.make_model('luna')


def test_unknown_config_and_expensive_profile_fail_closed(monkeypatch):
    monkeypatch.setenv('MSTY_LLM_GATEWAY_DEEPSEEK_CONFIG_ID', 'wrong')
    for profile in ('deepseek', 'sonnet'):
        with pytest.raises(models.ModelAdapterError):
            models.make_model(profile)


def test_explicit_operator_rollback(monkeypatch):
    monkeypatch.setenv('MSTY_LLM_GATEWAY_ENABLED', '0')
    monkeypatch.setenv('OPENAI_API_KEY', 'synthetic-direct')
    assert models.make_model('luna').openai_api_base == 'https://api.openai.com/v1'


@pytest.mark.parametrize('status', [301,401,402,403,429,500,503])
def test_gateway_failure_is_one_attempt_no_fallback(status):
    model=models.make_model('luna',32)
    seen=[]
    async def respond(request):
        seen.append(str(request.url))
        return httpx.Response(status,headers={'Location':'https://untrusted.invalid'},
                              json={'error':{'message':'synthetic failure','type':'test'}})
    async def run():
        async with httpx.AsyncClient(transport=httpx.MockTransport(respond),follow_redirects=False) as client:
            model.root_async_client._client=client
            with pytest.raises(Exception):
                await model.ainvoke([HumanMessage(content='synthetic')])
    asyncio.run(run())
    assert seen == [gateway.HOST+'/openai/v1/chat/completions']


def test_gateway_requires_reported_model_identity():
    reply=AIMessage(content='OK',usage_metadata={'input_tokens':1,'output_tokens':1,'total_tokens':2})
    with pytest.raises(models.ModelAdapterError,match='не подтвердил модель'):
        models.stamp_usage('luna',reply)


@pytest.mark.parametrize('profile,selector', [
    ('astra', 'openai/gpt-6-astra'),
    ('sol', 'openai/gpt-5.6-sol'),
    ('opus', 'anthropic/claude-opus-4-8'),
    ('fable', 'anthropic/claude-fable-5-1'),
])
def test_consult_profiles_wire(profile, selector, monkeypatch):
    monkeypatch.setenv('OPENAI_BASE_URL', 'https://untrusted.invalid')
    monkeypatch.setenv('LANGSMITH_GATEWAY_BASE_URL', 'https://untrusted.invalid')
    model = models.make_model(profile, 100)
    seen = []
    async def respond(request):
        seen.append(request)
        # Live-proven 2026-09-21: the gateway echoes the provider-prefixed
        # BYOK selector verbatim as the response model identity.
        return httpx.Response(200, json={'id':'synthetic', 'object':'chat.completion',
            'created':1, 'model':gateway.CONSULT_WIRE[profile],
            'choices':[{'index':0,'message':{'role':'assistant','content':'OK'},'finish_reason':'stop'}],
            'usage':{'prompt_tokens':3,'completion_tokens':1,'total_tokens':4}})
    async def run():
        async with httpx.AsyncClient(transport=httpx.MockTransport(respond)) as client:
            model.root_async_client._client = client
            return await model.ainvoke([HumanMessage(content='Synthetic probe')])
    reply = asyncio.run(run())
    assert len(seen) == 1
    request = seen[0]
    assert str(request.url) == gateway.HOST + '/v1/chat/completions'
    assert request.headers['authorization'] == 'Bearer synthetic-gateway-only'
    assert request.headers['x-tenant-id'] == gateway.WORKSPACE
    assert request.headers['x-gateway-app'] == 'sanare-msty'
    wire = json.loads(request.content)
    assert wire['model'] == selector
    assert model.max_retries == 0
    assert model.use_responses_api is False
    assert model.http_client.follow_redirects is False
    if profile == 'astra':
        # gpt-6-astra has no 'none' tier; 'low' is its minimal reasoning effort.
        assert wire['reasoning_effort'] == 'low' and wire['store'] is False
    elif profile == 'sol':
        assert wire['reasoning_effort'] == 'medium' and wire['store'] is False
    assert models.stamp_usage(profile, reply).usage_metadata['total_tokens'] == 4


@pytest.mark.parametrize('profile', ['astra', 'sol', 'opus', 'fable'])
def test_consult_identity_rejects_wrong_model(profile):
    reply = AIMessage(content='OK', response_metadata={'model_name': 'gpt-4o-mini'},
                      usage_metadata={'input_tokens': 1, 'output_tokens': 1, 'total_tokens': 2})
    with pytest.raises(models.ModelAdapterError, match='неожиданной модели'):
        models.stamp_usage(profile, reply)
