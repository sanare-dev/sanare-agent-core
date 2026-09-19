"""Runtime configuration regression tests."""

from deep_agent.graph import _sandbox_enabled


def test_sandbox_is_disabled_by_default(monkeypatch):
    monkeypatch.delenv("DEEP_AGENT_SANDBOX_ENABLED", raising=False)

    assert _sandbox_enabled() is False


def test_sandbox_can_be_enabled_explicitly(monkeypatch):
    monkeypatch.setenv("DEEP_AGENT_SANDBOX_ENABLED", "true")

    assert _sandbox_enabled() is True
