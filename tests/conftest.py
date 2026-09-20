import pytest


@pytest.fixture(autouse=True)
def preserve_legacy_sonnet_cases(request, monkeypatch):
    # Existing adapter-specific regression remains explicit Sonnet. New profile
    # and routing tests exercise the real new default without this override.
    if request.node.path.name in {'test_msty.py', 'test_msty_guards.py',
                                  'test_msty_continuity.py', 'test_msty_eval.py',
                                  'test_msty_execution.py'}:
        monkeypatch.setenv('MSTY_MODEL_PROFILE', 'sonnet')


@pytest.fixture(scope="session")
def anyio_backend() -> str:
    return "asyncio"
