import importlib.util
from pathlib import Path


def test_release_fingerprint_covers_lock_helpers_fixtures_and_instructions(tmp_path, monkeypatch):
    file = Path(__file__).parents[2] / 'tools/verify_release.py'
    spec = importlib.util.spec_from_file_location('release_fingerprint_test', file)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    monkeypatch.setattr(module, 'ROOT', tmp_path)
    for name in ('langgraph.json', 'pyproject.toml', 'uv.lock', 'LEARNING.md', 'README.md',
                 'tools/check.py', 'tools/fixtures.json', 'tools/GUIDE.md',
                 '.github/workflows/regression.yml', 'src/deep_agent/msty.py'):
        path = tmp_path / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text('original')
    original = module.fingerprint()
    for name in ('uv.lock', 'LEARNING.md', 'tools/check.py', 'tools/fixtures.json',
                 '.github/workflows/regression.yml'):
        path = tmp_path / name
        path.write_text('changed')
        assert module.fingerprint() != original, name
        path.write_text('original')
    (tmp_path / '.env').write_text('private synthetic marker')
    assert module.fingerprint() == original
