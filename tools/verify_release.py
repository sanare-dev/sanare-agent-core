"""Offline release gate; does not spend tokens or publish anything."""
import hashlib
import json
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]

def fingerprint():
    # Dependency locks, executable release/eval helpers, fixtures and active
    # instructions affect the release too. Never recurse into .env, .git or venv.
    paths = sorted(set([
        *ROOT.glob('src/**/*.py'), *ROOT.glob('tests/unit_tests/**/*.py'),
        *ROOT.glob('tools/**/*.py'), *ROOT.glob('tools/**/*.json'),
        *ROOT.glob('tools/**/*.md'), *ROOT.glob('.github/workflows/*.yml'),
        ROOT/'langgraph.json', ROOT/'pyproject.toml', ROOT/'uv.lock',
        ROOT/'LEARNING.md', ROOT/'README.md',
    ]))
    digest = hashlib.sha256()
    for path in paths:
        digest.update(str(path.relative_to(ROOT)).encode())
        digest.update(path.read_bytes())
    return digest.hexdigest()

def main():
    before = fingerprint()
    result = subprocess.run([sys.executable, '-m', 'pytest', '-q', 'tests/unit_tests'], cwd=ROOT)
    after = fingerprint()
    passed = result.returncode == 0 and before == after
    print(json.dumps({'passed': passed, 'source_sha256': after,
                      'scope': 'offline regression; model quality requires live evaluation',
                      'published': False}))
    return 0 if passed else 1

if __name__ == '__main__':
    raise SystemExit(main())
