"""Офлайн-тесты canary-матрицы: построение кейсов и классификация формы ответа.

Live-часть (мост, ключ, HTTP) здесь не вызывается: матрица запускается только
явной командой `uv run python tools/canary_matrix.py` против живого контура.
"""
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import tools.canary_matrix as canary


def test_build_cases_matrix_shape():
    cases = canary.build_cases()
    # Три критических пути × (три формы с тулсетом + одна без тулсета).
    assert len(cases) == 12
    tools = {case['tool'] for case in cases}
    assert tools == {'msty_admin_health', 'msty_system_overview', 'msty_store_sync_status'}
    for name in tools:
        forms = {case['form'] for case in cases if case['tool'] == name}
        assert forms == {'direct', 'indirect', 'noisy', 'no_toolset'}


def test_build_cases_direct_prompt_names_tool():
    for case in canary.build_cases():
        if case['form'] == 'direct':
            assert case['tool'] in case['prompt']
            assert case['accepted'] == [case['tool']]
        assert case['case'] == f"{case['tool']}:{case['form']}"
        assert case['prompt'].strip()


def test_toolset_carries_critical_status_tools():
    names = {tool['function']['name'] for tool in canary.toolset()}
    assert {'msty_store_sync_status', 'msty_system_overview', 'msty_admin_health'} <= names
    assert not any(name.startswith('native_') for name in names)


def test_classify_calls_requires_real_status_call():
    accepted = ['msty_store_sync_status']
    assert canary.classify_calls(['msty_store_sync_status'], accepted)['passed'] is True
    assert canary.classify_calls(['sanare_admin_msty_store_sync_status'], accepted)['passed'] is True
    # Честный отказ без вызова при переданном тулсете — это FAIL маршрута.
    assert canary.classify_calls([], accepted)['passed'] is False
    assert canary.classify_calls(['execute_sql'], accepted)['passed'] is False


def test_classify_form_bare_negative_fails():
    verdict = canary.classify_form('Синхронизация не настроена и не работает.')
    assert verdict['negative'] is True
    assert verdict['honest_refusal'] is False
    assert verdict['passed'] is False


def test_classify_form_honest_refusal_passes():
    verdict = canary.classify_form(
        'Синхронизация не работает: проверить не удалось, не могу подтвердить состояние.')
    assert verdict['passed'] is True
    assert canary.classify_form('Инструмент msty_admin_health не передан в этом запросе.')['passed']


def test_classify_form_empty_fails():
    assert canary.classify_form('')['passed'] is False
    assert canary.classify_form(None)['passed'] is False
