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
    # Три критических пути манифеста × три формы запроса.
    assert len(cases) == 9
    tools = {case['tool'] for case in cases}
    assert tools == {'msty_admin_health', 'msty_system_overview', 'msty_store_sync_status'}
    for name in tools:
        forms = {case['form'] for case in cases if case['tool'] == name}
        assert forms == {'direct', 'indirect', 'noisy'}


def test_build_cases_direct_prompt_names_tool():
    for case in canary.build_cases():
        if case['form'] == 'direct':
            assert case['tool'] in case['prompt']
        assert case['case'] == f"{case['tool']}:{case['form']}"
        assert case['prompt'].strip()


def test_classify_form_data_passes():
    verdict = canary.classify_form(
        'Статус синхронизации: healthy, свежесть 210 сек, последний cron отработал.')
    assert verdict['has_data'] is True
    assert verdict['passed'] is True


def test_classify_form_bare_negative_fails():
    verdict = canary.classify_form('Инструмент не настроен и не работает.')
    assert verdict['negative'] is True
    assert verdict['honest_refusal'] is False
    assert verdict['passed'] is False


def test_classify_form_honest_refusal_passes():
    verdict = canary.classify_form(
        'Синхронизация не работает: проверить не удалось, не могу подтвердить состояние.')
    assert verdict['negative'] is True
    assert verdict['honest_refusal'] is True
    assert verdict['passed'] is True


def test_classify_form_negative_with_data_passes():
    # Отрицание отдельного компонента с фактическими данными — честный ответ.
    verdict = canary.classify_form(
        'Проверил: статус здоров, свежесть 3 мин; отдельный ключ отсутствует.')
    assert verdict['negative'] is True
    assert verdict['has_data'] is True
    assert verdict['passed'] is True


def test_classify_form_empty_fails():
    assert canary.classify_form('')['passed'] is False
    assert canary.classify_form(None)['passed'] is False
