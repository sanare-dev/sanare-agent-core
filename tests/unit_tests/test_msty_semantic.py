"""TAU L2 (неделя 4): семантический слой — офлайн-тесты.

Энкодер подменяется детерминированной хеш-функцией (bag-of-stems по 4096
корзинам): ни сети, ни fastembed, ни скачанной модели. Ранжирование реальное —
стемы запроса и описаний реестра совпадают только при настоящем лексическом
родстве («синхронизируются»/«синхронизации» → «синхро»).
"""
import hashlib
import re

import pytest

from deep_agent import msty_registry, msty_semantic, msty_tool_routing


class _HashEncoder:
    """Детерминированный стенд-энкодер: совместимый с fastembed интерфейс."""
    def _vector(self, text):
        vector = [0.0] * 4096
        for token in re.findall(r'[a-zа-яё0-9]+', text.lower()):
            stem = token[:6]
            bucket = int.from_bytes(hashlib.sha256(stem.encode()).digest()[:4], 'big')
            vector[bucket % 4096] += 1.0
        return vector

    def passage_embed(self, texts):
        # fastembed возвращает генераторы — фейк повторяет тот же интерфейс.
        return iter([self._vector(text) for text in texts])

    def query_embed(self, texts):
        return iter([self._vector(text) for text in texts])


class _BrokenEncoder:
    def passage_embed(self, texts):
        raise RuntimeError('onnx exploded')

    def query_embed(self, texts):
        raise RuntimeError('onnx exploded')


@pytest.fixture
def encoder(monkeypatch):
    """Установить хеш-энкодер и чистое lazy-состояние; низкий порог, потому
    что скоры bag-of-stems ниже настоящих косинусов e5."""
    msty_semantic._reset_for_tests()
    monkeypatch.setattr(msty_semantic, '_ENCODER', _HashEncoder())
    monkeypatch.setenv('MSTY_SEMANTIC_MIN_SCORE', '0.05')
    # Индекс строится прогревом (деплой), не в веб-запросе.
    assert msty_semantic.warmup()
    yield _HashEncoder
    msty_semantic._reset_for_tests()


def _tools(names):
    return [{'type': 'function', 'function': {'name': name, 'description': 'Synthetic.'}}
            for name in names]


SYNC_QUESTION = 'Почему заказы не синхронизируются с магазином?'
SYNC_TOOLS = ('msty_store_sync_status', 'execute_sql', 'discover_tools', 'list_tables')


def test_semantic_ranks_status_tool_for_sync_question(encoder):
    candidates = {entry.name for entry in msty_registry.TOOLS}
    result = msty_semantic.select(SYNC_QUESTION, candidates)
    assert result['status'] == 'ok'
    assert result['hits'][0]['name'] == 'msty_store_sync_status'
    assert all(hit['score'] >= 0.05 for hit in result['hits'])


def test_semantic_ranks_health_tool_for_health_question(encoder):
    candidates = {entry.name for entry in msty_registry.TOOLS}
    result = msty_semantic.select('Проверь здоровье контура: сервисы и ключи.', candidates)
    assert result['hits'][0]['name'] == 'msty_admin_health'


def test_semantic_top_k_and_threshold_from_env(encoder, monkeypatch):
    candidates = {entry.name for entry in msty_registry.TOOLS}
    monkeypatch.setenv('MSTY_SEMANTIC_TOP_K', '2')
    assert len(msty_semantic.select(SYNC_QUESTION, candidates)['hits']) <= 2
    monkeypatch.setenv('MSTY_SEMANTIC_MIN_SCORE', '0.99')
    assert msty_semantic.select(SYNC_QUESTION, candidates)['hits'] == []


def test_semantic_disabled_by_env(monkeypatch):
    """MSTY_SEMANTIC=off выключает слой даже при рабочем энкодере (A/B, откат)."""
    msty_semantic._reset_for_tests()
    monkeypatch.setattr(msty_semantic, '_ENCODER', _HashEncoder())
    monkeypatch.setenv('MSTY_SEMANTIC', 'off')
    result = msty_semantic.select(SYNC_QUESTION, {'msty_store_sync_status'})
    assert result['status'] == 'disabled' and result['reason'] == 'env_off'
    assert result['hits'] == []
    msty_semantic._reset_for_tests()


def test_semantic_disabled_without_encoder(monkeypatch):
    """Нет fastembed/модели → lexical-only, слой молча отключён на процесс."""
    msty_semantic._reset_for_tests()
    monkeypatch.setattr(msty_semantic, '_ENCODER', 'encoder_unavailable')
    result = msty_semantic.select(SYNC_QUESTION, {'msty_store_sync_status'})
    assert result['status'] == 'disabled' and result['reason'] == 'encoder_unavailable'
    msty_semantic._reset_for_tests()


def test_semantic_never_raises(encoder, monkeypatch):
    """Сбой ONNX/энкодера не имеет права ломать маршрут."""
    monkeypatch.setattr(msty_semantic, '_ENCODER', _BrokenEncoder())
    msty_semantic._INDEX = None
    result = msty_semantic.select(SYNC_QUESTION, {'msty_store_sync_status'})
    assert result['status'] == 'disabled'
    assert msty_semantic.select('пустой запрос', set())['status'] == 'skipped'


def test_router_adds_semantic_hits_without_losing_deterministic(encoder):
    """Инвариант L2: детерминированная проекция неизменна, семантика лишь
    добавляет. Запрос без доменной лексики: роутер даёт лишь resolver'ы,
    а msty_admin_health подмешивается семантикой (стемы «здоровье», «сервисы»,
    «ключи» совпадают с описанием записи реестра)."""
    tools = _tools(('msty_admin_health', 'msty_admin_route_request',
                    'msty_admin_memory_search', 'execute_sql'))
    # «Здоровье контура» теперь ловит детерминированный маршрут статуса
    # системы; здесь — формулировка без его лексики.
    question = 'Проверь сервисы и ключи.'
    selected, route, _ = msty_tool_routing.select_tools(
        [{'role': 'user', 'content': question}], tools)
    names = route['selected_names']
    # Детерминированная часть без изменений (resolver'ы пустого домена).
    assert 'msty_admin_route_request' in names and 'msty_admin_memory_search' in names
    # Добавление семантикой, с наблюдаемым скором.
    assert 'msty_admin_health' in names
    assert route['semantic']['status'] == 'ok'
    assert route['semantic']['hits'][0]['name'] == 'msty_admin_health'

    # Тот же запрос с запретом слоя — msty_admin_health не выбирается.
    baseline, route_off, _ = _select_with_semantic_off(question, tools)
    assert 'msty_admin_health' not in route_off['selected_names']
    assert route_off['semantic']['status'] == 'disabled'


def _select_with_semantic_off(question, tools):
    # select() читает env в момент вызова; подмена через monkeypatch нельзя
    # использовать вне теста, поэтому env ставится и снимается вручную.
    import os
    previous = os.environ.get('MSTY_SEMANTIC')
    os.environ['MSTY_SEMANTIC'] = 'off'
    try:
        selected, route, prompt = msty_tool_routing.select_tools(
            [{'role': 'user', 'content': question}], tools)
    finally:
        if previous is None:
            del os.environ['MSTY_SEMANTIC']
        else:
            os.environ['MSTY_SEMANTIC'] = previous
    return selected, route, prompt


def test_router_semantic_only_adds(encoder, monkeypatch):
    """Семантика не исключает: при пустых hits выбор идентичен lexical-only."""
    monkeypatch.setenv('MSTY_SEMANTIC_MIN_SCORE', '0.99')
    with_semantic, route_with, _ = msty_tool_routing.select_tools(
        [{'role': 'user', 'content': SYNC_QUESTION}], _tools(SYNC_TOOLS))
    assert route_with['semantic']['hits'] == []
    monkeypatch.setenv('MSTY_SEMANTIC', 'off')
    without_semantic, route_without, _ = msty_tool_routing.select_tools(
        [{'role': 'user', 'content': SYNC_QUESTION}], _tools(SYNC_TOOLS))
    assert route_without['semantic']['status'] == 'disabled'
    assert route_with['selected_names'] == route_without['selected_names']


def test_router_semantic_excludes_known_only(encoder, monkeypatch):
    """known_only не протекает через широкое (эмбеддинговое) совпадение."""
    captured = {}
    real_select = msty_semantic.select

    def spy(query_text, candidates):
        captured['candidates'] = set(candidates)
        return real_select(query_text, candidates)

    monkeypatch.setattr(msty_tool_routing.msty_semantic, 'select', spy)
    known_only = sorted(msty_registry.group('known_only'))[:2]
    msty_tool_routing.select_tools(
        [{'role': 'user', 'content': SYNC_QUESTION}],
        _tools(SYNC_TOOLS + tuple(known_only)))
    assert not (set(known_only) & captured['candidates'])


def test_route_semantic_key_always_present():
    """Continuation/direct маршруты не вызывают слой, но ключ диагностики есть."""
    _, route, _ = msty_tool_routing.select_tools(
        [{'role': 'user', 'content': 'Расскажи, что такое канонический дайджест.'}],
        _tools(SYNC_TOOLS))
    assert route['semantic']['status'] == 'skipped'
    continued, route_continued, _ = msty_tool_routing.select_tools(
        [{'role': 'user', 'content': 'Продолжай.'}], _tools(SYNC_TOOLS),
        prior_route={'version': msty_tool_routing.ROUTE_VERSION, 'intent': 'read',
                     'domains': ['commerce'], 'selected_names': ['execute_sql']})
    assert route_continued['semantic']['status'] == 'skipped'
    assert route_continued['selected_names'] == ['execute_sql']


def test_semantic_never_adds_write_tools_without_mutation_intent(encoder):
    tools = _tools(('msty_admin_health', 'execute_sql', 'apply_migration',
                    'msty_admin_route_request', 'msty_admin_memory_search'))
    _, route, _ = msty_tool_routing.select_tools(
        [{'role': 'user', 'content': 'Покажи что в базе: таблицы и миграции.'}], tools)
    hits = {hit['name'] for hit in route['semantic'].get('hits', [])}
    assert not hits & {'execute_sql', 'apply_migration'}


def test_semantic_hits_resolve_namespaced_client_names(encoder):
    candidates = {'sanare_admin_msty_store_sync_status', 'sanare_admin_msty_brain_lessons'}
    result = msty_semantic.select(SYNC_QUESTION, candidates)
    assert result['status'] == 'ok'
    assert result['hits'] and result['hits'][0]['name'] == 'sanare_admin_msty_store_sync_status'


def test_request_path_never_loads_encoder_synchronously(monkeypatch):
    msty_semantic._reset_for_tests()
    started = []
    monkeypatch.setattr(msty_semantic, '_start_background',
                        lambda: started.append(1) or 'warming')
    result = msty_semantic.select(SYNC_QUESTION, {'msty_store_sync_status'})
    assert result['status'] == 'disabled' and result['reason'] == 'warming'
    assert started == [1]
    msty_semantic._reset_for_tests()
