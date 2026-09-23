"""TAU, слой L2 (неделя 4): семантический retrieval поверх роутера.

Детерминированный роутер (`msty_tool_routing`) остаётся источником проекции:
critical_path, доменная и лексическая логика ничего не теряют. Этот модуль —
ДОПОЛНЕНИЕ: top-K инструментов по косинусной близости эмбеддинга запроса к
эмбеддингам описаний реестра L1. Принцип L2 соблюдён структурно: семантика
может только добавлять инструменты в loadout, никогда — исключать.

Эмбеддинги — опциональный extra (`fastembed`, ONNX-runtime, без torch).
Если зависимость не установлена, модель не скачана или слой выключен через
`MSTY_SEMANTIC=off`, роутер молча работает как раньше (lexical-only), а в
диагностике маршрута стоит `semantic: {'status': 'disabled', ...}`.

Модель по умолчанию — sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2
(384-мерные векторы, 50+ языков, включая русский). Прежний дефолт
intfloat/multilingual-e5-small в fastembed 0.8.1 не поддерживается — слой
не включился бы ни при каком деплое. Калибровка 2026-09-23 на реестре:
релевантные совпадения 0.35–0.80, шум ≤0.30, медиана ≈0.05–0.2; query ≈3 мс.

Инварианты деплоя (AGENTS.md: «веб-сервер, без обращений к ФС в рантайме»):
модуль сам файловую систему не трогает. Веб-запрос НИКОГДА не загружает
модель синхронно: первый запрос запускает фоновую загрузку (поток) и получает
`disabled/warming`, пока индекс не готов; сбой загрузки или индекса отключает
слой до конца процесса. Кеш — MSTY_SEMANTIC_CACHE_DIR (иначе стандартный кеш
fastembed). Прогрев на деплое — `msty_semantic.warmup()` (синхронно).

Конфигурация (env, как MSTY_NEGATIVE_MARKERS):
- MSTY_SEMANTIC=off — принудительно выключить слой (A/B, откат);
- MSTY_SEMANTIC_TOP_K — сколько инструментов максимум добавить (дефолт 5);
- MSTY_SEMANTIC_MIN_SCORE — минимальный косинус для добавления (дефолт 0.35);
- MSTY_SEMANTIC_MODEL — иная модель fastembed (дефолт e5-small).

Эмбеддинги описаний реестра вычисляются один раз на процесс и держатся в
памяти; инвалидация не нужна — манифест статичен в пределах процесса.
"""
from __future__ import annotations

import math
import os
import threading

from . import msty_registry


MODEL_ID = 'sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2'
DEFAULT_TOP_K = 5
DEFAULT_MIN_SCORE = 0.35

_LOCK = threading.Lock()
# Состояния энкодера: None — ещё не пробовали; объект fastembed — готов;
# строка — причина отказа, повторных попыток не будет (процесс lexical-only).
_ENCODER = None
# Индекс реестра: каноническое имя → вектор (tuple[float, ...]).
_INDEX: dict[str, tuple[float, ...]] | None = None
# Причина окончательного отказа индекса (сбой passage_embed): без повторов.
_INDEX_FAILED: str | None = None
_BACKGROUND: threading.Thread | None = None


def _env_int(name: str, default: int) -> int:
    try:
        value = int(os.environ.get(name, ''))
    except ValueError:
        return default
    return value if value > 0 else default


def _env_float(name: str, default: float) -> float:
    try:
        value = float(os.environ.get(name, ''))
    except ValueError:
        return default
    return value if 0.0 < value < 1.0 else default


def config() -> dict:
    """Действующие параметры слоя с учётом env-переопределений."""
    return {'top_k': _env_int('MSTY_SEMANTIC_TOP_K', DEFAULT_TOP_K),
            'min_score': _env_float('MSTY_SEMANTIC_MIN_SCORE', DEFAULT_MIN_SCORE),
            'model': os.environ.get('MSTY_SEMANTIC_MODEL') or MODEL_ID}


def _load_encoder_unlocked():
    """Тело ленивой инициализации; вызывать только под _LOCK."""
    global _ENCODER
    if _ENCODER is not None:
        return _ENCODER if not isinstance(_ENCODER, str) else None
    try:
        from fastembed import TextEmbedding
        cache_dir = os.environ.get('MSTY_SEMANTIC_CACHE_DIR') or None
        _ENCODER = TextEmbedding(model_name=config()['model'], cache_dir=cache_dir)
    except Exception:
        _ENCODER = 'encoder_unavailable'
        return None
    return _ENCODER


def _load_encoder():
    """Ленивая инициализация fastembed, один раз на процесс.

    Любой сбой (нет пакета, не скачалась модель, ошибка ONNX) переводит слой
    в disabled до конца процесса: веб-запрос не должен повторно дёргать сеть
    или диск. Детали исключения в ответ не уходят (content-free дисциплина).
    """
    with _LOCK:
        return _load_encoder_unlocked()


def _entry_text(entry: msty_registry.ToolEntry) -> str:
    """Текст записи для эмбеддинга: имя, описание, алиасы, домены."""
    return '. '.join([entry.name, entry.description, *entry.aliases, *entry.domains])


def _index() -> dict[str, tuple[float, ...]] | None:
    """Кеш эмбеддингов описаний реестра; None, если энкодер недоступен."""
    global _INDEX, _INDEX_FAILED
    if _INDEX is not None or _INDEX_FAILED:
        return _INDEX
    with _LOCK:
        if _INDEX is not None or _INDEX_FAILED:
            return _INDEX
        encoder = _load_encoder_unlocked()
        if encoder is None:
            return None
        texts = [_entry_text(entry) for entry in msty_registry.TOOLS]
        try:
            vectors = [tuple(float(value) for value in vector)
                       for vector in encoder.passage_embed(texts)]
        except Exception:
            _INDEX_FAILED = 'index_unavailable'
            return None
        _INDEX = {entry.name: vector for entry, vector in zip(msty_registry.TOOLS, vectors)}
        return _INDEX


def _cosine(left: tuple[float, ...], right: tuple[float, ...]) -> float:
    dot = sum(a * b for a, b in zip(left, right))
    norm = math.sqrt(sum(a * a for a in left)) * math.sqrt(sum(b * b for b in right))
    return dot / norm if norm else 0.0


def select(query_text: str, candidates: set[str]) -> dict:
    """Top-K релевантных инструментов из `candidates` по эмбеддингу запроса.

    Возвращает диагностический словарь для route['semantic']:
    - status 'ok' — hits есть (возможно пустой: всё ниже порога);
    - status 'disabled' — слой выключен или энкодер недоступен (reason);
    - status 'skipped' — пустой запрос или пустые кандидаты.
    Функция никогда не падает: семантика не имеет права ломать маршрут.
    """
    cfg = config()
    result = {'status': 'ok', 'model': cfg['model'], 'top_k': cfg['top_k'],
              'min_score': cfg['min_score'], 'hits': []}
    if os.environ.get('MSTY_SEMANTIC', '').strip().lower() == 'off':
        return {**result, 'status': 'disabled', 'reason': 'env_off'}
    if not (query_text or '').strip() or not candidates:
        return {**result, 'status': 'skipped'}
    try:
        index = _INDEX
        if index is None:
            reason = _start_background()
            return {**result, 'status': 'disabled', 'reason': reason}
        encoder = _ENCODER
        vector = tuple(float(value) for value in next(encoder.query_embed([query_text])))
        # Клиент может неймспейсить имена (sanare_admin_msty_admin_health):
        # вектор берётся по канонической записи, в hits — имя из запроса.
        scored = []
        for name in candidates:
            entry = msty_registry.find(name)
            if entry is not None and entry.name in index:
                scored.append((name, _cosine(vector, index[entry.name])))
        scored.sort(key=lambda item: (-item[1], item[0]))
        hits = [{'name': name, 'score': round(score, 4)}
                for name, score in scored if score >= cfg['min_score']][:cfg['top_k']]
        return {**result, 'hits': hits}
    except Exception:
        return {**result, 'status': 'disabled', 'reason': 'encoder_unavailable'}


def _start_background() -> str:
    """Запустить фоновую загрузку один раз; причина текущего отказа."""
    global _BACKGROUND
    if isinstance(_ENCODER, str):
        return _ENCODER
    if _INDEX_FAILED:
        return _INDEX_FAILED
    with _LOCK:
        if _BACKGROUND is None:
            _BACKGROUND = threading.Thread(target=_index, name='msty-semantic-warmup',
                                           daemon=True)
            _BACKGROUND.start()
    return 'warming'


def warmup() -> bool:
    """Прогрев на деплое: загрузить модель и построить индекс заранее."""
    return _index() is not None


def _reset_for_tests():
    """Сброс lazy-состояния; только для юнит-тестов (мок энкодера)."""
    global _ENCODER, _INDEX, _INDEX_FAILED, _BACKGROUND
    with _LOCK:
        _ENCODER = None
        _INDEX = None
        _INDEX_FAILED = None
        _BACKGROUND = None
