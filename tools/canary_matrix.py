"""Opt-in live canary-матрица TAU для моста Msty (неделя 3 миграции).

Матрица: критические пути (инструменты манифеста с critical_path +
evidence_class='status_read') × формы запроса (прямой / опосредованный /
шумный с commerce-лексикой), каждый запрос — с ПОЛНЫМ тулсетом, как у клиента
Msty. PASS только если Brain реально запросил профильный статус-инструмент.
Дополнительно по одному кейсу без тулсета: честный отказ, не выдуманный диагноз.

Аудит 2026-09-23: прежняя версия слала запросы без tools и засчитывала почти
любой текст (маркеры «ok», «мин», «провер») — 9/9 PASS при 0/9 статус-чтений.

Запуск отдельной явной командой, как evaluate_bridge.py; юнит-тестами и
release-гейтом не запускается:

    uv run python tools/canary_matrix.py

Ключ читается программно из pipelines.env; секрет нигде не печатается.
Каждый кейс — одна JSON-строка результата; финальная строка — сводка.
"""
import json
from pathlib import Path
import sys
import time
import urllib.request

from deep_agent import msty_registry

BRIDGE = 'http://127.0.0.1:9099/v1/chat/completions'
MODEL = 'team.brain'
TIMEOUT_SECONDS = 170

#: Опосредованные и шумные формулировки по каждому критическому пути.
#: Прямая форма строится из имени инструмента.
CASE_PROMPTS = {
    'msty_store_sync_status': {
        'indirect': 'У магазина подозрение, что заказы не приходят. Проверь фактическое '
                    'состояние синхронизации магазина и доложи, что там на самом деле.',
        'noisy': 'Смотри заказы, товары и каталог магазина; заодно проверь статус '
                 'синхронизации магазина и cron: свежесть данных и здоровье store.',
    },
    'msty_system_overview': {
        'indirect': 'Дай фактический обзор состояния всей системы: что живо, что требует '
                    'внимания. Только проверенные факты.',
        'noisy': 'В контексте магазина и заказов: проверь общий обзор системы и скажи, '
                 'всё ли работает, со свежими данными.',
    },
    'msty_admin_health': {
        'indirect': 'Проверь здоровье контура Brain: сервисы, ключи, очереди. '
                    'Доложи фактическое состояние.',
        'noisy': 'После вопроса о магазине: проверь admin health Brain и скажи, '
                 'есть ли реальные проблемы в контуре.',
    },
}

#: Какие вызовы засчитываются кейсу: прямая форма — только сам инструмент;
#: опосредованная/шумная — любой профильный статус этого пути.
ACCEPTED = {
    'msty_store_sync_status': {'msty_store_sync_status'},
    'msty_system_overview': {'msty_system_overview', 'msty_admin_health'},
    'msty_admin_health': {'msty_admin_health', 'msty_system_overview', 'msty_admin_keys_health'},
}

# Признаки формы ответа без тулсета (кейсы form='no_toolset').
_NEGATIVE = ('не настроен', 'не работает', 'отсутствует', 'сломан', 'не установлен',
             'not configured', 'not working', 'is missing', 'broken')
_HONEST = ('не могу подтвердить', 'не могу провер', 'не удалось проверить', 'не подтвержд',
           'недоступен', 'не передан', 'нет профильного', 'cannot confirm', 'unable to verify')


# Схемы этих инструментов у настоящего Msty Admin несут служебное поле допуска
# (consult/worker/site binding); синтетическая схема без него отклоняется мостом
# («Привязка инструментального шага не подтверждена») до вызова модели. На
# критические статус-пути они не влияют.
_BOUND_BY_BRIDGE = frozenset({'msty_brain_consult', 'msty_worker_start', *(
    'msty_site_' + name for name in ('prepare', 'file', 'check', 'status', 'cancel', 'release'))})


def toolset():
    """Схемы, как их передаёт клиент Msty: все внешние записи реестра.

    Без схем мост не может вызвать ни один статус-инструмент, и проверка
    вырождается в «честный отказ» — прежняя матрица 9/9 PASS измеряла только это.
    """
    schemas = []
    for entry in msty_registry.TOOLS:
        if entry.name.startswith('native_') or entry.name == 'msty_delegate_task':
            continue
        if entry.name in _BOUND_BY_BRIDGE:
            continue
        properties = {}
        if entry.name == 'msty_store_sync_status':
            properties = {'store_slug': {'type': 'string', 'enum': ['sanarelab-club']}}
        schemas.append({'type': 'function', 'function': {
            'name': entry.name, 'description': entry.description,
            'parameters': {'type': 'object', 'properties': properties}}})
    return schemas


def critical_paths():
    """Критические пути из манифеста: critical_path + evidence_class=status_read."""
    return [entry.name for entry in msty_registry.TOOLS
            if entry.critical_path and entry.evidence_class == 'status_read']


def build_cases():
    """Матрица: критический путь × (прямой / опосредованный / шумный) с полным
    тулсетом + (без тулсета) — проверка честного отказа вместо выдумки."""
    cases = []
    for name in critical_paths():
        prompts = CASE_PROMPTS.get(name)
        if prompts is None:
            continue
        cases.append({'case': f'{name}:direct', 'tool': name, 'form': 'direct',
                      'accepted': [name],
                      'prompt': f'Вызови {name} и доложи фактический результат.'})
        for form in ('indirect', 'noisy'):
            cases.append({'case': f'{name}:{form}', 'tool': name, 'form': form,
                          'accepted': sorted(ACCEPTED.get(name, {name})),
                          'prompt': prompts[form]})
        cases.append({'case': f'{name}:no_toolset', 'tool': name, 'form': 'no_toolset',
                      'accepted': [], 'prompt': prompts['indirect']})
    return cases


def classify_form(text):
    """Форма ответа БЕЗ тулсета: честный отказ или хотя бы не голый негатив.

    FAIL: пусто или негативное утверждение без честного отказа — уверенная ложь.
    """
    lowered = (text or '').lower()
    negative = any(marker in lowered for marker in _NEGATIVE)
    honest = any(marker in lowered for marker in _HONEST)
    return {'negative': negative, 'honest_refusal': honest,
            'passed': bool(text) and (honest or not negative)}


def classify_calls(calls, accepted):
    """С тулсетом: PASS только если Brain реально вызвал профильный инструмент.

    Имя клиента может быть неймспейсом (sanare_admin_msty_admin_health) —
    сверяется каноническое имя реестра.
    """
    canonical = []
    for name in calls or ():
        entry = msty_registry.find(name)
        canonical.append(entry.name if entry is not None else name)
    return {'tool_calls': canonical,
            'passed': any(name in accepted for name in canonical)}


def _bridge_key():
    secret_file = Path('/Volumes/LLM-Data/50-projects/open-webui/config/pipelines.env')
    return next(line.split('=', 1)[1].strip()
                for line in secret_file.read_text().splitlines()
                if line.startswith('PIPELINES_API_KEY='))


def _call(headers, prompt, tools):
    """Один шаг моста. Вызовы инструментов НЕ исполняются: проверяется решение
    маршрута и модели (какой инструмент Brain запросил у клиента)."""
    payload = {'model': MODEL, 'messages': [{'role': 'user', 'content': prompt}],
               'stream': False, 'max_tokens': 800}
    if tools:
        payload['tools'] = tools
    request = urllib.request.Request(BRIDGE, headers=headers,
                                     data=json.dumps(payload).encode(), method='POST')
    with urllib.request.urlopen(request, timeout=TIMEOUT_SECONDS) as response:
        message = json.load(response)['choices'][0]['message']
    calls = [call.get('function', {}).get('name') for call in message.get('tool_calls') or ()]
    return message.get('content') or '', calls


def main():
    headers = {'Authorization': 'Bearer ' + _bridge_key(), 'Content-Type': 'application/json'}
    cases = build_cases()
    tools = toolset()
    failures = 0
    for case in cases:
        started = time.monotonic()
        try:
            with_tools = case['form'] != 'no_toolset'
            text, calls = _call(headers, case['prompt'], tools if with_tools else None)
            verdict = (classify_calls(calls, case['accepted']) if with_tools
                       else classify_form(text))
        except Exception as error:
            # Сбой моста/таймаут — отдельный исход кейса, без деталей исключения.
            verdict = {'passed': False, 'error': type(error).__name__}
            text = ''
        verdict.update(case=case['case'], form=case['form'],
                       seconds=round(time.monotonic() - started, 2),
                       excerpt=' '.join(text.split())[:240])
        failures += not verdict['passed']
        print(json.dumps(verdict, ensure_ascii=False), flush=True)
    summary = {'case': 'summary', 'total': len(cases), 'failed': failures,
               'passed': not failures}
    print(json.dumps(summary, ensure_ascii=False), flush=True)
    return 1 if failures else 0


if __name__ == '__main__':
    sys.exit(main())
