"""Opt-in live canary-матрица TAU для моста Msty (неделя 3 миграции).

Матрица: критические пути (инструменты манифеста с critical_path +
evidence_class='status_read') × формы запроса (прямой / опосредованный /
шумный с commerce-лексикой). Assert по ФОРМЕ ответа, не по точному тексту:
ответ обязан содержать фактические данные или честный отказ; голый негатив
(«не настроен/не работает» без признаков статус-чтения) — FAIL.

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

# Признаки формы ответа (см. classify_form).
_NEGATIVE = ('не настроен', 'не работает', 'отсутствует', 'сломан', 'не установлен',
             'not configured', 'not working', 'is missing', 'broken', 'unavailable')
_HONEST = ('не могу подтвердить', 'не удалось проверить', 'не подтвержд', 'недоступен',
           'не получилось', 'cannot confirm', 'unable to verify')
_DATA = ('статус', 'свежест', 'здоров', 'healthy', 'ok', 'cron', 'синхронизац',
         'провер', 'последн', 'timestamp', 'мин', 'сек', 'status')


def critical_paths():
    """Критические пути из манифеста: critical_path + evidence_class=status_read."""
    return [entry.name for entry in msty_registry.TOOLS
            if entry.critical_path and entry.evidence_class == 'status_read']


def build_cases():
    """Матрица кейсов: критический путь × (прямой / опосредованный / шумный)."""
    cases = []
    for name in critical_paths():
        prompts = CASE_PROMPTS.get(name)
        if prompts is None:
            continue
        cases.append({'case': f'{name}:direct', 'tool': name, 'form': 'direct',
                      'prompt': f'Вызови {name} и доложи фактический результат.'})
        for form in ('indirect', 'noisy'):
            cases.append({'case': f'{name}:{form}', 'tool': name, 'form': form,
                          'prompt': prompts[form]})
    return cases


def classify_form(text):
    """Форма финального ответа: данные/честный отказ/голый негатив.

    PASS: есть фактические данные или честный неподтверждённый отказ.
    FAIL: негативное утверждение без данных и без честного отказа — та самая
    уверенная ложь, которую TAU обязан исключить.
    """
    lowered = (text or '').lower()
    negative = any(marker in lowered for marker in _NEGATIVE)
    honest = any(marker in lowered for marker in _HONEST)
    data = any(marker in lowered for marker in _DATA)
    return {'negative': negative, 'honest_refusal': honest, 'has_data': data,
            'passed': bool(text) and (not negative or honest or data)}


def _bridge_key():
    secret_file = Path('/Volumes/LLM-Data/50-projects/open-webui/config/pipelines.env')
    return next(line.split('=', 1)[1].strip()
                for line in secret_file.read_text().splitlines()
                if line.startswith('PIPELINES_API_KEY='))


def _call(headers, prompt):
    payload = {'model': MODEL, 'messages': [{'role': 'user', 'content': prompt}],
               'stream': False, 'max_tokens': 800}
    request = urllib.request.Request(BRIDGE, headers=headers,
                                     data=json.dumps(payload).encode(), method='POST')
    with urllib.request.urlopen(request, timeout=TIMEOUT_SECONDS) as response:
        return json.load(response)['choices'][0]['message'].get('content') or ''


def main():
    headers = {'Authorization': 'Bearer ' + _bridge_key(), 'Content-Type': 'application/json'}
    cases = build_cases()
    failures = 0
    for case in cases:
        started = time.monotonic()
        try:
            text = _call(headers, case['prompt'])
            verdict = classify_form(text)
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
