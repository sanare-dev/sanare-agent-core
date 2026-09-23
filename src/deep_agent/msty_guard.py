"""TAU, слой L3: Guard — валидатор на границе «вызов → исполнение».

Имя инструмента (включая алиасы) и схема аргументов сверяются с реестром
(`msty_registry`) ДО исполнения. Провал возвращается модели как структурированное
корректирующее ToolMessage с fuzzy-кандидатами из реестра (top-3, difflib), а не
как сырое исключение: «Unknown tool» как сырой отказ до модели доходить не должен
(псевдокод §4.1 эталонной архитектуры).

Guard — корректирующий, а не блокирующий слой:
- имя вне реестра → unknown_tool + ближайшие кандидаты + правило одной смены
  способа вызова;
- аргументы вне схемы → invalid_arguments + список нарушений для исправления;
- сломанная или чужая схема не считается провалом вызова (fail-open): исполнитель
  сам вернёт свою ошибку, Guard не подменяет её догадкой.

Содержимое корректирующих сообщений не включает значения аргументов: путь поля
и текст нарушения достаточны модели, а сырые значения в контекст не разносим.
"""
from __future__ import annotations

from jsonschema import Draft202012Validator, FormatChecker
from jsonschema.validators import validator_for
from langchain_core.messages import ToolMessage
from referencing import Registry

from . import msty_registry

_MAX_ERRORS = 3
_MAX_ERROR_CHARS = 200


def _correction(call: dict, error: str, content: str) -> ToolMessage:
    """Структурированный корректирующий ToolMessage вместо сырого исключения."""
    return ToolMessage(content=f'error={error}. {content}', name=call.get('name'),
                       tool_call_id=call.get('id') or 'unknown', status='error')


def unknown_tool(call: dict) -> ToolMessage:
    """Имя не резолвится в реестре ни точно, ни по алиасу, ни по суффиксу."""
    candidates = msty_registry.fuzzy(call.get('name') or '', top=3)
    nearest = (', '.join(candidates) if candidates else
               'кандидатов нет; выбери имя из переданных схем шага')
    return _correction(call, 'unknown_tool',
        f"Инструмент '{call.get('name')}' не существует в реестре Brain. "
        f'Ближайшие из реестра: {nearest}. Правило: смени способ вызова один раз, '
        'прямым именем из реестра; одинаковый неуспешный вызов не повторяй.')


def _schema_errors(schema, args) -> list[str]:
    """Нарушения аргументов по JSON Schema; непроверяемая схема — не провал."""
    if not isinstance(schema, dict):
        return []
    try:
        validator_class = validator_for(schema, default=None) if '$schema' in schema \
            else Draft202012Validator
        if validator_class is None:
            return []
        validator_class.check_schema(schema)
        validator = validator_class(schema, registry=Registry(),
                                    format_checker=FormatChecker())
        errors = []
        for error in sorted(validator.iter_errors(args), key=lambda e: list(e.absolute_path)):
            path = '.'.join(str(part) for part in error.absolute_path) or '<args>'
            # error.message содержит значение аргумента (в т.ч. секретоподобное):
            # модели отдаём только ключевое слово схемы и её ожидание.
            expected = error.validator_value if error.validator in (
                'type', 'enum', 'required', 'minimum', 'maximum', 'minLength',
                'maxLength', 'pattern', 'format', 'const', 'minItems', 'maxItems') else None
            detail = f'нарушено «{error.validator}»' + (
                f', ожидается {expected!r}'[:_MAX_ERROR_CHARS] if expected is not None else '')
            if error.validator == 'additionalProperties':
                detail = 'передано поле вне схемы'
            errors.append(f'{path}: {detail}')
            if len(errors) >= _MAX_ERRORS:
                break
        return errors
    except Exception:
        # Схема непроверяема (битый $ref, чужой диалект): Guard не блокирует
        # вызов догадкой — исполнитель вернёт собственную ошибку.
        return []


def guard_validate(call: dict, schema=None) -> ToolMessage | None:
    """Сверка вызова с реестром до исполнения; None — пропуск к исполнению.

    ``schema`` — живая схема шага (для MCP-инструментов её поставляет клиент);
    если не передана, используется статическая схема записи реестра. Алиасы и
    суффиксный неймспейсинг клиента резолвятся в каноническое имя реестра.
    """
    name = call.get('name')
    entry = msty_registry.find(name)
    if entry is None:
        return unknown_tool(call)
    return guard_arguments(call, schema=schema)


def guard_arguments(call: dict, schema=None) -> ToolMessage | None:
    """Проверка только аргументов: для инструментов, которых клиент допустил,
    но реестр не знает (будущие коннекторы Msty), Guard не судит вызов —
    реестр является источником истины лишь для покрытых им инструментов."""
    name = call.get('name')
    entry = msty_registry.find(name)
    if entry is None:
        return None
    errors = _schema_errors(entry.schema if schema is None else schema,
                            call.get('args'))
    if errors:
        return _correction(call, 'invalid_arguments',
            f"Аргументы вызова '{name}' (реестр: '{entry.name}') нарушают схему: "
            + '; '.join(errors)
            + '. Исправь аргументы по схеме и повтори вызов один раз.')
    return None
