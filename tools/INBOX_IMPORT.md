# Входящие Brain: приватная очередь готовых выжимок (#151, часть #72)

Первый источник для общего входящего потока — уже созданные мостом выжимки
длинных чатов (#49). `import_compactions.py` выполняется **только на Mac**:
читает `~/Library/Application Support/BrainDesk/threads/*.json`, выбирает
`additional_kwargs.brain_desk_compaction.summary` из ответов Brain и кладёт
кандидаты в приватный каталог
`~/Library/Application Support/SanareOrchestrator/owner-inbox/pending/`.
Второй источник — уже сделанные Kimi сводки `context.apply_compaction.summary`
из `kimi-desktop/.../sessions/**/agents/main/wire.jsonl`. Импортёр читает
только это поле; `turn.prompt`, `contextSummary` и прочие записи в очередь не
попадают. Полная переписка и вложения остаются на месте. В stdout идут только
счётчики. Модели и внешние API не вызываются.

Третий источник — помеченные `isCompactSummary: true` сводки Claude Code
в `~/.claude/projects/*/*.jsonl`. Записи подагентов и обычные сообщения не
выбираются. Длинные сводки до 30 тысяч символов допускаются только как
кандидаты на проверку.

Четвёртый источник — поле `chat_message.context_summary` из **закрытого снимка**
SQLite Open WebUI. Импортёр читает только это поле и идентификаторы сообщений;
`content`, `output` и таблица `chat` не читаются. Путь к снимку передаётся явно
через `--openwebui-db`. Живую базу с незавершёнными WAL-записями не использовать:
режим SQLite `immutable` намеренно читает только согласованный снимок.
Проверенные локальные снимки за 17–20 сентября содержали 0 заполненных
`context_summary`; это адаптер для будущих сводок, а не заявление об уже
импортированных чатах Open WebUI.

```sh
python3 tools/import_compactions.py --dry-run
python3 tools/import_compactions.py
python3 tools/import_compactions.py --source kimi --dry-run
python3 tools/import_compactions.py --source claude --dry-run
python3 tools/import_compactions.py --source openwebui --openwebui-db /путь/к/снимку/webui.db --dry-run
python3 -m unittest discover -s tools -p 'test_import_compactions.py'
```

Каждый кандидат содержит тип источника, ссылку на исходный файл/чат, идентификатор
сжатия, SHA-256 выжимки, время и текст выжимки. Имя файла —
SHA-256 от `(source_kind, thread_id, compaction_id, source_sha256)`; повторный запуск не
создаёт копии. Файлы имеют права `0600`, каталог `0700`. Повреждённые,
слишком большие и символьные файлы чатов пропускаются. Если простая локальная
проверка замечает ключ, пароль, адрес почты или телефон, кандидат пропускается.

Это **очередь на проверку**, не общая память и не источник полномочий. Простая
проверка не гарантирует удаления всех секретов или чужих персональных данных.
До отдельного просмотра владельцем или безопасной проверки Brain не получает
текст выжимки, и она не записывается в `/memories/` либо Engram. Этот предел
сохраняет правило «сырые чаты остаются на месте» и не делает новую базу
источником истины. Последующая поставка должна добавить контролируемое
принятие кандидата с квитанцией и затем адаптеры Msty и Codex.
В проверенных локальных Codex JSONL сжатие хранится в
`encrypted_content`, а не в доступной текстовой выжимке; нельзя выдавать его
за готовый импорт. Рабочие `.db` Msty содержат knowledge stacks, а не
таблицу чатов. Отдельный `Backups/automatic-context-*.sqlite` содержит
`conversationTextMessages`, но это снимок сырой переписки без поля готовой
выжимки, поэтому он не подходит для инкрементального импорта. У Claude Code
пять найденных локальных сводок были отклонены предварительным фильтром;
ничего из них не копировалось. Для оставшихся приложений нужен проверенный
экспорт или отдельный разрешённый путь формирования сводки через Brain.

Использован существующий формат compaction Brain Desk и подход кандидатов из
`src/deep_agent/consolidator.py`; новая библиотека/служба не добавлялась.

## Чтение очереди Brain Desk (#238)

`src/deep_agent` разворачивается на удалённом сервере и не имеет доступа к
файлам Mac (см. `AGENTS.md` репозитория: «avoid calls to actual file system»),
поэтому очередь читает не граф, а серверная часть Brain Desk — так же, как она
уже напрямую вызывает `src/lib/server/memory/engram.py` для памяти. Единый
источник формата — этот файл и `import_compactions.py`, а не отдельная копия
парсинга в TypeScript.

```sh
python3 tools/import_compactions.py --list-pending
python3 tools/import_compactions.py --resolve-pending <id> --decision written
python3 tools/import_compactions.py --resolve-pending <id> --decision rejected
```

- `--list-pending` печатает в stdout JSON-массив уже застейдженных кандидатов
  (тот же формат записи, плюс `id` — имя файла без `.json`). Не читает
  источники заново и не вызывает модель.
- `--resolve-pending ID --decision written|rejected` переносит один файл из
  `pending/` в `reviewed/<decision>/` (права `0700`/`0600` сохраняются). После
  переноса кандидат больше не показывается `--list-pending` и не появляется
  заново при повторном запуске обычного импорта того же источника — имя файла
  проверяется также в `reviewed/`, а не только в `pending/`. Возвращает
  `{"moved": true|false}`; `false` — неверный id/decision или кандидат уже не
  в очереди (гонка, не ошибка).
- Ожидаемый вызывающий — Brain Desk (`src/lib/server/import/pending-inbox.ts`,
  brain-desk#238-inbox): при отметке владельца сначала пишет через
  существующий адаптер памяти, затем помечает исход здесь. До первого
  успешного вызова адаптера кандидат остаётся в `pending/`.
