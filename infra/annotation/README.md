# SpeechLLM: данные, Airflow и AGY

## Текущее размещение и просмотр (2026-10-13)

Runtime работает только на dev-1 (`exp-srv-rtx6000x4-02`), в
`/mnt/local/drive2/speech-annotation-20261012/code`. Локальная машина используется
только для доступа; прежний локальный стенд удалён после проверенного переноса.
Используется существующий S3/MinIO, второй экземпляр не разворачивается.

В `audio_labeling_pipeline` обход исходников и импорт готовых manifests запускаются
независимо. Подготовщик публикует manifests отдельно, поэтому импорт не ждёт
повторного обхода всего каталога S3. После импорта последовательно выполняются
сравнение проходов и публикация; обработчик импорта сохраняет проверку holdout,
контроль неизменяемости и блокировку от параллельного импорта.
PostgreSQL хранит служебную очередь и состояние Airflow; аналитические таблицы
и датасеты публикуются в Parquet на S3.

## Архивирование накопленной истории CLI

При длительной работе CLI общий индекс разговоров и protobuf summary растут.
15 октября зафиксированы SQLite WAL на несколько GB и постоянное переписывание
индекса на десятки MB; фиксация транзакций очереди иногда ждала диск 7 секунд.
На AGY-06 после паузы и полного завершения заданий история перенесена в
`~/.gemini/annotation-history-backups/<timestamp>/`. OAuth, настройки,
installation identity и caches сохранены побайтово. Рабочий контейнер продолжил
работу без перезапуска и получил новые CLI/API-ответы. В коротких замерах запись,
учтённая у его процесса, снизилась с 238 до 6 MB/s; это не гарантированная скорость.

Для следующего слота прочитайте прежний `accounts.next_request_at` под блокировкой
строки, затем задайте maintenance hold. Сохраните прежнее и новое точные значения
и дождитесь нуля RUNNING-заданий.
Внутри его контейнера запустите
`python -m lalm.annotation.cli_history --hold-until <exact-UTC-timestamp>`;
`--apply` выполняет перенос. Модуль сам проверяет hold под блокировкой строки,
отсутствие RUNNING-заданий и живых процессов AGY. При ошибке переноса возвращает
уже перемещённые элементы. Список переносимых файлов ограничен историей;
профиль проверяется до и после. Резервная папка имеет mode 0700. Файлы остаются
на том же томе, model requests не отправляются.

После переноса освободите только свой точный hold с сохранением прежнего ожидания
и появившегося backoff. Транспортный сбой может оставить задание в NEEDS_REVIEW,
одновременно задав паузу аккаунта: учитывайте его сохранённый `next_attempt_at`,
а не только задания RETRYABLE_FAILED. Проверьте новые CLI-ответы. Не обрывайте
запросы ради ротации.
Фоновая автоматическая ротация не включена; обслуживание выполняется по одному
слоту с проверкой свежих результатов перед продолжением.
Пилоты 15 октября: AGY-06 в 01:55 UTC и AGY-13 в 02:03 UTC. На втором
слоте protobuf уменьшился с 44.9 MB до 9 KB, WAL — с 5.1 GB до 1.1 MB;
после возобновления получены новые ответы обоими транспортами. Старые файлы
сохранены в резервных папках каждого слота.

Продолжение после проверки пилотов, 15 октября (UTC):

| Слот | Время архивирования | Перенесено элементов |
| --- | --- | ---: |
| AGY-08 | 02:08:38 | 5 843 |
| AGY-14 | 02:11:27 | 5 970 |
| AGY-21 | 02:15:13 | 5 590 |
| AGY-10 | 02:20:59 | 5 087 |
| AGY-16 | 02:23:51 | 642 |
| AGY-05 | 02:34:04 | 3 803 |
| AGY-19 | 02:37:38 | 818 |
| AGY-03 | 02:39:13 | 10 |
| AGY-18 | 02:41:38 | 298 |
| AGY-12 | 02:45:31 | 295 |
| AGY-11 | 02:47:16 | 232 |
| AGY-25 | 02:49:39 | 686 |
| AGY-07 | 02:51:36 | 1 622 |
| AGY-24 | 02:53:18 | 10 |
| AGY-23 | 02:55:15 | 10 |

Для каждого слота сначала прекращена выдача новых заданий, затем подтверждены
ноль RUNNING и отсутствие AGY-процессов. Проверка профиля до/после прошла;
worker-процессы и их API cooldown не перезапускались. Резервные папки и точные
времена находятся в локальных `archive-manifest.json` этих слотов.

После окончания обслуживания всех 17 рабочих слотов (около 02:56 UTC)
проверен один свежий сохранённый результат каждого: хеш аудио и модель
совпали с заданием, сырой CLI/API-ответ присутствовал в S3. Просроченных lease
и повторных попыток после неоднозначного результата за 30 минут не было.
В коротких замерах `vmstat` запись снизилась примерно с 700–800 до
120–140 MB/s, I/O wait — до 3%. Operational snapshot 02:55 UTC содержит
4 608 сохранённых ответов за 5 минут и 10 413 за 15 минут; это отдельные окна,
не обещание постоянной скорости. После ротации мониторинг продолжается.

## Снимки и просмотр данных

Отдельный DAG `inspection_snapshot_pipeline` каждые 10 минут публикует
согласованный снимок таблиц `sources`, `passes`, `examples`,
`progress`, `publications`. `s3://speech-datasets/analytics/latest.json`
указывает на конкретные неизменяемые файлы с SHA256 и числом строк.
Нельзя читать glob всех версий: это повторно посчитает одни и те же записи.
Событие Airflow Asset для этого URI содержит ключ manifest и счётчики.
Снимок предназначен для инспекции, не означает разрешение на обучение.
Экспорт задаёт `SET LOCAL work_mem = '512MB'` только внутри своей read-only
транзакции: широкие агрегаты при стандартных 4 MB создавали временные файлы.
Глобальные настройки PostgreSQL и соединения воркеров остаются прежними.
Первый полный запуск после изменения (15 октября, 01:22:52–01:30:57 UTC)
успешно завершился за 8 минут 6 секунд; предыдущий занял 15 минут 29 секунд.
Это наблюдение одного запуска, а не гарантированный предел задержки.

Для просмотра используйте `notebooks/speech_data_progress.ipynb` на dev-1
в [JupyterHub](../jupyter/README.md), развёрнутом на том же сервере.
Airflow и JupyterHub импортируют общий пакет `lalm.annotation` из IFAO-lalm.
Read-only AWS credentials передаются
через окружение/profile, endpoint через `S3_ENDPOINT`, CA через `AWS_CA_BUNDLE`.
Ноутбук читает Parquet, показывает первоначальные метки и ответы отдельных
проходов, воспроизводит фрагмент и показывает происхождение/границы.
Исходные standalone-аудиофайлы доступны для прослушивания; для архивов/Parquet
пока выводятся метаданные контейнера и идентификатор элемента.
Перед сохранением notebook в Git очищайте outputs с содержимым данных.

Полный прогон notebook проверен 2026-10-13 на dev-1 во временном контейнере:
все семь исполняемых ячеек успешны, DuckDB прочитал пять таблиц, SHA256
аудиофрагмента совпал, объект Audio создан, HEAD исходника успешен.
Проверенный снимок содержал 129 165 источников, 2 340 примеров и 5 218 проходов.
Это проверка выполнения кода; визуальное воспроизведение звука в браузере
отдельно не проверялось. Секреты и результаты с содержимым данных не сохранены
в notebook.

Исходная метка без связи с фрагментом обозначается `not_linked_or_not_available`,
а не «не размечено». Машинный ответ и ручное подтверждение разделены.
Пустая `publications` означает отсутствие опубликованного релиза.

Для Golos исходные записи из `source-index.jsonl` связываются по SHA256
исходных аудиобайтов. Исходный индекс остаётся в S3; локальный SQLite-файл
служит только удаляемым кешем поиска внутри обработчика. В provenance
сохраняются версия индекса, исходные записи и признаки eligibility.
При единственной непротиворечивой текстовой метке она доступна также как
`original_label`; все дополнительные поля — в `original_label_metadata`.
Отсутствующий текст не восстанавливается догадкой, конфликтующие варианты
сохраняются без автоматического выбора. Эти признаки не снимают review-only.

Production-файлы с явным путём `sessions/YYYY/MM/DD/<UUID>/agent.opus`
или `customer.opus` получают общий `conversation_id=<UUID>` для всех фрагментов
обоих каналов. Реестр исключённых разговоров проверяется до чтения аудио;
совпадение получает состояние источника `EXCLUDED_HOLDOUT`. Для неизвестных
форматов пути сохраняется `unresolved-*`: идентификатор не угадывается.

Обработчик сначала возобновляет источник в `PREPARING`, затем выбирает следующий
из корпуса, который дольше всего не обрабатывался. Прежнего приоритета всех
SOVA-файлов нет. Переключение происходит между исходными объектами, поэтому
один большой архив ещё может занимать подготовку длительное время.

Слой оркестрации в существующем IFAO-lalm. Модель разметки, инструкция и проверка
нативного трейса взяты из `lalm/benchmark_langfuse_asr.py`. Зафиксирована
`gemini-3.8-flash-low`; каждый файл получает отдельный процесс/разговор AGY.
Общий контекст между заданиями не переносится. Первые два прохода (`verbatim`,
`coverage`) не получают текст другого ASR. При расхождении адаптивный третий
проход использует `source_correction`, если есть исходная строковая метка
длиной до 12 000 символов; иначе — независимый `literal_audit`.
Вложенные наборы нескольких ASR-меток пока не выбираются автоматически.
Уже назначенный `literal_audit` не заменяется и не дополняется четвёртым запросом.
Исходная метка может относиться к более длинной записи: инструкция требует
оставлять только услышанное в текущем фрагменте. Её хеш входит в ID задания
`source_correction`; отчёт явно помечает этот проход как зависимый от метки.
Совпадение такого прохода с источником не является независимым подтверждением.

## Что переиспользуем

- `lalm/benchmark_langfuse_asr.py`: Langfuse-форматы, аудио и `agy_result`.
- `lalm/BENCHMARK_ASR.md`: правила сравнения и сохранения происхождения данных.
- Существующий `rolling_asr_collect.py` и systemd timer в соседнем
  `asr-benchmark-worktree/lalm`: экспорт Langfuse prod/test, FFmpeg, Silero, GigaAM.
  Сборщик находится в незакоммиченной работе другой сессии; здесь не меняется.
- На dev результаты сборщика лежат в
  `/mnt/local/drive2/asr-rolling-collector-v1`, готовые корпуса — в соседних
  `prod-asr-benchmark-*`/`test-asr-benchmark-*`. Это оценочные данные: по умолчанию
  в новую обучающую очередь не импортируются.

## Поток

```text
существующий сборщик → нормализованный WAV + проверка наличия речи
  → stage.py → S3 incoming/*.jsonl
  → Airflow (каждые 5 минут) → PostgreSQL jobs
  → AGY worker аккаунта → его VLESS proxy → Gemini
  → исходный stream-json + stderr в приватном S3
  → проверка agy_result → VALIDATED / NEEDS_REVIEW
  → Airflow → неизменяемый Parquet manifest
```

Airflow не запускает процесс на каждую фразу и не держит worker во время Gemini.
Он регистрирует manifests, проверяет зависшие попытки и публикует партии до 1000
результатов. Асинхронные AGY workers забирают задания сами.

## Развёртывание — только dev-1

Рабочий сервер: `exp-srv-rtx6000x4-02` (`dev-1`, tailnet `bitmanager.ai`).
Корень установки: `/mnt/local/drive2/speech-annotation-20261012/code`.
`minipc` используется только для доступа; `dc` запрещает запуск на нём.
Новый MinIO не создаётся. Используется существующий S3 на dev-1: TLS,
CA и учётная запись ingest из `/mnt/local/drive2/speech-data-storage-20261012/config`.
Тестовые расшифровки в `speech-benchmarks` этой учётной записи недоступны.

На dev-1, из корня репозитория:

```bash
python3 infra/annotation/render.py \
  --agy /mnt/local/drive2/speech-annotation-20261012/bin/agy \
  --storage-env /mnt/local/drive2/speech-data-storage-20261012/config/ingest.env \
  --storage-ca /mnt/local/drive2/speech-data-storage-20261012/config/ca.pem
infra/annotation/dc build init airflow-api
infra/annotation/dc up -d
infra/annotation/dc run --rm control status
```

Секреты находятся в исключённых из Git `.env` и `runtime/`. Не выводите
`docker compose config` без `--quiet`: он раскрывает environment.
UI привязан к `127.0.0.1:18080` сервера; доступ — через SSH tunnel.
Пароль admin: `runtime/airflow-auth/passwords.json` на сервере.
DAG первоначально paused: до включения должен существовать реестр
`speech-datasets/benchmarks/holdouts.json`.

## Подготовка корпусов и независимые проходы

DAG регистрирует native audio/Parquet/tar из `speech-corpora/staging/` в
PostgreSQL `source_assets`. Внешний CPU worker `source-preparer` готовит
каналы отдельно, приводит их к PCM16/16 kHz и режет длинные записи возле пауз
с перекрытием до 0,5 с. Курсор обработки сохраняется вместе с регистрацией подготовленной партии; повторный запуск использует те же идентификаторы.

```bash
infra/annotation/dc up -d source-preparer
```

Подготовка ограничена суммой активных заданий и ещё не импортированных строк manifests;
она не ждёт отдельного запуска Airflow после каждого аудиофайла.
`PREPARE_MAX_PENDING` задаёт этот запас: в Compose по умолчанию 6000.
При 15 воркерах прежние 1500 заданий регулярно заканчивались раньше
пятиминутного импорта Airflow. Запас 6000 покрывает несколько циклов подачи
при наблюдаемом темпе 20–33 тысяч ответов в час; это не лимит параллельных запросов.
После изменения проверить рост подготовленных manifests и отсутствие пустой очереди
на нескольких циклах. Лимит по-прежнему ограничивает объём подготовки заранее.
Исходные `.txt`/`.json` рядом с аудио в OpenSTT/WebDataset сохраняются
в provenance вместе с именем архивного элемента. Для нестандартного порядка
элементов или отдельно лежащих таблиц меток ещё требуется связывание по ключам;
пустое поле метки не доказывает отсутствие разметки в источнике.

Исходники остаются в `staging/`, подготовленные WAV — в
`speech-corpora/prepared/`, manifests — в `speech-datasets/incoming/`,
сырые ответы и сравнительные отчёты — в `speech-annotations`.

Два отдельных диалога AGY используют `verbatim` и `coverage`. При расхождении
нормализованных слов Airflow добавляет третий `literal_audit`; предыдущие
ответы и исходная метка в промпт не передаются. Согласие моделей не считается
эталоном. Все новые корпусные примеры имеют `review_only=true`: публикация
обучающего датасета ожидает проверки групп разговоров, PCM-дубликатов и аудита.
Сбой декодирования или неподдерживаемая схема сохраняются для проверки.

Настроенный лимит каждого из четырёх аккаунтов — 20 одновременных запросов
(суммарный потолок 80). Фактическая параллельность зависит от доступности
аккаунтов, квот и cooldown; заблокированные аккаунты не включаются автоматически. Каждый аккаунт имеет отдельный home, proxy и внутреннюю сеть.

## Один аккаунт + один VLESS

Добавьте `agy-02`, `agy-03` и т.д. в accounts JSON и повторите render. У каждого
аккаунта свой home volume, private network, proxy service и настройки нагрузки.
До настройки proxy его единственный outbound — blackhole.

```bash
python3 infra/annotation/configure_vless.py \
  --link-file /secure/agy-01.vless \
  --output infra/annotation/runtime/agy-01/xray.json
infra/annotation/dc run --rm --no-deps agy-01-proxy run -test -config /etc/xray/config.json
infra/annotation/dc up -d agy-01-proxy
infra/annotation/dc run --rm --no-deps --entrypoint agy agy-01
```

Последняя команда — интерактивный официальный вход для этого аккаунта. Завершите
вход и выйдите из AGY. Его состояние останется только в home volume аккаунта.
Конвертер поддерживает TCP/WS/gRPC+TLS и TCP/gRPC+REALITY; неизвестные параметры
отвергаются. Для других транспортов нужен проверенный полный Xray JSON.

Затем зарегистрируйте аккаунт и запустите worker:

```bash
infra/annotation/dc run --rm control account --account-id agy-01 \
  --concurrency 2 --min-interval 1 --enable
infra/annotation/dc up -d agy-01
```

Concurrency в accounts JSON ограничивает число процессов в контейнере; значение
в PostgreSQL дополнительно ограничивает ВСЕ процессы данного аккаунта, даже
если случайно запущен второй контейнер. При настройке меняйте оба значения.
Для загрузочного теста увеличивайте 2 → 5 → 10 → 20, измеряя завершённые
аудиочасы/час, latency и rate limits. Оптимальный предел нельзя установить до
подключения конкретных подписок. 429 включает общий cooldown аккаунта;
исчерпание квоты/отказ авторизации отключает аккаунт до явного повторного enable.
Уже выполняющиеся запросы могут закончиться. Новые не отправляются.

Worker имеет только internal network. Удаление proxy или сбой VLESS не открывают
прямой маршрут в интернет. У proxy нет опубликованных портов и direct fallback.
Это egress через HTTP CONNECT поверх VLESS, не системный TUN VPN. Соединения,
игнорирующие proxy, не должны проходить наружу; это проверяется сетевым тестом.

## Benchmark и подготовка данных

Обязательный объект `s3://speech-data/benchmarks/holdouts.json`:

```json
{"conversation_ids": ["prod-EXCLUDED-TRACE"], "audio_sha256": ["64-hex-audio-hash"]}
```

Пустые массивы допустимы только для пустого тестового стенда. Для реальных данных
сначала включите все benchmark conversations и хеши, включая алиасы. Реестр в
PostgreSQL только пополняется. Защита действует при ingest, claim и publication.
Если benchmark изменён после публикации, старые manifests требуют отдельного
аудита и отзыва: задним числом они автоматически не переписываются.

Сборщик публикует `assets.jsonl` и mono PCM16 16 kHz в `user-audio/`. Его завершённый
Silero отчёт должен иметь `complete=true`, `results[]` с `audio_sha256`,
`speech_detected`, `backend`, `version`. `stage` читает именно эти форматы.

Пример с новым выделенным TRAIN-корпусом (не с текущим benchmark):

```bash
infra/annotation/dc run --rm --entrypoint python \
  -v /path/to/new-corpus:/corpus:ro control \
  -m lalm.annotation.stage --root /corpus --speech-qc /corpus/speech-qc.json --split train
infra/annotation/dc run --rm control ingest
infra/annotation/dc run --rm control publish
```

Разделение train/validation задаётся по разговорам до stage. Очередь отвергает
разговоры или одинаковое аудио, встречающиеся в разных splits. Несколько алиасов
звонка удерживаются до явной канонизации. Исходники и контекст на S3 хранятся
раздельно; эта версия разметки — blind, без полной истории.

После подключения реального source/holdout включите расписание:

```bash
infra/annotation/dc exec airflow-scheduler airflow dags unpause audio_labeling_pipeline
infra/annotation/dc exec airflow-scheduler airflow dags unpause inspection_snapshot_pipeline
```

Текущее расписание сборщика остаётся systemd. Перенос запуска самого сборщика под
Airflow требует deployment на хост с его SSH-доступами. Граница интеграции уже
есть: завершённый snapshot → stage → S3. Не запускайте два сборщика одновременно.

## Состояния и восстановление

- `PENDING → RUNNING → VALIDATED → PUBLISHED`.
- `RETRYABLE_FAILED`: только явный rate limit, максимум 3 попытки.
- `NEEDS_REVIEW`: тишина, невалидный результат, квота, неизвестный исход/timeout,
  истёкший lease. Автоматического повтора потенциально оплаченного запроса нет.
- `PERMANENT_FAILED`: неверный hash/размер входа либо исчерпанные retries.

В БД сохраняются аккаунт, номер попытки, fencing token, время и причина. Сырой
нативный ответ сохраняется в S3 до принятия результата. Если процесс упал между
запросом и сохранением, lease через 10 минут переводит работу в NEEDS_REVIEW.
У AGY CLI нет подтверждённого идемпотентного job API: exactly-once для внешнего
вызова не заявляется. S3 publication переживает повтор после DB rollback.

Текст stdout worker содержит только job IDs и типы ошибок. Сырые traces содержат
речь/пути и доступны как приватные артефакты. `human_gold=false` сохраняется в
результатах: прохождение структурных проверок не делает транскрипцию эталоном.

## Проверки

```bash
python3 -m unittest tests.test_annotation_contract tests.test_benchmark_langfuse_asr -q
infra/annotation/dc run --rm --entrypoint python \
  -v "$PWD/tests:/tests:ro" control /tests/annotation_integration.py
infra/annotation/dc exec airflow-scheduler airflow dags list-import-errors
```

Интеграционные тесты создают отдельные schema и S3 bucket, затем удаляют только
их. Проверяются конкурентные claims, квоты, разделение данных, потерянные leases
и полный путь до Parquet с фиктивным ответом AGY. Они не тратят подписку.

Это локальный рабочий стенд. Перед production-развёртыванием нужны выбранный
хост/корпоративный S3, отдельные DB/S3 роли для сервисов, TLS, backup и policy
хранения. В текущем стенде внутренние сервисы используют общую административную
учётку хранилища; изоляция login-state аккаунтов не означает изоляцию от общей БД.

Источники: [AGY headless](https://www.antigravity.google/docs/cli/headless/),
[Airflow Docker](https://airflow.apache.org/docs/apache-airflow/3.3.1/howto/docker-compose/index.html),
[Xray VLESS](https://xtls.github.io/en/config/outbounds/vless.html).

## Исторический локальный smoke test 2026-10-12 (стенд уже удалён)

На `/home/alexesn` поднят Compose project `speech-data`. API, scheduler,
dag-processor, PostgreSQL и MinIO работают. Ручной запуск
`infra-smoke-20261012` и плановый запуск завершились `success`; обе задачи
`ingest_manifests`/`publish_validated` успешны. Ошибок импорта DAG нет.

30 unit/contract проверок прошли, 3 существующих FFmpeg проверки пропущены
локальным окружением. Все 5 интеграционных проверок на реальных PostgreSQL и S3
прошли с фиктивным AGY. AGY 1.3.3 запускается в worker image. Xray проверил
закрытый конфиг; соединение worker напрямую с внешним IP не прошло.

В реестр исключений внесён консервативный superset существующей оценки:
1186 conversation IDs и 1009 audio SHA256 из месячных/rolling корпусов и
контрольного «учусь». Обучающие задания не загружены. Активная конфигурация
содержит семь слотов `agy-01`–`agy-07`, выключенных для массовой обработки.
Старые VLESS заменены последним набором в порядке `fr / it / po / bu / au / pt / lon`.
Прокси `agy-08` удалён, его запись в БД выключена; home volumes сохранены.
Конфиги находятся только в игнорируемом `runtime/`, с правами 0600;
Xray запускается с UID/GID владельца файлов и без capabilities.

Все семь новых прокси запущены и прошли проверки конфигурации, внешнего IP,
Google HTTPS и публичного endpoint Google Login. Выходные IP у всех разные.
Прямое соединение с внешним IP заблокировано во всех семи workers.
Подробности сетевого прогона — `runtime/proxy-health.json`.

По явному запросу владельца текущая авторизация AGY скопирована в отдельный
home volume слота `agy-07`; исходная авторизация на хосте сохранена.
Реальный текстовый вызов Gemini завершился SUCCESS за 2.8 с. Отдельный вызов
на коротком benchmark-аудио с рабочим разметочным промптом прошёл `agy_result`
за 11.7 с: русский непустой транскрипт, один turn, подтверждённое чтение аудио.
После замены на новый выход `lon` повторный аудиовызов прошёл тот же валидатор
за 14.6 с. Это технический smoke test, не оценка качества или максимальной нагрузки.
Аудио не добавлялось в обучающую очередь. Артефакты находятся в `smoke/`
внутри home volume `agy-07`. Массовый worker и остальные аккаунты не включены.
Комбинация транспорта соответствует
[официальному примеру Xray](https://github.com/XTLS/Xray-examples/blob/main/VLESS-gRPC-REALITY/config_client.jsonc).

## Paused quality audit and prompt experiments (2026-10-13)

Bulk inference is paused: all account rows are disabled and worker containers
are stopped. Do not resume it as part of a deployment or notebook refresh.
The configured per-account limit remains 20. Diagnostic calls are separate,
sequential, explicitly requested and never imported as training annotations.

Five flagged SOVA examples were traced back to their exact Parquet rows. Source
audio hashes matched, source labels matched, and re-running preprocessing
reproduced the prepared WAV hashes. This verifies identity, not transcription
accuracy. No listening-based correctness claim follows from this audit.

The notebook now identifies the selected slice, exposes languages/error codes,
and has `matched`, `three_matched`, `flagged`, and `latest` views. Agreement uses
the same normalization as reconciliation: case, punctuation and ё/е. Two equal
passes are not three; a source-conditioned pass is marked explicitly. The live
`speech_data_progress.ipynb` is a separate analyst copy, preserving existing work.

Explicit language aliases such as `Russian` are normalized to `ru`; original
language strings remain in `language_raw`. Old validation metadata was corrected
without changing transcript strings, historical attempts, or human-review state.
Empty text and non-Russian text now have separate error codes.

Small prompt tests did not solve the user-identified short Russian clip. The
RU/reference-RU/EN comparison comprised six calls on two fixed audio files.
The short clip matched the user's reading only in the reference-conditioned arm;
that is not evidence against copying. No prompt winner or quality gate was accepted.
Exact requests and outputs are in private S3 under
`debug/ru-en-reference-6f53147e6449/`; earlier candidates are under
`debug/ru-candidates-cc0206dc8f81/` and `debug/prompt-ru-20261013/`.

A bounded experiment CLI is available (dry-run unless `--execute` is passed):

```sh
python -m lalm.annotation.prompt_ablation --sample-prefix SAMPLE_PREFIX
```

It accepts 1–4 unambiguous audio prefixes, requires source labels for the reference
arm, performs three sequential requests per sample, and writes only debug artifacts.
Use a one-off account container; keep bulk workers disabled.

Research supporting the experimental design:
- [Google transcription guidance](https://ai.google.dev/gemini-api/docs/transcribe)
  recommends known-language hints for its transcription endpoint. AGY's text
  instruction is not proof that its native endpoint exposes equivalent controls.
- [ClozeGER, ACL 2024](https://aclanthology.org/2024.findings-acl.37/) incorporates
  source speech into correction to reduce acoustically inconsistent rewrites.
- [DUST](https://arxiv.org/abs/2011.13439) uses prediction consistency under
  dropout for uncertainty filtering; it does not validate three prompt paraphrases
  as independent human-quality annotations.
- [Interspeech 2025 human-assisted annotation](https://www.isca-archive.org/interspeech_2025/johnson25_interspeech.pdf)
  reports that naive text correction can worsen WER and evaluates selective human
  input and acceptance filtering. Its English-domain results are not our benchmark.

### Mixed-language prompt audit (2026-10-13)

Bulk inference remains paused. A 12-request diagnostic compared Russian weak-prior,
reference-conditioned Russian, and English weak-prior prompts on four user-listened
samples. Run: `debug/mixed-language-90bd17c1f3fd/`. User listening notes and a
consolidated report are saved there; exact text was confirmed only for «я умею».
Portuguese and Ukrainian language controls remained foreign in all three arms;
the ambiguous tile sample remained inconsistent. These are not corpus-wide metrics.

Two additional controls on «я умею» are stored under
`debug/short-russian-control-3e523464f7b0/`. A verified-Russian-language-only prompt
returned «Ё-моё», matching the incorrect English-prompt output. An intentionally
wrong draft «я гуляю» instead produced «я умею». This demonstrates instability;
neither agreement nor empty uncertainty fields establish correctness.

User priority is precision over recall: retain conflicting evidence, abstain on
unstable examples, and never use draft-conditioned agreement to clear a conflict.
No automatic acceptance rule or training promotion was enabled. Precision needs a
separate human-checked evaluation set; these four examples are prompt-development
controls. The shared diagnostic CLI supports `--profile mixed-language`; its default
is still dry-run. `notebooks/speech_data_progress.ipynb` reads the 14 saved results
and renders the four audio clips without issuing any inference requests.

### Exact three-arm precision pilot

`--profile precision-triad` runs English automatic-language transcription, Russian
automatic-language transcription, and source-conditioned correction. The English
arm has no Russian prior. The Russian arm permits foreign languages; neither blind
arm sees the source label or the other outputs. The correction arm sees only the
original annotation and audio. Twelve requests on the same four development clips
are stored under `debug/precision-triad-6f3319f29d81/`.

`notebooks/speech_data_progress.ipynb` shows transcripts, language consistency,
three-way word agreement, uncertainty, and prior conflicts separately. It never
accepts or publishes training examples. Language consistency compares the primary
BCP-47 language (pt and pt-BR are consistent), while word agreement uses the shared
normalizer without stemming or fuzzy matching. A three-way candidate is not a
human-verified label, and prior conflicts remain visible.

### Overnight Russian-only run: precision-ru-triad-v1

The existing Airflow DAG and queue now run an English, audio-only first pass.
Only a nonempty Russian answer without uncertainty schedules the Russian blind
pass and source correction. Missing source annotations stay for review without
inventing a correction input. All three answers must agree after the shared word
normalization, be Russian and contain no uncertainty; any conflicting historical
answer blocks candidacy. Errors do not become votes. Nothing is automatically
published for training. `triad_decision` is shared by review and the night notebook.

Old responses are retained in their old batches. Previously pending old-prompt
jobs were quarantined as `superseded_prompt_version`; 3,873 prepared samples got
one versioned first-pass job. The source preparer now emits only the first pass;
Airflow schedules the remaining two conditionally every five minutes.

Five authenticated accounts (01,02,03,04,07) are configured at 20 concurrent
requests each. Empty slots 05/06 remain disabled. Native rate-limit cooldowns and
quota/auth circuit breakers remain active. Runtime stays on dev-1. All workers,
Airflow, and Jupyter import the same read-only source directory; no new service
was added. `notebooks/speech_data_progress.ipynb` shows strict candidates and audio.

Night-watch adjustment: paid-tier checks confirmed 01/02 as Pro and 03/07 as
Ultra; 04 currently reports Free. Limits are 5/5/20/5/20 for 01/02/03/04/07.
05/06 are intentionally disabled empty slots. If a native quota error includes
an explicit `Resets in ...` duration, the existing `next_request_at` enforces
that cooldown plus 60 seconds without retrying the failed audio job. Unknown
quota/auth errors still block the account. Claims now follow `created_at` FIFO:
measurements showed that prioritizing follow-ups let fresh work keep jumping
ahead of first passes, leaving the oldest first passes waiting over 30 minutes.
FIFO reduces that starvation while still completing each triad as follow-ups
arrive.

An ordinary partial unique index on prepared audio SHA256 and pass prevents
cross-manifest duplicate inference within this prompt version. Source provenance
remains in the source inventory and immutable manifests. This is exact-byte
prepared-audio deduplication, not semantic/near-duplicate audio detection.

Overnight audit found that a bare `429` substring in a conversation ID could
misclassify a rejected native result as a rate limit. Classification now requires
an explicit provider code/status or rate-limit message. Four worker error tests
cover this false-positive case, genuine service errors, auth/quota separation,
and bounded reset times. Rejected native turns remain rejected; no transcript
validation was relaxed. Historical misclassifications are corrected with audit
metadata while raw artifacts and attempt history are retained.

Starter account 04 repeatedly reached provider capacity limits in short bursts.
Its launch interval is now 2 seconds with the same concurrency cap of 5; existing
provider cooldowns remain in force. Ultra accounts keep their configured limits.

Archive preparation uses bounded S3 range reads with the original ETag and closes
responses before queue backpressure can pause iteration. A stalled archive can
resume through its existing source cursor; prepared-audio job uniqueness prevents
repeated inference. Eight preparation tests cover archive identity, closed range
responses, changed source rejection, segment coverage, and holdout exclusion.
For archive JSON sidecars, the correction pass can read an explicit `rover` text
(or an explicit transcription field). Conflicting sidecars remain for review;
raw source metadata is preserved. Twelve contract tests cover these cases and
the existing precision gates.

## Native account usage

From the operator host with `pexpect` installed, run:

```bash
python3 infra/annotation/check_usage.py --output /tmp/agy-usage-latest.json
```

Reads native `/usage` for the twelve original slots through `ssh-gpu dev-1`, with three
concurrent read sessions. It does not submit model prompts or change worker
limits. Output includes UTC timestamps and remaining weekly/five-hour quota;
account emails are redacted. `--accounts 5 6` restricts the check. Interactive
CLI setup must already be complete; failures produce an error and nonzero exit
status rather than guessing quota. This is an on-demand check, not a scheduled
collector. The output path is for operational data, not a Git artifact.
The collector waits for EOF after `/exit` and records `cleanup_confirmed`.
An in-container 180-second timeout with a five-second kill grace also bounds
the TUI if SSH disconnects. This prevents accumulating idle AGY processes.

## Intermittent headless authentication (2026-10-13)

Observed on pinned AGY 1.3.3: a fresh print process enters OAuth and times out,
while adjacent requests and `agy models` authenticate without a new login.
Upstream reports a matching intermittent startup failure in
https://github.com/google-antigravity/antigravity-cli/issues/944 and related
container credential initialization failures in
https://github.com/google-antigravity/antigravity-cli/issues/479.
These reports do not prove our internal root cause or a token-refresh race.

The worker now permits one recovery only when the entire stream is one ERROR
result with zero turns, zero native duration and the exact authentication timeout,
with no init/tool/generation events and less than 180 seconds elapsed. The bound
includes observed 126–134 second startup timeouts that previously bypassed recovery. It saves
that failed startup under `auth-startup-failure/`, checks native `agy models`
(timeout 40s), and retries once only after access to the configured model succeeds.
A second failure follows the existing account breaker. Ambiguous inference,
permission failures, quota, and verification-required failures are not retried
by this path. Eligibility/verification errors now explicitly block the account.
This mitigates the observed startup case; it is not a claim that AGY internals
are fixed. Raw failed and recovered outputs remain auditable in private S3.

Six extra slots 08–13 are staged (not enabled) using existing VPN proxies and
independent home volumes. See `ACCOUNTS.md` and `pending-ultra-slots.json`.
`check_usage.py --accounts 8 9 10 11 12 13` supports them after authentication.

## Model routing, 2026-10-13

Routing support is prepared and tested; bulk Argon ingestion remains disabled
pending an isolated quota comparison. Three manual Argon audio smoke calls
passed native audio/schema checks, which does not establish better ASR quality.

Ultra accounts expose `gemini-4-argon-low`; Pro account 02 does not. New
manifests can target Argon through `ANNOTATION_MODEL=gemini-4-argon-low` in
the Airflow scheduler. Each queued sample records `annotation_model`; absent
values mean the original `gemini-3.8-flash-low`. Follow-up passes inherit that
field, and job IDs, contract hashes and deduplication include the model.
Argon batches append the model ID to `pilot_id`, preserving older reports.

Workers use `ANNOTATION_ALLOWED_MODELS`, a comma-separated list (default Flash).
Set Ultra workers to both Flash and Argon while old queued samples drain;
Pro workers remain Flash-only. The render account config supports a `models`
list. Native validation checks the requested model in the init envelope and
rejects fallback to a different model. Existing results are immutable.

Existing databases need `precision_v1_audio_pass_model_uq` created concurrently
before the older `precision_v1_audio_pass_uq` is dropped concurrently. The
new schema already uses the model-aware index. The transactional check
`tests/annotation_model_routing_check.py` exercises real claims and deduplication
without leaving test rows or a test schema.

CLI 1.3.4 passed six sequential audio calls (three per model) on one isolated
Ultra identity with all its queue replicas drained and `useG1Credits=false`.
Native schemas and audio access validated in all six. Quota percentages were
delayed/non-monotonic, so no Argon/Flash quota multiplier was established.
Bulk Argon remains off. Raw canary streams and summaries are private S3 objects
under `evaluation/argon-quota-canary-20261013/` and are not training labels.

The verified CLI1.3.4 binary SHA256 is
`5b93565c5f04ea4f7b79d49d5c3237449bf6ce47b6779e8cf46c1fae97957c98`.
When changing the binary mount, update `AGY_EXPECTED_SHA256` in the same
configuration change; the worker refuses a mismatching binary. The renderer
pins the supplied binary automatically. Workers record the pin in raw requests
and results as `cli_sha256`.

The rolling CLI 1.3.4 deployment completed on 2026-10-13. All 21 running
containers were checked for version, matching pin, and a Flash-only model
allowlist. Slot 17 remains stopped. Workers mount an immutable copy of the
annotation modules and native parser under `runtime/cli134-eligibility500-20261014-v2/` on dev-1;
the producer still submits Flash jobs. The upgrade does not establish that
Google authentication timeouts are resolved; verify actual saved results after
each recovery.

Worker memory defaults to 16 GiB for concurrency above 10, otherwise 8 GiB.
Twenty native CLI processes can exceed 8 GiB. Check cgroup `memory.events`
for increases in `oom_kill` when investigating signal-9 exits, including streams
with a SUCCESS result. Raising a running container's limit with `docker update`
does not require interrupting in-flight requests; persist the same limit in
runtime Compose. A SUCCESS stream followed by a failed process exit remains
quarantined, without automatically repeating inference.

CLI 1.3.4 can emit an init followed by a zero-turn ERROR for Google's exact
eligibility `UNAVAILABLE (code 503)` response, with an optional native
`failed to send message` wrapper. The job stays in NEEDS_REVIEW and is not
replayed; the account receives a randomized 30–90 second cooldown instead of
an authentication block. Verification and 401/403 failures keep their blocks.
The transactional check `tests/annotation_transport_cooldown_check.py` verifies
that cooldown prevents dispatch and that the next claim cannot replay that job.

Confirmed native ERROR results after generation starts also receive this
cooldown for the observed `INTERNAL (code 500)`, model-capacity `UNAVAILABLE
(code 503)`, and interrupted-stream messages. These jobs remain NEEDS_REVIEW
without replay, under `inference_service_unavailable`. Only exact native error
messages qualify; model response text cannot trigger the pause. Authentication
failures and ambiguous timeouts keep their existing handling.

When replacing an identity, drain and disable its worker before changing auth.
Slot 03's first in-place replacement reverted to the previous identity during
CLI use. Its second import uses a clean home with the old volume retained for
rollback; audio generation and a subsequent identity/tier check confirmed the
new Ultra account. Test both an actual response and the identity after it.
For new code mounts, make public module directories readable to UID 10001,
then verify an import in the canary container before rolling out other workers.

Inspection Parquet publication and pass reconciliation run in parallel after
manifest ingestion. Inspection reads its own REPEATABLE READ, READ ONLY database
snapshot and does not consume reconciliation reports; `snapshot_at` identifies
the data's observation time. Training publication still depends on reconciliation.
`max_active_runs=1` keeps inspection publishers from overlapping across DAG runs.
All Airflow components mount the repository's `airflow/dags` directory read-only.

Reconciliation enqueues follow-up passes in transactions of at most 128 jobs,
using the same `enqueue` checks and IDs. Reports and cursors are written only
after all batches succeed. If report publication fails after a batch commits,
the next reconciliation sees those children and does not duplicate them.
`tests/annotation_review_batch_check.py` checks this recovery and raw-audio
holdout rejection against PostgreSQL in a rolled-back schema, without inference.

Eligibility-check transport EOF is handled separately from denied access. A single
ERROR result with zero turns and zero duration, and the exact HTTPS POST EOF
message, is retried through the existing queue (at most three total attempts).
The delay is random 30–90 seconds, doubled for each subsequent attempt; the
account also receives the queue cooldown. Raw artifacts remain per attempt.
Verification requests, forbidden responses, ambiguous output, and any model
activity do not qualify. Existing blocked accounts require a successful access
check and inspection of their failure before being re-enabled.

The same bounded pre-generation retry covers the observed eligibility response
`UNAVAILABLE (code 503): The service is currently unavailable.`
It also covers the exact observed load-code-assist `INTERNAL (code 500):
Internal error encountered.` response. If an init event already exists, the
job is quarantined and the account cooled down rather than replaying it.
Explicit access denials still take priority over these service errors.

The observed CLI catalog miss (`invalid model selection` for the requested model)
also permits a bounded retry, but only with exit code 1, the exact stderr message,
and a sole zero-turn, zero-duration ERROR envelope. The requested model is preserved;
any init event, generated turn, timeout, or different model/message prevents retry.
This handles transient catalog discovery without replaying uncertain inference.

PostgreSQL uses `max_connections=512` for the worker fleet and Airflow. The
default 100 produced connection-limit errors and lost finish updates during
bursts. Apply this startup parameter only after draining AGY and source workers
and allowing the current DAG run to finish. Retain account reset timestamps and
existing auth blocks when resuming. Expired attempts remain for review without
inference replay. No global `work_mem` change is required.

## One maintained notebook

`notebooks/speech_data_progress.ipynb` is the sole maintained notebook. It includes
progress, corpus yield, source/raw inspection, listening examples and optional
historical prompt experiments. Retired Jupyter copies are backed up outside the
visible workspace. Git versions must not contain cell outputs or embedded audio.
Use `python3 infra/jupyter/sync_notebook.py --check` to detect browser edits,
`--pull` to bring them into Git for review, or `--push` to deploy the repository
version. Synchronization preserves a backup before replacement.
The pre-generation allowlist also includes profile-picture HTTPS GET EOF and the
observed user-info `500 Internal Server Error`. They use the same three-attempt
queue bound and randomized cooldown, without disabling the account. A 401/403,
verification requirement, extra native event or started generation still does
not qualify for this recovery path.

### Восстановление после сбоев проверки доступа (13.10.2026)

Worker автоматически повторяет подтверждённые ошибки до начала генерации:
eligibility EOF / unexpected EOF, TLS handshake timeout, Google UNAVAILABLE 503
и user-info 500, а также точный ответ Google HTML 502. Условие: единственный native result, ERROR, ноль turns и нулевая
длительность; таймауты с неизвестным результатом не повторяются. Пауза с jitter
30–90 секунд и экспоненциальным увеличением; максимум три попытки задания.
Артефакты каждой попытки сохраняются в S3. Настоящие 401/403 и запрос
верификации по-прежнему блокируют аккаунт, исчерпанная квота ждёт reset.

Для аккаунтов, остановленных старой версией: прочитать последний raw result,
подтвердить этот тип временной ошибки, проверить `probe_auth()` внутри его
контейнера, затем снять блокировку. Проверить новые сохранённые результаты.
Не очищать все блокировки без проверки и не менять VPN автоматически.
Соответствие аккаунтов proxy записано в ACCOUNTS.md: сравнивать ошибки
аккаунтов на одном proxy и на разных proxy, отличая их от сбоя управляющего
SSH/Tailscale-соединения. Один EOF или 503 не доказывает неисправность VPN.

14.10.2026 обнаружился крайний случай: CLI может вернуть нулевой код процесса
вместе с единственным zero-turn eligibility ERROR. Поэтому безопасный повтор
проверяется по содержимому native result, а не по коду выхода. Любой прочий
`Eligibility check failed` без явного признака 401/403, отказа доступа или
верификации получает короткий transport cooldown; только явные отказы блокируют
аккаунт. Это не повторяет запрос с неизвестным результатом и не затрагивает
ошибки после начала генерации.

Точный `authentication failed or timed out` с нулём turns проходит тот же
ограниченный путь: успешный read-only catalog probe разрешает одну попытку
повтора сразу; неуспешный probe считается неопределённостью сети, поэтому
задание получает bounded retry и слот — transport cooldown. Явный отказ доступа
и verification requirement по-прежнему не повторяются.

Сетевой EOF, connection reset или broken pipe после начала модели также не
переигрывается: задание остаётся `NEEDS_REVIEW`, а аккаунт получает короткий
transport cooldown. Так мы не рискуем оплатить повторно запрос с неизвестным
результатом и временно не отправляем следующий запрос по тому же маршруту.

## Annotation operations monitor

The `annotation_ops_monitor` Airflow DAG runs every five minutes and publishes a
compact, transcript-free snapshot to
`s3://speech-datasets/analytics/ops/latest.json`; timestamped history is stored
under `analytics/ops/history/`. It records queue states, expired leases, saved
outputs per 5/15/60 minutes, per-account concurrency and throughput, median
successful request duration, cooldowns, recent error codes, timeout counts, and
source-preparation states. `pilot_review_only` is counted separately as a
completed review output, not as an operational error. Recent failure traces
contain only job ID, account, error code, timestamps, and duration; raw
transcripts and audio are excluded.
`native_api_failures_15m` includes the API `native_` response-validation errors
such as `native_schema_keys` and `native_incomplete_or_ambiguous_candidate`,
as well as HTTP/transport failures. The generic CLI `native_result_rejected`
is excluded from that API counter. This was corrected on 15 October; earlier
immutable snapshots can undercount API response-validation failures.

The unified `notebooks/speech_data_progress.ipynb` reads the latest snapshot,
shows account-level speeds and recent failures, and plots the recent throughput
trend. `/usage` quota reads remain a separate read-only check via
`infra/annotation/check_usage.py`.

## Bounded native API canary

`lalm.annotation.native_canary` temporarily reserves an existing account for a
fixed manifest of at most six distinct, unprotected training audios by default,
or 30 when `--max-samples 30` is explicit. The manifest
contains `sample`, explicitly attached source annotation, optional listening
audit metadata, and cached CLI results. It executes three independent prompts
per sample by default. With `--adaptive`, it first runs the blind prompt for all
samples and runs the other two only when that pass is a usable Russian result.
At most five native `generateContent` POSTs run concurrently, through
the account's existing network and local OAuth file. Auth refresh remains owned
by the native CLI; a catalog request is made only if the cached access token has
less than 15 minutes left. Project discovery is once per batch. Generation has
no automatic retry. A 401, 403, 429, server error or ambiguous transport failure
stops dispatch of further passes.

```bash
python -m lalm.annotation.native_canary \
  --manifest /private/canary-manifest.json \
  --prefix evaluation/native-api-canary-v1/unique-run-id
```

The manifest and logs belong in private storage, outside Git. `ACCOUNT_ID` and
the normal DB/S3 environment come from the existing account container. Existing
CLI tasks drain before canary dispatch. The account's queue cooldown reserves
15 minutes; the `finally` block restores it after the test, while an expired
reservation resumes CLI even if the canary process dies. Restoration never
overwrites a newer cooldown or quota block. Production jobs and their contract
IDs are untouched; results under the private annotation bucket's canary prefix
are separate review artifacts and never training publications.

`native_api.py` derives the existing English/Russian/source-correction prompts,
replacing file opening with an attached audio part. It checks the returned model,
STOP completion, JSON field types and `audio_heard=true`, and rejects output tool
calls. A Markdown JSON fence observed in the live gateway is handled explicitly.
The same three-pass Russian precision decision is reported as `native_decision`;
model agreement alone is not a listening audit. Inspect cached CLI contract
versions before treating historical results as a controlled comparison.

The account worker checks its local OAuth JSON before starting AGY. If a complete
credential object is followed only by trailing garbage, it saves the original
privately in the same home and atomically restores the complete object. Ambiguous
or incomplete files are rejected before inference. A one-minute shared catalog
cache prevents concurrent startup failures from starting many identical auth
probes. No tokens are logged or stored in the canary artifacts.

Validation: `tests/test_annotation_native_api.py` covers native validation and
single-dispatch timeout behavior. `tests/annotation_native_canary_check.py` checks
reservation expiry and guarded rollback in a disposable PostgreSQL transaction.
`tests/test_annotation_native_auth.py` covers the narrow credential repair.

For a larger parallel shadow check, `native_shadow_manifest.py` selects 100
distinct, unprotected training audios: six listening controls from the prior
manifest, 80 recent CLI results tagged `ru`, and 14 tagged as other languages.
The CLI language tags only stratify the sample; they are not ground truth. The
manifest is saved privately in S3 and is rejected if too few eligible audios
exist. Run the same canary with `--max-samples 100 --adaptive` on one account;
other accounts continue their normal CLI queue. This account reserves up to
15 minutes and returns to CLI automatically. Shadow results remain separate
review artifacts and must not enter the training publication automatically.

The recurring `native_api_shadow_pipeline` schedules one 30-audio review-only
batch every six hours. Its PostgreSQL registry prevents duplicate audio and
overlapping batches. The AGY-03 worker consumes the batch through its own VPN
and OAuth, reserving the account while its CLI requests drain; other accounts
continue CLI. It sends up to five native POSTs at once, with two follow-up
passes only after a usable Russian blind pass. Interrupted or incomplete
batches become `NEEDS_REVIEW` and are never replayed automatically. Private
manifests, raw responses, and reports live under
`speech-annotations/evaluation/native-api-canary-v1/recurring/<batch>/`.
`analytics/ops/latest.json` includes `native_shadow_recent`; the notebook's
`latest_recurring` view shows the latest completed batch with audio playback.
Three-pass agreement remains a review candidate, never a human-verified label.

After the initial shadow check, AGY-03 routed a deterministic 20% of new,
review-only, short Flash jobs through the same native API transport. This is a
transport substitution in the existing `jobs`/`attempts` queue, not a duplicate
inference: the result records `transport=native_api_v1` and keeps the raw
response under its usual private attempt prefix. Other jobs and accounts keep
using CLI. API results stay `NEEDS_REVIEW`, including valid Russian text; the
dataset publisher cannot include them. HTTP quota/auth/server errors suspend
the API route for ten minutes while new jobs use CLI. Three consecutive bad API
responses do the same. Responses after dispatch are never automatically
retried. Operational account metrics include `native_api_saved_5m` and
`native_api_saved_15m` and `native_api_failures_15m`. After 25 consecutive
strictly valid API responses the route rises to 35%; after 100 it rises to
50%, at most about five of this account's ten active jobs. A bad API response
resets this ramp. These checks measure transport/schema stability, not acoustic
accuracy; all routed labels remain outside training publication.
AGY-16 uses the same initial 20% route after its successful bounded canary;
each account reads only its own OAuth file and uses its own VPN. Other accounts
continue through CLI.
AGY-21 is added at 20% after its bounded canary and quota check. A per-container
semaphore limits native API inference to five concurrent requests even when an
account has more than ten CLI worker slots; excess selected jobs use CLI.
AGY-08 starts at 20% after its bounded canary and a fresh quota check.
AGY-25 starts at 20% after a bounded six-audio canary and quota check; its
two listened non-Russian controls were rejected and its listened Russian
control became a review candidate.
AGY-19 starts at 20% after a 30-audio canary. One of 78 responses failed strict
schema validation and was not retried; both listened non-Russian controls were
rejected, and the listened Russian control was also rejected. An account-local
20% ceiling initially held its route below the normal successful-response ramp.
On 15 October at 03:17 UTC, AGY-19 began a bounded 35% base/ceiling trial with
the existing five-call semaphore. The preceding 45-minute inspection contained
205 valid API outputs and two response-validation failures, with no API
transport failures observed. Those invalid responses stay quarantined; this is
not an error-free or human-gold quality claim. Existing response checks and
account-local fallback remain enabled.
After several error-free operational snapshots, AGY-08 and AGY-25 route up to
70% of eligible short review-only jobs via API, while AGY-21 routes up to 60%.
The normally five-call-per-container semaphore still limits simultaneous API requests;
overflow takes the CLI route. This increases throughput, but one API result
still consumes a model generation request and quota savings are unproven.
AGY-08's eight-call API pilot was returned to five after six post-dispatch
`RemoteDisconnected` responses in a burst on 14 October. The affected attempts
remain quarantined; the change drains in-flight work before recreation.
An AGY-25 eight-call pilot was rolled back after malformed model JSON responses;
that account remains at five. Its later 100% selection pilot, started after 30
minutes without API failures, was returned to 70% after an HTTP-200 JSON decode
failure and a post-dispatch disconnect. Both uncertain attempts remain for review.
Overflow still takes the CLI route. Both changes used a pause of new claims and a
complete in-flight drain before recreation. The worker enforces a hard ceiling
of eight; all deployed accounts now use five. AGY-12's eight-call pilot was
returned to five on 15 October after three fresh post-dispatch
`RemoteDisconnected` artifacts. Its pending work was drained and the existing
error cooldown preserved during recreation. HTTP 429 and server failures suspend that
account's API route and new jobs fall back to CLI.
AGY-06 and AGY-14 were then added as 20%-capped pilots after adaptive
six-audio canaries (12/12 and 14/14 valid generation responses respectively)
and account-local in-flight drains. The known foreign controls were rejected;
the known Russian control was also rejected by disagreement, so these canaries
show transport viability, not measured transcription recall. AGY-18 passed
14/14 canary responses and was then added at the same 20% cap after AGY-06's
shared VPN remained error-free. AGY-14 had one immediate post-dispatch
transport disconnect; its API route paused for ten minutes and new work
continued through CLI without replaying the uncertain attempt.
One AGY-06 native call later timed out after 90 seconds without an HTTP
response. The account-local safeguard paused its API route, and AGY-18 was
returned to CLI after an in-flight drain to test whether the shared VPN remains
stable without the extra API workload. Causality between the two is unproven.
After its ten-minute pause, AGY-06 automatically resumed and saved three
new API responses without another API failure by 19:42 UTC.

The 100-audio shadow comparison paired 99 native blind results with saved
earlier CLI blind results. Raw strings matched in 40/99, normalized word
sequences in 62/99, and language tags in 86/99. The calls were not simultaneous;
agreement is not a human-gold accuracy measure. The native API remains
review-only while these differences are audited.

The next expansion used the same six private control audios per account. AGY-05,
10, 12, 13, 18 and 24 each returned 18/18 complete API responses; AGY-11 also
returned 18/18. Both listened non-Russian controls were rejected on every
account. This is a transport and precision smoke check, not a gold accuracy
measurement: the listened Russian control was rejected by the three-pass rule
on most accounts. AGY-12, 13 and 18 route eligible short review-only
jobs to API. AGY-06's route
was raised to 50% after continued error-free API operation. AGY-10 was rolled
back to CLI after five concurrent post-dispatch `RemoteDisconnected` responses;
AGY-11 was rolled back after a post-dispatch 90-second `TimeoutError`. AGY-05
was rolled back after a second post-dispatch `RemoteDisconnected`; AGY-14 was
rolled back after another such failure before its proposed ramp. AGY-24 was
rolled back after a post-dispatch 90-second timeout on the LA route. Those
uncertain jobs remain `NEEDS_REVIEW` without automatic replay. The effective
API share must be read from `native_api_saved_5m / saved_5m`: route percentages
are caps, and API cooldowns or the concurrency semaphore can lower the actual
share. HTTP/server and ambiguous transport failures still suspend an affected
account's API route while CLI continues.
AGY-12 has five simultaneous API calls with a 100% selection cap; AGY-16
has a 70% selection cap with five simultaneous calls. AGY-21 briefly piloted
eight calls and a 100% cap, then returned to five and 60% after five concurrent
post-dispatch disconnections. The uncertain results were quarantined. At 21:15 UTC,
AGY-12, 13 and 14 had near
simultaneous post-dispatch disconnections, so their API routes were allowed to
cool down instead of forcing the traffic target. A successful six-audio canary
does not establish reliability under production load.

For a VPN migration, probe audio through the target proxy with the account's
own OAuth before changing Compose. `route_account.py` checks that the target
network reaches the proxy, PostgreSQL, and S3 relay; `--apply` saves a private
backup of the runtime Compose file. Pause new claims and wait for the account's
`RUNNING` count to reach zero before recreating its container. A stop timeout
can otherwise leave already-dispatched jobs ambiguous; these must remain in
`NEEDS_REVIEW` without automatic replay. Validate with `dc config --quiet`
before restart and check the account's actual proxy, DB/S3 access, fresh API
outputs, and error codes afterward.
Per-account restart floors are explicit in [render.py](render.py) and the private
runtime Compose configuration. AGY-19's current trial floor and ceiling are 35%.
Explicit settings keep a container restart from resetting a proven route to
the original 20%. A bad API response temporarily
reduces the effective share to 20% until 25 fresh valid responses; quota/auth
or server errors additionally suspend the API route for ten minutes.
An ambiguous transport failure without any HTTP response also suspends the
route immediately; its dispatched request is quarantined without replay.
