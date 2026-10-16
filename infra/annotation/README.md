# SpeechLLM: данные, Airflow и AGY

## Текущее размещение и просмотр (2026-10-13)

Runtime работает только на dev-1 (`exp-srv-rtx6000x4-02`), в
`/mnt/local/drive2/speech-annotation-20261012/code`. Локальная машина используется
только для доступа; прежний локальный стенд удалён после проверенного переноса.
Используется существующий S3/MinIO, второй экземпляр не разворачивается.

`annotation_ingest_pipeline` каждые пять минут импортирует готовые manifests
независимо от обхода S3 и сравнения проходов в `audio_labeling_pipeline`.
У каждого DAG остаётся `max_active_runs=1`: долгий review больше не задерживает
следующее пополнение очереди. Обработчик импорта сохраняет проверку holdout,
контроль неизменяемости и блокировку от параллельного импорта. Сравнение проходов
и последующая публикация остаются в `audio_labeling_pipeline`.
При переходе сначала дождитесь завершения старого `ingest_manifests`, проверьте
новый сериализованный граф без этого задания и только затем включайте новый DAG.
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

16 октября в 12:41–12:58 UTC обнаружено повторное накопление индексов CLI.
Пятисекундный замер `/proc/diskstats` показал на drive1 99,95% занятости и
12,39 ms среднего ожидания. Cgroup `io.stat` отнёс основную запись к AGY;
счётчики родительских процессов с унаследованным I/O не суммировались.
Существующим `cli_history` по одному обслужены три слота:

| Слот | Hold UTC | Архив UTC | Возобновление UTC | Элементов |
|---|---|---|---|---:|
| 16 | 12:46:41 | 12:47:50 | 12:48:05 | 1 012 |
| 11 | 12:51:27 | 12:52:04 | 12:52:26 | 241 |
| 19 | 12:54:04 | 12:55:52 | 12:56:12 | 231 |

Перед переносом проверены ноль RUNNING и отсутствие AGY-процессов; профиль
проверен по хешам до/после. История сохранена на том же томе в закрытых backups.
Освобождался только собственный точный hold с учётом прежней паузы и новых
пауз завершавшихся заданий. Worker и его API backoff не перезапускались.
На каждом слоте проверены три новых CLI и два API результата на пяти разных
аудио: хеш звука, модель, контракт и сохранённый текст совпали с raw.

После обслуживания в отдельных окнах drive1 показал 50,46%/1,18 ms (5 s)
и 80,25%/1,72 ms (20 s). Это разная текущая нагрузка, не контролируемый
бенчмарк ускорения. На 16 в активном окне с тремя AGY запись составила
3,411 MiB/s; protobuf уменьшился с 25 338 702 до 15 594 bytes.
Общий отчёт с ledger, backup-путями и ссылками на 15 проверенных результатов:
`speech-datasets/diagnostics/cli-history-20261016/three-profile-maintenance-c0e003298800c07f0f356185b1414c3503cc932dccca5b8b7f600f0ffa74e262.json`.

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
Это порог остановки подготовки новых аудио, а не жёсткий потолок всей очереди:
сравнение уже готовых ответов может добавить следующие проходы и увеличить
число заданий сверх порога. При таком запасе пауза source-preparer штатная;
проверять нужно также наличие неимпортированных manifests и продвижение ingest.
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
infra/annotation/dc exec airflow-scheduler airflow dags unpause annotation_ingest_pipeline
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

The helper uses the existing noninteractive `sudo -n` access for `dc`, because
the private runtime Compose file is not readable by the SSH user. Failure
records include a `stage` field, distinguishing startup/interactive-setup errors
from errors after reaching `/usage`. No inference request is submitted.

Reads native `/usage` for the twelve original slots through `ssh-gpu dev-1`, with three
concurrent read sessions. It does not submit model prompts or change worker
limits. Output includes UTC timestamps and remaining weekly/five-hour quota;
account emails are redacted. `--accounts 5 6` restricts the check. Interactive
CLI setup must already be complete; failures produce an error and nonzero exit
status rather than guessing quota. This is an on-demand check, not a scheduled
collector. The output path is for operational data, not a Git artifact.
The collector waits for EOF after `/exit` and records `cleanup_confirmed`.
Timeouts and setup failures also interrupt only the probe’s own PTY and record
whether it exited. If shutdown cannot be confirmed, `cleanup_confirmed` stays
false; an unsuccessful read never receives old quota values under a new date.
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

On 2026-10-15, a completed AGY-10 startup timeout took 215 seconds and fell
outside this immediate retry window. The exact one-result, zero-turn,
zero-duration authentication timeout is now classified as `auth_startup_timeout`
with a short transport cooldown instead of disabling the account. Its job stays
`NEEDS_REVIEW`, without replay; the 180-second immediate retry bound is unchanged.
Explicit denial, verification requirements, process timeouts and streams with
model activity retain their existing handling. Thirty-nine worker tests passed
inside the deployed worker image, including a process test that preserves all
three raw artifacts and makes no second model call. Read-only reclassification
of the original failed stream is recorded at
`diagnostics/auth-recovery-20261015/late-timeout-e85c90f14964b8c052c77668b66ce426f8cfe3c1dd5fb07b041363d868990799.json`.
The source is staged for the normal drain-and-canary rollout; staging alone
does not change already-running workers.
The initial classifier revision was loaded on AGY-07 at 21:19:50 UTC after
draining all jobs and CLI processes. Four ordinary CLI outputs passed strict
model/schema, audio hash, prompt/reference, contract and stored-text checks;
its previous concurrency 3 / interval 2s was restored at 21:21:52 UTC without
changing the current readiness deadline. Evidence:
`diagnostics/auth-recovery-20261015/agy07-late-timeout-canary-0ae259abcf5b610c5bdf611a0fb31e5c5ea3fcaf1120738028daba34214b208a.json`.
A subsequent source-only refinement preserves quota-error precedence even if
stderr contains both a quota marker and the exact startup timeout. All 39
worker tests passed again, including these mixed-log cases. The refinement is
staged; the already-running AGY-07 pilot still has its initial revision.
The final revision (`c437af5c94e1b0c455f22dab612fdc12f0e92baed5fd9920a7fbf5e1610086f5`)
was loaded on AGY-10 at 21:33:12 UTC after 43 valid native responses following
its token recovery, no active native pause, and a complete drain with no errors.
It also loads the existing proactive token-expiry check and account claim
`SKIP LOCKED` that this older process lacked. Four new native API responses
passed raw model/schema, audio SHA, prompt/reference, contract and stored-text
checks, with one HTTP attempt and no retry each. Concurrency 5 / interval 1s
was restored at 21:34:08 UTC without modifying the current readiness deadline.
Evidence:
`diagnostics/auth-recovery-20261015/agy10-refresh-roll-95abbcdfd135cf2d02424707c5b18c5e4186178cb3f861628f53c61075b903b6.json`.
Token recovery itself used stock interactive `/usage`, confirmed Ultra and
fresh expiry, and was followed by five independently verified raw responses:
`diagnostics/auth-recovery-20261015/agy10-expiry-ec18c44729355cae731c6fe9c46c502dd37c1277af84e0844369f93ce045a4fa.json`.
This confirms current recovery; the next natural expiry remains a monitoring
check rather than a claimed production refresh test.

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

For raw CLI audits, extract `agy_result` from the deployed worker parser,
including its `expected_model` parameter. The scheduler image can contain an
older helper with a different signature; it is not the inference validator.
Do not classify that diagnostic mismatch as a failed annotation. A read-only
audit on 2026-10-16 at 08:40 UTC confirmed fourteen current rejections against
the worker parser: multi-turn successes or an error result, all retained for
review without replay or acceptance changes. Evidence:
`speech-datasets/diagnostics/provider-errors-20261016/cli-format-rejections-ee2398580e27c545725fc42a8cb4b92928b4b5f017df88fb80e51e61a4dddecf.json`.

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

Queue dispatch uses the partial `jobs_claim_fifo(created_at,id)` index for
`PENDING`/`RETRYABLE_FAILED` rows with attempts below three. It preserves the
existing FIFO order, due-time/model filters and lease checks. The older
`jobs_queue` index remains available for state and due-time queries.
For an existing database, build this index with `CREATE INDEX CONCURRENTLY`
outside a transaction, following the
[PostgreSQL 17 documentation](https://www.postgresql.org/docs/17/sql-createindex.html).
The versioned schema includes the ordinary `IF NOT EXISTS` definition for init.

On 15 October at 05:24 UTC the live concurrent build completed in 13.27 seconds;
128 attempts finished during that interval. Both the new index and the existing
deduplication index were valid afterward. The exact claim SELECT, including
model filtering and `FOR UPDATE SKIP LOCKED`, changed from scanning/sorting
6,845 rows in 23.915 ms to an ordered index scan returning one row in 0.081 ms.
These are individual query-plan measurements, not an overall throughput claim.
Private plans are saved in `speech-datasets/diagnostics/claim-fifo-plan-20261015/`.

Recent attempt diagnostics use `attempts_started_at(started_at)`. Existing
databases should also build this index with `CREATE INDEX CONCURRENTLY` outside
a transaction, so monitoring does not scan the full attempt history or block
worker writes. Check `pg_index.indisvalid` before treating the build as complete.
`IF NOT EXISTS` does not repair an invalid index left by an interrupted build;
use `REINDEX INDEX CONCURRENTLY attempts_started_at` for that case. Concurrent
builds can wait for the inspection export's old snapshot to finish. Do not
restart the export merely because the index is waiting.

The live index became valid on 15 October after a concurrent rebuild lasting
126 seconds; 669 attempts finished during that interval. The five-minute
per-account count used the index and took 1.486 ms afterward. The original plan
scanned the entire 429 MB attempt heap. This measures one diagnostic query,
not model throughput. Plans and the build receipt are stored under
`speech-datasets/diagnostics/queue-contention-20261015/`.

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
The additive `storage` field records total, used and available bytes of the
filesystem containing the deployed code bind mount, using Python's standard
`shutil.disk_usage`. On the current deployment this is dev-1's local drive2;
it is not a capacity measurement of any S3 endpoint. An OS read failure is
reported by exception type without aborting the operational snapshot. Existing
timestamped history provides a disk-space trend without another service.
The scheduled 04:20 UTC snapshot was read back with this field present and
approximately 502 GiB available. The unified notebook displays available and
total GiB from that snapshot, while retaining compatibility with older snapshots
that lack `storage`. Deployment used the existing notebook synchronizer after
checking that browser and repository sources agreed; the previous live copy
was backed up before replacement.

The unified `notebooks/speech_data_progress.ipynb` reads the latest snapshot,
shows account-level speeds and recent failures, and plots the recent throughput
trend. `/usage` quota reads remain a separate read-only check via
`infra/annotation/check_usage.py`.
Add `--publish` to publish those same observations through the existing remote
Airflow scheduler, without another usage read or inference request:

```bash
umask 077
python3 infra/annotation/check_usage.py --accounts 1 2 3 4 5 6 7 8 9 10 11 12 23 --output /tmp/agy-usage.json --publish
```

Publication writes an immutable JSON snapshot and updates `latest.json` with
an ETag precondition. It replaces the raw screen and measured fields together,
clears absent quota/tier observations, rejects older observations, and leaves
unmeasured accounts unchanged. A publication failure preserves the local output
and reports a separate error; it does not alter account or queue state.
The helper also returns structured Gemini quota fields (`weekly_pct`,
`weekly_reset`, and the five-hour fields when present). It parses only the
Gemini section, so another model group's quota cannot replace an exhausted
Gemini quota. After closing its usage TUI, it makes one read-only
`loadCodeAssist` call and records `paid_tier_id`, `paid_tier_name`,
`paid_tier_checked_at`, and `paid_tier_http_status`. A failed discovery is
reported separately as `paid_tier_error`; the observed quota is retained.
This distinction matters when Google changes an account from Ultra to Starter:
an earlier percentage belongs to the earlier plan and does not prove that
the account can still generate. Discovery never submits audio or inference.
The read-only fleet poll at 04:23–04:25 UTC on 15 October successfully read all
17 active slots. Its PII-free numeric observation is saved in
`analytics/usage/latest.json`, with immutable history under
`analytics/usage/history/2026/10/15/0424-20d476eb4c7a.json`. This is an operator
observation, not a new scheduled DAG; each row carries its own check time.
Do not sum percentages across duplicate slots of the same subscription.
AGY-03's TUI did not confirm EOF within the helper's ten-second exit wait;
an authoritative process check afterward found no remaining plain AGY TUI.

## Bounded native API canary

A fresh worker starts with a native routing share capped at 20%, respecting a
lower configured account ceiling. After 25 valid native results it releases
that startup cap and uses the existing routing policy. Controlled rolling
updates also drain current requests and wait for old in-memory guards to
expire; the startup cap itself does not persist those guards across restarts.

Before native inference, the worker checks that OAuth remains valid for the
request budget. On the explicit predispatch expiry error, it uses the existing
bounded `agy models` catalog probe to let the native CLI refresh credentials,
then checks token validity again. This also supports native-only traffic.
Concurrent threads share the existing locks and probe cache. Failed refresh
or discovery does not send audio; failures after generation starts remain
quarantined without replay. Loaded worker versions are recorded in ACCOUNTS.md.

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
Provider-declared prompt blocks are classified separately as
`native_prompt_blocked_<reason>`. A recognized `promptFeedback.blockReason`
with HTTP 200, the expected model and no candidates remains `NEEDS_REVIEW`,
retains the raw response, and is never retried. It does not change route failure
counters, ramp state or an existing cooldown. Missing candidates without this
feedback, unknown reasons, truncation and transport failures keep the existing
guard behavior. Google defines `OTHER` as a block for an unknown reason; this
does not establish that the audio or draft contains prohibited content. See the
[GenerateContent response reference](https://ai.google.dev/api/generate-content#BlockReason).
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
The 35% trial was returned to 20% at about 03:28 UTC after two fresh
post-dispatch `RemoteDisconnected` responses at 03:18. The affected attempts
were not replayed. CLI continued during the API cooldown; the account then
drained before recreation at 20%, preserving both the prior queue backoff and
the ten-minute API error cooldown. These observations do not establish whether
the higher fraction caused the disconnects.
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
runtime Compose configuration. AGY-19's current floor and ceiling are 20%.
Explicit settings keep a container restart from resetting a proven route to
the original 20%. A bad API response temporarily
reduces the effective share to 20% until 25 fresh valid responses; quota/auth
or server errors additionally suspend the API route for ten minutes.
An ambiguous transport failure without any HTTP response also suspends the
route immediately; its dispatched request is quarantined without replay.

### CLI web-tool guard pilot, 2026-10-15

Strict result validation rejects unexpected tools, but a rejected response can
already have executed `search_web`. A read-only audit found seven completed web
searches in a fixed cohort of forty rejected CLI streams. Those outputs remain
quarantined; they are never automatically replayed or admitted as labels.

The installed AGY 1.3.4 documents workspace `.agents/hooks.json` and synchronous
`PreToolUse` command hooks. A single public-query smoke test on AGY-07 verified
an attempted `search_web` changed from ACTIVE to ERROR with the denial reason
visible, no DONE event, and no timeout. This test contained no production audio.
Private evidence is stored under
`speech-annotations/evaluation/no-web-hook-probe-20261015/agy-07/035502-58a36c9e/`.

`lalm.annotation.cli_hooks` installs that stock hook only for AGY-07, in each
temporary CLI workspace. It denies the exact `search_web` tool and leaves
`view_file` unchanged. Request artifacts record `tool_policy_sha256`; prompts,
schemas, result validation and the no-uncertain-replay rule remain unchanged.
Other slots receive no hook until ordinary production jobs confirm the pilot.
The two hook tests and all 26 worker tests passed inside the worker image;
worker tests unset deployment-specific native percentage caps in their test
process. An initial test invocation inherited AGY-07's disabled API cap and
failed its unrelated positive API-route assertion; no live setting was changed.

AGY-07 was drained before restart and released at 04:00:28 UTC with its prior
readiness and transport backoffs preserved. At 04:02:37 UTC, the first ten
completed production jobs all recorded the policy hash and a completed
`view_file`: nine Russian review-only outputs and one non-Russian rejection,
with no web-tool events. This is an initial compatibility check, not evidence
that the guard has been exercised by a production web-search attempt. The
metadata-only readback is stored alongside the probe as
`production-pilot-first-readback.json`.

The pilot is extended to AGY-25 after a separate fixed cohort of 22 rejected
AGY-25 streams contained seven web-search attempts, six completed. Its API
selection cap and result validators remain unchanged; pause/drain and any
outstanding native API cooldown must be preserved during the rollout.
AGY-25 was released at 04:10:07 UTC after its previous API cooldown expired.
The first twenty inspected CLI outputs all recorded the policy hash and a
completed audio read, without web-tool events. Its API route also resumed.
Readback metadata is in
`evaluation/no-web-hook-probe-20261015/agy-25/production-pilot-first-readback.json`.

### Global CLI tool policy, 2026-10-15 04:39 UTC

The installed CLI customization guide also documents global discovery at
`~/.gemini/config/`. A separate public-query probe on AGY-07 verified the same
`hooks.json` policy from this global location, with no workspace hook: an actual
`search_web` attempt became ERROR with the denial reason, and no search became
DONE. Evidence is stored under
`evaluation/no-web-global-hook-probe-20261015/agy-07/043551-10f8ee1e/`.

The global policy was then installed on all 17 active slots at 04:39:17–18 UTC.
No worker was restarted, no inference was interrupted, and API traffic caps
and cooldowns were unchanged. It applies to new CLI sessions; already-running
sessions retain their earlier configuration. The profile installation report
is retained in the private runtime directory and S3 evaluation evidence.

`python -m lalm.annotation.cli_hooks` installs the stock global configuration.
It preserves unrelated named hooks, privately backs up an existing file before
merging, refuses a conflicting audio hook, and writes atomically. Eight hook
tests cover idempotence, executable denial, preserved owner hooks, invalid or
conflicting configuration, failed replacement and concurrent file creation;
all eight plus the 26 worker tests passed in the actual worker image.

New worker starts install/check the same global policy before claiming jobs.
CLI request metadata recognizes the global policy hash; the earlier two-slot
workspace policy remains a fallback. Already-loaded older workers do not gain
that metadata field without restart, so use the timestamped fleet installation
record to identify their global policy. Missing request metadata alone does
not mean the profile lacks its verified global configuration.
The immutable installation report is
`evaluation/no-web-global-hook-probe-20261015/fleet/install-2bd1fef537fe749b92893bf0af887e456670593556f8fb09bf46e93b808797a7.json`.
At 04:41:54 UTC, a fresh completed CLI job from every active slot was read
back using the actual worker parser. All seventeen input hashes and raw
transcripts matched their saved records, and all passed the existing strict
audio-only validator. This establishes initial fleet compatibility; the
public diagnostic remains the direct evidence of a denied search attempt.
The metadata-only readback is stored alongside the installation report as
`production-first-readback.json`.
At 04:48:54–55 UTC the same policy was installed on the seven quota-waiting
profiles (01, 02, 04, 09, 15, 20, 22), without a model request. Their existing
quota waits remain in effect; these profiles are not counted as active workers.
The disabled duplicate 17 was not started. New account containers receive the
policy through the worker startup path.

At 05:54:55 UTC, a read-only audit of 24 rejected CLI streams from slots
07, 24 and 25 found a production search attempt on slot 25. The stream recorded
`search_web` ACTIVE followed by ERROR, the exact configured denial reason, and
no DONE event. Its job remained quarantined; it was not replayed. The other
23 streams reported SUCCESS with two to four turns and were rejected by the
existing one-turn contract. These are contract rejections, not evidence that
the account could not authenticate. The metadata-only evidence is in
`speech-datasets/diagnostics/cli-rejects-20261015/0555-2597afe02a28c785bdc3a7f4d4b6dc250a876454d62bdd82694bd708c0bc9d4f.json`.

### Native response-object prompt investigation, 2026-10-15

A metadata-only audit of twelve recent `native_schema_keys` responses found
JSON Schema fields (`type`, `properties`, `required`) instead of, or alongside,
the result. Over the preceding hour, 29 of 31 such failures used the English
`verbatim` prompt. The shared English prompt says “Return the supplied JSON
schema.” CLI's separate structured-output mechanism usually disambiguates
this, but a plain API text instruction can reasonably elicit the schema itself.

The [public Gemini generation reference](https://ai.google.dev/api/generate-content)
documents structured response configuration. It does not establish which
fields the internal AGY gateway enforces. One diagnostic request with
`responseJsonSchema` returned HTTP 200 but used `text` instead of the required
`transcript`; it failed the existing strict validator. That setting is not
enabled in production and the failed output remains diagnostic-only.

`native_prompt` now explicitly requests a result object conforming to the
schema and names the required fields, including `transcript` rather than
`text`. The English instruction explicitly excludes returning a schema
definition. The same-audio `verbatim` probe then passed the unchanged validator
without `responseJsonSchema`. This is a format compatibility observation, not
an accuracy or reliability measurement. Artifacts live under
`speech-annotations/evaluation/native-response-schema-probe-20261015/agy-12/`.
Generation results record `prompt_format=result_object_v1`; their existing
contract hash includes the exact changed prompt. Prompt changes must first
be observed on one worker before expanding or increasing API traffic.
The additional Russian and source-correction probes also passed the unchanged
validator on the same audio. No diagnostic response was inserted into the
annotation queue. Four native API tests, three hook tests and all 26 worker
tests passed in the actual worker environment. AGY-12 alone was restarted
after draining; its prior native API cooldown had already expired. It resumed
at 04:12:30 UTC with its existing 100% selection cap and five-call semaphore.
At 04:12:52 UTC, fifteen new saved responses identified `result_object_v1`.
At that point, other workers retained their already-loaded native module; no
traffic cap was increased. Future restarts load the current source.

At 06:13:46 UTC, AGY-08 resumed with the same result-object prompt. Its preceding
30-minute window contained 865 native outcomes: five schema-echo rejections and
one HTTP-200 JSON decoding rejection, with no consecutive failure pair. All
running jobs drained before recreation; no new API failure occurred during the
drain. The last old CLI job remained review-only after a contract rejection and
was not replayed. Compose bytes, worker concurrency 20, native selection 70/70
and the five-call semaphore were unchanged. At 06:14:44 UTC, 55 new attempts had
completed without errors. The first four native results passed strict response
validation; their input hashes, raw/saved text and contract hashes matched, with
one HTTP attempt and no retries each. This short readback does not establish
long-term reliability or transcript accuracy. Metadata-only evidence:
`speech-datasets/diagnostics/native-result-object-prompt-20261015/agy08-544ea175ff294ca046abeef9869f2a339fbcabf516cc314b285ffb196336a0e1.json`.

### LA service degradation and lower-burst trial, 2026-10-15

Slots 11 and 24 shared intermittent eligibility/generation 503 failures while
their LA proxy remained running. Plain HTTPS health checks succeeded, and
read-only subscription discovery succeeded. A same-account diagnostic through
LA, NY and LA again took 2.048, 1.664 and 2.554 seconds, respectively, without
changing the worker route or submitting generation. These observations did
not establish a broken VPN or invalid account authorization.

From 06:20:00 to 06:35:00 UTC, only slot 11's queue limits changed from
10 concurrent jobs / 0.2-second interval to 5 / 2 seconds. Slot 24 stayed at
5 / 0.2 seconds. Existing requests, cooldowns, VPN routes and workers were not
restarted. In the preceding fifteen-minute window, slots 11/24 saved 124/74
Russian responses; during the trial they saved 138/165. Neither slot recorded
an eligibility/generation service-unavailable failure during the trial. Slot
11 still had four startup/eligibility transport failures and one contract
rejection; slot 24 had six contract rejections. Both recovered, including the
unchanged control, so this observation did not demonstrate a benefit from
lowering slot 11's limits. The comparison was not randomized and included
existing cooldowns and requests started before each window.

At 06:36:16 UTC, slot 11's original 10 / 0.2 limits were restored through the
queue without restarting or altering its cooldown. Restoration checked that
the two trial values were still owned. Evidence is stored under
`speech-datasets/diagnostics/agy11-lower-burst-20261015/`:
`window-53ba43719533efe8d771dcdddf91f997bac6a039e59b78da8e089314ba34d70a.json`
and `restore-0ac8c73b8a4a1047ce6f31f172a4d387b0b62169d6d419d87b6a94e3dda3d91f.json`.

### Post-dispatch 429 replay fix, 2026-10-15

A live audit found 429 errors after audio access, model activity and structured
output. The generic rate-limit classifier had marked them retryable; 23
distinct jobs from the overnight window had already retried before detection.
This consumed additional generation attempts even though the audio/pass/model
unique index still kept one durable job. The original raw attempts remain
available; historical attempts were not rewritten to hide the retries.

Generic 429 errors now become `NEEDS_REVIEW` with
`ambiguous_rate_limit_after_dispatch`, retaining a five-minute account cooldown.
No generic 429 alone proves that generation did not start, including a
zero-turn envelope. Existing narrowly verified predispatch retries retain their
separate codes and behavior.

At 07:01:52 UTC, small BEFORE triggers on jobs and attempts enforced the same
quarantine for old workers still emitting `RETRYABLE_FAILED / rate_limited`.
This closes the retry path without restarting the fleet or clearing live API
guards. The triggers leave unrelated retry codes unchanged. Temporary-table
INSERT/UPDATE tests passed, and rollback probes on both live tables confirmed
the stored state becomes `NEEDS_REVIEW`. No model calls were used for these
tests. All 28 worker tests passed; their process-level API routing overrides
were unset to avoid inheriting slot 07's disabled-API test environment.

Slot 25 was briefly held during investigation. A real provider quota cooldown
subsequently superseded that hold across shared-quota slots 23/24/25; it must
not be cleared as if it were an operator maintenance hold. Diagnostics:
`speech-datasets/diagnostics/rate-limit-retry-incident-20261015/guard-bfaa8e3526f9433668b3ca9a99751ce6363b0278fe4efe2de9e3c5d70c25ba54.json`.

### Повторное обслуживание истории CLI, 15.10.2026, 07:55–08:00 UTC

На работающих слотах обнаружены WAL 8.23 GB (08), 9.35 GB (21) и 4.62 GB (14); большой WAL сам по себе не доказывает сбой. По одному обслужены 08 и 21: точный maintenance hold, полный drain, проверка отсутствия AGY-процессов внутри `cli_history`, архивирование разрешённых файлов и проверка неизменности профиля. Контейнеры не перезапускались, внутренние API backoff не сбрасывались. Слот 14 пока не обслужен.

- 08: архив `20261015T075528Z-d5406ce5`, 10 элементов; приём заданий возобновлён в 07:56:08 UTC. После возобновления активный WAL 6.13 MB, protobuf 97 KB.
- 21: архив `20261015T075910Z-b2b0e96e`, 13 элементов; приём возобновлён в 07:59:37 UTC. После возобновления WAL 1.55 MB, protobuf 19 KB.

Для каждого слота три новых штатных CLI-ответа повторно проверены строгим парсером: текст, SHA256 аудио и контракт совпали с сохранёнными результатами. Проверка не отправляла модельных запросов. Отчёты: `speech-datasets/diagnostics/cli-history-20261015/agy08-post-0756-65fe72ea86674f9ed33231d3b34b54c630199337012be4333bb4fe86e4b2dab4.json` и `agy21-post-0759-545172c94460954d4c720b729f0b08e6b65aec30b8dfd322033cb98ee5590b0d.json`.

Отдельный HTTP 502 на 21 в 07:43:47 UTC остался в NEEDS_REVIEW, attempt=1. Защитная пауза API истекла без операторского сброса; первые новые native API результаты появились в 07:54 UTC. До этого CLI сохранил 536 результатов.

### Выход из служебного `/usage`, 15.10.2026, 08:40 UTC

В `check_usage.py` после таймаута обычного `/exit` добавлена попытка штатного двойного Ctrl-C только в созданном этим скриптом интерактивном PTY. Этот PTY не запускает print-mode задания разметки. Внешнее ограничение `timeout 180s --kill-after=5s` сохранено. При ошибке выхода сами успешно прочитанные квоты сохраняются с отдельным флагом `cleanup_confirmed`.

Живая проверка на 14 рабочих слотах: 13 закрылись через `/exit`, 21-й через резервный выход. Независимая проверка `/proc` во всех 14 контейнерах подтвердила ноль оставшихся процессов agy без аргументов. Процессы разметки не завершались; дополнительные генерации для проверки не создавались. Свежие квоты опубликованы в `speech-datasets/analytics/usage/latest.json` с неизменяемой историей.


### Три исходных архива в архивном MinIO, 16.10.2026

`minio-source-tier.json` содержит три добавочных lifecycle-правила для
`speech-corpora`: точные ключи неактивных исходных TAR, всего 74 344 155 342
байта. Перед применением каждый prefix проверен на единственный объект,
состояние источников — `QUEUED`; текущий подготавливаемый источник не затронут.
Переход — `Days: 0` в существующий `ANNOTATION-OVERFLOW`, без expiration.
Это фрагмент для объединения с текущими правилами, а не замена всей bucket policy:
при применении в 15:51 UTC сохранено существующее правило диагностического canary.
Исходные S3 URI, ETag и VersionId сохраняются; чтение идёт через основной MinIO.
Архивный tier не является резервной копией: для чтения нужны оба MinIO.

Перенос асинхронный. На 16:09 UTC первый архив deepspeech-balalaika
(25 691 307 251 байт) перенесён и полностью прочитан через прежний URI.
Совпали три SHA256 диапазона до/после переноса, размер, ETag и VersionId.
Из полного чтения восстановлен исходный multipart ETag с частями 16 MiB;
SHA256 полного чтения записан, но исходного полного SHA256 не было.
Остальные два архива ещё ожидают обхода scanner; это не ошибка переноса.
Свободное место drive2 выросло примерно с 77 до 102 GB, однако параллельные
записи других проектов продолжаются, поэтому это не точный учёт освобождения.
Временные ускорения scanner завершены; прочитаны `speed=default` и восемь
transition workers. Контейнеры и сервисы не перезапускались.

Исходные размер/версия/диапазоны:
`speech-datasets/diagnostics/storage-tier-20261016/source-preflight-4e452cd3ac55c036788b7904c2555683ebef95fac9e8652bd3f55678d7b3e321.json`.
Сохранённые правила до изменения и план:
`speech-datasets/diagnostics/storage-tier-20261016/source-policy-before-0ac7572fb44fd866240d0c103cb3f44814b6ddcddd6828c567cb0cb7a84790f2.json`.
Проверка первого архива и список двух ожидающих:
`speech-datasets/diagnostics/storage-tier-20261016/source-transition-batch-4f63d6d8c35d94e02bd289f9b56ad0a40ae877b029a3fed6cc0a301179392c82.json`.
Отключение этих трёх правил останавливает будущие переходы; оно не возвращает
уже перенесённые байты на основной диск. Не удалять tier или backend для отката.

### Перенос старых inspection-снимков на второй MinIO, 15.10.2026

`minio-analytics-tier.json` задаёт стандартное lifecycle-правило для
`speech-datasets/analytics/tables/`: переход в `ANNOTATION-OVERFLOW` с `Days: 0`
(первоначально использовался `Days: 1`; нулевой срок подтверждён 15.10 в 16:40 UTC).
Удаления и expiration нет. MinIO применяет правило своим scanner.
Для ненулевого возраста действует округление S3 lifecycle. Сам перенос
асинхронный, поэтому освобождение места не мгновенное. Указатели
`analytics/latest.json`, входные manifests, аудио и
бенчмарки этим правилом не затронуты.

При `Days: 0` текущий MinIO в ответе GET lifecycle опускает поле Days;
StorageClass без Date/Days соответствует тому же нулевому transition, который
проверен на synthetic canary. Не считать отсутствие поля расхождением с JSON.
Перед изменением измерено около 85 GB ещё не перенесённых current Parquet при
более 500 GiB свободного места на destination. Для `reviews/` первоначально
оставлен `Days: 1`; последующее изменение на нулевой срок описано ниже.
Проверять оба диска по `df`, а не
считать логический размер S3 равным физическому освобождению.

После ограничения ILM до восьми workers повторное пятиминутное окно scanner
`fastest` в 16:41–16:46 UTC прошло без memory pressure; возврат в `default`
прочитан из конфигурации в 16:48 UTC. Старые Parquet ещё ждут обхода большого
`annotations/raw/`. В 16:50 UTC включено ограниченное 30-минутное окно с timer
`speech-minio-scanner-restore-20261015e` (возврат около 17:20 UTC), при сохранённых
ILM workers=8 и лимитах памяти 48/64 GiB. Перед включением: память 30.1 GiB,
pressure=0, HEAD bucket 1–6 ms. Проверять S3 latency и cgroup pressure в ходе
обхода; для раннего возврата использовать приватный `restore-scanner.py`.

Окно e завершилось штатно: в 17:24 UTC прочитаны scanner `default` и ILM
workers `8`. Из примерно 85 GB STANDARD Parquet осталось 0.11 GB; текущий
перенесённый объём analytics составил 102.7 GB (включая ранее перенесённые
снимки). Это логические размеры current S3 objects. Фактически свободно
286 GiB на drive2 и 364 GiB на drive1; cgroup memory pressure=0.
`speech-datasets/diagnostics/storage-tier-20261015/after-window-e-2811c6e1edf055f33a8cdbad56f887f8244eb32a31dbfa514eb211c33a20ce56.json`.

Tier использует уже существующий второй MinIO на drive1, bucket
`speech-annotation-archive`, prefix `primary-annotation/`. Его отдельная учётка
ограничена этим префиксом; credentials хранятся только в закрытом server audit,
не в Git. Исходные S3 URI, ETag и VersionId сохраняются; обычный GET и Range
прозрачно читают перенесённый объект. Это не backup: для чтения нужны metadata
первого MinIO и доступность второго. Перед изменением правил сохранить текущую
lifecycle-конфигурацию и объединять правила, а не перезаписывать чужие.

До включения проверены синтетические JSON, случайный бинарный объект и объект
144 MiB: полные SHA256, ETag, VersionId и Range совпали после фактического
перехода; у большого объекта подтверждён multipart destination из двух частей.
Учётка tier получает 403 при чтении вне своего префикса. Временный ускоренный
scanner возвращён в `default` в 15:44 UTC. Закрытые доказательства и backup
исходной конфигурации находятся на dev-1 в audit-каталоге
`annotation-tier-canary-20261015` второго MinIO. В 16:00 UTC подтверждён переход
рабочих объектов. Для шести файлов из `passes`, `examples` и `reviews`, включая
Parquet 106 MB и JSON 112 MB, полный SHA256 совпал с хешем в исходном ключе,
ETag сохранился, Parquet footer читается через исходный endpoint. Доказательство:
`speech-datasets/diagnostics/storage-tier-20261015/production-readback-0ee7608f5274f045361ad8e78959ea125c2b93f19d38986013065e1c3da1e1df.json`.

В 15:57 UTC аналогичное правило `minio-review-tier.json` добавлено для
`speech-annotations/reviews/` (только старые JSON-отчёты сравнения проходов,
`Days: 1`, без expiration). Исходная конфигурация сохранена в том же закрытом
audit-каталоге. Сырые ответы `annotations/raw/` не входят в этот префикс.
Для ускорения первого обхода включено ограниченное окно `speed=fastest`;
автовозврат в `default` назначен timer
`speech-minio-scanner-restore-20261015c` на 16:09 UTC. После включения измеренные
Range-запросы занимали 2–4 ms, а обработка продолжалась без просроченных leases.
Проверять окончание окна по фактической конфигурации MinIO, а не только timer.

Позже срок перехода для `reviews/` изменён на `Days: 0`: перед изменением
физические файлы префикса занимали 105.7 GB, на destination было 322.6 GB
свободного места. Нативный MinIO transition передаёт хранимый размер объекта,
включая сжатие, поэтому логический размер JSON не равен требуемому месту.
Правило применено и прочитано обратно; удаления нет, scanner оставлен
`default`, ILM workers — 8. Перенос асинхронный; сам факт изменения правила
не означает освобождение всего объёма. Доказательство изменения:
`speech-datasets/diagnostics/storage-tier-20261015/review-days0-46eac81a89ff6db785dc84953c5770288a87d50b5a0a46d913f5d13b7682283a.json`.
После изменения полностью прочитан архивированный JSON размером 167 MB:
SHA256 совпал с исходным ключом, Range вернул 1024 bytes. Доказательство:
`speech-datasets/diagnostics/storage-tier-20261015/review-days0-readback-ab847f4d180fc0d8fcea25c056752aa36eb32b72ec86cab9b78809e40cba386a.json`.

Во время первого массового переноса возникла просадка S3: исходные
`MemoryHigh=12G`, `MemoryMax=16G` вызвали cgroup memory reclaim (около 48% full
memory pressure), очередь listener выросла до 535 подключений. Промежуточного
лимита 20/24 GiB тоже оказалось мало. Процесс MinIO не перезапускался. Scanner
досрочно возвращён в `default` в 16:06 UTC. Штатный ILM ограничен с 100 до 8
параллельных переносов; изменение применено динамически и прочитано обратно.
Для primary MinIO сохранены лимиты 48/64 GiB на сервере с 251 GiB RAM; перед
увеличением проверено более 129 GiB MemAvailable. Secondary не менялся.

Воспроизведение ресурсных настроек на dev-1 (не требует restart):

```sh
sudo systemctl set-property speech-data-minio.service MemoryHigh=48G MemoryMax=64G
# В приватном admin-окружении существующего MinIO:
mc admin config set speech ilm transition_workers=8
```

`ilm` поддерживает динамические настройки в используемом MinIO release:
https://github.com/minio/minio/blob/RELEASE.2025-09-07T16-13-09Z/internal/config/config.go
После изменения health отвечал за 3–7 ms, чтение ops JSON за 52 ms. В 16:09 UTC
свободное место drive2 выросло с 199 до 211 GiB; проверять дальнейшую скорость,
обновление inspection pointer и memory pressure, а не считать перенос законченным.

Из 61 задания с `ambiguous_worker_error` восстановлены 32 без новых генераций:
19 API и 13 CLI. Результаты остались `NEEDS_REVIEW`, human gold не выставлялся,
новых attempts и публикаций нет. Для API проверены модель, схема, исходный
контракт, SHA256 аудио и совпадение structured output с текстом; у 12 из них
полные validated native metadata сохранились, но исходный HTTP response
отсутствует. Это явно отмечено `recovery_raw_response_available=false`.
Исторический API-контракт до `374ad79` проверяется без добавления отсутствующего
в его исходном хеше generation config; записанный контракт не заменяется новым.

Первые шесть восстановлены отдельно. Первая диагностическая попытка ошибочно
отклонила 13 CLI: образ scheduler содержит старый `agy_result(text,path)`, а
работающий worker использует параметр `expected_model`. Для повторной проверки
взят точный worker parser, SHA256
`3fea624d0c08eebbd36adb8753c2b2be2cfec7b5eb216fa1c990a9ff839ffb2a`, совпавший с
Git. Проверены записанные prompt, contract, audio SHA, exit code и штатный
строгий парсер. Время CLI inference не восстанавливается из предположений:
`inference_seconds=null` для восстановленных CLI. Production parser не менялся.

Остальные 29 не переотправлялись: для 23 есть stream без request.json, для шести
артефактов нет. Полное соответствие prompt/contract для них не подтверждено.
Исходный список ошибок сохранён до изменений; итоговый receipt:
`speech-datasets/diagnostics/storage-tier-20261015/recovery-receipt2-6a444e6fecc0ff5b84bf6e8af31af982abfe83262226f92ae31c48d3cfdc1b86.json`.
Все 32 восстановленных задания найдены в snapshot `2026-10-15T16:33:50.227911Z`:
проверены 15 полей, включая текст, модель, contract SHA, human flag, длительность,
raw prefix и nullable CLI inference time; пропусков и расхождений нет.
`speech-datasets/diagnostics/storage-tier-20261015/recovered-parquet-eb16d2efcd91858e5f9a473372366d69214f2c2f05255c680ed81f30b07dcb3f.json`.
В 16:12 UTC подтверждены scanner `default`, ILM workers `8`, тот же PID MinIO,
отсутствие OOM kills и нулевое текущее cgroup memory pressure:
`speech-datasets/diagnostics/storage-tier-20261015/recovery-9138fb43f64d4be132291bdc42eaf6ae874649ceb57b1f8535ab076414d2db7a.json`.


### Обслуживание истории AGY-03, 15.10.2026, 20:10 UTC

На фоне насыщения drive2 записью (около 400 MB/s в коротком замере)
измерена история CLI слота 03: protobuf 15.8 MB, SQLite 41.6 MB,
активный WAL около 300 MB. Большой WAL сам по себе не доказывает сбой.
Штатный `cli_history` заархивировал 10 разрешённых элементов после точного
maintenance hold и полного drain DB/AGY. Профиль не изменён; файлы сохранены
на том же volume, поэтому это обслуживание истории, а не освобождение диска.
Воркер не перезапускался, его API guard и реальные паузы сохранились.
В 20:10:28 UTC возвращены DB concurrency=10 и interval=0.2; единственная
ошибка во время drain — терминальный `native_result_rejected`, без replay.
Проверены четыре новых результата (три CLI, один API): raw output, модель,
контракт и сохранённый текст согласованы. Для API также проверены SHA256
аудио, один HTTP-запрос и отсутствие автоматического повтора.
Доказательства: `speech-datasets/diagnostics/cli-history-20261015/`:
`agy03-5a2d67444c2c94d48a8928cb83af50de100b1b2ca0c59cd5eff3f20e3d66be73.json`
и `agy03-post-6e9ef03825d71a0f9a41d0dec857edab5c63a73b776121c0cc1b632f65cdd0c6.json`.
Замер не доказывает устранение общей нагрузки диска: остальные слоты и
PostgreSQL продолжают писать. Обновление старого claim-кода 03 отложено
после свежего API failure; перезапуск для сброса guard не выполнялся.


### Обновление выдачи заданий AGY-07, 15.10.2026, 20:18 UTC

Слот 07 загружен с account `SKIP LOCKED` после точного maintenance hold,
нулевого DB RUNNING и отсутствия AGY-процессов. Во время drain ошибок нет.
Контейнер перезапущен в 20:15:18 UTC; новый StartedAt подтверждён.
Это CLI-only слот: `NATIVE_API_PERCENT=0`, аккаунт отсутствует в native route
map, за предыдущий час API-ответов нет. Действующая пауза API не сбрасывалась.
Проверочные задания запущены в 20:15:44 UTC с DB concurrency=1.
Четыре последовательных CLI-ответа прошли проверки модели, схемы, SHA256
исходного аудио в request, контракта, единственного успешного turn и
разрешённого чтения аудио. Сохранённый текст совпал с raw structured output.
В 20:18:00 UTC возвращены concurrency=3 и interval=2 без изменения текущего
next_request_at. Доказательство:
`speech-datasets/diagnostics/queue-contention-20261015/agy07-canary-a2691705908a0ea1e9854891d4f7f654b188524f487b8dc7e9086b0beb9df249.json`.
Остальные старые воркеры ещё могут создавать очереди блокировок;
это обновление одного слота не доказывает устранение всех ожиданий PostgreSQL.


### Обновление AGY-06, 15.10.2026, 20:25 UTC

Перед maintenance после последней native API ошибки подтверждены 166
успешных native результатов. Точный hold установлен в 20:21:47 UTC,
полностью завершены DB RUNNING и AGY-процессы; во время drain ошибок нет.
Перезапущен только 06, новый StartedAt — 20:22:56 UTC. Загружены
account `SKIP LOCKED`, обработка provider prompt blocks и ранее проверенный
worker-код; CLI остаётся pinned 1.3.4. Проверочные задания начались
в 20:23:34 UTC с concurrency=1. Четыре результата (один CLI, три API)
прошли проверки raw output, модели, схемы, контракта, аудио SHA256 и
сохранённого текста. Для CLI проверены один успешный turn, audio-only tools
и точный prompt; для API — generation config, один HTTP-запрос, ноль retries.
В 20:25:38 UTC возвращены concurrency=10 и interval=1 без изменения
текущего next_request_at. Доказательство:
`speech-datasets/diagnostics/queue-contention-20261015/agy06-canary-e88fedce99ea83ab9d040908919be7760df01e0b3e76d23271d39d77f3335d48.json`.

Подача данных отдельно проверена: свежие manifests по 128 строк выходят
примерно каждые 20–25 секунд, ошибок текущего source cursor нет.
Проверка pending_work заняла 14 ms; запас ограничен PREPARE_MAX_PENDING=6000.
Редкие поступления в предыдущем окне не означали остановку ingest:
при большом запасе preparer штатно ждёт, а затем продолжает выпуск.


### Схема identity indexes, 15.10.2026, 20:47 UTC

В schema.sql сохранены три уже существующих производственных индекса:
`jobs_audio_identity`, `jobs_conversation_identity` и частичный уникальный
`precision_v1_audio_pass_model_uq`. Первые два используются split guard
(живой EXPLAIN ANALYZE: 0.219 ms), третий запрещает повторную обработку
одного audio/pass/model в текущем precision-прогоне, включая другой
conversation_id и явное указание модели по умолчанию. Рабочие индексы
подтверждены valid/ready; DDL на production не выполнялся.

Полная схема проверена в изолированной транзакции: два дубля отклонены,
четыре независимых контроля (другой pass, model, audio, исторический pilot)
приняты. Транзакция откачена, тестовая схема отсутствует. Main и frozen
копии schema.sql синхронизированы после проверки исходного SHA256;
backup — runtime/backups/schema-identity-20261015. Новый SHA256:
`9de926ea71a0bc73af6a08ce81c666d253a6e6e6ea49d5281f9c75dbd649f9d8`.
Доказательство:
`speech-datasets/diagnostics/queue-contention-20261015/schema-identity-d37183836ea32ba564dea3f9d736b51e59dec0ad4d632d43d598b5d6f08e0d9d.json`.

### Проверка полной входной партии, 15.10.2026, 21:26 UTC

Для manifest `incoming/c366d8842f6a50100c8333e67a11f7ce947d373e3dee24fb1bee8639c88f9ecb.jsonl`
проверены все 128 WAV и decoded PCM: SHA256 уникальны, длительность 441.6 s,
исходные метки и provenance сохранены, пересечений с защищёнными identities нет.
103 допустимых первых ответа получили все 206 дополнительных проходов.
Всего 334 задания завершены: 325 сохранённых ответов полностью сверены с raw,
девять ошибок остались в карантине без автоматического повтора.
28 совпадений трёх проходов — кандидаты до проверки исторических конфликтов,
не human-gold и не разрешение публикации train-набора.

После завершения штатного Airflow export все 334 задания найдены в immutable
Parquet со срезом 21:06:34 UTC; 11 полей, включая текст, исходную разметку,
state/error, attempt, contract и review flags, совпали с DB. Пропусков,
повторных job IDs и расхождений в проверенной партии нет. Проверка прочитала
14.5 MB через Range, без повторного inference или перезапуска exporter.
Доказательства в `speech-datasets/diagnostics/input-conservation-20261015/`:
`fresh-c366-followups-74c0933dfbe177ecdec438cd93f87b1316249ebbdb80e96cec643c508b243961.json`
и `fresh-c366-final-parquet-a5d74a59bcd42b5520d3e274f0c892408c63bd2836f4cea8bf0a24cd39d383a5.json`.

### Обновление AGY-12, 15.10.2026, 21:45 UTC

Перед обновлением подтверждены как минимум 77 успешных native ответов после
последней ошибки, без действующей паузы маршрута. Maintenance hold установлен
в 21:39:19 UTC; полностью завершены DB RUNNING и AGY-процессы, ошибок при
drain нет. Перезапущен только 12, новый StartedAt — 21:40:33 UTC. Загружен
worker `c437af5c94e1b0c455f22dab612fdc12f0e92baed5fd9920a7fbf5e1610086f5`
с account `SKIP LOCKED`, обработкой provider prompt blocks и проверкой OAuth
expiry перед native запросом. В 21:41:07 UTC начат canary с concurrency=1.

Четыре native результата сверены с raw: model/schema, фактический WAV SHA256,
prompt/reference, contract и сохранённый текст совпали. Каждый использовал
один HTTP-запрос и ноль retries; в артефактах generation config — 8192 токена.
В 21:41:50 UTC возвращены concurrency=10 и interval=0.2 без изменения текущего
next_request_at. К 21:45:54 UTC после восстановления сохранены ещё 128
review-only ответов без ошибок; два задания выполняются. Это проверка
работоспособности и сохранности, а не человеческое подтверждение транскрипций.
Доказательства в `speech-datasets/diagnostics/queue-contention-20261015/`:
`agy12-canary-c8f165670999526d97059fb4f88c15acc09d3760059db5ca071085637eec9057.json`
и `agy12-post-fd442cb1167ddf514bf42e37527f532d608552f26f83a735a6019febcf29919d.json`.

Отдельно проверен прежний incomplete-ответ 03: HTTP 200, ноль кандидатов,
`promptFeedback=OTHER`, без исчерпания 8192-токенного бюджета. Новый validator
на том же immutable ответе возвращает `native_prompt_blocked_OTHER`; исходное
задание не изменено и не повторялось. Это один подтверждённый случай provider
sample rejection, не доказательство причины всех неполных ответов или VPN-сбоя.
Доказательство:
`speech-datasets/diagnostics/native-incomplete-20261015/agy03-classification-80acacf14c429a6bbef63ee46872aaa8a6e900939c6a11a9eb85d4d877add1fd.json`.

Более поздний сбой 03 в 21:43:47 UTC имеет другую причину: `RemoteDisconnected`
после 15.123 s, HTTP-ответа нет, один вызов inference и ноль auto retries.
Это неоднозначно завершившаяся генерация: задание остаётся в карантине,
действующий native cooldown сохранён, обновление воркера отложено. CLI продолжает
работать. Не следует переносить диагноз `OTHER` на этот сетевой случай.
Доказательство:
`speech-datasets/diagnostics/native-incomplete-20261015/agy03-route-f228a377ad04cbab173a983f8a313d390b6297ad5379702cbf4d5cdbec0ba008.json`.

### Inspection snapshot v2, 15.10.2026, 22:17 UTC

Full inspection exports repeatedly aggregated the complete JSON-heavy job
history in PostgreSQL. The export now reads sources, passes and publications
in the same read-only repeatable-read transaction, then derives examples and
progress from that exact pass Parquet using DuckDB 1.4.1. It reuses
`review_views.sql`; notebook and pipeline counts have one definition.
The aggregation uses one CPU thread, a 2 GB memory limit and bounded temporary
spill. No new service is introduced. DuckDB is already used by Jupyter;
the Airflow image pins the same standalone package without upgrading its
existing boto3/psycopg/pyarrow/numpy dependencies.

Source annotations are recovered from structured provenance: a string that
looks like JSON stays a string, and JSON null differs from a missing label.
Snapshot manifest version 2 records the engine and annotation serialization.
Examples' annotation JSON may have different whitespace/key order; tiny
floating-point aggregation rounding differences are expected. Raw pass
serialization, transcript text, review flags and training publication rules
are unchanged. All five immutable files precede the discovery pointer.

The actual new Parquet writer was compared against one existing consistent
snapshot with 1,497,771 passes and 565,720 examples. Both output schemas match;
all examples match after JSON normalization, with no missing groups or other
field differences. All 157 progress groups have identical integer counts,
timestamps and nullness. Maximum differences were 1.03e-10 audio-hours and
2.92e-13 mean inference seconds. Deriving and writing the two tables took
8.11 seconds; this excludes PostgreSQL scanning and S3 uploads, and is not a
claim that the whole export completes in eight seconds.

Four runtime tests cover typed labels/review flags, empty tables, single pass
reading and pointer-last publication, and failed aggregation leaving the old
pointer intact. They passed in the live scheduler's Python environment.
Evidence:
`speech-datasets/diagnostics/inspection-duckdb-20261015/full-fixture-e3d414fb82b2c8b5fecc27853f4a7c3f7a2dd9d6bb0c9320f918dcb1678e80f5.json`.

The inspection DAG's previous unpaused state was restored at 22:17:33 UTC
after a source-update hold. Its already-running 22:00 export was preserved;
no scheduler or annotation worker was restarted. The next export loads v2.
The first live v2 run finished successfully in 522.81 seconds. Its 22:26:09 UTC
snapshot contains 1,522,944 passes and 575,311 examples. Full S3 readback verified
all five file SHA256 values, all footer counts, conservation of pass/text counts
across examples and progress, and zero human-verified examples. Evidence:
`speech-datasets/diagnostics/inspection-duckdb-20261015/first-live-v2-51c3d78ba988aac7fd12d17f97607ab4bc0c9ef1b13b556536a16ac3d77c335c.json`.

The view writer now resolves Arrow string-column names once per view rather
than inspecting Arrow field objects for every cell. Three paired runs over
10,000 real pass rows produced byte-identical Parquet files. Warm preparation
and writing took 0.20–0.21 seconds instead of 0.35–0.36 seconds; database reading
and S3 upload are excluded. All four snapshot tests passed in the actual
scheduler environment before deployment. This does not establish a speedup
for the complete export. Benchmark evidence:
`speech-datasets/diagnostics/inspection-serialization-20261015/benchmark-c09aac9a75d77833dbcf5907e973210d26cd2be4f5758c563ea73046f16b2b69.json`.

The first completed export starting after this source update took 609.73
seconds and published 1,560,907 passes and 591,504 examples. All five schemas
and footer counts matched the live views; example/progress request and text
counts agreed, and human-approval flags remained false. This is one observed
runtime, not a measured whole-export speedup. Verification:
`speech-datasets/diagnostics/inspection-serialization-20261015/first-live-498bd45124c021e4f2c033bc20a04a7cbf4db81c959fda73037da567fcabdffb.json`.

### Native organization-call pilot

16 October: user excluded OpenSTT from Gemini annotation. Source preparation
was stopped gracefully, 4,890 queued jobs were quarantined with
`corpus_excluded_by_user`, and 31 source assets were marked `EXCLUDED_POLICY`.
Existing results and in-flight requests were preserved. `source_policy.sql`
guards new inserts and retry transitions in PostgreSQL, including older live
workers; source selection excludes both `openstt-phone2` and
`openstt-balalaika`. Fresh initialization installs the same guard. Temporary
PostgreSQL fixtures verified insert/retry exclusion and source scheduling,
including preservation of allowed MTEDx work. The first combined DDL/backfill
transaction deadlocked and rolled back; guard installation and bounded
`SKIP LOCKED` backfill then committed separately.
Receipt: `speech-datasets/diagnostics/source-policy-20261016/exclude-openstt-62aab0d6039cbe2245f6d755532b2bb7432d15eb23c84a2d6711a2d9c6d82fd6.json`.
Meteor dialogue context is the next priority; MTEDx has not been started.

`python -m lalm.annotation.contextual_calls --manifest <incoming-key>` builds
a separate review-only Parquet from completed, conversation-timed pilot windows.
It makes no inference requests. Up to five nonempty Russian predictions from
the same conversation are attached only when their audio ends before the current
window starts. Uncertain predictions are excluded from history. Channel IDs
remain unverified speaker IDs; these windows are not aligned conversational turns.
The full native transcript stays in the source index, separate from targets.
Records include the source job IDs, result hashes, audio hashes and contract.
This does not mean the annotation requests themselves used dialogue history.

The first Meteor-only context export contains 99 windows from six calls,
77 with history. Full Parquet readback SHA256 and an audio readback through the
Jupyter read-only identity passed. The shared notebook now has a Meteor section
with the history, target and audio player; its prior cells/outputs were preserved
and a checkpoint created. All rows remain human-unverified and not released
for training. Receipt:
`speech-datasets/diagnostics/contextual-review/0f47e99776a5a1dde46da6e7de50821ca40c472aea12493029759edf9774f3eb.json`.

A second Meteor-only batch was staged on 16 October at 16:29 UTC using the
same adapter: 12 previously unseen conversations, 2,791 declared seconds,
301 distinct audio windows. Encoded bytes matched the existing native archive
inventory. Scheduled Airflow ingest imported all 301 at 16:30:23 UTC.
Manifest: `incoming/7b81de1c75a102a8fe40a2c3801c47008e535e95f5093ddb8f50afef5976ee5a.jsonl`.
At 16:31, 252 results were saved (251 Russian review-only, one foreign), four
failed jobs were quarantined and 45 remained running. This is a partial progress
observation, not a completed batch or a new history export. Two fresh CLI and
two API outputs passed raw text/contract/audio checks:
`speech-datasets/diagnostics/storage-integrity-20261016/fresh-native-cli-91d5b17fd7cb780f38c7dd84818eb78d1d41e95f643f14875f59880c4ef0b271.json`.
Import/state receipt:
`speech-datasets/diagnostics/organization-calls-20261016/batch2-import-b79b9330acfd51a4c95a2be81450cf55d32a2d859330725844f9c4d7d89ef0ba.json`.

Native metadata joined against the encoded-audio inventory currently identifies
2,161 Meteor conversations with audio and source text (2,161 unique encoded
hashes; 82.812 declared hours). HashTesh has 1,917 such conversation IDs but only
1,477 encoded hashes, so conversation counts do not measure unique audio there.
These are candidate inventory statistics, not measured durations or human-gold
quality estimates. Test/holdout checks remain mandatory during staging.

Additional native inventory, 16 October at approximately 17:48 UTC:

- All 1,708 Meteor calls of 60–1,800 declared seconds have already been selected
  across the pilot and active feeds; selection is not annotation completion.
  Another 450 unselected calls are shorter than 60 seconds. One unselected call
  declares 7,429 seconds and needs a bounded preparation check before feeding.
- Meteor has 1,374 additional native dialogue rows whose current audio key is
  empty. The 18:16 UTC follow-up found no archived audio through previous
  relations either. All have positive declared duration and 1,327 have source
  text, but neither proves a usable recording. Of these rows, 1,210 have URLs
  on the CRM host `meteor.s20.online`; none of those URL paths matches an audio
  member in the archive. The URLs were not followed. Recovering a verified audio
  identity/source is required before counting these rows as runnable input.
- HashTesh has 1,919 native rows, 1,917 distinct conversation IDs, and 1,477
  unique encoded audio hashes (134,498 declared seconds, about 37.36 hours).
  Of these hashes, 1,471 have not been selected in saved pilots/feed selections:
  758 calls of 60–1,800 seconds and 713 shorter calls. Choose one native row per
  audio hash rather than submitting all dialogue rows.
- No encoded audio hash overlaps between Meteor and HashTesh, and no native
  conversation ID points to multiple encoded hashes in this inventory. This
  does not establish decoded-PCM uniqueness or holdout isolation; both remain
  staging/release checks. No extra model requests were made for this inventory.
- MTEDx is present in the source registry as both an archive and HF Parquet
  shards, including validation/test shards. It remains unstarted. Do not feed
  both representations or validation/test as training input. Native production
  conversations take priority for the dialogue-history work.

Inventory evidence:
`speech-datasets/diagnostics/organization-calls-20261016/additional-native-inventory-73322ad14b2ba566210c3a9ac09763b10232aacfb579eee6075979b66cc783ac.json`.

Missing-audio follow-up:
`speech-datasets/diagnostics/organization-calls-20261016/meteor-missing-audio-classification-f0033a15f53c36edaa0967b5fdbbe445ec37e60a0021dfc43a0004babadd3098.json`.

At 17:50 UTC a HashTesh reserve was prepared on dev-1 in
`/mnt/local/drive2/speech-native-calls-dedup-20261013/hashtesh-context-batch-20261016-1750-reserve`:
758 calls, 103,681 declared seconds (28.80 hours), 416,445,696 encoded bytes.
Each selected audio hash and conversation ID is unique and excludes saved prior
selections. Resolving 95 archive hardlinks was necessary; actual audio bytes were
read from the regular target members and verified against the original inventory
SHA256. An initial extraction stopped at the regular-file check, without a
selection manifest or inference requests; the resumed preparation verified any
previously written file before keeping it. S3 and database holdout checks both
found zero matches. This is an extracted reserve, not yet submitted, and the
existing native feeder must finish before another starts. Holdout checks run
again during normal staging; decoded-PCM overlap is still unproven.

Reserve preparation/holdout evidence:
`speech-datasets/diagnostics/organization-calls-20261016/hashtesh-reserve-b8ff2139228e2c2ba08b559a21cc176ba719ec0161aad8fa769fa4a5838441fb.json`.

The inspection DAG now chains `publish_hashtesh_contexts` after
`publish_meteor_contexts`, reusing the same exporter and canonical receipt
checks. It issues no inference requests. The updated DAG parsed successfully
in the deployed Airflow runtime. The PostgreSQL fixture check also verifies
that HashTesh cohort selection excludes Meteor, with all fixtures rolled back.
The first scheduled HashTesh task succeeded in the 17:50 UTC run, executing
17:55:03–17:55:10 UTC (7.32 seconds). Meteor's preceding context task also
succeeded in 14.44 seconds.

An explicit first export of the already finished mixed native pilot published
only its six HashTesh conversations: 61 Russian windows, 41 with prior history.
The shared notebook's last section now selects `Meteor` or `HashTesh` through
`context_organization`, keeping the other 30 cells intact. A Hub checkpoint was
created before updating it. Both choices were executed in a separate read-only
verification process (not the user's notebook kernel): all cohort Parquet
hashes, row counts, duplicate IDs, chronological history and one audio preview
hash per organization passed. Meteor then had 74 cohorts / 881 conversations /
14,901 windows / 11,815 history-bearing windows; these remain predicted,
review-only records and are not human-gold accuracy measurements.

Notebook verification evidence:
`speech-datasets/diagnostics/contextual-review/notebook-both-organizations-a92617c1b9d7ce3cb75d7338550e382c8f6d02989eb2bb654780d27a50068fd0.json`.

MTEDx split protection was applied before enabling any generic preparation:
the source selector now allows only explicitly named `mtedx-ru-hf` train shards.
Its validation/test shards, unknown shard names and the mixed-split `mtedx-ru`
archive are excluded from this generic path, which assigns `split=train`.
The three existing non-train/mixed sources were moved from QUEUED to NEEDS_REVIEW
with `policy_reason=mtedx_official_split_requires_adapter`; all 18 train shards
remain queued. Source bytes were preserved and no inference was performed.
PostgreSQL fixtures verified exclusion of held-out/mixed/unknown inputs while
retaining train selection and existing source rotation, then rolled back.

Policy receipt:
`speech-datasets/diagnostics/source-policy-20261016/mtedx-split-guard-5e56dd9d5c0d18b185b21a0be02879e3d0afd7302609b4969aa6acc337118068.json`.

At 18:01 UTC the user's OpenSTT exclusion was extended from the two initially
observed corpus names to every `openstt-*` prefix, including `openstt-phone1`.
Both the source selector and the existing database insert/retry trigger now
apply that rule. The live function was replaced without dropping its trigger;
temporary PostgreSQL fixtures verified phone1/phone2/balalaika/future OpenSTT
names and preservation of unrelated Balalaika and MTEDx inputs. A bounded
backfill found no additional queued jobs or queued/preparing source assets,
and the final pending OpenSTT count was zero. Previous outputs remain intact.
This does not exclude the separately catalogued proprietary/Youtube Balalaika
corpus whose name has no OpenSTT prefix.

Policy receipt:
`speech-datasets/diagnostics/source-policy-20261016/all-openstt-27bdfe21ce96aeb83e05e17a7896d2851caa862346f81ea2ca955ab3f106c2d4.json`.

The 1,560-call remaining Meteor feed exited successfully at approximately
18:05 UTC after staging its final 12-call cohort (216 windows). This is input
preparation completion, not completion of all annotation jobs. The prepared
758-call HashTesh reserve then started through the same feeder. Its first
12 calls produced 193 windows in
`incoming/c389422549d40580b5df730f4733d9577efb717f3b6e15bf80dc612a496b080c.jsonl`;
the feeder paused normally when combined pending work reached 6,033.

Use separate mount destinations for private metadata and audio selections.
Nesting a metadata mount under the read-only `/pilot` mount fails at container
creation. The private 0700/0600 source files also require this one-off container
to use `--user 0:0`; permissions on the host files remain unchanged. Both initial
startup failures happened before publishing inputs or dispatching inference.
The working invocation on dev-1, from the deployed repository, is:

```bash
sudo -n infra/annotation/dc run --rm --no-deps --user 0:0 \
  --name speech-data-hashtesh-feed-reserve-20261016 \
  --volume /mnt/local/drive2/speech-native-calls-dedup-20261013/meteor-pilot-metadata:/native-metadata:ro \
  --volume /mnt/local/drive2/speech-native-calls-dedup-20261013/hashtesh-context-batch-20261016-1750-reserve:/pilot:ro \
  --entrypoint python source-preparer -u -m lalm.annotation.organization_feed \
  --metadata /native-metadata --selection /pilot/selection.json --audio-directory /pilot/audio
```

Do not launch this again while its container is running. The PostgreSQL feeder
lock and verified S3 checkpoints retain the existing exclusion/resume behavior.

At 18:10:54 UTC Airflow imported the first HashTesh manifest as 193 pending
jobs. It did not bypass the earlier Meteor jobs in FIFO order. Native source
selection and input preparation continue independently with pending-work
backpressure.

The 17:59 UTC usage collection succeeded for all 13 measured profiles, including
paid-tier discovery and interactive-PTY cleanup. Its canonical receipt is
`speech-datasets/analytics/usage/28a0f79fb9372852737feb8ec92e66eaabccf24b17b64bb4067a4f1b5077d414.json`.
Subsequent provider denials paused 01 until 20:49 UTC and 02 until 20:51 UTC
on 16 October (five-hour quota), and 10/16 until approximately 11:49 UTC on
20 October (weekly quota). These cooldowns were not cleared.

At 18:11 UTC the effective database cap for 18 was reduced from 10 concurrent
jobs / 1-second minimum interval to 3 / 2 seconds. In the preceding 15-minute
sample it completed 30 attempts with 13 saved outputs, median 66.72 seconds
and p95 307.17 seconds, alongside provider-capacity and transport failures.
The other slots in the same identity family were already capped at three.
Existing requests were allowed to finish, and `next_request_at` was unchanged.
This bounds outstanding work; no recovery or throughput improvement is claimed
until subsequent observations.

Concurrency receipt:
`speech-datasets/diagnostics/provider-errors-20261016/agy18-concurrency-d99b5b9273a79252d087828fd5f3e3a4e0774bbe6563c49cb6808b39bcc023b3.json`.

The 18:20 UTC review cycle completed 200 groups in 59.70 seconds, leaving 350
changed groups awaiting review (oldest update about 535 seconds). Before that
cycle, the observed queue was 418 groups / 750 seconds. To give this five-minute
task more room to catch up with native call groups, its Airflow invocation now
passes `reconcile(max_pilots=300)` explicitly; the library default remains 200.
No acceptance rules, prompts, cursors or inference settings changed. The deployed
DAG parsed, and its callable passed 300 to a mocked reconciler without running
extra reviews or restarting services. The first real 300-group cycle at 18:25 UTC
completed in 68.47 seconds, requested no third passes, and left 254 changed
groups / 343 seconds oldest age. It fits the five-minute interval; sustained
backlog clearance remains under observation.

Baseline review receipt:
`speech-datasets/diagnostics/review-performance-20261016/native-batch200-2b0ce017967f6ee0517487926854a65ad91ded58b0c4efad1f92237f071f0a2d.json`.

First 300-group live receipt:
`speech-datasets/diagnostics/review-performance-20261016/native-batch300-9f9a5799025c53899b089bca192f8adf9e47d5c3110999d26b9e7441a70b3338.json`.

At 18:28 UTC the first new HashTesh cohort had saved 113 of its 193 windows,
with remaining work still queued/running/retrying. Four fresh HashTesh-only
results (two native API, two CLI) passed exact raw-text, model/prompt-contract
and audio-hash checks. This proves storage/protocol integrity for the sampled
outputs, not transcription accuracy or completed-cohort acceptance.
Evidence:
`speech-datasets/diagnostics/storage-integrity-20261016/fresh-native-cli-a075c66c47c22a238682febccb83f6721c8d8769ca98ee82969ab1e2cd1fa628.json`.

Before resuming the existing generic source preparer after HashTesh, a live
selection check found Golos crowd/train as the next source, but also queued
test/validation archives in other corpora. The selector now excludes named
`test`, `valid`, `validation`, `dev`, `eval`, and `evaluation` path components
or shard-name prefixes, case-insensitively. The existing database policy also
blocks pending/retried jobs whose source key or archive member names such a
split, because generic preparation currently assigns train. OpenSTT retains
its separate corpus-wide exclusion.

At 18:40 UTC, 42 additional sources were moved from QUEUED to NEEDS_REVIEW:
31 `golos-balalaika`, four `balalaika-longform`, and seven
`qwen-answer-nohistory-v1-20261015` tables. No corresponding jobs were pending
or running, no inference was submitted, and all source objects/results remain
intact. PostgreSQL fixtures verify source selection and insertion/retry guards,
including a held-out member inside a mixed archive and a nonmatching
`testament.wav` name; fixtures were rolled back.
Receipt:
`speech-datasets/diagnostics/source-policy-20261016/named-heldout-05d1e8f2e7012e2d85259830e37d292e07419bfb72149113a0865023f68b0b7a.json`.

A one-off operator wait is attached to the running HashTesh feeder's Docker
exit. Only exit code zero resumes the existing `source-preparer` service; its
current pending-work threshold is 6,000. The source selector's deployed hash is
checked before resume. No new service or separate preparation code is added.
The generic preparer remains stopped while this wait is pending.

The HashTesh feeder subsequently exited with code zero and the existing generic
preparer resumed automatically at approximately 18:52 UTC. The final HashTesh
audit verified all 64 checkpoint/manifests and registry row counts: 758 selected
calls, 757 represented calls, 11,152 windows and 11,152 distinct prepared-audio
hashes. One selected call produced no VAD windows. All input rows match their
selected original-audio hashes; this is completed preparation, not completed
annotation or a training release. Receipt:
`speech-datasets/diagnostics/organization-calls-20261016/hashtesh-feed-complete-dd98fd5f2332f9029f0d6810d2aac643e7f45bad80678284a8012e1850188bcc.json`.

The generic worker selected the expected Golos crowd/train archive and had
prepared 768 examples with zero errors at the 18:57 UTC check. Its first
128-row manifest was verified in full and imported as 128 pending jobs. Every
row names the train source, has no held-out member path, remains review-only,
and preserves unverified conversation identity rather than inventing dialogue
history. Receipt:
`speech-datasets/diagnostics/source-handoff-20261016/golos-start-2e26b80f6c5a8d0d4fe72eeba09522f15326b6bdee815c9f2c8ae231876cd16b.json`.

At 19:16 UTC the primary MinIO background scanner was temporarily changed from
`default` to `fast`, with a 30-minute systemd restore timer armed before the
change (`speech-data-minio-scan-restore-20261016-1915.timer`). The private restore
script changes it back only if it is still `fast`, preserving a different
setting subsequently chosen by another operator. No service restart or ILM
worker-count change was made. The current scan cycle had started at 15:55 UTC
and remained active; the selected two source archives were still pending.

The subsequent 20-second I/O sample showed average device waits of 0.75 / 1.77 ms
and busy fractions of 21.28% / 68.16% on the primary / secondary devices. This
sample did not show disk saturation. At 19:20 UTC the scanner reported 15,483
objects checked in the last minute; these global counters do not prove that the
two selected archives have transitioned. Continue checking their storage class
and complete original-URI readback before counting their space as freed.
Control receipt:
`speech-datasets/diagnostics/storage-tier-20261016/scanner-fast-ed60fd285b00e6ad84c1cc34bc2af05591f6225f338c62c9c5d94d954074a35f.json`.

For a bounded scanner-status read, use the installed mc command's
`admin scanner status -n 1 --interval 1 speech` with the existing private client
environment. Its default is an endless stream; waiting for EOF under a short
timeout does not establish a server failure. Parse the complete JSON object,
not individual lines of its pretty-printed output.

At 19:22 UTC the first Golos manifest reached inference after the older
HashTesh queue: 35 saved results, 41 running jobs and 52 pending. Four fresh
Golos-only outputs (two native API and two CLI, from 03/19/25/11) subsequently
passed exact raw-text, contract/model and audio-hash checks. This verifies the
stock-preparer handoff through result storage without changing acceptance
criteria or claiming listening-verified accuracy. Receipt:
`speech-datasets/diagnostics/storage-integrity-20261016/fresh-native-cli-523504c62d015eff38e09eb0d773719150b6837ffd95c1d2380c6e644fa31999.json`.

At 18:42 UTC the read-only Jupyter process verified all current cohort Parquet
hashes and one audio preview per organization: Meteor 143 cohorts / 1,709
conversations / 28,809 windows / 22,813 with history; HashTesh 11 cohorts / 125
conversations / 1,630 windows / 1,186 with history. The first new 12-call
HashTesh cohort now has its canonical context receipt: 182 Russian nonempty
windows, 139 with history, from 189 saved outputs across 193 source jobs.
No future or overlapping windows entered history. The user's notebook kernel
was not executed or changed, and all records remain review-only.
Evidence:
`speech-datasets/diagnostics/contextual-review/notebook-new-hashtesh-908d47606ebad3a0a633155c83bc7456c99b5ed81cc2a11f150919f36f64417a.json`.

`python -m lalm.annotation.organization_feed --metadata <native-csv-directory>
--selection <selection.json> --audio-directory <audio-directory>` feeds explicit
native selections in batches of at most 12 calls through the same adapter.
It checks the existing pending-work counter (including staged manifests) before
each batch; the default threshold is 6,000, with at most one batch of overshoot.
A PostgreSQL session advisory lock prevents concurrent feeders. S3 checkpoints
bind the selection and native CSV hashes; resume verifies each referenced
manifest SHA256 and registry row count before skipping it. No model request is
made by this process. Existing holdout, native join and hash checks still apply.

The first feed selected 120 previously unseen Meteor conversations, 19,180
declared seconds (5.328 hours), and staged 2,115 windows in ten manifests.
These are prepared inputs, not completed annotations or measured unique hours.
The live resume check skipped all ten saved batches and staged zero new manifests.
Receipt: `speech-datasets/diagnostics/organization-calls-20261016/feed120-7888cbaf582693d1848042e63c5f58c11aa1875b80ea64cb122cd45bfbf4319b.json`.

`inspection_snapshot_pipeline.publish_meteor_contexts` follows the existing
inspection export every ten minutes. Only terminal native-call cohorts are
exported; already indexed cohorts are skipped. Verified per-cohort receipts are
stored below `datasets/contextual-review/meteor/conversation-window-history-v1/manifests/`.
The shared notebook discovers these receipts and checks the referenced Parquet
hashes instead of requiring manual links. It retains review-only/human flags.
Initial read-only notebook validation found 137 conversations, 2,473 windows,
1,992 with history across 12 completed cohorts; target and history are model
predictions, not human-gold. Repeating the exporter skipped all 12 cohorts:
`speech-datasets/diagnostics/contextual-review/repeated-export-b40b1e7682b56b36a0d4bff159a9fe4c44f2f5d7159133df7551d9de4f293dea.json`.

The exporter selects by verified native provenance: ingress normalizes the pilot
IDs. PostgreSQL fixture checks cover normalized IDs, waiting work, another
organization and unverified joins. The candidate DAG imported with the expected
two-task dependency; feeder tests cover backpressure, resume and corrupt saved
manifests. Live dedup audit counted 2,879 native-source jobs, all covered by the
existing precision unique index, with zero audio/pass/model duplicate groups:
`speech-datasets/diagnostics/input-conservation-20261016/native-dedup-6722522bd08747c67c4ec285e58f8227d7c8709a7f0391d191a62b2e257b1042.json`.

The scheduled 17:00 UTC run completed the inspection export at 17:04:24 and
the dependent context task at 17:04:36: one new cohort published, twelve existing
cohorts skipped, zero inference requests. Scheduled-run evidence:
`speech-datasets/diagnostics/contextual-review/first-scheduled-91667266422198ab91619897ed286d87c55aff035bf4cf7ed0046bedd7aa3f7b.json`.
After the bounded feed/resume checks, another 1,560 previously unselected Meteor
calls (60–1,800 declared seconds per call; 245,187 seconds total, 68.108 hours)
were extracted with encoded-byte hashes verified. Their existing-adapter feeder
started around 17:01 UTC and waits when pending work reaches 6,000. Source
preparation for unrelated corpora remains stopped; OpenSTT stays excluded.

### Native throughput checks and review batching, 16.10.2026, 17:21 UTC

Four newly imported Meteor calls were decoded and segmented again in the
existing preparer image: all 36 windows matched the stored S3 bytes, audio
hashes, channels, time ranges and native conversation IDs. This verifies audio
provenance, not transcription accuracy. Receipt:
`speech-datasets/diagnostics/organization-calls-20261016/window-reproduction-932fe050f6a76b301d76b790ceccfaa0af8d7a6475298d278240bb637db4b21a.json`.

Native calls create one pilot per conversation. The old 25-pilot review cap
could process only 75 groups per 15 minutes while 288 new groups arrived;
285 groups were waiting at 17:14, oldest update about nine minutes old.
`reconcile(max_pilots=200)` now uses a bounded, validated, caller-tunable limit.
PostgreSQL fixtures verified a 40/20 split of 60 groups, invalid-limit rejection,
failed-report recovery, follow-up deduplication, holdout guards, negations,
numbers and historical conflicts. No model prompts or acceptance rules changed.
The runtime file was replaced atomically with a private backup; no services
were restarted. The scheduled 17:20 and 17:25 cycles each produced 200 reports in 65.374 and
64.756 seconds, within their five-minute interval. The waiting count fell from
429 at 17:21 to 405 at 17:27 while new work continued; the oldest waiting update
was six to seven minutes old. The backlog has not cleared. First-cycle evidence:
`speech-datasets/diagnostics/review-performance-20261016/native-batch200-6f318515dbdc016bd9182c819f008f526e5a9e211bc2b5eb0a7bda1e95115b2a.json`.
Second cycle: `speech-datasets/diagnostics/review-performance-20261016/native-batch200-f318c7e510d4a1a258d67be7ba6f3c7181bda55f4848ec92c45db10a803b5e5e.json`.

Live feeder logs also verified automatic backpressure: pending work reached
6,115 at offset 708, then staging resumed and paused again at 6,045 for the next
batch. The threshold is soft, checked between batches:
`speech-datasets/diagnostics/input-conservation-20261016/native-backpressure-316e8970e5c00f2d84125619efb276efb6525251c2b2d9da29b2ab5ce28b5141.json`.

### Manifest lookup index for context exports

The exporter loads each completed source cohort by `sample->>'source_manifest_key'`.
Without an index, one 265-result cohort took 3,341 ms and scanned the full jobs
table. `jobs_source_manifest_key` was created CONCURRENTLY on the live database
in about 12 seconds, with normal writes continuing and no service restart.
The valid/ready/live B-tree occupies 15,958,016 bytes. Fresh databases create
the same index from `schema.sql`; existing live databases should use
`CREATE INDEX CONCURRENTLY jobs_source_manifest_key ON public.jobs ((sample->>'source_manifest_key'))`
outside a transaction rather than rerunning schema initialization.

The identical query then used a bitmap index scan and took 4.714 ms. The full
ordered `(id,sample,result)` digest of all 265 rows matched before and after;
this measures one cohort lookup, not the speedup of the entire DAG.
Plan before: `speech-datasets/diagnostics/contextual-review/cohort-query-plan-322764c05f1649ff61d09cd8db5b5f5f1c009793720eed84f9206d8b0950eb8d.json`.
Creation: `speech-datasets/diagnostics/contextual-review/manifest-index-created-450ea8035fa608318cba634253c97f3b2c50e81b37f1568e9a49d07f4e445a71.json`.
Content and plan after: `speech-datasets/diagnostics/contextual-review/cohort-index-after-9e067e409b0b5061a6fba63cec7847f605cde6a46b5cab26895c50615de503f5.json`.

`lalm.annotation.organization_calls` stages a bounded pilot (at most 12 calls)
through the existing `incoming/` manifests, prepared-manifest registry and
Airflow ingest DAG. It reuses the production audio segmentation and queue
holdout checks; it adds no service or scheduler. The operator supplies native
CSV metadata, an explicit selection with encoded hashes, and verified audio.

The adapter joins `dialogs.current_relation_id = relations.id`, verifies both
organization and dialogue ownership, and preserves the native conversation ID,
channel number and segment times. Channels have no inferred agent/customer
roles. Each source index retains the complete selected native CSV rows and
archive provenance; the source index itself has a content-addressed S3 key.
Whole-dialogue text is **not** attached as a short segment's source label.

Consequently this pilot gets the existing blind first pass and remains
review-only. The existing triad reviewer reports missing segment annotation;
it cannot accept these examples as three-pass candidates or training data.
Canonical decoded-PCM overlap checking is explicitly false and remains a
training-release requirement. The adapter checks the known encoded-audio and
conversation holdouts before publishing an incoming manifest. Failed
preparation may leave immutable source/audio objects, but publishes no work
manifest until all selected inputs pass. Retries produce the same manifest;
the existing queue still enforces audio/pass/model uniqueness.

This is a pilot of dialogue-preserving **segmentation**, not whole-call
contextual annotation: current windows remain at most 20 seconds, with up to
0.5 seconds overlap. Full-call recognition and aligned source correction need
separate validation before expanding this path.

The first Meteor/HashTesh pilot used 12 calls (1,519.704 seconds of original
stereo recordings) and produced 164 distinct windows. Scheduled Airflow ingest
imported all 164 at 22:51:38 UTC on 15 October. All completed with one queue
attempt: 109 native API results and 55 CLI results, all using
`gemini-3.8-flash-low`. Full raw-response audit verified model, schema, prompt
contract, encoded-audio identity and exact agreement with stored output for
every result. There are 160 nonempty Russian transcripts, four empty results
and 19 results reporting uncertainty (these categories may overlap). None is
human-verified or a training release; recognition accuracy is not established.
The 3,076.688 seconds summed over windows include both channels and overlaps
and must not be reported as unique call duration.

Manifest: `incoming/e2e269ca978e67102173d8bfb5433d384921b3f4492051f1374b84fcf88c780c.jsonl`.
Raw audit:
`speech-datasets/diagnostics/organization-calls-20261015/pilot-raw-audit-1d5474959e948f0ea233cc469fa5263602fa0fbfd1a19be9d59c83e7acec2e3a.json`.
Three native results use the legacy contract that did not bind the generation
budget; their raw completed responses were still independently validated.
All other native results explicitly record the 8,192-token configuration.

The subsequent 23:08:25 UTC inspection snapshot contains all 164 completed
pilot passes and examples. Every cohort field matched the live read-only
views, including text, hashes, provenance, errors and review flags; the
progress table retained four empty results and zero human approvals. Pass
and example checks used ETag-pinned range reads, not full-object SHA256
readbacks; the smaller progress object was fully checksum-verified. Evidence:
`speech-datasets/diagnostics/organization-calls-20261015/pilot-parquet-dbfbbc66886c91ab4cc9753b852057f77d010eef27fdf2a697868b9a1f8e6dfe.json`.

### PostgreSQL memory for inspection exports

Inspection transactions use `SET LOCAL work_mem = '64MB'` before fetching
10,000-row batches. This changes only the export transaction; the cluster and
other sessions retain their defaults. There is no database or Airflow restart.
The default 4 MB caused a temporary file of about 16.4 MB per measured batch.
Two paired, read-only 10,000-row probes, including reversed execution order,
produced identical row hashes within each transaction. PostgreSQL logged a
temporary file for each 4 MB fetch and none for either 64 MB fetch.

Fetch timings varied under concurrent load: 4/64 MB took 12.12/4.55 seconds in
the first pair; reversed 64/4 MB took 8.06/5.92 seconds. These observations prove
spill removal for the measured batches, not a whole-export speedup or absence
of spills for arbitrarily wide rows. All four snapshot checks passed with the
candidate in the deployed scheduler environment, including typed labels and
publication-pointer ordering. A real transaction verified 4 MB before, 64 MB
during the local setting, and 4 MB after rollback.

The source was atomically deployed at 2026-10-16 01:24:51 UTC, with the previous
file saved under `infra/annotation/runtime/backups/inspection-work-mem-20261016-0125/`.
Already-running exports retain their existing transaction settings.
Evidence: `speech-datasets/diagnostics/snapshot-io-20261016/work-mem-probe-8eb07008ed4b6f9c6165fba1389f320cf7c3a26476cbd0e33e1b799d90cf92db.json`
and `speech-datasets/diagnostics/snapshot-io-20261016/work-mem-reverse-115d4271645510bd50ea446724de6989b284578fb66fb0b288b7d724c3edc197.json`.

The first completed export starting after deployment ran from 01:30:18 to
01:34:36 UTC on 16 October: 258.77 seconds, with 1,600,938 passes and 607,801
examples. The preceding export took 595.18 seconds. This is an observed pair
under changing load, not a controlled whole-export speed benchmark. A bounded
12-sample trace during the new export observed no `BuffileWrite`, and database
temporary-file counters did not increase during that sample.

All five published Parquet schemas and footer counts matched their live views
and manifest. Example/progress request and text counters agreed; every pass
and example retained `human_verified=false`. The progress object was fully
checksum-verified; larger objects used ETag-pinned range reads, not full-object
SHA256 readback. Evidence:
`speech-datasets/diagnostics/snapshot-io-20261016/first-live-full-f10921272f992f8f64b8bbb19674f17da3d861c73ba537a070425e49ca185f2b.json`.


### Размер служебных SQLite-журналов AGY, 16.10.2026

При контроле свободного места отдельно проверять `conversation_summaries.db-wal`
в home volumes AGY: размер основной базы не отражает выделенное журналу место.
В 09:48 UTC у простаивающего 14 журнал занимал 2 110 931 472 bytes при базе
около 64 MB. Штатный `PRAGMA wal_checkpoint(TRUNCATE)` из Python SQLite,
`timeout=0`, `busy_timeout=0`, `query_only=ON`, завершился успешно за 43,8 с;
журнал стал нулевым, `quick_check(1)` вернул `ok`, SHA256 файла авторизации
не изменился. Задания и записи истории не удалялись, контейнер не перезапускался.

Для занятого 13 проверен только `PASSIVE`: возврат `(1,-1,-1)` означал занятый
checkpoint, освобождение места этим вызовом не подтверждено. Затем штатный
AGY сам уменьшил WAL с 5,98 GB до 0,34 GB. Размеры WAL меняются скачками;
несколько замеров роста не доказывают постоянную утечку. Проверять фактические
`st_blocks` и свободное место `df`, а также свежие результаты разметки.

Использовать только штатные операции SQLite:
<https://www.sqlite.org/pragma.html#pragma_wal_checkpoint>.
`PASSIVE` не ждёт завершения читателей/писателей; `TRUNCATE` может блокировать
писателей во время checkpoint даже при нулевом busy timeout. Поэтому для
`TRUNCATE` сначала остановить выдачу **только выбранному** слоту, дождаться
завершения его заданий и служебных CLI-сессий, затем вернуть прежние настройки.
Занятые профили принудительно не обслуживать. Файлы `-wal`/`-shm` вручную
не удалять; не менять режим журналирования и не выполнять VACUUM в рабочем потоке.
Новый daemon или периодическое принудительное обслуживание не добавлены.


### Резерв ext4 на диске данных dev-1, 16.10.2026

В 10:43 UTC для `/mnt/local/drive1` (`/dev/nvme3n1`, ext4) штатный резерв
для привилегированных процессов уменьшен с 5% до 1% командой
`sudo tune2fs -m 1 /dev/nvme3n1`. Перед операцией повторно проверены mount,
тип FS, UUID и исходные 48 837 732 reserved blocks; параметры сохранены в
закрытом runtime backup `drive1-ext4-reserve-20261016-1043.json` на drive2.
После операции прочитаны обратно 9 767 546 reserved blocks; UUID, block count,
block size, filesystem features, reserved UID/GID и mount options не менялись.

Обычным процессам стало доступно ещё 160 031 481 856 bytes: около 225 GB
вместо 65 GB. Физически свободных блоков осталось около 265 GB; это изменение
доступности существующего свободного места, не увеличение ёмкости или удаление
файлов. Оставлен резерв около 40 GB. Сервисы не перезапускались; source cursor
продолжил расти, S3 diagnostics записаны. У drive2 исходный reserved block count
равен нулю; его настройка не менялась.

Документация параметра: <https://man7.org/linux/man-pages/man8/tune2fs.8.html>.
Откат исходного значения на этом же проверенном устройстве:
`sudo tune2fs -r 48837732 /dev/nvme3n1`. Не переносить имя устройства на другой
сервер без проверки mount и UUID. Метаданные до/после и доказательство:
`speech-datasets/diagnostics/storage-integrity-20261016/ext4-reserve-8de00b30a7fb521575c1cfc926d25864281c1022c4534b2edab066e8753fc0fe.json`.
Продолжать учитывать отдельно `f_bfree`, `f_bavail`, фактические размеры WAL,
работу lifecycle transition и оставшееся место на обоих дисках.


### Ускорение сравнения проходов, 16.10.2026

Запуск `compare_independent_passes` за 11:00 UTC занял 349,84 с при интервале
300 с. Патч `2c52190` убирает вычисление подробных `SequenceMatcher` diff
для булевого признака согласия: используются те же списки нормализованных слов
и их точное равенство. Функция `compare()` с подробными различиями сохранена.

При одном изменившемся pilot SQL не загружает его собственные ответы в
историческую выборку: цикл Python и раньше отбрасывал эти строки. Для
нескольких изменившихся pilots история сохранена целиком; `IS DISTINCT FROM`
сохраняет старые строки без pilot ID. Если изменений нет, возвращается прежний
пустой результат без полной загрузки истории. Правила принятия, порядок
follow-up, holdout guards, курсоры и формат отчёта сохранены.

Проверки старой и новой версий на изолированной схеме PostgreSQL откатились
без записей в рабочую очередь/S3. Проверены повторный запуск после ошибки
записи отчёта, отсутствие дублей, holdout, отрицания, числа, пустые ответы,
конфликт с историей, NULL pilot ID и конфликт двух одновременно изменившихся
pilots. Дополнительно прошли 17 unit-тестов. Временный тестовый layout содержит
только Python helpers; его файлы удаляются после проверки.

На 640 парах из 128 реальных транскриптов решения старого и нового признака
совпали; 12 800 вычислений каждого варианта дали 0,1523 и 0,0555 с (2,75 раза).
Это microbenchmark признака, а не ускорение всего pipeline.
Патч атомарно применён в 11:16:50 UTC, без перезапуска воркеров. SHA256 нового
`review.py`: `0e086ab83fd4eefd07b41fba4e5518226207e6070d8d65af0e6bfb573b8ec624`.
Исходник и параметры сохранены в закрытом runtime backup
`review-before-perf-20261016-1116.py` / `review-perf-20261016-1116.json`.

Первые новые циклы 11:20/11:25 завершились за 235,55/253,10 с
(3 мин 56 с / 4 мин 13 с), оба внутри интервала 300 с. Предыдущие
старые циклы 11:10/11:15 заняли 257,29/264,79 с. Это наблюдение на живом
потоке, не контролируемый benchmark: входные данные и нагрузка меняются.
Счётчики нового отчёта согласуются с поступлением новых ответов; человеческая
верификация и автоматическая публикация не включались. Доказательство:
`speech-datasets/diagnostics/review-performance-20261016/first-live-bf1298b4922f1d65b0f399cfc122deb9f13c50f8f9d76cf595682a7a13d04f35.json`.

Два последовательных новых запуска завершились до следующего пятиминутного
тика. Сводка с временами, версиями кода, результатами проверок и счётчиками:
`speech-datasets/diagnostics/review-performance-20261016/two-live-17bc538500c99ce3c24a30405b779d6e9f0d12f2913caedb397389c83474531b.json`.
