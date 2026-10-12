# SpeechLLM: данные, Airflow и AGY

## Текущее размещение и просмотр (2026-10-13)

Runtime работает только на dev-1 (`exp-srv-rtx6000x4-02`), в
`/mnt/local/drive2/speech-annotation-20261012/code`. Локальная машина используется
только для доступа; прежний локальный стенд удалён после проверенного переноса.
Используется существующий S3/MinIO, второй экземпляр не разворачивается.
PostgreSQL хранит служебную очередь и состояние Airflow; аналитические таблицы
и датасеты публикуются в Parquet на S3.

После каждого успешного прохода DAG задача `publish_inspection_parquet`
публикует согласованный снимок таблиц `sources`, `passes`, `examples`,
`progress`, `publications`. `s3://speech-datasets/analytics/latest.json`
указывает на конкретные неизменяемые файлы с SHA256 и числом строк.
Нельзя читать glob всех версий: это повторно посчитает одни и те же записи.
Событие Airflow Asset для этого URI содержит ключ manifest и счётчики.
Снимок предназначен для инспекции, не означает разрешение на обучение.

Для просмотра используйте `notebooks/speech_data_inspection.ipynb` на dev-1
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
`speech_data_matches.ipynb` is a separate analyst copy, preserving existing work.

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
is still dry-run. `notebooks/speech_prompt_review.ipynb` reads the 14 saved results
and renders the four audio clips without issuing any inference requests.
