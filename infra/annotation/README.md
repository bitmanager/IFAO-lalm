# SpeechLLM: данные, Airflow и AGY

Слой оркестрации в существующем IFAO-lalm. Модель разметки, инструкция и проверка
нативного трейса взяты из `lalm/benchmark_langfuse_asr.py`. Зафиксирована
`gemini-3.8-flash-low`; каждый файл получает отдельный процесс/разговор AGY.
Общий контекст между заданиями не переносится. Текст другого ASR не подсказывается.

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

## Поднять локальный стенд

Из корня репозитория:

```bash
python3 infra/annotation/render.py --accounts infra/annotation/accounts.example.json
infra/annotation/dc build init airflow-api minio
infra/annotation/dc up -d
infra/annotation/dc ps
infra/annotation/dc run --rm control status
```

`render.py` создаёт `.env` со случайными паролями, конфиги закрытых proxy и отдельные
volumes для login-state каждого аккаунта. Повторный запуск не меняет секреты.
Runtime и секреты исключены из Git и Docker build context. Не публикуйте вывод
`docker compose config` без `--quiet`: он раскрывает environment.

Airflow UI: `http://127.0.0.1:18080`, пользователь `admin`. Пароль хранится локально
в `runtime/airflow-auth/passwords.json`. Для другого компьютера используйте SSH
forward; внешнего публичного порта нет. DAG первоначально paused до установки
реестра benchmark. Базы и S3 доступны только внутри Docker-сетей.

Образ AGY использует установленный бинарник (проверен 1.3.3), только бинарник
монтируется read-only. Домашняя директория и авторизация текущего пользователя
не копируются. MinIO собирается из официального release tarball с проверкой
SHA256, поскольку публичный prebuilt image недоступен. Базовые версии закреплены;
после проверки на целевом хосте сохраняйте image digests для воспроизведения.

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
Конвертер поддерживает TCP/WS/gRPC+TLS и TCP+REALITY; неизвестные параметры
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

## Проверенный стенд 2026-10-12

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
контрольного «учусь». Обучающие задания не загружены. `agy-01` зарегистрирован,
но выключен; реальный вход в подписку, туннель с предоставленным VLESS и замер
максимальной параллельности ещё не выполнялись. Для них нужны будущие доступы.
