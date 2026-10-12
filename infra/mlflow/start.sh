#!/usr/bin/env bash
set -euo pipefail

MLFLOW_ROOT="${MLFLOW_ROOT:-/workspace/speech-llm/mlflow}"
MLFLOW_HOST="${MLFLOW_HOST:-127.0.0.1}"
MLFLOW_PORT="${MLFLOW_PORT:-5000}"
MLFLOW_ALLOWED_HOSTS="${MLFLOW_ALLOWED_HOSTS:-localhost,127.0.0.1,127.0.0.1:${MLFLOW_PORT}}"
MLFLOW_CORS_ALLOWED_ORIGINS="${MLFLOW_CORS_ALLOWED_ORIGINS:-http://localhost:${MLFLOW_PORT},http://127.0.0.1:${MLFLOW_PORT}}"

exec "${MLFLOW_ROOT}/venv/bin/mlflow" server \
  --backend-store-uri "sqlite:///${MLFLOW_ROOT}/mlflow.db" \
  --default-artifact-root "${MLFLOW_ROOT}/artifacts" \
  --serve-artifacts \
  --host "${MLFLOW_HOST}" \
  --port "${MLFLOW_PORT}" \
  --allowed-hosts "${MLFLOW_ALLOWED_HOSTS}" \
  --cors-allowed-origins "${MLFLOW_CORS_ALLOWED_ORIGINS}"
