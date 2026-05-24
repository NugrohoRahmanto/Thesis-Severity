#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export PYTHONPATH="${PROJECT_ROOT}:${PYTHONPATH:-}"
STREAMLIT_PORT="${STREAMLIT_PORT:-8501}"

if [[ ! -f "${PROJECT_ROOT}/config.json" ]]; then
  echo "Missing config.json at ${PROJECT_ROOT}/config.json" >&2
  exit 1
fi

STREAMLIT_BIN="${PROJECT_ROOT}/.venv/bin/streamlit"
if [[ ! -x "${STREAMLIT_BIN}" ]]; then
  STREAMLIT_BIN="$(command -v streamlit || true)"
fi

if [[ -z "${STREAMLIT_BIN}" ]]; then
  echo "streamlit is not installed in the current environment." >&2
  exit 1
fi

exec "${STREAMLIT_BIN}" run "${PROJECT_ROOT}/apps/ecg_inference_app.py" --server.port "${STREAMLIT_PORT}" --server.headless true --browser.gatherUsageStats false