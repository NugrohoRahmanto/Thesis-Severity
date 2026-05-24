#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export STREAMLIT_PORT="${STREAMLIT_PORT:-8501}"

exec bash "${PROJECT_ROOT}/run_app.sh" "$@"