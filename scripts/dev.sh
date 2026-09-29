#!/usr/bin/env bash
# Development mode: API with auto-reload on :8000 + Vite hot-reload UI on :5173
set -euo pipefail
cd "$(dirname "$0")/.."
(cd backend && ../.venv/bin/python -m uvicorn app.main:app --reload --port 8000) &
API=$!
trap 'kill $API 2>/dev/null' EXIT
cd web && npm run dev
