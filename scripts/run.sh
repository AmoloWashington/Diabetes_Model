#!/usr/bin/env bash
# Serve API + built web app on http://localhost:8000
set -euo pipefail
cd "$(dirname "$0")/../backend"
exec ../.venv/bin/python -m uvicorn app.main:app --host 127.0.0.1 --port "${PORT:-8000}"
