#!/usr/bin/env bash
# One-time setup for macOS / Linux: Python virtualenv + web build.
set -euo pipefail
cd "$(dirname "$0")/.."

PY=${PYTHON:-python3}
"$PY" -c 'import sys; assert (3, 11) <= sys.version_info[:2] <= (3, 13), f"Python 3.11-3.13 required, found {sys.version.split()[0]}"'
command -v node >/dev/null || { echo "Node.js 20+ is required (https://nodejs.org)"; exit 1; }

echo "==> Creating Python virtualenv in .venv"
"$PY" -m venv .venv
.venv/bin/python -m pip install --upgrade pip >/dev/null
.venv/bin/pip install -r requirements-dev.txt

echo "==> Installing and building the web app"
(cd web && npm ci && npm run build)

[ -f .env ] || { cp .env.example .env; echo "==> Created .env from .env.example (add your API keys there)"; }
echo "==> Done. Start the app with: scripts/run.sh   (then open http://localhost:8000)"
