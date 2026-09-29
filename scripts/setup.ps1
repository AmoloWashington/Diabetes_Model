# One-time setup for Windows (PowerShell): Python virtualenv + web build.
# Run from the repository root:  powershell -ExecutionPolicy Bypass -File scripts\setup.ps1
$ErrorActionPreference = "Stop"
Set-Location (Join-Path $PSScriptRoot "..")

$py = if ($env:PYTHON) { $env:PYTHON } else { "py" }
$ver = & $py -c "import sys; print('%d.%d' % sys.version_info[:2])"
if ([version]$ver -lt [version]"3.11" -or [version]$ver -gt [version]"3.13") { throw "Python 3.11-3.13 required, found $ver (install from python.org)" }
if (-not (Get-Command node -ErrorAction SilentlyContinue)) { throw "Node.js 20+ is required (https://nodejs.org)" }

Write-Host "==> Creating Python virtualenv in .venv"
& $py -m venv .venv
& .\.venv\Scripts\python.exe -m pip install --upgrade pip | Out-Null
& .\.venv\Scripts\pip.exe install -r requirements-dev.txt

Write-Host "==> Installing and building the web app"
Push-Location web
npm ci
npm run build
Pop-Location

if (-not (Test-Path .env)) { Copy-Item .env.example .env; Write-Host "==> Created .env from .env.example (add your API keys there)" }
Write-Host "==> Done. Start the app with: scripts\run.ps1   (then open http://localhost:8000)"
