# Development mode on Windows: API with auto-reload on :8000 + Vite hot-reload UI on :5173
$ErrorActionPreference = "Stop"
$root = Join-Path $PSScriptRoot ".."
$api = Start-Process -PassThru -NoNewWindow -WorkingDirectory (Join-Path $root "backend") `
  -FilePath (Join-Path $root ".venv\Scripts\python.exe") -ArgumentList "-m", "uvicorn", "app.main:app", "--reload", "--port", "8000"
try { Set-Location (Join-Path $root "web"); npm run dev } finally { Stop-Process -Id $api.Id -ErrorAction SilentlyContinue }
