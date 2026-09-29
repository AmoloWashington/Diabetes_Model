# Serve API + built web app on http://localhost:8000 (Windows)
$ErrorActionPreference = "Stop"
Set-Location (Join-Path $PSScriptRoot "..\backend")
$port = if ($env:PORT) { $env:PORT } else { "8000" }
& ..\.venv\Scripts\python.exe -m uvicorn app.main:app --host 127.0.0.1 --port $port
