# Run without activating venv (avoids ExecutionPolicy)
Set-Location $PSScriptRoot
if (-not (Test-Path ".venv\Scripts\python.exe")) {
    Write-Host "Creating venv..."
    python -m venv .venv
}
Write-Host "Installing dependencies..."
& ".venv\Scripts\python.exe" -m pip install -q -r requirements.txt
Write-Host "Starting DeepSea eDNA AI at http://127.0.0.1:8000"
& ".venv\Scripts\python.exe" web_app.py
