@echo off
cd /d "%~dp0"
if not exist ".venv\Scripts\python.exe" (
    echo Creating venv...
    python -m venv .venv
)
echo Installing dependencies...
".venv\Scripts\python.exe" -m pip install -q -r requirements.txt
echo Starting DeepSea eDNA AI at http://127.0.0.1:8000
".venv\Scripts\python.exe" web_app.py
