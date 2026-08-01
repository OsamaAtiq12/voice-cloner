@echo off
setlocal
cd /d "%~dp0"

if not exist ".venv\Scripts\python.exe" (
  echo Creating virtual environment...
  py -3.12 -m venv .venv 2>nul || python -m venv .venv
  if errorlevel 1 (
    echo Failed to create .venv. Install Python 3.12 and try again.
    pause
    exit /b 1
  )
)

if not exist "models\OmniVoice\model.safetensors" (
  echo OmniVoice model not found. Downloading now...
  ".venv\Scripts\python.exe" scripts\download_models.py
  if errorlevel 1 (
    echo Model download failed. See README.md
    pause
    exit /b 1
  )
)

echo Starting Voice Cloner...
echo Open http://127.0.0.1:7870 in your browser
echo.
".venv\Scripts\python.exe" -m uvicorn app.main:app --host 127.0.0.1 --port 7870
endlocal
