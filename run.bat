@echo off
setlocal
cd /d "%~dp0"

set "VENV_PY=.venv\Scripts\python.exe"
if not exist "%VENV_PY%" (
    echo [ERROR] The virtual environment is missing. Run setup.bat first.
    pause
    exit /b 1
)

REM Never fall back to a system Python: always run the project's own interpreter.
"%VENV_PY%" -c "import sys; sys.exit(0 if (3,11) <= sys.version_info[:2] <= (3,12) else 1)" >nul 2>nul
if errorlevel 1 (
    echo [ERROR] The virtual environment uses an unsupported Python version.
    echo         Delete the .venv folder and run setup.bat again with Python 3.11 or 3.12.
    pause
    exit /b 1
)

if not exist ".env" copy ".env.example" ".env" >nul

REM ---- Check configuration ---------------------------------------------
"%VENV_PY%" -c "from config.settings import load_settings; load_settings()"
if errorlevel 1 (
    echo [ERROR] Your .env file contains invalid values. See the message above.
    pause
    exit /b 1
)

REM ---- Check Ollama connectivity (the app also shows this in Settings) ----
"%VENV_PY%" -c "import sys; from config.settings import load_settings; from core.ollama_client import OllamaClient; sys.exit(0 if OllamaClient(load_settings().ollama_base_url).is_running() else 1)"
if errorlevel 1 (
    echo [WARNING] Ollama is not reachable. Start the Ollama app ^(or run "ollama serve"^).
    echo           The app will still open and show instructions.
) else (
    echo [OK] Ollama is running.
)

echo.
echo Starting Local RAG AI Assistant at http://localhost:8501
echo Press Ctrl+C in this window to stop.
"%VENV_PY%" -m streamlit run app.py --server.port 8501
endlocal
