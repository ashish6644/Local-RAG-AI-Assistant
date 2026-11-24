@echo off
setlocal EnableDelayedExpansion
cd /d "%~dp0"
echo ==============================================
echo  Local RAG AI Assistant - Setup
echo ==============================================

REM ---- 1. Find Python 3.11 or 3.12 ------------------------------------
set "PYCMD="
where py >nul 2>nul
if not errorlevel 1 (
    for %%V in (3.12 3.11) do (
        if not defined PYCMD (
            py -%%V -c "import sys" >nul 2>nul
            if not errorlevel 1 set "PYCMD=py -%%V"
        )
    )
)
if not defined PYCMD (
    where python >nul 2>nul
    if not errorlevel 1 (
        python -c "import sys; sys.exit(0 if (3,11) <= sys.version_info[:2] <= (3,12) else 1)" >nul 2>nul
        if not errorlevel 1 set "PYCMD=python"
    )
)
if not defined PYCMD (
    echo [ERROR] Python 3.11 or 3.12 was not found.
    echo         Install Python 3.12 from https://www.python.org/downloads/windows/
    echo         and tick "Add python.exe to PATH" in the installer, then run setup.bat again.
    echo         ^(Python 3.13+ is not supported by the pinned dependencies.^)
    pause
    exit /b 1
)
echo [OK] Using: %PYCMD%

REM ---- 2. Virtual environment ------------------------------------------
if exist ".venv\Scripts\python.exe" (
    REM Reuse the environment only if it was created with a supported Python.
    ".venv\Scripts\python.exe" -c "import sys; sys.exit(0 if (3,11) <= sys.version_info[:2] <= (3,12) else 1)" >nul 2>nul
    if errorlevel 1 (
        echo [ERROR] The existing .venv uses an unsupported Python version.
        echo         Delete the .venv folder and run setup.bat again with Python 3.11 or 3.12.
        pause
        exit /b 1
    )
) else (
    echo Creating virtual environment...
    %PYCMD% -m venv .venv
    if errorlevel 1 (
        echo [ERROR] Could not create the virtual environment.
        pause
        exit /b 1
    )
)
set "VENV_PY=.venv\Scripts\python.exe"
if not exist "%VENV_PY%" (
    echo [ERROR] The virtual environment is incomplete: %VENV_PY% is missing.
    echo         Delete the .venv folder and run setup.bat again.
    pause
    exit /b 1
)

REM ---- 3. Dependencies -------------------------------------------------
echo Installing dependencies ^(this can take several minutes the first time^)...
"%VENV_PY%" -m pip install --upgrade pip
"%VENV_PY%" -m pip install -r requirements.txt
if errorlevel 1 (
    echo [ERROR] Dependency installation failed. Check your internet connection and the messages above.
    pause
    exit /b 1
)

if not exist ".env" copy ".env.example" ".env" >nul
if not exist "data\uploads" mkdir "data\uploads"
if not exist "data\chroma_db" mkdir "data\chroma_db"
if not exist "logs" mkdir "logs"

REM ---- 4. Ollama check -------------------------------------------------
set "CHAT_MODEL=llama3.2:3b"
set "EMBED_MODEL=nomic-embed-text"
for /f "delims=" %%M in ('.venv\Scripts\python.exe -c "from config.settings import load_settings; print(load_settings().chat_model)" 2^>nul') do set "CHAT_MODEL=%%M"
for /f "delims=" %%M in ('.venv\Scripts\python.exe -c "from config.settings import load_settings; print(load_settings().embedding_model)" 2^>nul') do set "EMBED_MODEL=%%M"

where ollama >nul 2>nul
if errorlevel 1 (
    echo.
    echo [WARNING] Ollama was not found.
    echo           Install it from https://ollama.com/download , start it, then run:
    echo               ollama pull %CHAT_MODEL%
    echo               ollama pull %EMBED_MODEL%
    goto :done
)
echo [OK] Ollama found.
ollama list >"%TEMP%\rag_models.txt" 2>nul
if errorlevel 1 (
    echo [WARNING] Could not query Ollama. Start the Ollama app, then run:
    echo               ollama pull %CHAT_MODEL%
    echo               ollama pull %EMBED_MODEL%
    goto :done
)
findstr /i /c:"%CHAT_MODEL%" "%TEMP%\rag_models.txt" >nul
if errorlevel 1 echo [MISSING MODEL] Run: ollama pull %CHAT_MODEL%
findstr /i /c:"%EMBED_MODEL%" "%TEMP%\rag_models.txt" >nul
if errorlevel 1 echo [MISSING MODEL] Run: ollama pull %EMBED_MODEL%

:done
echo.
echo Setup finished. Start the app with run.bat
pause
endlocal
