#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"

VENV_PY=".venv/bin/python"
if [ ! -x "$VENV_PY" ]; then
    echo "[ERROR] The virtual environment is missing. Run ./setup.sh first."
    exit 1
fi

if ! "$VENV_PY" -c 'import sys; sys.exit(0 if (3, 11) <= sys.version_info[:2] <= (3, 12) else 1)'; then
    echo "[ERROR] The virtual environment uses an unsupported Python version."
    echo "        Delete the .venv folder and run ./setup.sh again with Python 3.11 or 3.12."
    exit 1
fi

[ -f .env ] || cp .env.example .env

# ---- Check configuration ----------------------------------------------
if ! "$VENV_PY" -c 'from config.settings import load_settings; load_settings()'; then
    echo "[ERROR] Your .env file contains invalid values. See the message above."
    exit 1
fi

# ---- Check Ollama connectivity (the app also shows this in Settings) --
if "$VENV_PY" -c 'import sys; from config.settings import load_settings; from core.ollama_client import OllamaClient; sys.exit(0 if OllamaClient(load_settings().ollama_base_url).is_running() else 1)'; then
    echo "[OK] Ollama is running."
else
    echo "[WARNING] Ollama is not reachable. Start the Ollama app (or run \"ollama serve\")."
    echo "          The app will still open and show instructions."
fi

echo
echo "Starting Local RAG AI Assistant at http://localhost:8501"
echo "Press Ctrl+C to stop."
exec "$VENV_PY" -m streamlit run app.py --server.port 8501
