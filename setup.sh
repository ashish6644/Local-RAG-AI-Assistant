#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"

echo "=============================================="
echo " Local RAG AI Assistant - Setup"
echo "=============================================="

# ---- 1. Find Python 3.11 or 3.12 --------------------------------------
PYTHON=""
for candidate in python3.12 python3.11 python3 python; do
    if command -v "$candidate" >/dev/null 2>&1; then
        if "$candidate" -c 'import sys; sys.exit(0 if (3, 11) <= sys.version_info[:2] <= (3, 12) else 1)' 2>/dev/null; then
            PYTHON="$candidate"
            break
        fi
    fi
done

if [ -z "$PYTHON" ]; then
    echo "[ERROR] Python 3.11 or 3.12 was not found."
    echo "        Install it (for example: brew install python@3.12) and run ./setup.sh again."
    echo "        Python 3.13+ is not supported by the pinned dependencies."
    exit 1
fi
echo "[OK] Using: $("$PYTHON" --version)"

# ---- 2. Virtual environment -------------------------------------------
if [ -x .venv/bin/python ]; then
    # Reuse the environment only if it was created with a supported Python.
    if ! .venv/bin/python -c 'import sys; sys.exit(0 if (3, 11) <= sys.version_info[:2] <= (3, 12) else 1)' 2>/dev/null; then
        echo "[ERROR] The existing .venv uses an unsupported Python version."
        echo "        Delete the .venv folder and run ./setup.sh again with Python 3.11 or 3.12."
        exit 1
    fi
else
    echo "Creating virtual environment..."
    "$PYTHON" -m venv .venv
fi

VENV_PY=".venv/bin/python"
if [ ! -x "$VENV_PY" ]; then
    echo "[ERROR] The virtual environment is incomplete: $VENV_PY is missing."
    echo "        Delete the .venv folder and run ./setup.sh again."
    exit 1
fi

# ---- 3. Dependencies --------------------------------------------------
echo "Installing dependencies (this can take several minutes the first time)..."
"$VENV_PY" -m pip install --upgrade pip
"$VENV_PY" -m pip install -r requirements.txt

[ -f .env ] || cp .env.example .env
mkdir -p data/uploads data/chroma_db logs

# ---- 4. Ollama check --------------------------------------------------
CHAT_MODEL="$("$VENV_PY" -c 'from config.settings import load_settings; print(load_settings().chat_model)' 2>/dev/null || echo 'llama3.2:3b')"
EMBED_MODEL="$("$VENV_PY" -c 'from config.settings import load_settings; print(load_settings().embedding_model)' 2>/dev/null || echo 'nomic-embed-text')"

if ! command -v ollama >/dev/null 2>&1; then
    echo
    echo "[WARNING] Ollama was not found."
    echo "          Install it from https://ollama.com/download , start it, then run:"
    echo "              ollama pull $CHAT_MODEL"
    echo "              ollama pull $EMBED_MODEL"
    exit 0
fi
echo "[OK] Ollama found."

if ! ollama list >/dev/null 2>&1; then
    echo "[WARNING] Could not query Ollama. Start the Ollama app, then run:"
    echo "              ollama pull $CHAT_MODEL"
    echo "              ollama pull $EMBED_MODEL"
    exit 0
fi

ollama list | grep -qi -- "$CHAT_MODEL" || echo "[MISSING MODEL] Run: ollama pull $CHAT_MODEL"
ollama list | grep -qi -- "$EMBED_MODEL" || echo "[MISSING MODEL] Run: ollama pull $EMBED_MODEL"

echo
echo "Setup finished. Start the app with ./run.sh"
