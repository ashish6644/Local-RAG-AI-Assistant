# Local RAG AI Assistant

[![CI](https://github.com/ashish6644/Local-RAG-AI-Assistant/actions/workflows/ci.yml/badge.svg)](https://github.com/ashish6644/Local-RAG-AI-Assistant/actions/workflows/ci.yml)
[![Python](https://img.shields.io/badge/python-3.11%20%7C%203.12-blue)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/license-MIT-green)](LICENSE)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.40-FF4B4B)](https://streamlit.io/)

Chat with your own documents **entirely on your computer**. Retrieval-Augmented Generation with
Streamlit, Ollama (local LLM + embeddings), ChromaDB (vectors) and SQLite (chat history & metadata).
No API keys, no cloud services, no Docker.

## Highlights
- **Fully local**: Ollama + ChromaDB + SQLite; no API keys, no telemetry, no cloud calls.
- **Grounded answers with citations**: filename, PDF page / DOCX section / sheet+rows, similarity
  score and the exact chunk text.
- **Production-minded ingestion**: extension and size validation, SHA-256 duplicate detection,
  per-file status, atomic settings saves, stale-vector cleanup and a whole-index rebuild.
- **Configurable retrieval**: similarity or MMR, Top-K, similarity threshold, optional multi-query
  expansion and per-document search scope.
- **Streaming UX**: token-by-token answers, stop button that keeps the partial answer, persistent
  conversations that survive restarts.
- **Quality gates**: 94 pytest tests (Ollama and Chroma faked for speed), GitHub Actions CI on
  Python 3.11 and 3.12, and Ruff linting.
- **Cross-platform**: one-command setup on Windows, macOS and Linux.

## Features
- Upload PDF, DOCX, TXT, CSV, XLSX (multiple files, size/extension validation, SHA-256 duplicate detection)
- Per-document status, size, upload time, chunk count; delete (removes vectors + file) and re-index
- Real token-by-token streaming from Ollama, **Stop generating** button (partial answer is saved)
- Multiple persistent conversations: create, rename, delete, clear, resume after restart
- Source citations: filename, page (PDF) / section / sheet+rows, similarity score and chunk text
- History-aware question rewriting, optional multi-query retrieval, MMR or similarity search,
  Top-K, similarity threshold, per-document search scope
- Model settings: pick chat/embedding models, temperature, top-p, max tokens, connection test
- One-click **Rebuild vector index** after switching embedding models
- Retrieval evaluation utility (`python -m core.evaluation cases.json`)

## Requirements
- Windows 10/11 (also works on macOS/Linux), **Python 3.11 or 3.12**
- [Ollama](https://ollama.com/download)
- ~8 GB RAM recommended for `llama3.2:3b` (CPU only is fine, just slower)

## Install & run (Windows)
1. Extract the ZIP and open the folder.
2. Double-click **`setup.bat`** (creates `.venv`, installs dependencies, checks Ollama).
3. Install Ollama, then download the models:
   ```powershell
   ollama pull llama3.2:3b
   ollama pull nomic-embed-text
   ```
4. Double-click **`run.bat`** and open <http://localhost:8501>.

### Same steps in PowerShell
```powershell
cd path\to\local-rag-ai-assistant
py -3.12 -m venv .venv            # or: py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1      # if blocked: Set-ExecutionPolicy -Scope Process Bypass
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
Copy-Item .env.example .env
ollama pull llama3.2:3b
ollama pull nomic-embed-text
python -m streamlit run app.py --server.port 8501
```

## Install & run (macOS / Linux)
1. Open a terminal in the project folder.
2. Make the scripts executable once: `chmod +x setup.sh run.sh`.
3. Run **`./setup.sh`** (creates `.venv`, installs dependencies, checks Ollama).
4. Install Ollama, then download the models:
   ```bash
   ollama pull llama3.2:3b
   ollama pull nomic-embed-text
   ```
5. Run **`./run.sh`** and open <http://localhost:8501>.

Both scripts require Python 3.11 or 3.12 (`python3.12`, `python3.11` or `python3` on PATH).
If an existing `.venv` was created with another version, delete it and run the setup script again.

## Using the app
1. **Documents** page -> choose files -> *Upload & index*.
2. **Chat** page -> ask questions. Expand *Sources & retrieval details* under any answer.
3. **Settings** -> change models, chunking, retrieval; *Test models* checks Ollama end to end.

Notes
- Chunk settings apply when a document is indexed; after changing chunking, press 🔄 on that document to re-index it.
- After changing the **embedding model**, open **Documents → 🛠 Maintenance → Rebuild vector index**.
  ChromaDB fixes the vector size when the index is created, so a per-document re-index alone can fail.
- A rebuild is destructive-first: the existing index is deleted, then every stored document is
  re-indexed. If it is interrupted, run it again; documents stay marked *processing* until indexed.
- If nothing relevant is found (similarity below threshold) the app says the documents lack
  sufficient information, without calling the LLM.

## Configuration
Copy `.env.example` to `.env` (setup does this). Settings changed in the UI are saved to
`data/user_settings.json` and override `.env` values for the editable options.

## Architecture
See [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) for the component diagram, upload/question flows,
data model, design decisions and testing strategy.

## Project layout
```
app.py                Streamlit entry point
config/               Pydantic settings, logging
core/                 Ollama client, embeddings, ChromaDB store, document processing,
                      chunking, retriever, RAG chain, prompts, evaluation
database/             SQLAlchemy models, connection, repository
services/             chat, ingestion, documents, model checks, service container
ui/                   sidebar, chat, documents, settings, shared components
tests/                pytest suite (Ollama and Chroma are faked where needed)
docs/                 architecture and design notes
examples/             sample retrieval-evaluation cases
.github/workflows/    CI: Ruff + pytest on Python 3.11 and 3.12
data/                 created automatically: uploads/, chroma_db/, app.db
                      (user_settings.json appears after you save Settings)
logs/                 created automatically: app.log
```

## Tests
```powershell
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements-dev.txt
python -m pytest
```
On macOS/Linux use `.venv/bin/python -m pip install -r requirements-dev.txt` and
`.venv/bin/python -m pytest`.

Most tests use fakes and need no Ollama. `test_real_chroma_persistence` uses the real ChromaDB
and is skipped when it is not installed. Lint with `ruff check .`.

## Retrieval evaluation
`core/evaluation.py` scores a labelled question set with **hit rate**, **recall@k** and **MRR**:
```powershell
python -m core.evaluation examples/evaluation_cases.json
```
Edit `examples/evaluation_cases.json` so the questions, filenames and keywords match your indexed
documents. The command needs Ollama running and at least one indexed document.

## Limitations
- **Scanned (image-only) PDFs are not OCR'd.** They are rejected with a clear message; run them
  through an OCR tool (e.g. OCRmyPDF, Adobe Acrobat) first.
- Legacy `.doc` / `.xls` files and password-protected PDFs are not supported.
- DOCX has no page numbers; citations use the nearest heading ("section: ...") instead.
- CSV/XLSX are indexed as row groups, so they suit lookups better than numeric aggregation.
- Streaming stop is implemented via Streamlit reruns: clicking *Stop* ends generation and keeps what was produced.
- Dependencies are pinned for Python 3.11/3.12; Python 3.13+ is not supported.

## Troubleshooting
| Problem | Fix |
|---|---|
| "Cannot connect to Ollama" | Start the Ollama app or run `ollama serve` |
| "Model ... is not installed" | `ollama pull <model>` (shown in the message) |
| Slow answers | Use a smaller model (`llama3.2:1b`), lower Top K or context window |
| Weird vector-size error after changing embedding model | **Documents → 🛠 Maintenance → Rebuild vector index** |
| Rebuild stopped part-way | Run **Rebuild vector index** again; unfinished documents are marked *processing* |
| Details of any error | `logs/app.log` |

## License
MIT — see [LICENSE](LICENSE).
