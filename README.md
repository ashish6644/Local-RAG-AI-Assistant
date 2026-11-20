# Local RAG AI Assistant

Chat with your own documents **entirely on your computer**. Retrieval-Augmented Generation with
Streamlit, Ollama (local LLM + embeddings), ChromaDB (vectors) and SQLite (chat history & metadata).
No API keys, no cloud services, no Docker.

## Features
- Upload PDF, DOCX, TXT, CSV, XLSX (multiple files, size/extension validation, SHA-256 duplicate detection)
- Per-document status, size, upload time, chunk count; delete (removes vectors + file) and re-index
- Real token-by-token streaming from Ollama, **Stop generating** button (partial answer is saved)
- Multiple persistent conversations: create, rename, delete, clear, resume after restart
- Source citations: filename, page (PDF) / section / sheet+rows, similarity score and chunk text
- History-aware question rewriting, optional multi-query retrieval, MMR or similarity search,
  Top-K, similarity threshold, per-document search scope
- Model settings: pick chat/embedding models, temperature, top-p, max tokens, connection test
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

## Using the app
1. **Documents** page -> choose files -> *Upload & index*.
2. **Chat** page -> ask questions. Expand *Sources & retrieval details* under any answer.
3. **Settings** -> change models, chunking, retrieval; *Test models* checks Ollama end to end.

Notes
- Chunk settings apply when a document is indexed. After changing chunking or the **embedding model**,
  press 🔄 on each document to re-index (vectors of different models are incompatible).
- If nothing relevant is found (similarity below threshold) the app says the documents lack
  sufficient information, without calling the LLM.

## Configuration
Copy `.env.example` to `.env` (setup does this). Settings changed in the UI are saved to
`data/user_settings.json` and override `.env` values for the editable options.

## Project layout
```
app.py            Streamlit entry point
config/           Pydantic settings, logging
core/             Ollama client, embeddings, ChromaDB store, document processing,
                  chunking, retriever, RAG chain, prompts, evaluation
database/         SQLAlchemy models, connection, repository
services/         chat, ingestion, documents, model checks, service container
ui/               sidebar, chat, documents, settings, shared components
tests/            pytest suite (Ollama and Chroma are faked where needed)
data/             created automatically: uploads/, chroma_db/, app.db, user_settings.json
logs/             created automatically: app.log
```

## Tests
```powershell
.\.venv\Scripts\Activate.ps1
python -m pytest
```
Most tests use fakes and need no Ollama. `test_real_chroma_persistence` uses the real ChromaDB.

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
| Weird vector-size error after changing embedding model | Re-index documents (🔄) |
| Details of any error | `logs/app.log` |
