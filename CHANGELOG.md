# Changelog

All notable changes to **Local RAG AI Assistant**, grouped by the four-day build plan.
The project is feature-complete: the full pytest suite passes locally (80 passed, 1 skipped —
the skip is the real-ChromaDB persistence test and requires `chromadb`).

## Final hardening pass (2026-11-24)

### Fixed
- Long filenames keep their extension when truncated, so a valid file is no longer rejected as
  an unsupported type.
- A `user_settings.json` containing a non-object (e.g. `[]` or `null`) is ignored instead of
  crashing startup, and settings saves are atomic.
- The embedding-model selector only offers detected embedding models (with a fallback when none
  is recognised), preventing a chat model from being selected as the embedder.
- Failed uploads no longer abort the rest of a multi-file batch, and sidebar document listing is
  guarded.
- Deleting a document whose metadata deletion fails marks the row failed instead of leaving it
  "ready"; re-indexing a missing file removes its stale vectors.
- Whole-index rebuild validates the embedding model before resetting and marks documents
  *processing* first, so an interrupted rebuild is recoverable and visible.
- `history_window = 0` disables query rewriting, malformed Ollama JSON becomes a friendly
  `OllamaError`, and the Streamlit upload cap matches the app's 1024 MB limit.
- The Stop button sets an explicit cancellation flag before closing the generator.

### Added
- `setup.sh` / `run.sh` for macOS and Linux, plus a README section for them.
- `setup.bat` / `run.bat` validate the existing `.venv` Python version and always use the
  virtual environment interpreter (no silent fallback to the system Python).
- Regression tests for every fix above (94 passed, 1 skipped).

## Day 4 — Tests and final hardening (2026-11-24)

### Added
- Whole-index rebuild: **Documents → 🛠 Maintenance → Rebuild vector index**. ChromaDB fixes a
  collection's vector size at creation, so switching to an embedding model with a different
  dimension now resets the collection and re-indexes every stored document in one action.
- `VectorStore.reset()` to delete and recreate the collection safely.
- `DocumentService.rebuild_index()` returning indexed/failed counts, with missing files marked failed.
- Tests for settings, model service, document re-index/delete/rebuild, and vector-store reset.

### Changed
- README: document the rebuild action and clarify the per-document re-index workflow.

## Day 3 — Evaluation and core coverage (2026-11-23)

### Added
- Real `recall@k` in `core/evaluation.py`, multi-source cases (`expected_sources`), keyword-only
  cases and optional top-k truncation. Hit rate and MRR are preserved.
- Tests for embeddings (task prefixes, batching, error wrapping), text splitting (metadata,
  strategies, validation) and evaluation metrics.

## Day 2 — RAG pipeline, services and UI (2026-11-21)

### Added
- Document processing for PDF, DOCX, TXT, CSV and XLSX with page/section/sheet locations.
- Configurable chunking, Ollama embeddings with task prefixes, and a persistent ChromaDB store.
- Retriever with similarity/MMR, multi-query merging, thresholds and per-document filtering.
- Ollama client with real NDJSON streaming and clear connection/model errors.
- SQLite persistence (conversations, messages, documents) and repository layer.
- Services for ingestion, documents, chat and model checks, plus the dependency container.
- Streamlit UI: chat with citations, document management and settings.
- Pytest suite using fakes for Ollama and ChromaDB.

## Day 1 — Foundation (2026-11-20)

### Added
- Repository scaffolding, `.env.example`, `.gitignore` and Streamlit configuration.
- Pinned dependencies and pytest configuration for Python 3.11/3.12.
- Windows `setup.bat` and `run.bat` scripts.
- README with features, installation and troubleshooting.
- Application error hierarchy and Pydantic settings with environment/UI overrides and logging.
