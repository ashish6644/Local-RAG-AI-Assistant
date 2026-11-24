# Changelog

All notable changes to **Local RAG AI Assistant**, grouped by the four-day build plan.
The project is feature-complete: the full pytest suite passes locally (80 passed, 1 skipped —
the skip is the real-ChromaDB persistence test and requires `chromadb`).

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
