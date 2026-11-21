"""Document ingestion: validate -> dedupe -> extract -> chunk -> embed -> store."""
from __future__ import annotations

import hashlib
import logging
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from config.settings import ALLOWED_EXTENSIONS, Settings
from core.document_processor import process_file
from core.embeddings import OllamaEmbedder
from core.errors import AppError, EmptyDocumentError
from core.text_splitter import split_segments
from core.vector_store import VectorStore
from database.models import Document
from database.repository import Repository

logger = logging.getLogger(__name__)


@dataclass
class IngestResult:
    status: str  # "ready" | "duplicate" | "failed" | "rejected"
    message: str
    filename: str
    document: Optional[Document] = None
    warnings: tuple[str, ...] = ()

    @property
    def ok(self) -> bool:
        return self.status == "ready"


def sanitize_filename(name: str) -> str:
    base = Path(name.replace("\\", "/")).name
    base = re.sub(r"[^A-Za-z0-9._() \-]", "_", base).strip(" .")
    return (base or "document")[:120]


class IngestionService:
    def __init__(self, settings: Settings, repo: Repository, embedder: OllamaEmbedder, store: VectorStore) -> None:
        self.settings = settings
        self.repo = repo
        self.embedder = embedder
        self.store = store

    def ingest_bytes(self, filename: str, data: bytes, cfg: Settings) -> IngestResult:
        safe_name = sanitize_filename(filename)
        ext = Path(safe_name).suffix.lower()
        if ext not in ALLOWED_EXTENSIONS:
            return IngestResult("rejected", f"Unsupported file type '{ext or 'none'}'. Allowed: {', '.join(ALLOWED_EXTENSIONS)}.", safe_name)
        if not data:
            return IngestResult("rejected", "The file is empty (0 bytes).", safe_name)
        if len(data) > cfg.max_upload_mb * 1024 * 1024:
            return IngestResult("rejected", f"The file is larger than the {cfg.max_upload_mb} MB limit.", safe_name)

        file_hash = hashlib.sha256(data).hexdigest()
        existing = self.repo.get_document_by_hash(file_hash)
        if existing is not None:
            if existing.processing_status == "ready":
                return IngestResult(
                    "duplicate",
                    f"'{safe_name}' was already uploaded (stored as '{existing.filename}'). It was not indexed again.",
                    safe_name, existing,
                )
            self._remove(existing)  # retry a failed / interrupted upload from scratch

        relative = f"uploads/{file_hash[:12]}_{safe_name}"
        path = self.settings.resolve_data_path(relative)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(data)
        doc = self.repo.create_document(
            filename=safe_name, file_hash=file_hash, file_path=relative,
            file_type=ext[1:], file_size=len(data),
        )
        return self.index_document(doc, cfg)

    def index_document(self, doc: Document, cfg: Settings) -> IngestResult:
        """(Re)build the vectors for an already-registered document. Never raises for document problems."""
        path = self.settings.resolve_data_path(doc.file_path)
        try:
            self.repo.update_document_status(doc.document_id, "processing")
            self.store.delete_document(doc.document_id)
            processed = process_file(path)
            chunks = split_segments(processed.segments, cfg.chunk_size, cfg.chunk_overlap, cfg.chunking_strategy)
            if not chunks:
                raise EmptyDocumentError("No text could be extracted from this document.")
            vectors = self.embedder.embed_documents([c.text for c in chunks], cfg.embedding_model)
            ids, metas = [], []
            for c in chunks:
                ids.append(f"{doc.document_id}:{c.chunk_index}")
                meta = {
                    "document_id": doc.document_id, "filename": doc.filename, "file_type": doc.file_type,
                    "chunk_index": c.chunk_index, "location": c.location,
                    "embedding_model": cfg.embedding_model,
                }
                if c.page is not None:
                    meta["page"] = c.page
                metas.append(meta)
            self.store.add_chunks(doc.document_id, ids, [c.text for c in chunks], vectors, metas)
            self.repo.update_document_status(doc.document_id, "ready", chunk_count=len(chunks))
            fresh = self.repo.get_document(doc.document_id)
            return IngestResult(
                "ready", f"Indexed '{doc.filename}' into {len(chunks)} chunks.", doc.filename,
                fresh, tuple(processed.warnings),
            )
        except AppError as exc:
            return self._fail(doc, str(exc))
        except Exception:
            logger.exception("Unexpected error while indexing %s", doc.filename)
            return self._fail(doc, "An unexpected error occurred while processing this document. See logs/app.log.")

    def _fail(self, doc: Document, message: str) -> IngestResult:
        try:
            self.store.delete_document(doc.document_id)
        except AppError:
            logger.warning("Could not clean partial vectors for document %s", doc.document_id)
        self.repo.update_document_status(doc.document_id, "failed", chunk_count=0, error=message)
        return IngestResult("failed", message, doc.filename, self.repo.get_document(doc.document_id))

    def _remove(self, doc: Document) -> None:
        try:
            self.store.delete_document(doc.document_id)
        except AppError:
            logger.warning("Could not remove vectors for %s", doc.document_id)
        self.repo.delete_document(doc.document_id)
        path = self.settings.resolve_data_path(doc.file_path)
        try:
            if path.resolve().is_relative_to(self.settings.upload_dir.resolve()):
                path.unlink(missing_ok=True)
        except OSError:
            logger.warning("Could not delete file %s", path)
