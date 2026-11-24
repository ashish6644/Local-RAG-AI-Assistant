"""Document listing, deletion (files + vectors + metadata) and re-indexing."""
from __future__ import annotations

import logging
from typing import Optional

from config.settings import Settings
from core.vector_store import VectorStore
from database.models import Document
from database.repository import Repository
from services.ingestion_service import IngestionService, IngestResult

logger = logging.getLogger(__name__)


class DocumentService:
    def __init__(self, settings: Settings, repo: Repository, store: VectorStore, ingestion: IngestionService) -> None:
        self.settings = settings
        self.repo = repo
        self.store = store
        self.ingestion = ingestion

    def list_documents(self, status: Optional[str] = None) -> list[Document]:
        return self.repo.list_documents(status)

    def delete_document(self, document_id: int) -> bool:
        """Remove vectors first (so a failure leaves everything consistent), then metadata and file."""
        doc = self.repo.get_document(document_id)
        if doc is None:
            return False
        self.store.delete_document(document_id)
        self.repo.delete_document(document_id)
        path = self.settings.resolve_data_path(doc.file_path)
        try:
            if path.resolve().is_relative_to(self.settings.upload_dir.resolve()):
                path.unlink(missing_ok=True)
        except OSError:
            logger.warning("Could not delete file %s", path)
        return True

    def reindex(self, document_id: int, cfg: Settings) -> Optional[IngestResult]:
        doc = self.repo.get_document(document_id)
        if doc is None:
            return None
        if not self.settings.resolve_data_path(doc.file_path).exists():
            self.repo.update_document_status(document_id, "failed", 0, "The stored file is missing; upload it again.")
            return IngestResult("failed", "The stored file is missing; upload it again.", doc.filename, doc)
        return self.ingestion.index_document(doc, cfg)

    def rebuild_index(self, cfg: Settings) -> tuple[int, int]:
        """Drop all vectors, then re-index every document. Returns (indexed, failed).

        ChromaDB fixes a collection's vector size at creation time, so switching to an embedding
        model with a different dimension needs a fresh collection rather than a per-document
        re-index. Documents whose stored file is missing are marked as failed and counted.
        """
        self.store.reset()
        indexed = failed = 0
        for doc in self.repo.list_documents():
            result = self.reindex(doc.document_id, cfg)
            if result is not None and result.status == "ready":
                indexed += 1
            else:
                failed += 1
        logger.info("Rebuilt vector index: %d indexed, %d failed", indexed, failed)
        return indexed, failed
