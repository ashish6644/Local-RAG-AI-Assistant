"""Persistent ChromaDB wrapper (cosine similarity, metadata filtering)."""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional, Sequence

import numpy as np

from core.errors import VectorStoreError

logger = logging.getLogger(__name__)


@dataclass
class RetrievedChunk:
    id: str
    text: str
    metadata: dict[str, Any]
    score: float  # cosine similarity, 1.0 = identical
    embedding: Optional[np.ndarray] = field(default=None, repr=False)

    @property
    def filename(self) -> str:
        return str(self.metadata.get("filename", "unknown"))

    @property
    def document_id(self) -> Optional[int]:
        value = self.metadata.get("document_id")
        return int(value) if value is not None else None

    @property
    def page(self) -> Optional[int]:
        value = self.metadata.get("page")
        return int(value) if value is not None else None

    @property
    def location(self) -> str:
        return str(self.metadata.get("location", ""))


class VectorStore:
    def __init__(self, persist_dir: Path, collection_name: str = "documents") -> None:
        self.collection_name = collection_name
        try:
            import chromadb
            from chromadb.config import Settings as ChromaSettings

            persist_dir.mkdir(parents=True, exist_ok=True)
            self._client = chromadb.PersistentClient(
                path=str(persist_dir), settings=ChromaSettings(anonymized_telemetry=False)
            )
            self._collection = self._client.get_or_create_collection(
                name=collection_name, metadata={"hnsw:space": "cosine"}
            )
        except Exception as exc:
            logger.exception("Could not open ChromaDB at %s", persist_dir)
            raise VectorStoreError(
                f"Could not open the vector database at {persist_dir}. "
                "If the problem persists, close other copies of the app or delete the chroma_db folder "
                "(documents can then be re-indexed)."
            ) from exc

    def reset(self) -> None:
        """Delete every vector and recreate the collection.

        ChromaDB fixes a collection's vector dimension when it is created, so switching to an
        embedding model with a different size requires a fresh collection. Callers are expected
        to re-index the documents afterwards (DocumentService.rebuild_index does both).
        """
        try:
            try:
                self._client.delete_collection(self.collection_name)
            except Exception as exc:  # already missing is fine
                if "does not exist" not in str(exc).lower() and "not found" not in str(exc).lower():
                    raise
            self._collection = self._client.get_or_create_collection(
                name=self.collection_name, metadata={"hnsw:space": "cosine"}
            )
        except Exception as exc:
            logger.exception("Could not reset the vector collection")
            raise VectorStoreError(
                "Could not rebuild the vector database. Close other copies of the app, "
                "delete the chroma_db folder, and restart."
            ) from exc

    def add_chunks(
        self, document_id: int, chunk_ids: Sequence[str], texts: Sequence[str],
        embeddings: Sequence[Sequence[float]], metadatas: Sequence[dict[str, Any]],
    ) -> None:
        if not (len(chunk_ids) == len(texts) == len(embeddings) == len(metadatas)):
            raise VectorStoreError("Internal error: chunk/embedding counts do not match.")
        try:
            step = 256
            for i in range(0, len(chunk_ids), step):
                self._collection.add(
                    ids=list(chunk_ids[i:i + step]),
                    documents=list(texts[i:i + step]),
                    embeddings=[list(e) for e in embeddings[i:i + step]],
                    metadatas=list(metadatas[i:i + step]),
                )
        except Exception as exc:
            logger.exception("Failed to add chunks for document %s", document_id)
            msg = "Failed to store document vectors."
            if "dimension" in str(exc).lower():
                msg += (
                    " The embedding model changed since earlier documents were indexed "
                    "(different vector size). Rebuild the vector index from "
                    "Documents → 🛠 Maintenance."
                )
            raise VectorStoreError(msg) from exc

    def delete_document(self, document_id: int) -> None:
        try:
            self._collection.delete(where={"document_id": document_id})
        except Exception as exc:
            logger.exception("Failed to delete vectors for document %s", document_id)
            raise VectorStoreError("Could not remove the document's vectors from the database.") from exc

    def count(self) -> int:
        try:
            return int(self._collection.count())
        except Exception as exc:
            raise VectorStoreError("Could not read the vector database.") from exc

    def count_document(self, document_id: int) -> int:
        try:
            res = self._collection.get(where={"document_id": document_id}, include=[])
            return len(res["ids"])
        except Exception as exc:
            raise VectorStoreError("Could not read the vector database.") from exc

    def query(
        self, embedding: Sequence[float], n_results: int,
        document_ids: Optional[Sequence[int]] = None, include_embeddings: bool = False,
    ) -> list[RetrievedChunk]:
        total = self.count()
        if total == 0:
            return []
        include = ["documents", "metadatas", "distances"]
        if include_embeddings:
            include.append("embeddings")
        kwargs: dict[str, Any] = {
            "query_embeddings": [list(embedding)],
            "n_results": max(1, min(n_results, total)),
            "include": include,
        }
        if document_ids:
            ids = [int(i) for i in document_ids]
            kwargs["where"] = {"document_id": ids[0]} if len(ids) == 1 else {"document_id": {"$in": ids}}
        try:
            res = self._collection.query(**kwargs)
        except Exception as exc:
            logger.exception("Vector query failed")
            msg = "Searching the vector database failed."
            if "dimension" in str(exc).lower():
                msg += (
                    " The selected embedding model produces vectors of a different size than the "
                    "indexed documents. Rebuild the vector index from Documents → 🛠 Maintenance."
                )
            raise VectorStoreError(msg) from exc

        out: list[RetrievedChunk] = []
        ids_ = res["ids"][0]
        embs = res.get("embeddings")
        for i, chunk_id in enumerate(ids_):
            distance = float(res["distances"][0][i])
            emb = None
            if include_embeddings and embs is not None:
                emb = np.asarray(embs[0][i], dtype=float)
            out.append(
                RetrievedChunk(
                    id=chunk_id,
                    text=res["documents"][0][i] or "",
                    metadata=dict(res["metadatas"][0][i] or {}),
                    score=max(-1.0, min(1.0, 1.0 - distance)),
                    embedding=emb,
                )
            )
        return out
