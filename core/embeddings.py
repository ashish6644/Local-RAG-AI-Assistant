"""Embedding generation through Ollama."""
from __future__ import annotations

import logging

from core.errors import EmbeddingError, OllamaError
from core.ollama_client import OllamaClient

logger = logging.getLogger(__name__)


def _prefix(model: str, kind: str) -> str:
    """nomic-embed-text was trained with task prefixes; other models use none."""
    if model.lower().startswith("nomic-embed-text"):
        return "search_document: " if kind == "document" else "search_query: "
    return ""


class OllamaEmbedder:
    def __init__(self, client: OllamaClient, batch_size: int = 16) -> None:
        self.client = client
        self.batch_size = batch_size

    def _embed(self, texts: list[str], model: str, kind: str) -> list[list[float]]:
        prefix = _prefix(model, kind)
        vectors: list[list[float]] = []
        try:
            for start in range(0, len(texts), self.batch_size):
                batch = [prefix + t for t in texts[start:start + self.batch_size]]
                vectors.extend(self.client.embed(model, batch))
        except OllamaError as exc:
            logger.error("Embedding failed: %s", exc)
            raise EmbeddingError(f"Embedding generation failed: {exc}") from exc
        return vectors

    def embed_documents(self, texts: list[str], model: str) -> list[list[float]]:
        return self._embed(texts, model, "document") if texts else []

    def embed_queries(self, texts: list[str], model: str) -> list[list[float]]:
        return self._embed(texts, model, "query") if texts else []

    def embed_query(self, text: str, model: str) -> list[float]:
        return self.embed_queries([text], model)[0]
