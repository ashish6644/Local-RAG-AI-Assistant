"""Semantic retrieval: similarity or MMR, multi-query merging, threshold and document filters."""
from __future__ import annotations

import logging
from typing import Optional, Sequence

import numpy as np

from core.embeddings import OllamaEmbedder
from core.vector_store import RetrievedChunk, VectorStore

logger = logging.getLogger(__name__)


def _mmr(
    candidates: list[RetrievedChunk], k: int, lambda_mult: float = 0.6
) -> list[RetrievedChunk]:
    """Maximal Marginal Relevance: balance relevance (score) against redundancy."""
    usable = [c for c in candidates if c.embedding is not None]
    if not usable:
        return sorted(candidates, key=lambda c: c.score, reverse=True)[:k]
    matrix = np.array([c.embedding for c in usable], dtype=float)
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    matrix = matrix / np.where(norms == 0, 1, norms)
    selected = [int(np.argmax([c.score for c in usable]))]
    while len(selected) < min(k, len(usable)):
        best_idx, best_val = None, -1e9
        for i, cand in enumerate(usable):
            if i in selected:
                continue
            redundancy = max(float(matrix[i] @ matrix[j]) for j in selected)
            value = lambda_mult * cand.score - (1 - lambda_mult) * redundancy
            if value > best_val:
                best_idx, best_val = i, value
        selected.append(best_idx)
    return [usable[i] for i in selected]


class Retriever:
    def __init__(self, store: VectorStore, embedder: OllamaEmbedder) -> None:
        self.store = store
        self.embedder = embedder

    def retrieve(
        self,
        queries: Sequence[str],
        embedding_model: str,
        top_k: int = 4,
        threshold: float = 0.3,
        strategy: str = "similarity",
        document_ids: Optional[Sequence[int]] = None,
    ) -> list[RetrievedChunk]:
        queries = [q.strip() for q in queries if q and q.strip()]
        if not queries:
            return []
        vectors = self.embedder.embed_queries(queries, embedding_model)
        use_mmr = strategy == "mmr"
        fetch_k = top_k * 4 if use_mmr else top_k * 2
        pool: dict[str, RetrievedChunk] = {}
        for vector in vectors:
            for hit in self.store.query(vector, fetch_k, document_ids, include_embeddings=use_mmr):
                current = pool.get(hit.id)
                if current is None or hit.score > current.score:
                    pool[hit.id] = hit
        candidates = [c for c in pool.values() if c.score >= threshold]
        logger.info("Retrieval: %d candidates, %d above threshold %.2f", len(pool), len(candidates), threshold)
        if not candidates:
            return []
        if use_mmr:
            return _mmr(candidates, top_k)
        candidates.sort(key=lambda c: c.score, reverse=True)
        return candidates[:top_k]
