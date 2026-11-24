"""In-memory stand-ins so tests run without Ollama or ChromaDB."""
from __future__ import annotations

import hashlib

import numpy as np

from core.vector_store import RetrievedChunk


def fake_vector(text: str, dim: int = 8) -> list[float]:
    digest = hashlib.sha256(text.lower().encode()).digest()
    return [b / 255.0 + 0.01 for b in digest[:dim]]


class FakeEmbedder:
    def __init__(self, fail: bool = False):
        self.fail = fail
        self.calls = 0

    def embed_documents(self, texts, model):
        self.calls += 1
        if self.fail:
            from core.errors import EmbeddingError
            raise EmbeddingError("embedding failed (fake)")
        return [fake_vector(t) for t in texts]

    def embed_queries(self, texts, model):
        return [fake_vector(t) for t in texts]


class FakeStore:
    """Mimics VectorStore using cosine similarity over numpy vectors."""

    def __init__(self):
        self.rows: dict[str, dict] = {}

    def add_chunks(self, document_id, chunk_ids, texts, embeddings, metadatas):
        for cid, text, emb, meta in zip(chunk_ids, texts, embeddings, metadatas):
            self.rows[cid] = {"text": text, "emb": np.array(emb, dtype=float), "meta": dict(meta)}

    def delete_document(self, document_id):
        self.rows = {k: v for k, v in self.rows.items() if v["meta"]["document_id"] != document_id}

    def reset(self):
        self.rows.clear()

    def count(self):
        return len(self.rows)

    def count_document(self, document_id):
        return sum(1 for v in self.rows.values() if v["meta"]["document_id"] == document_id)

    def query(self, embedding, n_results, document_ids=None, include_embeddings=False):
        q = np.array(embedding, dtype=float)
        out = []
        for cid, row in self.rows.items():
            if document_ids and row["meta"]["document_id"] not in document_ids:
                continue
            e = row["emb"]
            score = float(q @ e / (np.linalg.norm(q) * np.linalg.norm(e)))
            out.append(RetrievedChunk(cid, row["text"], row["meta"], score, e if include_embeddings else None))
        out.sort(key=lambda c: c.score, reverse=True)
        return out[:n_results]


def make_chunk(cid="1:0", text="Revenue grew 12% in 2023.", filename="report.pdf", page=3, score=0.8):
    meta = {"document_id": int(cid.split(":")[0]), "filename": filename, "location": f"page {page}", "page": page}
    return RetrievedChunk(cid, text, meta, score)


class FakeOllama:
    """Scripted chat client; records the messages it receives."""

    def __init__(self, tokens=("Hello", " world"), rewrite="standalone question?"):
        self.tokens = list(tokens)
        self.rewrite = rewrite
        self.stream_calls = []
        self.chat_calls = []

    def chat(self, model, messages, options=None):
        self.chat_calls.append(messages)
        return self.rewrite

    def chat_stream(self, model, messages, options=None):
        self.stream_calls.append(messages)
        yield from self.tokens
