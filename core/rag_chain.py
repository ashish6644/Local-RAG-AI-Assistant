"""RAG orchestration: history-aware rewriting, retrieval, grounded streaming generation."""
from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Iterator, Optional, Sequence

from core import prompt_templates as pt
from core.errors import OllamaError
from core.ollama_client import OllamaClient
from core.retriever import Retriever
from core.vector_store import RetrievedChunk

if TYPE_CHECKING:
    from config.settings import Settings

logger = logging.getLogger(__name__)
MAX_SNIPPET_CHARS = 1500


@dataclass(frozen=True)
class RAGParams:
    chat_model: str
    embedding_model: str
    temperature: float = 0.2
    top_p: float = 0.9
    max_tokens: int = 1024
    num_ctx: int = 4096
    top_k: int = 4
    threshold: float = 0.3
    strategy: str = "similarity"
    history_window: int = 6
    rewrite: bool = True
    multi_query: bool = False
    document_ids: Optional[tuple[int, ...]] = None

    @classmethod
    def from_settings(cls, s: Settings, document_ids: Optional[Sequence[int]] = None) -> "RAGParams":
        return cls(
            chat_model=s.chat_model, embedding_model=s.embedding_model,
            temperature=s.temperature, top_p=s.top_p, max_tokens=s.max_output_tokens,
            num_ctx=s.context_window, top_k=s.top_k, threshold=s.similarity_threshold,
            strategy=s.search_strategy, history_window=s.history_window,
            rewrite=s.use_query_rewriting, multi_query=s.use_multi_query,
            document_ids=tuple(document_ids) if document_ids else None,
        )

    def ollama_options(self, **overrides: Any) -> dict:
        options = {
            "temperature": self.temperature, "top_p": self.top_p,
            "num_predict": self.max_tokens, "num_ctx": self.num_ctx,
        }
        options.update(overrides)
        return options


@dataclass
class RetrievalResult:
    query: str
    variants: list[str]
    chunks: list[RetrievedChunk] = field(default_factory=list)
    reason: str = "ok"  # "ok" | "no_documents" | "no_match"

    def to_payload(self) -> dict:
        """JSON-serialisable record stored with the assistant message."""
        return {
            "rewritten_query": self.query,
            "queries": self.variants,
            "chunks": [
                {
                    "chunk_id": c.id, "document_id": c.document_id, "filename": c.filename,
                    "page": c.page, "location": c.location, "score": round(c.score, 4),
                    "text": c.text[:MAX_SNIPPET_CHARS],
                }
                for c in self.chunks
            ],
        }


class RAGPipeline:
    def __init__(self, client: OllamaClient, retriever: Retriever) -> None:
        self.client = client
        self.retriever = retriever

    # ---- query preparation --------------------------------------------
    def rewrite_question(self, question: str, history: Sequence[dict], params: RAGParams) -> str:
        recent = list(history)[-params.history_window:] if params.history_window > 0 else []
        prompt = pt.REWRITE_PROMPT.format(history=pt.format_history(recent), question=question)
        try:
            raw = self.client.chat(
                params.chat_model, [{"role": "user", "content": prompt}],
                params.ollama_options(temperature=0.0, num_predict=128),
            )
        except OllamaError as exc:
            logger.warning("Question rewriting failed, using the original question: %s", exc)
            return question
        lines = [ln.strip().strip('"').strip() for ln in raw.strip().splitlines() if ln.strip()]
        rewritten = lines[0] if lines else ""
        rewritten = re.sub(r"^(standalone question|question)\s*:\s*", "", rewritten, flags=re.I)
        if not rewritten or len(rewritten) > 400:
            return question
        return rewritten

    def expand_queries(self, question: str, params: RAGParams, n: int = 3) -> list[str]:
        prompt = pt.MULTI_QUERY_PROMPT.format(n=n, question=question)
        try:
            raw = self.client.chat(
                params.chat_model, [{"role": "user", "content": prompt}],
                params.ollama_options(temperature=0.4, num_predict=160),
            )
        except OllamaError as exc:
            logger.warning("Multi-query expansion failed: %s", exc)
            return []
        out = []
        for line in raw.splitlines():
            line = re.sub(r"^\s*(?:[-*\u2022]|\d+[.)])\s*", "", line).strip().strip('"')
            if line and line.lower() != question.lower() and len(line) < 300:
                out.append(line)
        return out[:n]

    # ---- retrieval -----------------------------------------------------
    def retrieve(self, question: str, history: Sequence[dict], params: RAGParams) -> RetrievalResult:
        if self.retriever.store.count() == 0:
            return RetrievalResult(question, [question], [], "no_documents")
        standalone = question
        if params.rewrite and history and params.history_window > 0:
            standalone = self.rewrite_question(question, history, params)
        variants = [standalone]
        if params.multi_query:
            variants += self.expand_queries(standalone, params)
        chunks = self.retriever.retrieve(
            variants, params.embedding_model, params.top_k, params.threshold,
            params.strategy, params.document_ids,
        )
        return RetrievalResult(standalone, variants, chunks, "ok" if chunks else "no_match")

    # ---- generation ----------------------------------------------------
    def build_messages(
        self, question: str, history: Sequence[dict], chunks: Sequence[RetrievedChunk],
        params: RAGParams,
    ) -> list[dict]:
        window = list(history)[-params.history_window:] if params.history_window > 0 else []
        messages = [{"role": "system", "content": pt.SYSTEM_PROMPT}]
        messages += [{"role": m["role"], "content": m["content"]} for m in window]
        messages.append({
            "role": "user",
            "content": pt.USER_TEMPLATE.format(context=pt.build_context(chunks), question=question),
        })
        return messages

    def stream_answer(
        self, question: str, history: Sequence[dict], retrieval: RetrievalResult, params: RAGParams
    ) -> Iterator[str]:
        if retrieval.reason == "no_documents":
            yield pt.NO_DOCUMENTS_MESSAGE
            return
        if not retrieval.chunks:
            yield pt.NO_RELEVANT_MESSAGE
            return
        messages = self.build_messages(question, history, retrieval.chunks, params)
        yield from self.client.chat_stream(params.chat_model, messages, params.ollama_options())
