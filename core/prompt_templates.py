"""Prompt templates and context formatting."""
from __future__ import annotations

from typing import Sequence

from core.vector_store import RetrievedChunk

SYSTEM_PROMPT = """You are a careful document assistant. Answer using ONLY the numbered context excerpts that accompany the user's question.

Rules:
- If the excerpts do not contain enough information to answer, say clearly that the uploaded documents do not contain sufficient information. Do not guess and do not use outside knowledge.
- Cite the excerpts you rely on with their numbers in square brackets, for example [1] or [2]. Never invent sources, page numbers or quotes.
- Use the conversation history only to understand follow-up questions.
- Be concise and well structured. Use Markdown lists or tables when they help."""

USER_TEMPLATE = """Context excerpts:
{context}

Question: {question}

Answer using only the context excerpts above."""

REWRITE_PROMPT = """Given the chat history and the latest user question, rewrite the question as a single standalone question that can be understood without the history. Do NOT answer it. If it is already standalone, return it unchanged. Output only the question.

Chat history:
{history}

Latest question: {question}

Standalone question:"""

MULTI_QUERY_PROMPT = """Generate {n} different search queries that would help find information to answer the question below. Use different wording for each. Output one query per line with no numbering or extra text.

Question: {question}"""

NO_DOCUMENTS_MESSAGE = (
    "There are no indexed documents yet. Upload a document on the **Documents** page "
    "and I can answer questions about it."
)
NO_RELEVANT_MESSAGE = (
    "The uploaded documents do not contain sufficient information to answer that. "
    "Try rephrasing the question, widening the search scope, or lowering the similarity "
    "threshold in Settings."
)


def build_context(chunks: Sequence[RetrievedChunk]) -> str:
    blocks = []
    for number, chunk in enumerate(chunks, start=1):
        where = f" ({chunk.location})" if chunk.location else ""
        blocks.append(f"[{number}] Source: {chunk.filename}{where}\n{chunk.text}")
    return "\n\n---\n\n".join(blocks)


def format_history(history: Sequence[dict], max_chars: int = 500) -> str:
    lines = []
    for item in history:
        role = "User" if item["role"] == "user" else "Assistant"
        content = item["content"].strip().replace("\n", " ")
        lines.append(f"{role}: {content[:max_chars]}")
    return "\n".join(lines)
