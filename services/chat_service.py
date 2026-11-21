"""Conversation management and the streaming question-answering flow."""
from __future__ import annotations

import logging
from typing import Iterator, Optional

from core.rag_chain import RAGParams, RAGPipeline
from database.models import DEFAULT_TITLE, Conversation, Message
from database.repository import Repository

logger = logging.getLogger(__name__)
STOPPED_NOTE = "\n\n*(Generation stopped before completion.)*"


class ChatService:
    def __init__(self, repo: Repository, pipeline: RAGPipeline) -> None:
        self.repo = repo
        self.pipeline = pipeline

    # ---- conversations -------------------------------------------------
    def create_conversation(self, title: str = DEFAULT_TITLE) -> Conversation:
        return self.repo.create_conversation(title)

    def list_conversations(self) -> list[Conversation]:
        return self.repo.list_conversations()

    def get_conversation(self, conversation_id: int) -> Optional[Conversation]:
        return self.repo.get_conversation(conversation_id)

    def rename_conversation(self, conversation_id: int, title: str) -> None:
        self.repo.rename_conversation(conversation_id, title)

    def delete_conversation(self, conversation_id: int) -> bool:
        return self.repo.delete_conversation(conversation_id)

    def clear_conversation(self, conversation_id: int) -> int:
        return self.repo.clear_messages(conversation_id)

    def get_messages(self, conversation_id: int) -> list[Message]:
        return self.repo.get_messages(conversation_id)

    # ---- question answering ---------------------------------------------
    def ask(self, conversation_id: int, question: str, params: RAGParams) -> Iterator[dict]:
        """Generator of events: status / retrieval / token / done.

        The user message is saved immediately; the assistant message is saved when the generator
        finishes, fails or is closed early (stop button), so partial answers are never lost.
        """
        history = [{"role": m.role, "content": m.content} for m in self.repo.get_messages(conversation_id)]
        self.repo.add_message(conversation_id, "user", question)
        conv = self.repo.get_conversation(conversation_id)
        if conv is not None and conv.title == DEFAULT_TITLE and not history:
            title = " ".join(question.split())[:50]
            self.repo.rename_conversation(conversation_id, title + ("…" if len(question) > 50 else ""))

        parts: list[str] = []
        payload: Optional[dict] = None
        completed = False
        try:
            yield {"type": "status", "text": "Searching your documents…"}
            retrieval = self.pipeline.retrieve(question, history, params)
            payload = retrieval.to_payload()
            yield {"type": "retrieval", "payload": payload}
            yield {"type": "status", "text": "Generating answer…"}
            for token in self.pipeline.stream_answer(question, history, retrieval, params):
                parts.append(token)
                yield {"type": "token", "text": token}
            completed = True
            yield {"type": "done"}
        finally:
            answer = "".join(parts)
            if answer.strip():
                if not completed:
                    answer += STOPPED_NOTE
                try:
                    self.repo.add_message(
                        conversation_id, "assistant", answer,
                        sources=payload if payload and payload.get("chunks") else None,
                    )
                except Exception:  # never mask the original error / stop signal
                    logger.exception("Could not save the assistant message")
