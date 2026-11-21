"""Data-access layer for conversations, messages and documents."""
from __future__ import annotations

from typing import Any, Optional, Sequence

from sqlalchemy import select

from database.connection import Database
from database.models import DEFAULT_TITLE, Conversation, Document, Message, utcnow


class Repository:
    def __init__(self, db: Database) -> None:
        self.db = db

    # ---- conversations -------------------------------------------------
    def create_conversation(self, title: str = DEFAULT_TITLE) -> Conversation:
        with self.db.session() as s:
            conv = Conversation(title=(title.strip() or DEFAULT_TITLE)[:200])
            s.add(conv)
            s.flush()
            return conv

    def list_conversations(self) -> list[Conversation]:
        with self.db.session() as s:
            stmt = select(Conversation).order_by(
                Conversation.updated_at.desc(), Conversation.conversation_id.desc()
            )
            return list(s.scalars(stmt))

    def get_conversation(self, conversation_id: int) -> Optional[Conversation]:
        with self.db.session() as s:
            return s.get(Conversation, conversation_id)

    def rename_conversation(self, conversation_id: int, title: str) -> None:
        with self.db.session() as s:
            conv = s.get(Conversation, conversation_id)
            if conv is not None:
                conv.title = (title.strip() or DEFAULT_TITLE)[:200]

    def delete_conversation(self, conversation_id: int) -> bool:
        with self.db.session() as s:
            conv = s.get(Conversation, conversation_id)
            if conv is None:
                return False
            s.delete(conv)  # messages removed by cascade
            return True

    # ---- messages ------------------------------------------------------
    def add_message(
        self, conversation_id: int, role: str, content: str, sources: Any = None
    ) -> Message:
        with self.db.session() as s:
            conv = s.get(Conversation, conversation_id)
            if conv is None:
                raise ValueError(f"Conversation {conversation_id} does not exist")
            now = utcnow()
            msg = Message(
                conversation_id=conversation_id, role=role, content=content,
                timestamp=now, sources=sources,
            )
            conv.updated_at = now
            s.add(msg)
            s.flush()
            return msg

    def get_messages(self, conversation_id: int) -> list[Message]:
        with self.db.session() as s:
            stmt = (
                select(Message)
                .where(Message.conversation_id == conversation_id)
                .order_by(Message.message_id)
            )
            return list(s.scalars(stmt))

    def clear_messages(self, conversation_id: int) -> int:
        with self.db.session() as s:
            msgs = list(s.scalars(select(Message).where(Message.conversation_id == conversation_id)))
            for m in msgs:
                s.delete(m)
            return len(msgs)

    # ---- documents -----------------------------------------------------
    def create_document(
        self, *, filename: str, file_hash: str, file_path: str, file_type: str, file_size: int
    ) -> Document:
        with self.db.session() as s:
            doc = Document(
                filename=filename, file_hash=file_hash, file_path=file_path,
                file_type=file_type, file_size=file_size, processing_status="processing",
            )
            s.add(doc)
            s.flush()
            return doc

    def get_document(self, document_id: int) -> Optional[Document]:
        with self.db.session() as s:
            return s.get(Document, document_id)

    def get_document_by_hash(self, file_hash: str) -> Optional[Document]:
        with self.db.session() as s:
            return s.scalars(select(Document).where(Document.file_hash == file_hash)).first()

    def list_documents(self, status: Optional[str] = None) -> list[Document]:
        with self.db.session() as s:
            stmt = select(Document).order_by(Document.uploaded_at.desc(), Document.document_id.desc())
            if status:
                stmt = stmt.where(Document.processing_status == status)
            return list(s.scalars(stmt))

    def update_document_status(
        self, document_id: int, status: str, chunk_count: Optional[int] = None,
        error: Optional[str] = None,
    ) -> None:
        with self.db.session() as s:
            doc = s.get(Document, document_id)
            if doc is None:
                return
            doc.processing_status = status
            doc.error_message = error
            if chunk_count is not None:
                doc.chunk_count = chunk_count

    def delete_document(self, document_id: int) -> bool:
        with self.db.session() as s:
            doc = s.get(Document, document_id)
            if doc is None:
                return False
            s.delete(doc)
            return True

    def count_documents(self, status: Optional[str] = None) -> int:
        return len(self.list_documents(status))


__all__: Sequence[str] = ["Repository"]
