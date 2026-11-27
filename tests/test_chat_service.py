import pytest

pytest.importorskip("sqlalchemy")

from fakes import FakeEmbedder, FakeOllama, FakeStore, fake_vector

from core.rag_chain import RAGParams, RAGPipeline
from core.retriever import Retriever
from database.connection import Database
from database.repository import Repository
from services.chat_service import ChatService

PARAMS = RAGParams(chat_model="m", embedding_model="e", threshold=0.0, rewrite=False)


def make_service(tmp_path, tokens=("A", "B", "C")):
    db = Database(tmp_path / "c.db")
    db.init_db()
    store = FakeStore()
    text = "Solar energy facts"
    store.add_chunks(1, ["1:0"], [text], [fake_vector(text)], [{"document_id": 1, "filename": "s.txt", "location": "text file"}])
    return ChatService(Repository(db), RAGPipeline(FakeOllama(tokens), Retriever(store, FakeEmbedder())))


def test_full_turn_is_saved_with_sources_and_auto_title(tmp_path):
    svc = make_service(tmp_path)
    conv = svc.create_conversation()
    events = list(svc.ask(conv.conversation_id, "Solar energy facts", PARAMS))
    assert [e["text"] for e in events if e["type"] == "token"] == ["A", "B", "C"]
    msgs = svc.get_messages(conv.conversation_id)
    assert [m.role for m in msgs] == ["user", "assistant"] and msgs[1].content == "ABC"
    assert msgs[1].sources["chunks"][0]["filename"] == "s.txt"
    assert svc.get_conversation(conv.conversation_id).title == "Solar energy facts"


def test_stopping_early_saves_partial_answer(tmp_path):
    svc = make_service(tmp_path)
    conv = svc.create_conversation()
    gen = svc.ask(conv.conversation_id, "Solar energy facts", PARAMS)
    for event in gen:
        if event["type"] == "token":
            break
    gen.close()
    last = svc.get_messages(conv.conversation_id)[-1]
    assert last.role == "assistant" and last.content.startswith("A") and "stopped" in last.content
