import pytest

pytest.importorskip("sqlalchemy")

from database.connection import Database
from database.repository import Repository


def make_repo(tmp_path):
    db = Database(tmp_path / "t.db")
    db.init_db()
    return db, Repository(db)


def test_conversation_and_message_roundtrip(tmp_path):
    _, repo = make_repo(tmp_path)
    conv = repo.create_conversation("Hello")
    repo.add_message(conv.conversation_id, "user", "Hi")
    repo.add_message(conv.conversation_id, "assistant", "Hey", sources={"chunks": [{"filename": "a.pdf"}]})
    msgs = repo.get_messages(conv.conversation_id)
    assert [m.role for m in msgs] == ["user", "assistant"]
    assert msgs[1].sources["chunks"][0]["filename"] == "a.pdf"


def test_rename_ordering_and_cascade_delete(tmp_path):
    _, repo = make_repo(tmp_path)
    a = repo.create_conversation("A")
    b = repo.create_conversation("B")
    repo.add_message(a.conversation_id, "user", "bump A")
    assert [c.title for c in repo.list_conversations()] == ["A", "B"]
    repo.rename_conversation(b.conversation_id, "Renamed")
    assert repo.get_conversation(b.conversation_id).title == "Renamed"
    assert repo.delete_conversation(a.conversation_id)
    assert repo.get_messages(a.conversation_id) == []


def test_persistence_after_restart(tmp_path):
    db, repo = make_repo(tmp_path)
    conv = repo.create_conversation("Persist")
    repo.add_message(conv.conversation_id, "user", "remember me")
    db.dispose()
    db2 = Database(tmp_path / "t.db")
    db2.init_db()
    repo2 = Repository(db2)
    assert repo2.get_messages(conv.conversation_id)[0].content == "remember me"


def test_documents_unique_hash_and_status(tmp_path):
    _, repo = make_repo(tmp_path)
    doc = repo.create_document(filename="a.txt", file_hash="h1", file_path="uploads/a.txt", file_type="txt", file_size=3)
    assert repo.get_document_by_hash("h1").document_id == doc.document_id
    repo.update_document_status(doc.document_id, "ready", chunk_count=5)
    assert repo.get_document(doc.document_id).chunk_count == 5
    assert len(repo.list_documents("ready")) == 1
    assert repo.delete_document(doc.document_id)
