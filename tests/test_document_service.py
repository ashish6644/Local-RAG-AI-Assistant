import pytest

pytest.importorskip("sqlalchemy")
pytest.importorskip("pydantic")
pytest.importorskip("langchain_text_splitters")

from fakes import FakeEmbedder, FakeStore

from config.settings import Settings
from core.errors import AppError, DatabaseError
from database.connection import Database
from database.repository import Repository
from services.document_service import DocumentService
from services.ingestion_service import IngestionService

TEXT = ("Solar energy is growing quickly. " * 30).encode()


def setup(tmp_path, chunk_size=200):
    cfg = Settings(data_dir=tmp_path / "data", chunk_size=chunk_size, chunk_overlap=20)
    (cfg.data_dir / "uploads").mkdir(parents=True)
    db = Database(cfg.db_path)
    db.init_db()
    repo, store = Repository(db), FakeStore()
    ingestion = IngestionService(cfg, repo, FakeEmbedder(), store)
    return cfg, repo, store, ingestion, DocumentService(cfg, repo, store, ingestion)


def test_reindex_unknown_document_returns_none(tmp_path):
    cfg, _, _, _, documents = setup(tmp_path)

    assert documents.reindex(999, cfg) is None


def test_reindex_with_missing_file_marks_the_document_failed(tmp_path):
    cfg, repo, _, ingestion, documents = setup(tmp_path)
    result = ingestion.ingest_bytes("notes.txt", TEXT, cfg)
    cfg.resolve_data_path(result.document.file_path).unlink()

    again = documents.reindex(result.document.document_id, cfg)

    assert again.status == "failed"
    assert "missing" in again.message
    assert repo.get_document(result.document.document_id).processing_status == "failed"


def test_reindex_with_missing_file_removes_stale_vectors(tmp_path):
    cfg, repo, store, ingestion, documents = setup(tmp_path)
    result = ingestion.ingest_bytes("notes.txt", TEXT, cfg)
    assert store.count_document(result.document.document_id) > 0
    cfg.resolve_data_path(result.document.file_path).unlink()

    documents.reindex(result.document.document_id, cfg)

    assert store.count_document(result.document.document_id) == 0
    assert repo.get_document(result.document.document_id).processing_status == "failed"


def test_reindex_rebuilds_vectors_with_the_new_chunking(tmp_path):
    cfg, repo, store, ingestion, documents = setup(tmp_path, chunk_size=200)
    result = ingestion.ingest_bytes("notes.txt", TEXT, cfg)
    first_count = result.document.chunk_count
    smaller = Settings(data_dir=cfg.data_dir, chunk_size=100, chunk_overlap=10)

    again = documents.reindex(result.document.document_id, smaller)

    assert again.status == "ready"
    assert again.document.chunk_count == store.count()
    assert again.document.chunk_count != first_count
    assert repo.get_document(result.document.document_id).processing_status == "ready"


def test_delete_document_removes_vectors_metadata_and_file(tmp_path):
    cfg, repo, store, ingestion, documents = setup(tmp_path)
    result = ingestion.ingest_bytes("notes.txt", TEXT, cfg)
    path = cfg.resolve_data_path(result.document.file_path)

    assert documents.delete_document(result.document.document_id) is True

    assert store.count() == 0
    assert repo.get_document(result.document.document_id) is None
    assert not path.exists()


def test_delete_unknown_document_returns_false(tmp_path):
    _, _, _, _, documents = setup(tmp_path)

    assert documents.delete_document(123) is False


def test_delete_marks_the_row_failed_when_metadata_deletion_fails(tmp_path, monkeypatch):
    cfg, repo, store, ingestion, documents = setup(tmp_path)
    result = ingestion.ingest_bytes("notes.txt", TEXT, cfg)

    def broken_delete(_document_id):
        raise DatabaseError("database is locked")

    monkeypatch.setattr(repo, "delete_document", broken_delete)

    with pytest.raises(AppError, match="locked"):
        documents.delete_document(result.document.document_id)

    assert store.count() == 0
    assert repo.get_document(result.document.document_id).processing_status == "failed"


def test_rebuild_index_resets_vectors_and_reindexes_every_document(tmp_path):
    cfg, repo, store, ingestion, documents = setup(tmp_path, chunk_size=200)
    first = ingestion.ingest_bytes("one.txt", TEXT, cfg)
    second = ingestion.ingest_bytes("two.txt", ("Wind power is expanding. " * 30).encode(), cfg)
    before = first.document.chunk_count + second.document.chunk_count
    smaller = Settings(data_dir=cfg.data_dir, chunk_size=100, chunk_overlap=10)

    indexed, failed = documents.rebuild_index(smaller)

    assert (indexed, failed) == (2, 0)
    assert store.count() == sum(doc.chunk_count for doc in repo.list_documents())
    assert store.count() > before


def test_rebuild_index_marks_missing_files_failed(tmp_path):
    cfg, repo, store, ingestion, documents = setup(tmp_path)
    keep = ingestion.ingest_bytes("keep.txt", TEXT, cfg)
    gone = ingestion.ingest_bytes("gone.txt", ("Wind power is expanding. " * 30).encode(), cfg)
    cfg.resolve_data_path(gone.document.file_path).unlink()

    indexed, failed = documents.rebuild_index(cfg)

    assert (indexed, failed) == (1, 1)
    assert repo.get_document(gone.document.document_id).processing_status == "failed"
    assert repo.get_document(keep.document.document_id).processing_status == "ready"
    assert store.count() == keep.document.chunk_count


def test_rebuild_index_aborts_before_reset_when_embedding_fails(tmp_path):
    cfg, repo, store, ingestion, documents = setup(tmp_path)
    result = ingestion.ingest_bytes("notes.txt", TEXT, cfg)
    before = store.count()
    ingestion.embedder.fail = True

    with pytest.raises(AppError):
        documents.rebuild_index(cfg)

    # A broken embedding model must not wipe a healthy index.
    assert store.count() == before
    assert repo.get_document(result.document.document_id).processing_status == "ready"
