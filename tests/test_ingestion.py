import pytest

pytest.importorskip("sqlalchemy")
pytest.importorskip("pydantic")
pytest.importorskip("langchain_text_splitters")

from config.settings import Settings
from database.connection import Database
from database.repository import Repository
from fakes import FakeEmbedder, FakeStore
from services.document_service import DocumentService
from services.ingestion_service import IngestionService


def setup(tmp_path, embedder=None):
    cfg = Settings(data_dir=tmp_path / "data", chunk_size=200, chunk_overlap=20)
    (cfg.data_dir / "uploads").mkdir(parents=True)
    db = Database(cfg.db_path)
    db.init_db()
    repo, store = Repository(db), FakeStore()
    ing = IngestionService(cfg, repo, embedder or FakeEmbedder(), store)
    return cfg, repo, store, ing, DocumentService(cfg, repo, store, ing)


TEXT = ("Solar energy is growing quickly. " * 30).encode()


def test_ingest_duplicate_delete_cycle(tmp_path):
    cfg, repo, store, ing, docs = setup(tmp_path)
    r = ing.ingest_bytes("notes.txt", TEXT, cfg)
    assert r.status == "ready" and r.document.chunk_count == store.count() > 1
    assert ing.ingest_bytes("copy.txt", TEXT, cfg).status == "duplicate"
    assert store.count() == r.document.chunk_count  # no double indexing
    path = cfg.resolve_data_path(r.document.file_path)
    assert path.exists()
    assert docs.delete_document(r.document.document_id)
    assert store.count() == 0 and not path.exists() and repo.list_documents() == []


def test_rejections_and_failures_do_not_crash(tmp_path):
    cfg, repo, store, ing, _ = setup(tmp_path)
    assert ing.ingest_bytes("virus.exe", b"x", cfg).status == "rejected"
    assert ing.ingest_bytes("empty.txt", b"", cfg).status == "rejected"
    bad = ing.ingest_bytes("broken.pdf", b"not a pdf", cfg)
    assert bad.status == "failed" and repo.list_documents()[0].processing_status == "failed"
    cfg2, repo2, store2, ing2, _ = setup(tmp_path / "b", FakeEmbedder(fail=True))
    failed = ing2.ingest_bytes("x.txt", TEXT, cfg2)
    assert failed.status == "failed" and store2.count() == 0
