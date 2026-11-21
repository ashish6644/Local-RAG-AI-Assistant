import pytest

from core.retriever import Retriever
from fakes import FakeEmbedder, FakeStore, fake_vector


def populated_store():
    store = FakeStore()
    texts = {"1:0": "solar panels convert sunlight", "1:1": "wind turbines generate power", "2:0": "cooking pasta recipes"}
    for cid, text in texts.items():
        store.add_chunks(int(cid[0]), [cid], [text], [fake_vector(text)], [{"document_id": int(cid[0]), "filename": f"d{cid[0]}.txt", "location": "text file"}])
    return store


def test_exact_query_ranks_first_and_respects_top_k():
    r = Retriever(populated_store(), FakeEmbedder())
    hits = r.retrieve(["wind turbines generate power"], "m", top_k=2, threshold=0.0)
    assert hits[0].id == "1:1" and len(hits) == 2


def test_threshold_filters_everything():
    r = Retriever(populated_store(), FakeEmbedder())
    assert r.retrieve(["something unrelated"], "m", top_k=3, threshold=1.01) == []


def test_document_filter():
    r = Retriever(populated_store(), FakeEmbedder())
    hits = r.retrieve(["wind turbines generate power"], "m", top_k=5, threshold=0.0, document_ids=[2])
    assert {h.document_id for h in hits} == {2}


def test_multi_query_merges_without_duplicates():
    r = Retriever(populated_store(), FakeEmbedder())
    hits = r.retrieve(["solar panels convert sunlight", "cooking pasta recipes"], "m", top_k=5, threshold=0.0)
    ids = [h.id for h in hits]
    assert len(ids) == len(set(ids)) == 3


def test_mmr_returns_k_results():
    r = Retriever(populated_store(), FakeEmbedder())
    assert len(r.retrieve(["solar"], "m", top_k=2, threshold=0.0, strategy="mmr")) == 2


def test_real_chroma_persistence(tmp_path):
    pytest.importorskip("chromadb")
    from core.vector_store import VectorStore

    store = VectorStore(tmp_path / "chroma", "t")
    store.add_chunks(7, ["7:0"], ["hello world"], [fake_vector("hello world")], [{"document_id": 7, "filename": "h.txt", "location": "x"}])
    assert VectorStore(tmp_path / "chroma", "t").count_document(7) == 1
    hit = store.query(fake_vector("hello world"), 3)[0]
    assert hit.id == "7:0" and hit.score > 0.99
    store.delete_document(7)
    assert store.count() == 0
