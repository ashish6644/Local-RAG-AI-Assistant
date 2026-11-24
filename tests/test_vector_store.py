from core.vector_store import VectorStore


class FakeCollection:
    name = "documents"


class FakeClient:
    def __init__(self):
        self.deleted: list[str] = []
        self.created: list[tuple[str, dict | None]] = []

    def delete_collection(self, name):
        self.deleted.append(name)

    def get_or_create_collection(self, name, metadata=None):
        self.created.append((name, metadata))
        return FakeCollection()


def make_store() -> VectorStore:
    store = VectorStore.__new__(VectorStore)
    store.collection_name = "documents"
    store._client = FakeClient()
    store._collection = FakeCollection()
    return store


def test_reset_deletes_and_recreates_the_collection():
    store = make_store()

    store.reset()

    assert store._client.deleted == ["documents"]
    assert store._client.created == [("documents", {"hnsw:space": "cosine"})]
    assert isinstance(store._collection, FakeCollection)


def test_reset_treats_a_missing_collection_as_already_clean():
    store = make_store()

    def missing(name):
        raise RuntimeError("Collection does not exist")

    store._client.delete_collection = missing

    store.reset()

    assert store._client.created == [("documents", {"hnsw:space": "cosine"})]
