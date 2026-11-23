import pytest

from core.embeddings import OllamaEmbedder, _prefix
from core.errors import EmbeddingError, OllamaError


class FakeClient:
    """Records the batches sent to Ollama and returns trivial vectors."""

    def __init__(self, fail: bool = False):
        self.fail = fail
        self.batches: list[tuple[str, list[str]]] = []

    def embed(self, model: str, texts: list[str]) -> list[list[float]]:
        self.batches.append((model, list(texts)))
        if self.fail:
            raise OllamaError("boom")
        return [[float(len(text)), 1.0] for text in texts]


def test_nomic_models_use_task_prefixes():
    assert _prefix("nomic-embed-text", "document") == "search_document: "
    assert _prefix("nomic-embed-text:latest", "query") == "search_query: "
    assert _prefix("all-minilm", "document") == ""


def test_documents_are_batched_and_prefixed():
    client = FakeClient()
    embedder = OllamaEmbedder(client, batch_size=2)

    vectors = embedder.embed_documents(["one", "two", "three"], "nomic-embed-text")

    assert len(vectors) == 3
    assert [texts for _, texts in client.batches] == [
        ["search_document: one", "search_document: two"],
        ["search_document: three"],
    ]


def test_queries_use_the_query_prefix():
    client = FakeClient()
    embedder = OllamaEmbedder(client)

    assert embedder.embed_queries(["hello"], "nomic-embed-text") == [
        [float(len("search_query: hello")), 1.0]
    ]
    assert client.batches == [("nomic-embed-text", ["search_query: hello"])]


def test_embed_query_returns_a_single_vector():
    embedder = OllamaEmbedder(FakeClient())

    assert embedder.embed_query("hi", "m") == [[float(len("hi")), 1.0]][0]


def test_empty_input_does_not_call_the_client():
    client = FakeClient()
    embedder = OllamaEmbedder(client)

    assert embedder.embed_documents([], "m") == []
    assert embedder.embed_queries([], "m") == []
    assert client.batches == []


def test_client_errors_are_wrapped_in_embedding_error():
    embedder = OllamaEmbedder(FakeClient(fail=True))

    with pytest.raises(EmbeddingError) as info:
        embedder.embed_documents(["x"], "m")

    assert "boom" in str(info.value)
    assert isinstance(info.value.__cause__, OllamaError)
