from core.rag_chain import RAGParams, RAGPipeline
from core.retriever import Retriever
from fakes import FakeEmbedder, FakeOllama, FakeStore, fake_vector

PARAMS = RAGParams(chat_model="m", embedding_model="e", threshold=0.0, history_window=2)


def pipeline(tokens=("Answer", " [1]"), docs=True):
    store = FakeStore()
    if docs:
        text = "Revenue grew 12% in 2023."
        store.add_chunks(1, ["1:0"], [text], [fake_vector(text)], [{"document_id": 1, "filename": "report.pdf", "location": "page 3", "page": 3}])
    ollama = FakeOllama(tokens)
    return RAGPipeline(ollama, Retriever(store, FakeEmbedder())), ollama


def test_answer_streams_real_tokens_with_context_and_sources():
    p, ollama = pipeline()
    retrieval = p.retrieve("Revenue grew 12% in 2023.", [], PARAMS)
    assert retrieval.reason == "ok"
    tokens = list(p.stream_answer("What grew?", [], retrieval, PARAMS))
    assert tokens == ["Answer", " [1]"]
    system, user = ollama.stream_calls[0][0], ollama.stream_calls[0][-1]
    assert system["role"] == "system" and "ONLY" in system["content"]
    assert "[1] Source: report.pdf (page 3)" in user["content"]
    payload = retrieval.to_payload()
    assert payload["chunks"][0]["page"] == 3 and payload["chunks"][0]["filename"] == "report.pdf"


def test_history_window_limits_messages_and_rewrite_runs():
    p, ollama = pipeline()
    history = [{"role": "user" if i % 2 == 0 else "assistant", "content": f"m{i}"} for i in range(6)]
    retrieval = p.retrieve("and the second one?", history, PARAMS)
    assert retrieval.query == "standalone question?" and ollama.chat_calls
    msgs = p.build_messages("q", history, retrieval.chunks, PARAMS)
    assert len(msgs) == 1 + 2 + 1


def test_history_window_zero_disables_rewriting():
    p, ollama = pipeline()
    history = [{"role": "user", "content": "earlier"}, {"role": "assistant", "content": "answer"}]
    no_history = RAGParams(chat_model="m", embedding_model="e", threshold=0.0, history_window=0)

    retrieval = p.retrieve("and the second one?", history, no_history)

    assert retrieval.query == "and the second one?"
    assert not ollama.chat_calls


def test_no_documents_and_no_match_never_call_the_llm():
    p, ollama = pipeline(docs=False)
    r = p.retrieve("anything", [], PARAMS)
    assert r.reason == "no_documents"
    assert "no indexed documents" in "".join(p.stream_answer("anything", [], r, PARAMS))
    p2, ollama2 = pipeline()
    strict = RAGParams(chat_model="m", embedding_model="e", threshold=1.01)
    r2 = p2.retrieve("zzz", [], strict)
    assert r2.reason == "no_match"
    assert "do not contain sufficient information" in "".join(p2.stream_answer("zzz", [], r2, strict))
    assert not ollama.stream_calls and not ollama2.stream_calls
