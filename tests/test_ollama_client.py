import json

import pytest
import requests

from core.errors import OllamaConnectionError, OllamaError, OllamaModelNotFoundError
from core.ollama_client import OllamaClient


class FakeResponse:
    def __init__(self, status=200, lines=(), body=None, text=""):
        self.status_code, self._lines, self._body, self.text = status, lines, body, text
        self.closed = False

    def iter_content(self, chunk_size=None):
        for line in self._lines:
            yield (json.dumps(line) + "\n").encode()

    def json(self):
        return self._body

    def close(self):
        self.closed = True


class FakeSession:
    def __init__(self, response=None, error=None):
        self.response, self.error = response, error

    def post(self, *a, **k):
        if self.error:
            raise self.error
        return self.response

    get = post


def client_with(**kw):
    c = OllamaClient()
    c._session = FakeSession(**kw)
    return c


def test_stream_yields_tokens_incrementally_and_stops_at_done():
    lines = [{"message": {"content": "He"}}, {"message": {"content": "llo"}}, {"done": True}, {"message": {"content": "ignored"}}]
    resp = FakeResponse(lines=lines)
    stream = client_with(response=resp).chat_stream("m", [])
    assert next(stream) == "He"
    assert next(stream) == "llo"
    assert list(stream) == [] and resp.closed


def test_closing_the_stream_closes_the_connection():
    resp = FakeResponse(lines=[{"message": {"content": "a"}}, {"message": {"content": "b"}}])
    stream = client_with(response=resp).chat_stream("m", [])
    next(stream)
    stream.close()
    assert resp.closed


def test_midstream_error_is_raised():
    resp = FakeResponse(lines=[{"error": "model crashed"}])
    with pytest.raises(OllamaError):
        list(client_with(response=resp).chat_stream("m", []))


def test_connection_refused_message():
    c = client_with(error=requests.exceptions.ConnectionError("refused"))
    with pytest.raises(OllamaConnectionError):
        c.chat("m", [])
    assert c.is_running() is False


def test_missing_model_gives_pull_instruction():
    resp = FakeResponse(status=404, text=json.dumps({"error": "model 'x' not found"}))
    with pytest.raises(OllamaModelNotFoundError) as err:
        client_with(response=resp).embed("x", ["t"])
    assert "ollama pull x" in str(err.value)


def test_embed_validates_count():
    resp = FakeResponse(body={"embeddings": [[0.1]]})
    with pytest.raises(OllamaError):
        client_with(response=resp).embed("m", ["a", "b"])


def test_model_installed_handles_latest_tag():
    assert OllamaClient.model_installed("nomic-embed-text", ["nomic-embed-text:latest"])
    assert not OllamaClient.model_installed("llama3.2:3b", ["llama3.2:1b"])
