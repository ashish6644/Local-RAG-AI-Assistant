import pytest

from core.errors import OllamaError
from services.model_service import ModelService, is_embedding_model, pull_command


class FakeOllama:
    """Ollama stand-in with per-operation failure switches."""

    def __init__(
        self,
        models=("llama3.2:3b", "nomic-embed-text"),
        fail_list=False,
        fail_version=False,
        fail_chat=False,
        fail_embed=False,
        reply="ok",
    ):
        self.models = list(models)
        self.fail_list = fail_list
        self.fail_version = fail_version
        self.fail_chat = fail_chat
        self.fail_embed = fail_embed
        self.reply = reply

    def is_running(self):
        return True

    def list_models(self):
        if self.fail_list:
            raise OllamaError("Ollama is offline")
        return list(self.models)

    def model_installed(self, name, installed):
        return name in installed

    def version(self):
        if self.fail_version:
            raise OllamaError("version unavailable")
        return "0.5.0"

    def chat(self, model, messages, options=None):
        if self.fail_chat:
            raise OllamaError("chat failed")
        return self.reply

    def embed(self, model, texts):
        if self.fail_embed:
            raise OllamaError("embed failed")
        return [[0.1, 0.2, 0.3, 0.4]]


def test_embedding_model_detection_and_pull_command():
    assert is_embedding_model("nomic-embed-text")
    assert is_embedding_model("bge-small-en")
    assert not is_embedding_model("llama3.2:3b")
    assert pull_command("llama3.2:3b") == "ollama pull llama3.2:3b"


def test_installed_models_returns_empty_on_connection_error():
    assert ModelService(FakeOllama(fail_list=True)).installed_models() == []


def test_chat_and_embedding_model_lists_are_separated():
    service = ModelService(FakeOllama(models=("llama3.2:3b", "nomic-embed-text", "bge-small")))

    assert service.chat_models() == ["llama3.2:3b"]
    # Embedding models are listed first, then the remaining (chat) models.
    assert service.embedding_models() == ["bge-small", "nomic-embed-text", "llama3.2:3b"]


def test_preflight_passes_when_required_models_exist():
    service = ModelService(FakeOllama())

    assert service.preflight("llama3.2:3b", "nomic-embed-text") == (True, "")


def test_preflight_reports_missing_models_with_pull_commands():
    service = ModelService(FakeOllama(models=()))

    ok, message = service.preflight("llama3.2:3b", "nomic-embed-text")

    assert ok is False
    assert "llama3.2:3b" in message and "nomic-embed-text" in message
    assert "ollama pull llama3.2:3b" in message
    assert "ollama pull nomic-embed-text" in message


def test_preflight_honours_optional_requirements():
    service = ModelService(FakeOllama(models=("nomic-embed-text",)))

    assert service.preflight("missing-chat", "nomic-embed-text", need_chat=False) == (True, "")


def test_preflight_surfaces_list_errors():
    ok, message = ModelService(FakeOllama(fail_list=True)).preflight("a", "b")

    assert ok is False and "offline" in message


def test_test_connection_success_reports_chat_and_embedding_details():
    result = ModelService(FakeOllama()).test_connection("llama3.2:3b", "nomic-embed-text")

    assert result["running"] is True
    assert result["version"] == "0.5.0"
    assert result["chat"]["ok"] is True
    assert result["embedding"]["ok"] is True
    assert "4 dimensions" in result["embedding"]["detail"]


def test_test_connection_stops_when_version_fails():
    result = ModelService(FakeOllama(fail_version=True)).test_connection("a", "b")

    assert result["running"] is False
    assert result["chat"] is None and result["embedding"] is None
    assert "version unavailable" in result["error"]


def test_test_connection_records_chat_and_embedding_failures_independently():
    chat = ModelService(FakeOllama(fail_chat=True)).test_connection("a", "b")
    embed = ModelService(FakeOllama(fail_embed=True)).test_connection("a", "b")

    assert chat["chat"]["ok"] is False and chat["embedding"]["ok"] is True
    assert embed["chat"]["ok"] is True and embed["embedding"]["ok"] is False
