"""Ollama availability, model discovery, pre-flight checks and connection tests."""
from __future__ import annotations

import time
from typing import Any

from core.errors import OllamaError
from core.ollama_client import OllamaClient

_EMBED_HINTS = ("embed", "bge-", "minilm", "mxbai", "e5-", "gte-", "arctic-embed")


def is_embedding_model(name: str) -> bool:
    return any(h in name.lower() for h in _EMBED_HINTS)


def pull_command(model: str) -> str:
    return f"ollama pull {model}"


class ModelService:
    def __init__(self, client: OllamaClient) -> None:
        self.client = client

    def is_running(self) -> bool:
        return self.client.is_running()

    def installed_models(self) -> list[str]:
        try:
            return self.client.list_models()
        except OllamaError:
            return []

    def chat_models(self) -> list[str]:
        return [m for m in self.installed_models() if not is_embedding_model(m)]

    def embedding_models(self) -> list[str]:
        models = self.installed_models()
        return sorted(models, key=lambda m: (not is_embedding_model(m), m))

    def preflight(
        self, chat_model: str, embedding_model: str, need_chat: bool = True, need_embedding: bool = True
    ) -> tuple[bool, str]:
        """Return (ok, actionable message). Cheap: one request to /api/tags."""
        try:
            installed = self.client.list_models()
        except OllamaError as exc:
            return False, str(exc)
        missing = []
        if need_chat and not self.client.model_installed(chat_model, installed):
            missing.append(chat_model)
        if need_embedding and not self.client.model_installed(embedding_model, installed):
            missing.append(embedding_model)
        if missing:
            cmds = "\n".join(pull_command(m) for m in dict.fromkeys(missing))
            return False, (
                "Required Ollama model(s) not installed: " + ", ".join(dict.fromkeys(missing))
                + ".\nDownload with:\n" + cmds
            )
        return True, ""

    def test_connection(self, chat_model: str, embedding_model: str) -> dict[str, Any]:
        result: dict[str, Any] = {"running": False, "version": None, "chat": None, "embedding": None}
        try:
            result["version"] = self.client.version()
            result["running"] = True
        except OllamaError as exc:
            result["error"] = str(exc)
            return result
        started = time.perf_counter()
        try:
            reply = self.client.chat(
                chat_model, [{"role": "user", "content": "Reply with the single word: ok"}],
                {"num_predict": 8, "temperature": 0},
            )
            result["chat"] = {"ok": True, "detail": f"responded in {time.perf_counter() - started:.1f}s ({reply.strip()[:30]!r})"}
        except OllamaError as exc:
            result["chat"] = {"ok": False, "detail": str(exc)}
        started = time.perf_counter()
        try:
            vector = self.client.embed(embedding_model, ["connection test"])[0]
            result["embedding"] = {"ok": True, "detail": f"{len(vector)} dimensions in {time.perf_counter() - started:.1f}s"}
        except OllamaError as exc:
            result["embedding"] = {"ok": False, "detail": str(exc)}
        return result
