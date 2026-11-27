"""Thin HTTP client for the local Ollama server (real NDJSON streaming, no SDK needed)."""
from __future__ import annotations

import json
import logging
from collections.abc import Iterator
from typing import Any

import requests

from core.errors import OllamaConnectionError, OllamaError, OllamaModelNotFoundError

logger = logging.getLogger(__name__)


class OllamaClient:
    def __init__(
        self, base_url: str = "http://localhost:11434",
        connect_timeout: float = 5.0, read_timeout: float = 600.0,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self._timeout = (connect_timeout, read_timeout)
        self._session = requests.Session()

    # ---- helpers -------------------------------------------------------
    def _url(self, path: str) -> str:
        return f"{self.base_url}{path}"

    def _connection_error(self) -> OllamaConnectionError:
        return OllamaConnectionError(
            f"Cannot connect to Ollama at {self.base_url}. Make sure Ollama is installed "
            "(https://ollama.com/download) and running (open the Ollama app or run `ollama serve`)."
        )

    @staticmethod
    def _error_text(response: Any) -> str:
        try:
            text = response.text
            try:
                return str(json.loads(text).get("error", text))
            except (ValueError, AttributeError):
                return text
        except Exception:  # pragma: no cover - defensive
            return ""

    @staticmethod
    def _json(response: Any, context: str) -> dict:
        """Decode a JSON object, converting malformed responses into a user-facing error."""
        try:
            data = response.json()
        except ValueError as exc:
            raise OllamaError(f"Ollama returned an invalid response ({context}).") from exc
        if not isinstance(data, dict):
            raise OllamaError(f"Ollama returned an invalid response ({context}).")
        return data

    def _raise_for_status(self, response: Any, model: str | None) -> None:
        if response.status_code == 200:
            return
        detail = self._error_text(response)
        response.close()
        if response.status_code == 404 or "not found" in detail.lower():
            name = model or "the requested model"
            raise OllamaModelNotFoundError(
                f"Model '{name}' is not installed in Ollama. Download it with: ollama pull {name}"
            )
        raise OllamaError(f"Ollama returned HTTP {response.status_code}: {detail or 'unknown error'}")

    def _post(self, path: str, payload: dict, *, stream: bool = False, model: str | None = None):
        try:
            response = self._session.post(
                self._url(path), json=payload, stream=stream, timeout=self._timeout
            )
        except requests.exceptions.ConnectionError as exc:
            raise self._connection_error() from exc
        except requests.exceptions.Timeout as exc:
            raise OllamaError("Ollama took too long to respond. The model may still be loading; try again.") from exc
        except requests.exceptions.RequestException as exc:
            raise OllamaError(f"Request to Ollama failed: {exc}") from exc
        self._raise_for_status(response, model)
        return response

    def _get(self, path: str, timeout: float = 5.0) -> dict:
        try:
            response = self._session.get(self._url(path), timeout=timeout)
        except requests.exceptions.ConnectionError as exc:
            raise self._connection_error() from exc
        except requests.exceptions.RequestException as exc:
            raise OllamaError(f"Request to Ollama failed: {exc}") from exc
        self._raise_for_status(response, None)
        return self._json(response, path)

    # ---- status / models ----------------------------------------------
    def is_running(self, timeout: float = 2.0) -> bool:
        try:
            self._get("/api/version", timeout=timeout)
            return True
        except OllamaError:
            return False

    def version(self) -> str:
        return str(self._get("/api/version").get("version", "unknown"))

    def list_models(self) -> list[str]:
        data = self._get("/api/tags")
        return sorted(m.get("name", "") for m in data.get("models", []) if m.get("name"))

    @staticmethod
    def model_installed(name: str, installed: list[str]) -> bool:
        wanted = name.strip().lower()
        names = {m.lower() for m in installed}
        return wanted in names or (":" not in wanted and f"{wanted}:latest" in names)

    # ---- embeddings ----------------------------------------------------
    def embed(self, model: str, texts: list[str]) -> list[list[float]]:
        response = self._post("/api/embed", {"model": model, "input": texts}, model=model)
        embeddings = self._json(response, "/api/embed").get("embeddings")
        if not embeddings or len(embeddings) != len(texts):
            raise OllamaError(
                f"Ollama returned an unexpected embedding response for model '{model}'."
            )
        return embeddings

    # ---- chat ----------------------------------------------------------
    @staticmethod
    def _chat_payload(model: str, messages: list[dict], options: dict | None, stream: bool) -> dict:
        return {"model": model, "messages": messages, "stream": stream, "options": options or {}}

    def chat(self, model: str, messages: list[dict], options: dict | None = None) -> str:
        response = self._post("/api/chat", self._chat_payload(model, messages, options, False), model=model)
        data = self._json(response, "/api/chat")
        if "error" in data:
            raise OllamaError(str(data["error"]))
        return str(data.get("message", {}).get("content", ""))

    def chat_stream(
        self, model: str, messages: list[dict], options: dict | None = None
    ) -> Iterator[str]:
        """Yield tokens as Ollama produces them. Closing the generator aborts generation."""
        response = self._post(
            "/api/chat", self._chat_payload(model, messages, options, True), stream=True, model=model
        )
        buffer = b""
        try:
            for piece in response.iter_content(chunk_size=None):
                if not piece:
                    continue
                buffer += piece
                while b"\n" in buffer:
                    line, buffer = buffer.split(b"\n", 1)
                    token, done = self._parse_line(line)
                    if token:
                        yield token
                    if done:
                        return
            if buffer.strip():
                token, _ = self._parse_line(buffer)
                if token:
                    yield token
        except requests.exceptions.RequestException as exc:
            raise OllamaError(f"The connection to Ollama was interrupted: {exc}") from exc
        finally:
            response.close()

    @staticmethod
    def _parse_line(line: bytes) -> tuple[str, bool]:
        line = line.strip()
        if not line:
            return "", False
        try:
            data = json.loads(line)
        except ValueError:
            logger.warning("Skipping malformed streaming line from Ollama")
            return "", False
        if "error" in data:
            raise OllamaError(str(data["error"]))
        return str(data.get("message", {}).get("content", "")), bool(data.get("done"))
