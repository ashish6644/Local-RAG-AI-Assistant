"""Configuration: environment variables (.env) + persisted UI overrides, validated by Pydantic."""
from __future__ import annotations

import json
import logging
import os
from logging.handlers import RotatingFileHandler
from pathlib import Path
from typing import Any, Literal

from dotenv import load_dotenv
from pydantic import BaseModel, Field, ValidationError, field_validator, model_validator

from core.errors import ConfigError

BASE_DIR = Path(__file__).resolve().parent.parent
load_dotenv(BASE_DIR / ".env")

ALLOWED_EXTENSIONS: tuple[str, ...] = (".pdf", ".docx", ".txt", ".csv", ".xlsx")
USER_SETTINGS_FILE = "user_settings.json"

# Settings the user may change from the UI (persisted in data/user_settings.json).
USER_EDITABLE = (
    "chat_model", "embedding_model", "temperature", "top_p", "max_output_tokens",
    "context_window", "chunk_size", "chunk_overlap", "chunking_strategy", "top_k",
    "similarity_threshold", "search_strategy", "history_window",
    "use_query_rewriting", "use_multi_query",
)


class Settings(BaseModel):
    ollama_base_url: str = "http://localhost:11434"
    chat_model: str = "llama3.2:3b"
    embedding_model: str = "nomic-embed-text"

    temperature: float = Field(0.2, ge=0.0, le=2.0)
    top_p: float = Field(0.9, gt=0.0, le=1.0)
    max_output_tokens: int = Field(1024, ge=16, le=32768)
    context_window: int = Field(4096, ge=512, le=131072)

    chunk_size: int = Field(1000, ge=100, le=8000)
    chunk_overlap: int = Field(150, ge=0, le=4000)
    chunking_strategy: Literal["recursive", "character"] = "recursive"

    top_k: int = Field(4, ge=1, le=20)
    similarity_threshold: float = Field(0.30, ge=0.0, le=1.0)
    search_strategy: Literal["similarity", "mmr"] = "similarity"
    history_window: int = Field(6, ge=0, le=50)
    use_query_rewriting: bool = True
    use_multi_query: bool = False

    max_upload_mb: int = Field(50, ge=1, le=1024)
    data_dir: Path = BASE_DIR / "data"
    log_dir: Path = BASE_DIR / "logs"
    collection_name: str = Field("documents", min_length=1)
    log_level: str = "INFO"

    @field_validator("data_dir", "log_dir", mode="before")
    @classmethod
    def _resolve_dir(cls, value: Any) -> Path:
        path = Path(str(value)).expanduser()
        return path if path.is_absolute() else BASE_DIR / path

    @field_validator("ollama_base_url")
    @classmethod
    def _url(cls, value: str) -> str:
        value = value.strip().rstrip("/")
        if not value.startswith(("http://", "https://")):
            raise ValueError("OLLAMA_BASE_URL must start with http:// or https://")
        return value

    @model_validator(mode="after")
    def _overlap_smaller_than_chunk(self) -> Settings:
        if self.chunk_overlap >= self.chunk_size:
            raise ValueError("CHUNK_OVERLAP must be smaller than CHUNK_SIZE")
        return self

    @property
    def upload_dir(self) -> Path:
        return self.data_dir / "uploads"

    @property
    def chroma_dir(self) -> Path:
        return self.data_dir / "chroma_db"

    @property
    def db_path(self) -> Path:
        return self.data_dir / "app.db"

    def resolve_data_path(self, relative: str) -> Path:
        return self.data_dir / relative


def _format_validation_error(exc: ValidationError) -> str:
    lines = []
    for err in exc.errors():
        loc = ".".join(str(p) for p in err["loc"]) or "settings"
        lines.append(f"- {loc.upper()}: {err['msg']}")
    return "Invalid configuration:\n" + "\n".join(lines)


def load_settings() -> Settings:
    """Read .env / environment, then apply UI overrides saved by the user."""
    env_values = {
        name: os.environ[name.upper()]
        for name in Settings.model_fields
        if os.environ.get(name.upper(), "").strip() != ""
    }
    # Allow trailing inline comments like "recursive   # note" in .env files.
    env_values = {k: v.split(" #")[0].split("\t#")[0].strip() for k, v in env_values.items()}
    try:
        base = Settings.model_validate(env_values)
    except ValidationError as exc:
        raise ConfigError(_format_validation_error(exc)) from exc

    overrides = load_user_overrides(base)
    if overrides:
        try:
            return Settings.model_validate({**base.model_dump(), **overrides})
        except ValidationError:
            logging.getLogger(__name__).warning("Ignoring invalid user_settings.json overrides")
    return base


def load_user_overrides(settings: Settings) -> dict[str, Any]:
    path = settings.data_dir / USER_SETTINGS_FILE
    if not path.exists():
        return {}
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        logging.getLogger(__name__).warning("Could not read %s", path)
        return {}
    if not isinstance(data, dict):
        logging.getLogger(__name__).warning("Ignoring %s: expected a JSON object", path)
        return {}
    return {k: v for k, v in data.items() if k in USER_EDITABLE}


def save_user_overrides(settings: Settings) -> None:
    """Persist the user-editable subset of `settings` so it survives restarts."""
    path = settings.data_dir / USER_SETTINGS_FILE
    path.parent.mkdir(parents=True, exist_ok=True)
    dumped = settings.model_dump()
    payload = json.dumps({k: dumped[k] for k in USER_EDITABLE}, indent=2)
    # Write atomically so an interrupted save cannot corrupt the overrides file.
    tmp = path.with_name(path.name + ".tmp")
    try:
        tmp.write_text(payload, encoding="utf-8")
        os.replace(tmp, path)
    finally:
        tmp.unlink(missing_ok=True)


def with_updates(settings: Settings, **updates: Any) -> Settings:
    """Return a validated copy of `settings` with `updates` applied."""
    try:
        return Settings.model_validate({**settings.model_dump(), **updates})
    except ValidationError as exc:
        raise ConfigError(_format_validation_error(exc)) from exc


def ensure_directories(settings: Settings) -> None:
    for path in (settings.data_dir, settings.upload_dir, settings.chroma_dir, settings.log_dir):
        path.mkdir(parents=True, exist_ok=True)


_logging_ready = False


def setup_logging(settings: Settings) -> None:
    global _logging_ready
    if _logging_ready:
        return
    settings.log_dir.mkdir(parents=True, exist_ok=True)
    level = getattr(logging, settings.log_level.upper(), logging.INFO)
    root = logging.getLogger()
    root.setLevel(level)
    fmt = logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s")
    file_handler = RotatingFileHandler(
        settings.log_dir / "app.log", maxBytes=2_000_000, backupCount=3, encoding="utf-8"
    )
    file_handler.setFormatter(fmt)
    root.addHandler(file_handler)
    logging.getLogger("chromadb").setLevel(logging.WARNING)
    logging.getLogger("httpx").setLevel(logging.WARNING)
    _logging_ready = True
