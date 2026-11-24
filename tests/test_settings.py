import json

import pytest
from pydantic import ValidationError

from config.settings import (
    BASE_DIR,
    USER_EDITABLE,
    Settings,
    ensure_directories,
    load_settings,
    load_user_overrides,
    save_user_overrides,
    with_updates,
)
from core.errors import ConfigError

FIELD_ENVS = [name.upper() for name in Settings.model_fields]


@pytest.fixture(autouse=True)
def clean_settings_env(monkeypatch):
    """Keep the developer's real environment out of the settings tests."""
    for name in FIELD_ENVS:
        monkeypatch.delenv(name, raising=False)


def test_relative_directories_resolve_against_the_project_root():
    settings = Settings(data_dir="relative/data", log_dir="relative/logs")

    assert settings.data_dir == BASE_DIR / "relative/data"
    assert settings.log_dir == BASE_DIR / "relative/logs"


def test_absolute_directories_are_kept(tmp_path):
    assert Settings(data_dir=tmp_path).data_dir == tmp_path


def test_url_is_normalised_and_must_be_http():
    assert Settings(ollama_base_url="http://localhost:11434/").ollama_base_url == "http://localhost:11434"

    with pytest.raises(ValidationError):
        Settings(ollama_base_url="localhost:11434")


def test_chunk_overlap_must_be_smaller_than_chunk_size():
    with pytest.raises(ValidationError, match="CHUNK_OVERLAP"):
        Settings(chunk_size=100, chunk_overlap=100)


def test_with_updates_returns_a_validated_copy():
    settings = Settings()

    assert with_updates(settings, temperature=0.7).temperature == 0.7
    with pytest.raises(ConfigError):
        with_updates(settings, top_k=100)


def test_load_settings_reads_environment_and_strips_inline_comments(tmp_path, monkeypatch):
    monkeypatch.setenv("DATA_DIR", str(tmp_path))
    monkeypatch.setenv("CHAT_MODEL", "custom:1b")
    monkeypatch.setenv("CHUNKING_STRATEGY", "character   # inline note")

    settings = load_settings()

    assert settings.chat_model == "custom:1b"
    assert settings.chunking_strategy == "character"
    assert settings.data_dir == tmp_path


def test_load_settings_wraps_validation_errors_in_config_error(tmp_path, monkeypatch):
    monkeypatch.setenv("DATA_DIR", str(tmp_path))
    monkeypatch.setenv("TEMPERATURE", "5")

    with pytest.raises(ConfigError, match="TEMPERATURE"):
        load_settings()


def test_save_user_overrides_persists_only_editable_fields(tmp_path):
    settings = Settings(data_dir=tmp_path)
    changed = settings.model_copy(update={"chat_model": "custom:1b", "top_k": 7})

    save_user_overrides(changed)

    data = json.loads((tmp_path / "user_settings.json").read_text(encoding="utf-8"))
    assert set(data) == set(USER_EDITABLE)
    assert data["chat_model"] == "custom:1b" and data["top_k"] == 7
    assert "data_dir" not in data


def test_load_user_overrides_filters_unknown_keys(tmp_path):
    settings = Settings(data_dir=tmp_path)
    (tmp_path / "user_settings.json").write_text(
        json.dumps({"chat_model": "custom:1b", "not_a_setting": 1}), encoding="utf-8"
    )

    assert load_user_overrides(settings) == {"chat_model": "custom:1b"}


def test_load_user_overrides_ignores_broken_json(tmp_path):
    settings = Settings(data_dir=tmp_path)
    (tmp_path / "user_settings.json").write_text("{not json", encoding="utf-8")

    assert load_user_overrides(settings) == {}


@pytest.mark.parametrize("payload", ["[]", "null", '"text"', "42"])
def test_load_user_overrides_ignores_non_object_json(tmp_path, payload):
    settings = Settings(data_dir=tmp_path)
    (tmp_path / "user_settings.json").write_text(payload, encoding="utf-8")

    assert load_user_overrides(settings) == {}


def test_save_user_overrides_is_atomic_and_leaves_no_temp_file(tmp_path):
    settings = Settings(data_dir=tmp_path)

    save_user_overrides(settings)

    assert (tmp_path / "user_settings.json").is_file()
    assert not (tmp_path / "user_settings.json.tmp").exists()


def test_ensure_directories_creates_every_data_path(tmp_path):
    settings = Settings(data_dir=tmp_path / "data", log_dir=tmp_path / "logs")

    ensure_directories(settings)

    assert settings.data_dir.is_dir()
    assert settings.upload_dir.is_dir()
    assert settings.chroma_dir.is_dir()
    assert settings.log_dir.is_dir()
