"""Application-level exceptions. Messages are written to be shown to end users."""
from __future__ import annotations


class AppError(Exception):
    """Base class for errors whose message is safe and useful to display."""


class ConfigError(AppError):
    pass


class DatabaseError(AppError):
    pass


class OllamaError(AppError):
    pass


class OllamaConnectionError(OllamaError):
    pass


class OllamaModelNotFoundError(OllamaError):
    pass


class EmbeddingError(AppError):
    pass


class VectorStoreError(AppError):
    pass


class DocumentProcessingError(AppError):
    """A document could not be read. `str(error)` is a user-facing explanation."""


class UnsupportedFileError(DocumentProcessingError):
    pass


class EmptyDocumentError(DocumentProcessingError):
    pass


class ScannedPDFError(DocumentProcessingError):
    pass
