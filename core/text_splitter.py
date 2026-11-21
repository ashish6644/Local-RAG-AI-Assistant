"""Configurable chunking that preserves page / location metadata."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

from core.document_processor import Segment

STRATEGIES = ("recursive", "character")


@dataclass(frozen=True)
class Chunk:
    text: str
    chunk_index: int
    page: int | None
    location: str


def make_splitter(strategy: str, chunk_size: int, chunk_overlap: int):
    if chunk_overlap >= chunk_size:
        raise ValueError("chunk_overlap must be smaller than chunk_size")
    if strategy == "recursive":
        from langchain_text_splitters import RecursiveCharacterTextSplitter

        return RecursiveCharacterTextSplitter(
            chunk_size=chunk_size, chunk_overlap=chunk_overlap,
            separators=["\n\n", "\n", ". ", " ", ""],
        )
    if strategy == "character":
        from langchain_text_splitters import CharacterTextSplitter

        return CharacterTextSplitter(
            separator="\n", chunk_size=chunk_size, chunk_overlap=chunk_overlap, strip_whitespace=True
        )
    raise ValueError(f"Unknown chunking strategy '{strategy}'. Choose from: {', '.join(STRATEGIES)}")


def split_segments(
    segments: Sequence[Segment], chunk_size: int, chunk_overlap: int, strategy: str = "recursive"
) -> list[Chunk]:
    splitter = make_splitter(strategy, chunk_size, chunk_overlap)
    chunks: list[Chunk] = []
    for segment in segments:
        for piece in splitter.split_text(segment.text):
            piece = piece.strip()
            if len(piece) < 3:
                continue
            chunks.append(Chunk(piece, len(chunks), segment.page, segment.location))
    return chunks
