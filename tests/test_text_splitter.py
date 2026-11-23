import pytest

pytest.importorskip("langchain_text_splitters")

from core.document_processor import Segment
from core.text_splitter import make_splitter, split_segments


def test_split_segments_preserves_metadata_and_indexes():
    segments = [
        Segment("Paragraph one is here.\n\nParagraph two is here.", page=1, location="page 1"),
        Segment("Second page content.", page=2, location="page 2"),
    ]

    chunks = split_segments(segments, chunk_size=30, chunk_overlap=0)

    assert chunks
    assert [chunk.chunk_index for chunk in chunks] == list(range(len(chunks)))
    assert chunks[0].page == 1 and chunks[0].location == "page 1"
    assert chunks[-1].page == 2 and chunks[-1].location == "page 2"
    assert all(len(chunk.text) >= 3 for chunk in chunks)


def test_empty_segments_produce_no_chunks():
    assert split_segments([], chunk_size=30, chunk_overlap=0) == []
    assert split_segments([Segment("", page=1, location="page 1")], 30, 0) == []


def test_overlap_must_be_smaller_than_chunk_size():
    with pytest.raises(ValueError, match="chunk_overlap"):
        make_splitter("recursive", 10, 10)


def test_unknown_strategy_is_rejected():
    with pytest.raises(ValueError, match="Unknown chunking strategy"):
        make_splitter("nope", 10, 0)


def test_character_strategy_splits_on_newlines():
    chunks = split_segments(
        [Segment("alpha beta\ngamma delta", location="text file")],
        chunk_size=20,
        chunk_overlap=0,
        strategy="character",
    )

    assert [chunk.text for chunk in chunks] == ["alpha beta", "gamma delta"]
