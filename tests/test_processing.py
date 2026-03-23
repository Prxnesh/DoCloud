"""Unit tests for CloudInsight processing services."""

from services.processing.chunker import TextChunker
from services.processing.text_cleaner import TextCleaner


def test_text_cleaner_removes_page_labels_and_repairs_hyphenation() -> None:
    cleaner = TextCleaner()
    result = cleaner.clean(
        "Page 1\n\nhyphen-\nated text\n\n2\n\nAnother paragraph",
        document_id="doc-1",
    )

    assert "Page 1" not in result.cleaned_text
    assert "hyphenated" in result.cleaned_text
    assert "Another paragraph" in result.cleaned_text


def test_text_chunker_creates_overlap_aware_chunks() -> None:
    chunker = TextChunker()
    result = chunker.chunk(
        text="one two three four five six seven eight nine ten",
        document_id="doc-1",
        chunk_size=4,
        chunk_overlap=1,
    )

    assert len(result.chunks) >= 2
    assert result.chunks[0].document_id == "doc-1"
    assert result.metadata["chunk_overlap"] == 1
