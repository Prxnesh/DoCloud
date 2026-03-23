"""Document-centric domain models used throughout the pipeline."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal


DocumentType = Literal["pdf", "docx", "txt", "md", "csv", "json", "html", "rtf", "log"]


@dataclass(slots=True, frozen=True)
class DocumentContent:
    """Standardized representation of extracted source content."""

    document_id: str
    source_path: str
    file_name: str
    document_type: DocumentType
    text: str
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True, frozen=True)
class CleanedDocument:
    """Standardized representation of cleaned document text."""

    document_id: str
    cleaned_text: str
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True, frozen=True)
class TextChunk:
    """A retrieval-ready chunk of document text."""

    chunk_id: str
    document_id: str
    chunk_index: int
    text: str
    start_char: int
    end_char: int
    token_count: int
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True, frozen=True)
class ChunkingResult:
    """Collection of text chunks generated from a document."""

    document_id: str
    chunks: list[TextChunk]
    metadata: dict[str, Any] = field(default_factory=dict)
