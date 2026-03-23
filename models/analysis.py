"""Analysis and retrieval models used across NLP, search, and chat flows."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from .document import ChunkingResult, CleanedDocument, DocumentContent


@dataclass(slots=True, frozen=True)
class ClassificationResult:
    """Predicted document class with confidence."""

    label: str
    score: float
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True, frozen=True)
class EntityRecord:
    """Named entity extracted from text."""

    text: str
    label: str
    start_char: int
    end_char: int
    score: float | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True, frozen=True)
class KeywordRecord:
    """Keyword extracted from text with relevance score."""

    keyword: str
    score: float
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True, frozen=True)
class DocumentAnalysis:
    """Aggregated NLP analysis for a document."""

    document_id: str
    summary: str
    classification: ClassificationResult
    entities: list[EntityRecord]
    keywords: list[KeywordRecord]
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True, frozen=True)
class SearchResult:
    """Semantic search hit returned from the vector store."""

    chunk_id: str
    document_id: str
    score: float
    text: str
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True, frozen=True)
class ChatResponse:
    """Answer returned by the conversational retrieval layer."""

    answer: str
    results: list[SearchResult]
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True, frozen=True)
class PipelineResult:
    """End-to-end output of the document intelligence pipeline."""

    document: DocumentContent
    cleaned_document: CleanedDocument
    chunking: ChunkingResult
    analysis: DocumentAnalysis
    metadata: dict[str, Any] = field(default_factory=dict)
