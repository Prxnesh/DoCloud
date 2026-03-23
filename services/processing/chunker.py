"""Chunking service for the CloudInsight processing layer."""

from __future__ import annotations

import hashlib
import re
from typing import Any

from models.document import ChunkingResult, TextChunk

_TOKEN_PATTERN = re.compile(r"\S+")


class TextChunkingError(Exception):
    """Raised when text chunking cannot be completed safely."""

    def __init__(self, message: str, document_id: str | None = None) -> None:
        super().__init__(message)
        self.document_id = document_id


class TextChunker:
    """Split cleaned text into overlap-aware chunks for embedding and retrieval."""

    def chunk(
        self,
        text: str,
        document_id: str,
        metadata: dict[str, Any] | None = None,
        chunk_size: int = 250,
        chunk_overlap: int = 40,
    ) -> ChunkingResult:
        """
        Split cleaned text into retrieval-ready chunks.

        Args:
            text: Cleaned document text.
            document_id: Stable identifier propagated from upstream.
            metadata: Optional metadata to enrich and carry forward.
            chunk_size: Maximum number of whitespace tokens per chunk.
            chunk_overlap: Number of trailing tokens repeated in the next chunk.

        Returns:
            A standardized chunking result.

        Raises:
            TextChunkingError: If configuration or text input is invalid.
        """

        if not document_id or not document_id.strip():
            raise TextChunkingError("A non-empty document_id is required.")

        if not isinstance(text, str):
            raise TextChunkingError(
                "Input text must be a string.",
                document_id=document_id,
            )

        if not text.strip():
            raise TextChunkingError(
                "Input text is empty after trimming whitespace.",
                document_id=document_id,
            )

        if chunk_size <= 0:
            raise TextChunkingError(
                "chunk_size must be greater than zero.",
                document_id=document_id,
            )

        if chunk_overlap < 0 or chunk_overlap >= chunk_size:
            raise TextChunkingError(
                "chunk_overlap must be non-negative and smaller than chunk_size.",
                document_id=document_id,
            )

        try:
            spans = list(self._tokenize_with_offsets(text))
            chunks = self._build_chunks(
                text=text,
                document_id=document_id,
                spans=spans,
                chunk_size=chunk_size,
                chunk_overlap=chunk_overlap,
                metadata=metadata or {},
            )
        except (TypeError, ValueError) as exc:
            raise TextChunkingError(
                f"Failed to chunk text: {exc}",
                document_id=document_id,
            ) from exc

        if not chunks:
            raise TextChunkingError(
                "Chunking produced no output.",
                document_id=document_id,
            )

        merged_metadata = dict(metadata or {})
        merged_metadata.update(
            {
                "chunk_count": len(chunks),
                "chunk_size": chunk_size,
                "chunk_overlap": chunk_overlap,
            }
        )

        return ChunkingResult(
            document_id=document_id,
            chunks=chunks,
            metadata=merged_metadata,
        )

    def _build_chunks(
        self,
        text: str,
        document_id: str,
        spans: list[tuple[int, int]],
        chunk_size: int,
        chunk_overlap: int,
        metadata: dict[str, Any],
    ) -> list[TextChunk]:
        """Create chunks using token spans and paragraph-aware boundary tuning."""

        chunks: list[TextChunk] = []
        start_index = 0
        chunk_index = 0

        while start_index < len(spans):
            candidate_end = min(start_index + chunk_size, len(spans))
            end_index = self._adjust_end_to_paragraph_boundary(text, spans, start_index, candidate_end)

            if end_index <= start_index:
                end_index = candidate_end

            start_char = spans[start_index][0]
            end_char = spans[end_index - 1][1]
            chunk_text = text[start_char:end_char].strip()

            if chunk_text:
                token_count = end_index - start_index
                chunks.append(
                    TextChunk(
                        chunk_id=self._build_chunk_id(document_id, chunk_index, start_char, end_char),
                        document_id=document_id,
                        chunk_index=chunk_index,
                        text=chunk_text,
                        start_char=start_char,
                        end_char=end_char,
                        token_count=token_count,
                        metadata={
                            **metadata,
                            "start_token_index": start_index,
                            "end_token_index": end_index - 1,
                        },
                    )
                )
                chunk_index += 1

            if end_index >= len(spans):
                break

            start_index = max(end_index - chunk_overlap, start_index + 1)

        return chunks

    def _adjust_end_to_paragraph_boundary(
        self,
        text: str,
        spans: list[tuple[int, int]],
        start_index: int,
        candidate_end: int,
    ) -> int:
        """Prefer paragraph boundaries when they are near the target window size."""

        if candidate_end >= len(spans):
            return candidate_end

        lower_bound = max(start_index + 1, candidate_end - 30)

        for index in range(candidate_end, lower_bound - 1, -1):
            boundary_char = spans[index - 1][1]
            next_start = spans[index][0] if index < len(spans) else len(text)
            boundary_text = text[boundary_char:next_start]

            if "\n\n" in boundary_text:
                return index

        return candidate_end

    def _tokenize_with_offsets(self, text: str) -> list[tuple[int, int]]:
        """Capture token start and end offsets for traceable retrieval chunks."""

        return [(match.start(), match.end()) for match in _TOKEN_PATTERN.finditer(text)]

    def _build_chunk_id(
        self,
        document_id: str,
        chunk_index: int,
        start_char: int,
        end_char: int,
    ) -> str:
        """Create a deterministic identifier for a chunk."""

        digest_input = f"{document_id}:{chunk_index}:{start_char}:{end_char}"
        return hashlib.sha256(digest_input.encode("utf-8")).hexdigest()
