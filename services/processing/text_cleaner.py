"""Text cleaning service for the CloudInsight processing layer."""

from __future__ import annotations

import re
from typing import Any

from models.document import CleanedDocument

_WINDOWS_NEWLINE_PATTERN = re.compile(r"\r\n?")
_MULTI_SPACE_PATTERN = re.compile(r"[^\S\r\n]+")
_MULTI_NEWLINE_PATTERN = re.compile(r"\n{3,}")
_PAGE_NUMBER_PATTERN = re.compile(r"^\s*\d+\s*$")
_PAGE_LABEL_PATTERN = re.compile(r"^\s*page\s+\d+(\s+of\s+\d+)?\s*$", re.IGNORECASE)
_SOFT_HYPHEN_BREAK_PATTERN = re.compile(r"(\w)-\n(\w)")


class TextCleaningError(Exception):
    """Raised when text cleaning cannot be completed safely."""

    def __init__(self, message: str, document_id: str | None = None) -> None:
        super().__init__(message)
        self.document_id = document_id


class TextCleaner:
    """Normalize extracted document text for downstream NLP and chunking."""

    def clean(
        self,
        text: str,
        document_id: str,
        metadata: dict[str, Any] | None = None,
    ) -> CleanedDocument:
        """
        Clean extracted text while preserving meaningful paragraph boundaries.

        Args:
            text: Raw extracted text from the ingestion layer.
            document_id: Stable identifier propagated from upstream.
            metadata: Optional metadata to enrich and carry forward.

        Returns:
            A standardized cleaned text result.

        Raises:
            TextCleaningError: If the input text is invalid or cleaning fails.
        """

        if not document_id or not document_id.strip():
            raise TextCleaningError("A non-empty document_id is required.")

        if not isinstance(text, str):
            raise TextCleaningError(
                "Input text must be a string.",
                document_id=document_id,
            )

        if not text.strip():
            raise TextCleaningError(
                "Input text is empty after trimming whitespace.",
                document_id=document_id,
            )

        try:
            normalized_text = self._normalize_newlines(text)
            repaired_hyphenation = self._repair_hyphenated_breaks(normalized_text)
            cleaned_text = self._clean_paragraphs(repaired_hyphenation)
        except (TypeError, ValueError) as exc:
            raise TextCleaningError(
                f"Failed to clean text: {exc}",
                document_id=document_id,
            ) from exc

        if not cleaned_text:
            raise TextCleaningError(
                "Cleaning removed all content from the document.",
                document_id=document_id,
            )

        merged_metadata = dict(metadata or {})
        merged_metadata.update(
            {
                "original_char_count": len(text),
                "cleaned_char_count": len(cleaned_text),
                "paragraph_count": len(cleaned_text.split("\n\n")),
            }
        )

        return CleanedDocument(
            document_id=document_id,
            cleaned_text=cleaned_text,
            metadata=merged_metadata,
        )

    def _normalize_newlines(self, text: str) -> str:
        """Convert all newline styles to Unix newlines."""

        return _WINDOWS_NEWLINE_PATTERN.sub("\n", text)

    def _repair_hyphenated_breaks(self, text: str) -> str:
        """Join words that were split across lines during extraction."""

        return _SOFT_HYPHEN_BREAK_PATTERN.sub(r"\1\2", text)

    def _clean_paragraphs(self, text: str) -> str:
        """Normalize line spacing and remove obvious extraction artifacts."""

        paragraphs = []

        for raw_block in text.split("\n\n"):
            normalized_block = self._normalize_block(raw_block)
            if normalized_block:
                paragraphs.append(normalized_block)

        return _MULTI_NEWLINE_PATTERN.sub("\n\n", "\n\n".join(paragraphs)).strip()

    def _normalize_block(self, block: str) -> str:
        """Clean a paragraph-like text block while preserving sentence flow."""

        lines = []
        for raw_line in block.splitlines():
            line = _MULTI_SPACE_PATTERN.sub(" ", raw_line).strip()

            # Drop isolated page-number lines that often appear in extracted text.
            if (
                not line
                or _PAGE_NUMBER_PATTERN.fullmatch(line)
                or _PAGE_LABEL_PATTERN.fullmatch(line)
            ):
                continue

            lines.append(line)

        if not lines:
            return ""

        joined = " ".join(lines)
        joined = _MULTI_SPACE_PATTERN.sub(" ", joined).strip()
        return joined
