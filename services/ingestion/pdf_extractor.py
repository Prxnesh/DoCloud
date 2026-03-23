"""PDF extraction service for the CloudInsight ingestion layer."""

from __future__ import annotations

from pathlib import Path
import hashlib
import re
from typing import Literal

from pypdf import PdfReader
from pypdf.errors import PdfReadError

from models.document import DocumentContent


_MULTI_SPACE_PATTERN = re.compile(r"[^\S\r\n]+")
_MULTI_NEWLINE_PATTERN = re.compile(r"\n{3,}")


class DocumentExtractionError(Exception):
    """Raised when a document cannot be extracted safely."""

    def __init__(self, message: str, source_path: str) -> None:
        super().__init__(message)
        self.source_path = source_path


class PDFExtractor:
    """Extract text from PDF documents in a memory-conscious way."""

    document_type: Literal["pdf"] = "pdf"

    def extract(self, source_path: str | Path) -> DocumentContent:
        """
        Extract normalized text from a PDF file.

        Args:
            source_path: Path to the PDF file.

        Returns:
            A standardized extraction result.

        Raises:
            DocumentExtractionError: If the file cannot be read or parsed.
        """

        path = Path(source_path).expanduser().resolve()
        self._validate_path(path)

        try:
            reader = PdfReader(str(path))
        except (PdfReadError, OSError, ValueError) as exc:
            raise DocumentExtractionError(
                f"Unable to open PDF document: {exc}",
                str(path),
            ) from exc

        if reader.is_encrypted:
            raise DocumentExtractionError(
                "Encrypted PDF documents are not supported.",
                str(path),
            )

        text_parts: list[str] = []
        extracted_pages = 0

        try:
            for page_number, page in enumerate(reader.pages, start=1):
                page_text = page.extract_text() or ""
                normalized_page = self._normalize_text(page_text)

                if normalized_page:
                    text_parts.append(normalized_page)

                extracted_pages = page_number
        except (PdfReadError, KeyError, ValueError) as exc:
            raise DocumentExtractionError(
                f"Failed while reading PDF pages: {exc}",
                str(path),
            ) from exc

        full_text = "\n\n".join(text_parts).strip()

        if not full_text:
            raise DocumentExtractionError(
                "No extractable text was found in the PDF document.",
                str(path),
            )

        return DocumentContent(
            document_id=self._build_document_id(path),
            source_path=str(path),
            file_name=path.name,
            document_type=self.document_type,
            text=full_text,
            metadata={
                "page_count": len(reader.pages),
                "pages_with_text": len(text_parts),
                "pages_processed": extracted_pages,
                "file_size_bytes": path.stat().st_size,
            },
        )

    def _validate_path(self, path: Path) -> None:
        """Validate the file before extraction starts."""

        if not path.exists():
            raise DocumentExtractionError("PDF file does not exist.", str(path))

        if not path.is_file():
            raise DocumentExtractionError("PDF path must point to a file.", str(path))

        if path.suffix.lower() != ".pdf":
            raise DocumentExtractionError("Expected a .pdf file.", str(path))

    def _build_document_id(self, path: Path) -> str:
        """Create a stable identifier from file metadata."""

        file_stats = path.stat()
        digest_input = f"{path.name}:{file_stats.st_size}:{file_stats.st_mtime_ns}"
        return hashlib.sha256(digest_input.encode("utf-8")).hexdigest()

    def _normalize_text(self, text: str) -> str:
        """Normalize extractor output while keeping paragraph boundaries."""

        cleaned_lines = []
        for raw_line in text.splitlines():
            line = _MULTI_SPACE_PATTERN.sub(" ", raw_line).strip()
            if line:
                cleaned_lines.append(line)
            elif cleaned_lines and cleaned_lines[-1] != "":
                cleaned_lines.append("")

        normalized = "\n".join(cleaned_lines).strip()
        return _MULTI_NEWLINE_PATTERN.sub("\n\n", normalized)
