"""DOCX extraction service for the CloudInsight ingestion layer."""

from __future__ import annotations

from pathlib import Path
import hashlib
import re
from typing import Literal
from xml.etree.ElementTree import ParseError, iterparse
from zipfile import BadZipFile, ZipFile

from models.document import DocumentContent


_WORD_NAMESPACE = "{http://schemas.openxmlformats.org/wordprocessingml/2006/main}"
_TEXT_TAG = f"{_WORD_NAMESPACE}t"
_PARAGRAPH_TAG = f"{_WORD_NAMESPACE}p"
_TAB_TAG = f"{_WORD_NAMESPACE}tab"
_BREAK_TAG = f"{_WORD_NAMESPACE}br"
_MULTI_SPACE_PATTERN = re.compile(r"[^\S\r\n]+")
_MULTI_NEWLINE_PATTERN = re.compile(r"\n{3,}")


class DocumentExtractionError(Exception):
    """Raised when a document cannot be extracted safely."""

    def __init__(self, message: str, source_path: str) -> None:
        super().__init__(message)
        self.source_path = source_path


class DOCXExtractor:
    """Extract text from DOCX files without loading a full document object model."""

    document_type: Literal["docx"] = "docx"

    def extract(self, source_path: str | Path) -> DocumentContent:
        """
        Extract normalized text from a DOCX file.

        Args:
            source_path: Path to the DOCX file.

        Returns:
            A standardized extraction result.

        Raises:
            DocumentExtractionError: If the file cannot be read or parsed.
        """

        path = Path(source_path).expanduser().resolve()
        self._validate_path(path)

        try:
            paragraphs = list(self._iter_paragraphs(path))
        except (BadZipFile, KeyError, OSError, ParseError, ValueError) as exc:
            raise DocumentExtractionError(
                f"Unable to extract DOCX content: {exc}",
                str(path),
            ) from exc

        text_parts = [paragraph for paragraph in paragraphs if paragraph]
        full_text = "\n\n".join(text_parts).strip()

        if not full_text:
            raise DocumentExtractionError(
                "No extractable text was found in the DOCX document.",
                str(path),
            )

        return DocumentContent(
            document_id=self._build_document_id(path),
            source_path=str(path),
            file_name=path.name,
            document_type=self.document_type,
            text=full_text,
            metadata={
                "paragraph_count": len(text_parts),
                "file_size_bytes": path.stat().st_size,
            },
        )

    def _iter_paragraphs(self, path: Path) -> list[str]:
        """Stream paragraphs from the DOCX XML payload."""

        paragraphs: list[str] = []

        with ZipFile(path) as archive:
            try:
                document_xml = archive.open("word/document.xml")
            except KeyError as exc:
                raise DocumentExtractionError(
                    "DOCX document.xml payload is missing.",
                    str(path),
                ) from exc

            with document_xml:
                paragraph_fragments: list[str] = []

                # Iterative parsing keeps memory usage predictable for large files.
                for event, element in iterparse(document_xml, events=("start", "end")):
                    if event == "start" and element.tag == _TAB_TAG:
                        paragraph_fragments.append("\t")
                    elif event == "start" and element.tag == _BREAK_TAG:
                        paragraph_fragments.append("\n")
                    elif event == "end" and element.tag == _TEXT_TAG:
                        paragraph_fragments.append(element.text or "")
                        element.clear()
                    elif event == "end" and element.tag == _PARAGRAPH_TAG:
                        paragraph_text = self._normalize_text("".join(paragraph_fragments))
                        if paragraph_text:
                            paragraphs.append(paragraph_text)
                        paragraph_fragments.clear()
                        element.clear()
                    elif event == "end":
                        element.clear()

        return paragraphs

    def _validate_path(self, path: Path) -> None:
        """Validate the file before extraction starts."""

        if not path.exists():
            raise DocumentExtractionError("DOCX file does not exist.", str(path))

        if not path.is_file():
            raise DocumentExtractionError("DOCX path must point to a file.", str(path))

        if path.suffix.lower() != ".docx":
            raise DocumentExtractionError("Expected a .docx file.", str(path))

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
