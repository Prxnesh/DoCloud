"""Document ingestion coordinator for CloudInsight."""

from __future__ import annotations

from pathlib import Path
from typing import Protocol

from models.document import DocumentContent

from .docx_extractor import DOCXExtractor
from .pdf_extractor import PDFExtractor
from .txt_extractor import TXTExtractor


class Extractor(Protocol):
    """Protocol implemented by all document extractors."""

    def extract(self, source_path: str | Path) -> DocumentContent:
        """Extract standardized content from a document path."""


class UnsupportedDocumentTypeError(Exception):
    """Raised when no extractor exists for the input file type."""


class DocumentIngestionService:
    """Route incoming document files to the correct extractor."""

    def __init__(
        self,
        pdf_extractor: Extractor | None = None,
        docx_extractor: Extractor | None = None,
        txt_extractor: Extractor | None = None,
    ) -> None:
        text_extractor = txt_extractor or TXTExtractor()
        self._extractors: dict[str, Extractor] = {
            ".pdf": pdf_extractor or PDFExtractor(),
            ".docx": docx_extractor or DOCXExtractor(),
            ".txt": text_extractor,
            ".md": text_extractor,
            ".csv": text_extractor,
            ".json": text_extractor,
            ".html": text_extractor,
            ".rtf": text_extractor,
            ".log": text_extractor,
        }

    def extract(self, source_path: str | Path) -> DocumentContent:
        """Extract text from the given document using the appropriate extractor."""

        path = Path(source_path).expanduser().resolve()
        extractor = self._extractors.get(path.suffix.lower())
        if extractor is None:
            raise UnsupportedDocumentTypeError(
                f"Unsupported document type: {path.suffix or 'unknown'}"
            )

        return extractor.extract(path)
