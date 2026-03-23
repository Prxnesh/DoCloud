"""Text-like document extraction service for the CloudInsight ingestion layer."""

from __future__ import annotations

from pathlib import Path
import hashlib
import re
from typing import Literal

from models.document import DocumentContent


_MULTI_SPACE_PATTERN = re.compile(r"[^\S\r\n]+")
_MULTI_NEWLINE_PATTERN = re.compile(r"\n{3,}")


class DocumentExtractionError(Exception):
    """Raised when a document cannot be extracted safely."""

    def __init__(self, message: str, source_path: str) -> None:
        super().__init__(message)
        self.source_path = source_path


class TXTExtractor:
    """Extract text from UTF-8 compatible text-like files."""

    document_type: Literal["txt"] = "txt"
    supported_suffixes = {".txt", ".md", ".csv", ".json", ".html", ".rtf", ".log"}

    def extract(self, source_path: str | Path) -> DocumentContent:
        """Read and normalize a UTF-8 compatible text file."""

        path = Path(source_path).expanduser().resolve()
        suffix = path.suffix.lower()
        self._validate_path(path, suffix)

        try:
            raw_text = path.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            try:
                raw_text = path.read_text(encoding="utf-8-sig")
            except (OSError, UnicodeDecodeError) as exc:
                raise DocumentExtractionError(
                    f"Unable to decode TXT document: {exc}",
                    str(path),
                ) from exc
        except OSError as exc:
            raise DocumentExtractionError(
                f"Unable to open TXT document: {exc}",
                str(path),
            ) from exc

        text = self._normalize_text(raw_text)
        if not text:
            raise DocumentExtractionError(
                "No extractable text was found in the TXT document.",
                str(path),
            )

        return DocumentContent(
            document_id=self._build_document_id(path),
            source_path=str(path),
            file_name=path.name,
            document_type=suffix.lstrip("."),
            text=text,
            metadata={
                "line_count": len(raw_text.splitlines()),
                "file_size_bytes": path.stat().st_size,
                "source_extension": suffix,
            },
        )

    def _validate_path(self, path: Path, suffix: str) -> None:
        """Validate the file before extraction starts."""

        if not path.exists():
            raise DocumentExtractionError("TXT file does not exist.", str(path))

        if not path.is_file():
            raise DocumentExtractionError("TXT path must point to a file.", str(path))

        if suffix not in self.supported_suffixes:
            supported = ", ".join(sorted(self.supported_suffixes))
            raise DocumentExtractionError(
                f"Expected one of: {supported}.",
                str(path),
            )

    def _build_document_id(self, path: Path) -> str:
        """Create a stable identifier from file metadata."""

        file_stats = path.stat()
        digest_input = f"{path.name}:{file_stats.st_size}:{file_stats.st_mtime_ns}"
        return hashlib.sha256(digest_input.encode("utf-8")).hexdigest()

    def _normalize_text(self, text: str) -> str:
        """Normalize whitespace while preserving paragraph boundaries."""

        cleaned_lines = []
        for raw_line in text.splitlines():
            line = _MULTI_SPACE_PATTERN.sub(" ", raw_line).strip()
            if line:
                cleaned_lines.append(line)
            elif cleaned_lines and cleaned_lines[-1] != "":
                cleaned_lines.append("")

        normalized = "\n".join(cleaned_lines).strip()
        return _MULTI_NEWLINE_PATTERN.sub("\n\n", normalized)
