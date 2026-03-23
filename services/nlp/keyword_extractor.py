"""Keyword extraction service built on KeyBERT."""

from __future__ import annotations

from models.analysis import KeywordRecord


class KeywordExtractionError(Exception):
    """Raised when keyword extraction fails."""


class KeywordExtractor:
    """Extract document keywords for faceting and search support."""

    def __init__(self, model_name: str) -> None:
        self.model_name = model_name
        self._extractor = None

    def extract(self, text: str, top_n: int = 10) -> list[KeywordRecord]:
        """Extract top keywords from the provided text."""

        if not text or not text.strip():
            return []

        extractor = self._get_extractor()

        try:
            results = extractor.extract_keywords(
                text,
                top_n=top_n,
                stop_words="english",
                use_mmr=True,
                diversity=0.4,
            )
        except Exception as exc:  # noqa: BLE001 - KeyBERT exceptions vary
            raise KeywordExtractionError(
                f"Failed to extract keywords: {exc}"
            ) from exc

        return [
            KeywordRecord(keyword=keyword, score=float(score))
            for keyword, score in results
        ]

    def _get_extractor(self):
        """Lazily load KeyBERT and its embedding backend."""

        if self._extractor is None:
            try:
                from keybert import KeyBERT
            except ImportError as exc:
                raise KeywordExtractionError(
                    "keybert is required for keyword extraction."
                ) from exc

            self._extractor = KeyBERT(model=self.model_name)

        return self._extractor
