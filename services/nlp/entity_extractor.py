"""Named entity extraction service powered by spaCy."""

from __future__ import annotations

import re

from models.analysis import EntityRecord


class EntityExtractionError(Exception):
    """Raised when NER processing fails."""


class EntityExtractor:
    """Extract named entities from cleaned document text."""

    def __init__(self, model_name: str) -> None:
        self.model_name = model_name
        self._nlp = None
        self._spacy_disabled = False

    def extract(self, text: str, max_chars: int = 100_000) -> list[EntityRecord]:
        """Extract named entities from text."""

        if not text or not text.strip():
            return []

        sample = text[:max_chars]

        try:
            nlp = self._get_nlp()
            if nlp is not None:
                doc = nlp(sample)
                return [
                    EntityRecord(
                        text=entity.text,
                        label=entity.label_,
                        start_char=entity.start_char,
                        end_char=entity.end_char,
                        score=None,
                    )
                    for entity in doc.ents
                ]
        except Exception:
            # Fall back to a lightweight regex-based extractor when spaCy is unavailable
            # or incompatible with the active Python runtime.
            self._spacy_disabled = True

        return self._extract_with_rules(sample)

    def _extract_with_rules(self, text: str) -> list[EntityRecord]:
        """Extract basic entities using deterministic regex heuristics."""

        patterns: list[tuple[str, str]] = [
            ("MONEY", r"\$\s?\d[\d,]*(?:\.\d+)?(?:\s?(?:million|billion|m|bn))?"),
            ("PERCENT", r"\b\d+(?:\.\d+)?%\b"),
            ("DATE", r"\b(?:\d{1,2}[/-]\d{1,2}[/-]\d{2,4}|\d{4}-\d{2}-\d{2})\b"),
            (
                "DATE",
                r"\b(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Sept|Oct|Nov|Dec)"
                r"(?:[a-z]+)?\s+\d{1,2}(?:,\s*\d{4})?\b",
            ),
            ("ORG", r"\b[A-Z][A-Za-z&.-]*(?:\s+[A-Z][A-Za-z&.-]*)*\s+(?:Inc|LLC|Ltd|Corp|Corporation|Company|Co)\b"),
            ("GPE", r"\b(?:USA|US|United States|UK|India|China|Germany|France|Canada|Australia)\b"),
            ("PERSON", r"\b[A-Z][a-z]+\s+[A-Z][a-z]+\b"),
        ]

        entities: list[EntityRecord] = []
        seen: set[tuple[int, int, str]] = set()

        for label, pattern in patterns:
            for match in re.finditer(pattern, text):
                span = (match.start(), match.end(), label)
                if span in seen:
                    continue
                seen.add(span)
                entities.append(
                    EntityRecord(
                        text=match.group(0),
                        label=label,
                        start_char=match.start(),
                        end_char=match.end(),
                        score=None,
                    )
                )

        entities.sort(key=lambda item: item.start_char)
        return entities

    def _get_nlp(self):
        """Lazily load the spaCy model."""

        if self._spacy_disabled:
            return None

        if self._nlp is None:
            try:
                import spacy
            except ImportError as exc:
                self._spacy_disabled = True
                return None

            try:
                self._nlp = spacy.load(self.model_name)
            except Exception:
                self._spacy_disabled = True
                return None

        return self._nlp
