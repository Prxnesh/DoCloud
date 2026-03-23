"""Summarization service backed by a local Hugging Face model."""

from __future__ import annotations

from collections import Counter
from math import ceil
import re
from typing import Any


class SummarizationError(Exception):
    """Raised when summarization fails."""


class Summarizer:
    """Summarize document text using a local transformer model."""

    def __init__(
        self,
        model_name: str,
        min_length: int = 50,
        max_length: int = 180,
    ) -> None:
        self.model_name = model_name
        self.min_length = min_length
        self.max_length = max_length
        self._pipeline = None
        self._pipeline_task: str | None = None

    def summarize(self, text: str) -> str:
        """Generate a summary for a document."""

        if not text or not text.strip():
            raise SummarizationError("Input text is empty.")

        sections = self._split_text(text, max_words=850)
        partial_summaries = [
            self._summarize_section(section)
            for section in sections
            if section.strip()
        ]

        if not partial_summaries:
            raise SummarizationError("No summary could be generated.")

        if len(partial_summaries) == 1:
            return partial_summaries[0]

        return self._summarize_section(" ".join(partial_summaries))

    def _summarize_section(self, text: str) -> str:
        """Summarize a single section of text."""

        pipeline = self._get_pipeline()
        if pipeline is None:
            return self._manual_summary(text)

        try:
            result = pipeline(
                text,
                min_length=self.min_length,
                max_length=self.max_length,
                truncation=True,
            )
        except Exception as exc:  # noqa: BLE001 - transformer exceptions vary
            return self._manual_summary(text)

        summary = self._extract_generated_text(result)
        if self._pipeline_task == "text-generation" or not self._looks_reasonable(summary):
            return self._manual_summary(text)

        return summary

    def _split_text(self, text: str, max_words: int) -> list[str]:
        """Break long text into word windows suited for local summarization."""

        words = text.split()
        if len(words) <= max_words:
            return [text]

        sections = []
        total_sections = ceil(len(words) / max_words)
        for index in range(total_sections):
            start = index * max_words
            end = start + max_words
            sections.append(" ".join(words[start:end]))

        return sections

    def _get_pipeline(self):
        """Lazily load the summarization model."""

        if self._pipeline is None:
            try:
                from transformers import pipeline
            except ImportError as exc:
                return None

            pipeline_factory = pipeline
            last_error: Exception | None = None

            for task_name in ("summarization", "text2text-generation"):
                try:
                    self._pipeline = pipeline_factory(
                        task_name,
                        model=self.model_name,
                    )
                    self._pipeline_task = task_name
                    break
                except Exception as exc:  # noqa: BLE001 - runtime support varies by version
                    last_error = exc

            if self._pipeline is None:
                return None

        return self._pipeline

    def _manual_summary(self, text: str) -> str:
        """Generate a lightweight extractive summary without model dependencies."""

        clean = " ".join(text.split())
        if not clean:
            raise SummarizationError("Input text is empty.")

        sentences = [segment.strip() for segment in re.split(r"(?<=[.!?])\s+", clean) if segment.strip()]
        if not sentences:
            return self._truncate_words(clean, self.max_length)

        if len(sentences) == 1:
            return self._truncate_words(sentences[0], self.max_length)

        stopwords = {
            "a", "an", "and", "are", "as", "at", "be", "by", "for", "from", "has", "he",
            "in", "is", "it", "its", "of", "on", "that", "the", "to", "was", "were", "will",
            "with", "this", "these", "those", "or", "but", "not", "we", "they", "you",
        }

        tokens = [word.lower() for word in re.findall(r"[A-Za-z0-9]+", clean)]
        frequencies = Counter(word for word in tokens if word not in stopwords and len(word) > 2)

        sentence_scores: list[tuple[int, float]] = []
        for index, sentence in enumerate(sentences):
            words = [word.lower() for word in re.findall(r"[A-Za-z0-9]+", sentence)]
            if not words:
                sentence_scores.append((index, 0.0))
                continue
            score = sum(frequencies.get(word, 0) for word in words) / len(words)
            sentence_scores.append((index, score))

        top_n = min(3, max(1, len(sentences) // 2))
        selected_indices = sorted(
            index for index, _ in sorted(sentence_scores, key=lambda item: item[1], reverse=True)[:top_n]
        )
        selected = [sentences[index] for index in selected_indices]
        candidate = " ".join(selected)

        return self._truncate_words(candidate, self.max_length)

    def _truncate_words(self, text: str, max_words: int) -> str:
        """Trim text to a maximum number of words while preserving readability."""

        words = text.split()
        if len(words) <= max_words:
            return text.strip()

        return " ".join(words[:max_words]).strip() + "..."

    def _looks_reasonable(self, text: str) -> bool:
        """Reject clearly malformed generation output and trigger manual fallback."""

        words = text.split()
        if len(words) < 6:
            return False

        unique_ratio = len(set(word.lower() for word in words)) / max(len(words), 1)
        return unique_ratio >= 0.45

    def _extract_generated_text(self, result: Any) -> str:
        """Handle both summarization and text2text-generation response formats."""

        if not isinstance(result, list) or not result or not isinstance(result[0], dict):
            raise SummarizationError("Unexpected summarization output format.")

        record = result[0]
        output_text = record.get("summary_text") or record.get("generated_text")
        if not output_text or not isinstance(output_text, str):
            raise SummarizationError("Model output did not include generated text.")

        return output_text.strip()
