"""Document classification service using a DistilBERT-based zero-shot model."""

from __future__ import annotations

from models.analysis import ClassificationResult


class ClassificationError(Exception):
    """Raised when document classification fails."""


class DocumentClassifier:
    """Classify documents into configurable business categories."""

    def __init__(self, model_name: str) -> None:
        self.model_name = model_name
        self._pipeline = None

    def classify(
        self,
        text: str,
        candidate_labels: list[str],
    ) -> ClassificationResult:
        """Predict the best-fitting label for the document."""

        if not text or not text.strip():
            raise ClassificationError("Input text is empty.")

        if not candidate_labels:
            raise ClassificationError("At least one candidate label is required.")

        pipeline = self._get_pipeline()

        try:
            result = pipeline(
                text[:4000],
                candidate_labels,
                multi_label=False,
            )
        except Exception as exc:  # noqa: BLE001 - transformer exceptions vary
            raise ClassificationError(f"Failed to classify document: {exc}") from exc

        return ClassificationResult(
            label=result["labels"][0],
            score=float(result["scores"][0]),
            metadata={"candidate_labels": candidate_labels},
        )

    def _get_pipeline(self):
        """Lazily load the classification model."""

        if self._pipeline is None:
            try:
                from transformers import pipeline
            except ImportError as exc:
                raise ClassificationError(
                    "transformers is required for document classification."
                ) from exc

            self._pipeline = pipeline(
                "zero-shot-classification",
                model=self.model_name,
            )

        return self._pipeline
