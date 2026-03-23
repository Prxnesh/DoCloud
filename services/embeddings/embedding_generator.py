"""Embedding generation service for semantic search and RAG."""

from __future__ import annotations


class EmbeddingGenerationError(Exception):
    """Raised when embeddings cannot be generated."""


class EmbeddingGenerator:
    """Generate dense vector embeddings from document chunks and queries."""

    def __init__(self, model_name: str, batch_size: int = 32) -> None:
        self.model_name = model_name
        self.batch_size = batch_size
        self._model = None

    def generate(self, texts: list[str]) -> list[list[float]]:
        """Generate embeddings for a batch of texts."""

        if not texts:
            return []

        model = self._get_model()

        try:
            embeddings = model.encode(
                texts,
                batch_size=self.batch_size,
                show_progress_bar=False,
                convert_to_numpy=True,
                normalize_embeddings=True,
            )
        except Exception as exc:  # noqa: BLE001 - external model exceptions vary
            raise EmbeddingGenerationError(
                f"Failed to generate embeddings: {exc}"
            ) from exc

        return embeddings.tolist()

    def embed_query(self, text: str) -> list[float]:
        """Generate an embedding for a single query string."""

        if not text or not text.strip():
            raise EmbeddingGenerationError("Query text is empty.")

        embeddings = self.generate([text])
        return embeddings[0]

    def _get_model(self):
        """Lazily load the sentence transformer model."""

        if self._model is None:
            try:
                from sentence_transformers import SentenceTransformer
            except ImportError as exc:
                raise EmbeddingGenerationError(
                    "sentence-transformers is required for embeddings."
                ) from exc

            self._model = SentenceTransformer(self.model_name)

        return self._model
