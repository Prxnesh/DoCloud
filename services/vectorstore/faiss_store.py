"""FAISS-backed vector store for semantic search."""

from __future__ import annotations

import json
from pathlib import Path

from models.analysis import SearchResult
from models.document import TextChunk


class VectorStoreError(Exception):
    """Raised when vector store operations fail."""


class FAISSVectorStore:
    """Persist document chunk embeddings in a local FAISS index."""

    def __init__(self, index_path: str | Path, metadata_path: str | Path) -> None:
        self.index_path = Path(index_path).expanduser().resolve()
        self.metadata_path = Path(metadata_path).expanduser().resolve()
        self.index_path.parent.mkdir(parents=True, exist_ok=True)
        self.metadata_path.parent.mkdir(parents=True, exist_ok=True)

        self._index = None
        self._records: list[dict[str, object]] = []
        self._dimension: int | None = None

        self._load()

    def add(self, chunks: list[TextChunk], embeddings: list[list[float]]) -> None:
        """Add document chunk embeddings to the index."""

        if not chunks:
            return

        if len(chunks) != len(embeddings):
            raise VectorStoreError("Chunk and embedding counts must match.")

        faiss = self._get_faiss()
        np = self._get_numpy()
        matrix = np.asarray(embeddings, dtype="float32")

        if matrix.ndim != 2:
            raise VectorStoreError("Embeddings must be a 2D matrix.")

        self._normalize(matrix, np)

        if self._index is None:
            self._dimension = int(matrix.shape[1])
            self._index = faiss.IndexFlatIP(self._dimension)
        elif int(matrix.shape[1]) != self._dimension:
            raise VectorStoreError("Embedding dimension does not match index dimension.")

        self._index.add(matrix)

        for chunk in chunks:
            self._records.append(
                {
                    "chunk_id": chunk.chunk_id,
                    "document_id": chunk.document_id,
                    "chunk_index": chunk.chunk_index,
                    "text": chunk.text,
                    "start_char": chunk.start_char,
                    "end_char": chunk.end_char,
                    "token_count": chunk.token_count,
                    "metadata": chunk.metadata,
                }
            )

        self._persist()

    def search(
        self,
        query_embedding: list[float],
        top_k: int = 5,
        document_id: str | None = None,
    ) -> list[SearchResult]:
        """Search the vector store for semantically similar chunks."""

        if top_k <= 0:
            raise VectorStoreError("top_k must be greater than zero.")

        if self._index is None or not self._records:
            return []

        np = self._get_numpy()
        vector = np.asarray([query_embedding], dtype="float32")
        self._normalize(vector, np)

        search_window = min(max(top_k * 5, top_k), len(self._records))
        distances, indices = self._index.search(vector, search_window)

        results: list[SearchResult] = []
        for score, index in zip(distances[0], indices[0], strict=False):
            if index < 0:
                continue

            record = self._records[index]
            if document_id and record["document_id"] != document_id:
                continue

            results.append(
                SearchResult(
                    chunk_id=str(record["chunk_id"]),
                    document_id=str(record["document_id"]),
                    score=float(score),
                    text=str(record["text"]),
                    metadata=dict(record["metadata"]),
                )
            )

            if len(results) >= top_k:
                break

        return results

    def _persist(self) -> None:
        """Persist the FAISS index and metadata to disk."""

        faiss = self._get_faiss()
        if self._index is None:
            return

        faiss.write_index(self._index, str(self.index_path))
        self.metadata_path.write_text(
            json.dumps(self._records, indent=2),
            encoding="utf-8",
        )

    def _load(self) -> None:
        """Load existing index state from disk if present."""

        if not self.index_path.exists() or not self.metadata_path.exists():
            return

        faiss = self._get_faiss()
        try:
            self._index = faiss.read_index(str(self.index_path))
            self._dimension = int(self._index.d)
            self._records = json.loads(self.metadata_path.read_text(encoding="utf-8"))
        except Exception as exc:  # noqa: BLE001 - faiss/json exceptions vary
            raise VectorStoreError(f"Failed to load vector store: {exc}") from exc

    def _get_faiss(self):
        """Import FAISS lazily so module import stays lightweight."""

        try:
            import faiss
        except ImportError as exc:
            raise VectorStoreError("faiss-cpu is required for vector search.") from exc

        return faiss

    def _get_numpy(self):
        """Import NumPy lazily for vector normalization."""

        try:
            import numpy as np
        except ImportError as exc:
            raise VectorStoreError("numpy is required for vector search.") from exc

        return np

    def _normalize(self, matrix, np) -> None:
        """Normalize vectors in-place for cosine similarity search."""

        norms = np.linalg.norm(matrix, axis=1, keepdims=True)
        norms[norms == 0.0] = 1.0
        matrix /= norms
