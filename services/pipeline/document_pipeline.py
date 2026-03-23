"""End-to-end document intelligence pipeline orchestration."""

from __future__ import annotations

from dataclasses import asdict
from datetime import UTC, datetime
import json
from pathlib import Path
from typing import Protocol

from ml.training.mlflow_tracker import MLflowTracker
from models.analysis import ChatResponse, DocumentAnalysis, PipelineResult, SearchResult
from models.document import ChunkingResult, CleanedDocument, DocumentContent


class IngestionService(Protocol):
    """Protocol for document ingestion."""

    def extract(self, source_path: str | Path) -> DocumentContent:
        """Extract standardized content from a source file."""


class TextCleanerService(Protocol):
    """Protocol for text cleaning."""

    def clean(
        self,
        text: str,
        document_id: str,
        metadata: dict[str, object] | None = None,
    ) -> CleanedDocument:
        """Clean extracted text."""


class TextChunkerService(Protocol):
    """Protocol for chunking."""

    def chunk(
        self,
        text: str,
        document_id: str,
        metadata: dict[str, object] | None = None,
        chunk_size: int = 250,
        chunk_overlap: int = 40,
    ) -> ChunkingResult:
        """Chunk cleaned text."""


class SummarizationService(Protocol):
    """Protocol for summarization."""

    def summarize(self, text: str) -> str:
        """Summarize a text body."""


class ClassificationService(Protocol):
    """Protocol for classification."""

    def classify(self, text: str, candidate_labels: list[str]):
        """Classify a text body."""


class EntityExtractionService(Protocol):
    """Protocol for entity extraction."""

    def extract(self, text: str):
        """Extract entities from a text body."""


class KeywordExtractionService(Protocol):
    """Protocol for keyword extraction."""

    def extract(self, text: str, top_n: int = 10):
        """Extract keywords from a text body."""


class EmbeddingService(Protocol):
    """Protocol for embeddings."""

    def generate(self, texts: list[str]) -> list[list[float]]:
        """Generate embeddings for many texts."""

    def embed_query(self, text: str) -> list[float]:
        """Generate an embedding for one text."""


class VectorStoreService(Protocol):
    """Protocol for vector search storage."""

    def add(self, chunks, embeddings) -> None:
        """Persist embeddings for chunks."""

    def search(
        self,
        query_embedding: list[float],
        top_k: int = 5,
        document_id: str | None = None,
    ) -> list[SearchResult]:
        """Search stored chunk embeddings."""


class StorageService(Protocol):
    """Protocol for binary file storage."""

    def save_bytes(self, file_name: str, content: bytes):
        """Persist bytes and return storage metadata."""


class ChatService(Protocol):
    """Protocol for RAG chat."""

    def answer(
        self,
        question: str,
        top_k: int = 5,
        document_id: str | None = None,
    ) -> ChatResponse:
        """Answer a grounded question."""


class DocumentPipeline:
    """Coordinate ingestion, NLP analysis, vector indexing, and chat."""

    def __init__(
        self,
        ingestion_service: IngestionService,
        cleaner: TextCleanerService,
        chunker: TextChunkerService,
        summarizer: SummarizationService,
        classifier: ClassificationService,
        entity_extractor: EntityExtractionService,
        keyword_extractor: KeywordExtractionService,
        embeddings: EmbeddingService,
        vector_store: VectorStoreService,
        storage: StorageService,
        chat_service: ChatService,
        tracker: MLflowTracker,
        processed_dir: str | Path,
        default_candidate_labels: list[str],
        chunk_size: int,
        chunk_overlap: int,
    ) -> None:
        self.ingestion_service = ingestion_service
        self.cleaner = cleaner
        self.chunker = chunker
        self.summarizer = summarizer
        self.classifier = classifier
        self.entity_extractor = entity_extractor
        self.keyword_extractor = keyword_extractor
        self.embeddings = embeddings
        self.vector_store = vector_store
        self.storage = storage
        self.chat_service = chat_service
        self.tracker = tracker
        self.processed_dir = Path(processed_dir).expanduser().resolve()
        self.processed_dir.mkdir(parents=True, exist_ok=True)
        self.default_candidate_labels = default_candidate_labels
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

    def process_document(
        self,
        file_name: str,
        content: bytes,
        candidate_labels: list[str] | None = None,
    ) -> PipelineResult:
        """Run the full document intelligence pipeline for an uploaded file."""

        stored_object = self.storage.save_bytes(file_name=file_name, content=content)
        run_name = f"process-{stored_object.object_id}"

        with self.tracker.run(run_name=run_name):
            document = self.ingestion_service.extract(stored_object.location)

            merged_metadata = {
                **document.metadata,
                "storage_location": stored_object.location,
                "storage_backend": stored_object.metadata.get("backend", "unknown"),
            }
            cleaned_document = self.cleaner.clean(
                text=document.text,
                document_id=document.document_id,
                metadata=merged_metadata,
            )
            chunking = self.chunker.chunk(
                text=cleaned_document.cleaned_text,
                document_id=document.document_id,
                metadata=cleaned_document.metadata,
                chunk_size=self.chunk_size,
                chunk_overlap=self.chunk_overlap,
            )

            summary = self.summarizer.summarize(cleaned_document.cleaned_text)
            entities = self.entity_extractor.extract(cleaned_document.cleaned_text)
            keywords = self.keyword_extractor.extract(cleaned_document.cleaned_text)
            classification = self.classifier.classify(
                cleaned_document.cleaned_text,
                candidate_labels or self.default_candidate_labels,
            )

            chunk_embeddings = self.embeddings.generate(
                [chunk.text for chunk in chunking.chunks]
            )
            self.vector_store.add(chunking.chunks, chunk_embeddings)

            analysis = DocumentAnalysis(
                document_id=document.document_id,
                summary=summary,
                classification=classification,
                entities=entities,
                keywords=keywords,
                metadata={"generated_at": datetime.now(UTC).isoformat()},
            )
            result = PipelineResult(
                document=document,
                cleaned_document=cleaned_document,
                chunking=chunking,
                analysis=analysis,
                metadata={
                    "stored_object_id": stored_object.object_id,
                    "stored_location": stored_object.location,
                },
            )

            self._write_processed_artifact(result)
            self._track_run(result)
            return result

    def search(
        self,
        query: str,
        top_k: int,
        document_id: str | None = None,
    ) -> list[SearchResult]:
        """Search indexed document chunks semantically."""

        query_embedding = self.embeddings.embed_query(query)
        return self.vector_store.search(
            query_embedding=query_embedding,
            top_k=top_k,
            document_id=document_id,
        )

    def chat(
        self,
        question: str,
        top_k: int,
        document_id: str | None = None,
    ) -> ChatResponse:
        """Answer grounded questions over the indexed corpus."""

        return self.chat_service.answer(
            question=question,
            top_k=top_k,
            document_id=document_id,
        )

    def _write_processed_artifact(self, result: PipelineResult) -> None:
        """Persist a lightweight JSON artifact for downstream inspection."""

        output_path = self.processed_dir / f"{result.document.document_id}.json"
        artifact = json.dumps(asdict(result), indent=2)
        output_path.write_text(artifact, encoding="utf-8")

    def _track_run(self, result: PipelineResult) -> None:
        """Record useful experiment metadata in MLflow."""

        self.tracker.log_params(
            {
                "document_type": result.document.document_type,
                "chunk_size": self.chunk_size,
                "chunk_overlap": self.chunk_overlap,
            }
        )
        self.tracker.log_metrics(
            {
                "chunk_count": float(len(result.chunking.chunks)),
                "entity_count": float(len(result.analysis.entities)),
                "keyword_count": float(len(result.analysis.keywords)),
                "summary_length": float(len(result.analysis.summary.split())),
            }
        )
        self.tracker.log_text(
            result.analysis.summary,
            artifact_file=f"summaries/{result.document.document_id}.txt",
        )
