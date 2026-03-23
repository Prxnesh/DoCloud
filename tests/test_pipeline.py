"""Pipeline smoke test with fake dependencies."""

from pathlib import Path

from ml.training.mlflow_tracker import MLflowTracker
from models.analysis import ChatResponse, ClassificationResult
from models.document import ChunkingResult, CleanedDocument, DocumentContent, TextChunk
from services.pipeline.document_pipeline import DocumentPipeline
from services.storage.storage_manager import LocalStorageManager


class FakeIngestionService:
    def extract(self, source_path: str | Path) -> DocumentContent:
        return DocumentContent(
            document_id="doc-1",
            source_path=str(source_path),
            file_name="sample.txt",
            document_type="txt",
            text="This is a test document for the pipeline.",
            metadata={},
        )


class FakeCleaner:
    def clean(self, text: str, document_id: str, metadata=None) -> CleanedDocument:
        return CleanedDocument(
            document_id=document_id,
            cleaned_text=text,
            metadata=dict(metadata or {}),
        )


class FakeChunker:
    def chunk(self, text: str, document_id: str, metadata=None, chunk_size=250, chunk_overlap=40) -> ChunkingResult:
        return ChunkingResult(
            document_id=document_id,
            chunks=[
                TextChunk(
                    chunk_id="chunk-1",
                    document_id=document_id,
                    chunk_index=0,
                    text=text,
                    start_char=0,
                    end_char=len(text),
                    token_count=len(text.split()),
                    metadata=dict(metadata or {}),
                )
            ],
            metadata={"chunk_size": chunk_size, "chunk_overlap": chunk_overlap},
        )


class FakeSummarizer:
    def summarize(self, text: str) -> str:
        return "summary"


class FakeClassifier:
    def classify(self, text: str, candidate_labels: list[str]) -> ClassificationResult:
        return ClassificationResult(label=candidate_labels[0], score=0.99)


class FakeEntityExtractor:
    def extract(self, text: str):
        return []


class FakeKeywordExtractor:
    def extract(self, text: str, top_n: int = 10):
        return []


class FakeEmbeddings:
    def generate(self, texts: list[str]) -> list[list[float]]:
        return [[0.1, 0.2, 0.3] for _ in texts]

    def embed_query(self, text: str) -> list[float]:
        return [0.1, 0.2, 0.3]


class FakeVectorStore:
    def __init__(self) -> None:
        self.records = []

    def add(self, chunks, embeddings) -> None:
        self.records.extend(zip(chunks, embeddings, strict=False))

    def search(self, query_embedding, top_k=5, document_id=None):
        return []


class FakeChatService:
    def answer(self, question: str, top_k: int = 5, document_id: str | None = None) -> ChatResponse:
        return ChatResponse(answer="answer", results=[], metadata={})


def test_document_pipeline_processes_document(tmp_path: Path) -> None:
    pipeline = DocumentPipeline(
        ingestion_service=FakeIngestionService(),
        cleaner=FakeCleaner(),
        chunker=FakeChunker(),
        summarizer=FakeSummarizer(),
        classifier=FakeClassifier(),
        entity_extractor=FakeEntityExtractor(),
        keyword_extractor=FakeKeywordExtractor(),
        embeddings=FakeEmbeddings(),
        vector_store=FakeVectorStore(),
        storage=LocalStorageManager(tmp_path / "uploads"),
        chat_service=FakeChatService(),
        tracker=MLflowTracker(
            enabled=False,
            experiment_name="test",
            artifact_dir=tmp_path / "mlruns",
        ),
        processed_dir=tmp_path / "processed",
        default_candidate_labels=["engineering", "finance"],
        chunk_size=250,
        chunk_overlap=40,
    )

    result = pipeline.process_document("sample.txt", b"content")

    assert result.document.document_id == "doc-1"
    assert result.analysis.summary == "summary"
    assert (tmp_path / "processed" / "doc-1.json").exists()
