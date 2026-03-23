"""FastAPI routes for the CloudInsight platform."""

from __future__ import annotations

from functools import lru_cache

from fastapi import APIRouter, File, Form, HTTPException, UploadFile
from pydantic import BaseModel, Field

from app.core.config import get_settings
from ml.training.mlflow_tracker import MLflowTracker
from services.chat.rag_pipeline import RAGPipeline
from services.embeddings.embedding_generator import EmbeddingGenerator
from services.ingestion.ingestion_service import DocumentIngestionService
from services.nlp.classifier import DocumentClassifier
from services.nlp.entity_extractor import EntityExtractor
from services.nlp.keyword_extractor import KeywordExtractor
from services.nlp.summarizer import Summarizer
from services.pipeline.document_pipeline import DocumentPipeline
from services.processing.chunker import TextChunker
from services.processing.text_cleaner import TextCleaner
from services.storage.storage_manager import build_storage_manager
from services.vectorstore.faiss_store import FAISSVectorStore


router = APIRouter(tags=["cloudinsight"])


class ProcessDocumentResponse(BaseModel):
    """API response for processed documents."""

    document_id: str
    file_name: str
    document_type: str
    summary: str
    classification_label: str
    classification_score: float
    keywords: list[str]
    entities: list[dict[str, str | int | float | None]]
    chunk_count: int
    metadata: dict[str, object]


class SearchRequest(BaseModel):
    """Request payload for semantic search."""

    query: str = Field(min_length=1)
    top_k: int = Field(default=5, ge=1, le=20)
    document_id: str | None = None


class SearchHit(BaseModel):
    """Single semantic search result."""

    chunk_id: str
    document_id: str
    score: float
    text: str
    metadata: dict[str, object]


class SearchResponse(BaseModel):
    """Response payload for semantic search."""

    results: list[SearchHit]


class ChatRequest(BaseModel):
    """Request payload for RAG chat."""

    question: str = Field(min_length=1)
    top_k: int = Field(default=5, ge=1, le=20)
    document_id: str | None = None


class ChatAPIResponse(BaseModel):
    """Response payload for RAG chat."""

    answer: str
    results: list[SearchHit]
    metadata: dict[str, object]


def _parse_candidate_labels(raw_labels: str | None) -> list[str] | None:
    """Parse candidate labels from a comma-separated form field."""

    if raw_labels is None or not raw_labels.strip():
        return None

    labels = [label.strip() for label in raw_labels.split(",") if label.strip()]
    return labels or None


@lru_cache(maxsize=1)
def get_pipeline() -> DocumentPipeline:
    """Construct the application pipeline once and reuse it per process."""

    settings = get_settings()
    storage = build_storage_manager(
        backend=settings.storage_backend,
        local_base_dir=settings.upload_dir,
        bucket_name=settings.aws_bucket_name,
        region_name=settings.aws_region,
    )
    vector_store = FAISSVectorStore(
        index_path=settings.faiss_index_path,
        metadata_path=settings.faiss_metadata_path,
    )
    embeddings = EmbeddingGenerator(
        model_name=settings.embedding_model_name,
        batch_size=settings.embedding_batch_size,
    )
    chat = RAGPipeline(
        model_name=settings.chat_model_name,
        embedder=embeddings,
        retriever=vector_store,
    )
    tracker = MLflowTracker(
        enabled=settings.enable_mlflow,
        experiment_name=settings.mlflow_experiment_name,
        tracking_uri=settings.mlflow_tracking_uri,
        artifact_dir=settings.mlruns_dir,
    )

    return DocumentPipeline(
        ingestion_service=DocumentIngestionService(),
        cleaner=TextCleaner(),
        chunker=TextChunker(),
        summarizer=Summarizer(
            model_name=settings.summarization_model_name,
            min_length=settings.summary_min_length,
            max_length=settings.summary_max_length,
        ),
        classifier=DocumentClassifier(settings.classifier_model_name),
        entity_extractor=EntityExtractor(settings.ner_model_name),
        keyword_extractor=KeywordExtractor(settings.keyword_model_name),
        embeddings=embeddings,
        vector_store=vector_store,
        storage=storage,
        chat_service=chat,
        tracker=tracker,
        processed_dir=settings.processed_dir,
        default_candidate_labels=settings.default_candidate_labels,
        chunk_size=settings.chunk_size,
        chunk_overlap=settings.chunk_overlap,
    )


@router.get("/health")
def health_check() -> dict[str, str]:
    """Return a basic application health signal."""

    return {"status": "ok"}


@router.post("/documents/process", response_model=ProcessDocumentResponse)
async def process_document(
    file: UploadFile = File(...),
    candidate_labels: str | None = Form(default=None),
) -> ProcessDocumentResponse:
    """Upload, process, analyze, and index a document."""

    pipeline = get_pipeline()

    try:
        content = await file.read()
        result = pipeline.process_document(
            file_name=file.filename or "uploaded_document",
            content=content,
            candidate_labels=_parse_candidate_labels(candidate_labels),
        )
    except Exception as exc:  # noqa: BLE001 - surface clean API errors
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    return ProcessDocumentResponse(
        document_id=result.document.document_id,
        file_name=result.document.file_name,
        document_type=result.document.document_type,
        summary=result.analysis.summary,
        classification_label=result.analysis.classification.label,
        classification_score=result.analysis.classification.score,
        keywords=[item.keyword for item in result.analysis.keywords],
        entities=[
            {
                "text": entity.text,
                "label": entity.label,
                "start_char": entity.start_char,
                "end_char": entity.end_char,
                "score": entity.score,
            }
            for entity in result.analysis.entities
        ],
        chunk_count=len(result.chunking.chunks),
        metadata=result.metadata,
    )


@router.post("/search", response_model=SearchResponse)
def semantic_search(request: SearchRequest) -> SearchResponse:
    """Run semantic retrieval over indexed chunks."""

    pipeline = get_pipeline()

    try:
        results = pipeline.search(
            query=request.query,
            top_k=request.top_k,
            document_id=request.document_id,
        )
    except Exception as exc:  # noqa: BLE001 - surface clean API errors
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    return SearchResponse(
        results=[
            SearchHit(
                chunk_id=result.chunk_id,
                document_id=result.document_id,
                score=result.score,
                text=result.text,
                metadata=result.metadata,
            )
            for result in results
        ]
    )


@router.post("/chat", response_model=ChatAPIResponse)
def chat(request: ChatRequest) -> ChatAPIResponse:
    """Answer grounded user questions using retrieval-augmented generation."""

    pipeline = get_pipeline()

    try:
        response = pipeline.chat(
            question=request.question,
            top_k=request.top_k,
            document_id=request.document_id,
        )
    except Exception as exc:  # noqa: BLE001 - surface clean API errors
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    return ChatAPIResponse(
        answer=response.answer,
        results=[
            SearchHit(
                chunk_id=result.chunk_id,
                document_id=result.document_id,
                score=result.score,
                text=result.text,
                metadata=result.metadata,
            )
            for result in response.results
        ],
        metadata=response.metadata,
    )
