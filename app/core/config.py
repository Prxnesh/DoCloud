"""Environment-driven configuration for CloudInsight."""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


PROJECT_ROOT = Path(__file__).resolve().parents[2]


class Settings(BaseSettings):
    """Application settings loaded from environment variables."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    app_name: str = "CloudInsight"
    app_env: str = "development"
    app_host: str = "0.0.0.0"
    app_port: int = 8000
    api_prefix: str = "/api/v1"

    data_dir: Path = Field(default_factory=lambda: PROJECT_ROOT / "data")
    upload_dir: Path = Field(default_factory=lambda: PROJECT_ROOT / "data" / "uploads")
    processed_dir: Path = Field(default_factory=lambda: PROJECT_ROOT / "data" / "processed")
    vectorstore_dir: Path = Field(default_factory=lambda: PROJECT_ROOT / "data" / "vectorstore")
    mlruns_dir: Path = Field(default_factory=lambda: PROJECT_ROOT / "mlruns")

    storage_backend: str = "local"
    vectorstore_backend: str = "faiss"
    aws_bucket_name: str | None = None
    aws_region: str | None = None

    summarization_model_name: str = "facebook/bart-large-cnn"
    classifier_model_name: str = "typeform/distilbert-base-uncased-mnli"
    embedding_model_name: str = "sentence-transformers/all-MiniLM-L6-v2"
    keyword_model_name: str = "sentence-transformers/all-MiniLM-L6-v2"
    ner_model_name: str = "en_core_web_sm"
    chat_model_name: str = "google/flan-t5-base"

    default_candidate_labels: list[str] = Field(
        default_factory=lambda: [
            "finance",
            "legal",
            "human resources",
            "engineering",
            "sales",
            "operations",
            "research",
            "compliance",
        ]
    )

    chunk_size: int = 250
    chunk_overlap: int = 40
    retrieval_top_k: int = 5
    summary_max_length: int = 180
    summary_min_length: int = 50
    embedding_batch_size: int = 32

    enable_mlflow: bool = True
    mlflow_tracking_uri: str | None = None
    mlflow_experiment_name: str = "cloudinsight-document-pipeline"

    @property
    def faiss_index_path(self) -> Path:
        """Path for the persisted FAISS index."""

        return self.vectorstore_dir / "cloudinsight.index"

    @property
    def faiss_metadata_path(self) -> Path:
        """Path for vector store metadata persistence."""

        return self.vectorstore_dir / "cloudinsight_metadata.json"


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Return a cached settings instance."""

    settings = Settings()
    settings.data_dir.mkdir(parents=True, exist_ok=True)
    settings.upload_dir.mkdir(parents=True, exist_ok=True)
    settings.processed_dir.mkdir(parents=True, exist_ok=True)
    settings.vectorstore_dir.mkdir(parents=True, exist_ok=True)
    settings.mlruns_dir.mkdir(parents=True, exist_ok=True)
    return settings
