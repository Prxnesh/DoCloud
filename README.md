# CloudInsight

CloudInsight is a modular AI document intelligence platform for local-first document ingestion, NLP analysis, semantic search, and retrieval-augmented chat. The codebase is structured to run locally with open-source components while staying portable to future AWS or GCP deployment.

## Features

- Ingests `PDF`, `DOCX`, and `TXT` documents.
- Normalizes extracted text and creates overlap-aware retrieval chunks.
- Runs summarization, classification, NER, and keyword extraction with local open-source models.
- Builds semantic search with Sentence Transformers plus FAISS.
- Supports conversational querying with a local FLAN-T5 RAG layer.
- Tracks pipeline runs with MLflow.
- Exposes an API-first FastAPI interface and a built-in dashboard UI.

## Project Structure

```text
cloudinsight/
├── app/
│   ├── api/
│   ├── core/
│   └── main.py
├── services/
│   ├── ingestion/
│   ├── processing/
│   ├── nlp/
│   ├── embeddings/
│   ├── vectorstore/
│   ├── chat/
│   ├── storage/
│   └── pipeline/
├── ml/
│   ├── training/
│   └── evaluation/
├── models/
├── data/
├── mlruns/
├── tests/
├── requirements.txt
└── README.md
```

## Setup

1. Create and activate a Python 3.11+ virtual environment.
2. Install dependencies:

```bash
pip install -r requirements.txt
python -m spacy download en_core_web_sm
```

3. Start the API:

```bash
uvicorn app.main:app --reload
```

4. Open the product UI:

```text
http://127.0.0.1:8000/
```

## Configuration

Configuration is environment-driven through `app/core/config.py`. Important settings include:

- `STORAGE_BACKEND`: `local` or `s3`
- `VECTORSTORE_BACKEND`: currently `faiss`
- `SUMMARIZATION_MODEL_NAME`
- `CLASSIFIER_MODEL_NAME`
- `EMBEDDING_MODEL_NAME`
- `CHAT_MODEL_NAME`
- `ENABLE_MLFLOW`
- `MLFLOW_TRACKING_URI`

## API Endpoints

- `GET /api/v1/health`
- `POST /api/v1/documents/process`
- `POST /api/v1/search`
- `POST /api/v1/chat`

## Frontend

The built-in frontend is served from `/` and includes:

- A drag-and-drop document upload flow
- Analysis overview cards and visualizations
- Semantic search results
- Retrieval-augmented chat
- A persisted dark mode toggle

## Notes

- The default classifier uses a DistilBERT-based zero-shot model so candidate labels can be supplied at runtime.
- The storage layer is abstracted to keep local disk and S3 implementations interchangeable.
- The vector store layer persists FAISS state on disk and is designed to be replaceable with managed vector databases later.
# DoCloud
