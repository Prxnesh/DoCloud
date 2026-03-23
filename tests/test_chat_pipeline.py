"""Unit tests for conversational RAG behavior."""

from models.analysis import SearchResult
from services.chat.rag_pipeline import RAGPipeline


class FakeEmbedder:
    def embed_query(self, text: str) -> list[float]:
        return [0.1, 0.2, 0.3]


class FakeRetriever:
    def search(self, query_embedding, top_k=5, document_id=None):
        return [
            SearchResult(
                chunk_id="chunk-1",
                document_id="doc-1",
                score=0.9,
                text="The policy requires quarterly compliance reporting.",
                metadata={},
            )
        ]


def test_rag_pipeline_uses_history_in_prompt() -> None:
    rag = RAGPipeline(
        model_name="dummy-model",
        embedder=FakeEmbedder(),
        retriever=FakeRetriever(),
    )

    captured_prompt = {"value": ""}

    def fake_generator(prompt: str, max_new_tokens: int, do_sample: bool):
        captured_prompt["value"] = prompt
        return [{"generated_text": "Answer: Quarterly compliance reporting is required."}]

    rag._get_pipeline = lambda: fake_generator

    response = rag.answer(
        question="What does the policy require?",
        history=[
            {"role": "user", "text": "Summarize the policy."},
            {"role": "assistant", "text": "It focuses on compliance controls."},
        ],
    )

    assert "Conversation history:" in captured_prompt["value"]
    assert "User: Summarize the policy." in captured_prompt["value"]
    assert "Assistant: It focuses on compliance controls." in captured_prompt["value"]
    assert "Quarterly compliance reporting" in response.answer
