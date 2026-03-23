"""Conversational retrieval pipeline for CloudInsight."""

from __future__ import annotations

from typing import Protocol

from models.analysis import ChatResponse, SearchResult


class QueryEmbedder(Protocol):
    """Protocol for query embedding services."""

    def embed_query(self, text: str) -> list[float]:
        """Generate an embedding for a single query."""


class Retriever(Protocol):
    """Protocol for vector retrieval backends."""

    def search(
        self,
        query_embedding: list[float],
        top_k: int = 5,
        document_id: str | None = None,
    ) -> list[SearchResult]:
        """Search for semantically similar chunks."""


class RAGPipelineError(Exception):
    """Raised when the conversational retrieval flow fails."""


class RAGPipeline:
    """Combine retrieval and local generation into a chat response."""

    def __init__(
        self,
        model_name: str,
        embedder: QueryEmbedder,
        retriever: Retriever,
    ) -> None:
        self.model_name = model_name
        self.embedder = embedder
        self.retriever = retriever
        self._pipeline = None

    def answer(
        self,
        question: str,
        top_k: int = 5,
        document_id: str | None = None,
        history: list[dict[str, str]] | None = None,
    ) -> ChatResponse:
        """Answer a question grounded in retrieved document chunks."""

        if not question or not question.strip():
            raise RAGPipelineError("Question cannot be empty.")

        query_embedding = self.embedder.embed_query(question)
        results = self.retriever.search(
            query_embedding=query_embedding,
            top_k=top_k,
            document_id=document_id,
        )

        if not results:
            return ChatResponse(
                answer="I could not find any relevant document context for that question.",
                results=[],
                metadata={"retrieved_chunks": 0},
            )

        prompt = self._build_prompt(question, results, history)
        generator = self._get_pipeline()

        try:
            output = generator(prompt, max_new_tokens=180, do_sample=False)
        except Exception as exc:  # noqa: BLE001 - transformer exceptions vary
            answer = self._manual_answer(question, results)
            return ChatResponse(
                answer=answer,
                results=results,
                metadata={"retrieved_chunks": len(results), "fallback": "manual"},
            )

        answer = self._extract_answer_text(output, prompt)
        if not answer:
            answer = self._manual_answer(question, results)
            metadata = {"retrieved_chunks": len(results), "fallback": "manual"}
        else:
            metadata = {"retrieved_chunks": len(results)}

        return ChatResponse(
            answer=answer,
            results=results,
            metadata=metadata,
        )

    def _build_prompt(
        self,
        question: str,
        results: list[SearchResult],
        history: list[dict[str, str]] | None = None,
    ) -> str:
        """Create a grounded prompt from retrieved passages."""

        context = "\n\n".join(
            f"Context {index + 1}:\n{result.text}"
            for index, result in enumerate(results)
        )

        conversation_block = self._format_history(history)

        return (
            "Answer the question using only the provided context. "
            "If the context is insufficient, say so clearly.\n\n"
            f"{conversation_block}"
            f"{context}\n\n"
            f"Question: {question}\n"
            "Answer:"
        )

    def _format_history(self, history: list[dict[str, str]] | None) -> str:
        """Convert prior conversation turns into a short prompt segment."""

        if not history:
            return ""

        lines = ["Conversation history:"]
        for turn in history[-6:]:
            role = str(turn.get("role", "user")).strip().lower()
            text = str(turn.get("text", "")).strip()
            if not text:
                continue

            speaker = "User" if role == "user" else "Assistant"
            lines.append(f"{speaker}: {text}")

        if len(lines) == 1:
            return ""

        return "\n".join(lines) + "\n\n"

    def _get_pipeline(self):
        """Lazily load the local chat generation model."""

        if self._pipeline is None:
            try:
                from transformers import pipeline
            except ImportError as exc:
                raise RAGPipelineError(
                    "transformers is required for conversational querying."
                ) from exc

            pipeline_factory = pipeline
            last_error: Exception | None = None

            for task_name in ("text2text-generation", "text-generation"):
                try:
                    self._pipeline = pipeline_factory(
                        task_name,
                        model=self.model_name,
                    )
                    break
                except Exception as exc:  # noqa: BLE001 - runtime support varies by version
                    last_error = exc

            if self._pipeline is None:
                raise RAGPipelineError(
                    "Unable to initialize a chat generation pipeline. "
                    "Tried 'text2text-generation' and 'text-generation'. "
                    f"Last error: {last_error}"
                )

        return self._pipeline

    def _extract_answer_text(self, output, prompt: str) -> str:
        """Normalize model output and reject prompt-echo responses."""

        if not isinstance(output, list) or not output or not isinstance(output[0], dict):
            return ""

        generated = str(output[0].get("generated_text", "")).strip()
        if not generated:
            return ""

        if generated.startswith(prompt):
            generated = generated[len(prompt):].strip()

        if generated.lower().startswith("answer:"):
            generated = generated.split(":", 1)[1].strip()

        if not generated:
            return ""

        # Reject responses that are mostly a repeated prompt/context block.
        if "Context 1:" in generated and "Question:" in generated:
            return ""

        return generated

    def _manual_answer(self, question: str, results: list[SearchResult]) -> str:
        """Provide a deterministic grounded response from top retrieved context."""

        top_text = results[0].text.strip() if results else ""
        if not top_text:
            return "I could not find enough context to answer that question."

        snippet = " ".join(top_text.split())
        if len(snippet) > 280:
            snippet = snippet[:277].rstrip() + "..."

        return f"Based on the retrieved document context: {snippet}"
