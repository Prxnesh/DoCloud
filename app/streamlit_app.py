"""Streamlit UI for CloudInsight AI Document Intelligence Platform.

This module focuses on presentation and lightweight UX enhancements while
reusing the existing FastAPI backend endpoints.
"""

from __future__ import annotations

import json
import math
import os
import re
from collections import Counter
from datetime import datetime
from typing import Any

import requests
import streamlit as st


DEFAULT_API_BASE_URL = os.getenv("CLOUDINSIGHT_API_URL", "http://127.0.0.1:8000")
API_PREFIX = "/api/v1"


def init_state() -> None:
    """Initialize session state containers used by optional UI features."""

    st.session_state.setdefault("processed_result", None)
    st.session_state.setdefault("document_text", "")
    st.session_state.setdefault("chat_answer", "")
    st.session_state.setdefault("chat_results", [])
    st.session_state.setdefault("question_history", [])
    st.session_state.setdefault("pending_question", "")
    st.session_state.setdefault("selected_keyword", None)


def estimate_reading_time_minutes(text: str, words_per_minute: int = 200) -> int:
    """Estimate reading time with a simple words-per-minute rule."""

    word_count = len(text.split())
    return max(1, math.ceil(word_count / max(1, words_per_minute)))


def infer_page_count(file_name: str, payload: bytes) -> int | None:
    """Infer page count when possible without extra heavy dependencies."""

    if not file_name.lower().endswith(".pdf"):
        return None

    try:
        # Lightweight heuristic: count common PDF page markers.
        return max(1, payload.count(b"/Type /Page"))
    except Exception:
        return None


def extract_text_preview(file_name: str, payload: bytes) -> str:
    """Best-effort text extraction for keyword sentence filtering and stats."""

    if file_name.lower().endswith((".txt", ".md", ".csv", ".json", ".html", ".rtf", ".log")):
        return payload.decode("utf-8", errors="ignore")

    # For non-TXT files, keep this lightweight and avoid additional parsers.
    return payload.decode("utf-8", errors="ignore")


def split_sentences(text: str) -> list[str]:
    """Split text into readable sentence-like chunks."""

    cleaned = " ".join(text.split())
    if not cleaned:
        return []
    parts = re.split(r"(?<=[.!?])\s+", cleaned)
    return [part.strip() for part in parts if part.strip()]


def call_process_document(api_base_url: str, file_name: str, payload: bytes) -> dict[str, Any]:
    """Call backend document processing endpoint with multipart upload."""

    endpoint = f"{api_base_url}{API_PREFIX}/documents/process"
    files = {"file": (file_name, payload)}
    response = requests.post(endpoint, files=files, timeout=300)
    response.raise_for_status()
    return response.json()


def call_chat(api_base_url: str, question: str, top_k: int = 5) -> dict[str, Any]:
    """Call backend chat endpoint."""

    endpoint = f"{api_base_url}{API_PREFIX}/chat"
    response = requests.post(
        endpoint,
        json={"question": question, "top_k": top_k},
        timeout=300,
    )
    response.raise_for_status()
    return response.json()


def call_search(api_base_url: str, query: str, top_k: int = 5) -> dict[str, Any]:
    """Call backend semantic search endpoint."""

    endpoint = f"{api_base_url}{API_PREFIX}/search"
    response = requests.post(
        endpoint,
        json={"query": query, "top_k": top_k},
        timeout=300,
    )
    response.raise_for_status()
    return response.json()


def render_header() -> tuple[str, bool]:
    """Render page title and top-level mode controls."""

    st.set_page_config(page_title="CloudInsight AI", page_icon="AI", layout="wide")

    st.title("CloudInsight AI Document Intelligence")
    st.caption("Analyze documents, extract insights, and ask grounded questions.")

    control_left, control_mid, control_right = st.columns([2, 1, 1])
    with control_left:
        api_base_url = st.text_input("Backend URL", value=DEFAULT_API_BASE_URL)
    with control_mid:
        advanced_mode = st.toggle("Advanced Mode", value=False)
    with control_right:
        st.write("")
        st.write("")
        st.markdown("**Mode:** Advanced" if advanced_mode else "**Mode:** Basic")

    return api_base_url.rstrip("/"), advanced_mode


def render_upload_and_process(api_base_url: str) -> None:
    """Render upload section and process documents with step-by-step status."""

    uploaded_file = st.file_uploader(
        "Upload a document",
        type=["txt", "pdf", "docx", "md", "csv", "json", "html", "rtf", "log"],
        help="Supported formats: PDF, DOCX, TXT, MD, CSV, JSON, HTML, RTF, LOG",
    )

    process_clicked = st.button("Process Document", type="primary", use_container_width=True)

    if not uploaded_file or not process_clicked:
        return

    file_name = uploaded_file.name
    payload = uploaded_file.getvalue()
    st.session_state.document_text = extract_text_preview(file_name, payload)

    try:
        with st.spinner("Extracting text..."):
            st.info("Step 1/3: Extracting text")

        with st.spinner("Running AI models..."):
            st.info("Step 2/3: Running AI models")
            result = call_process_document(api_base_url, file_name, payload)

        with st.spinner("Generating insights..."):
            st.info("Step 3/3: Generating insights")

        metadata = result.get("metadata", {})
        metadata["uploaded_file_name"] = file_name
        metadata["uploaded_file_size_bytes"] = len(payload)
        metadata["uploaded_file_pages"] = infer_page_count(file_name, payload)
        result["metadata"] = metadata

        st.session_state.processed_result = result
        st.success("Document processed successfully.")
    except requests.HTTPError as exc:
        detail = ""
        try:
            detail = exc.response.json().get("detail", "")
        except Exception:
            detail = exc.response.text if exc.response is not None else str(exc)
        st.error(f"Processing failed: {detail}")
    except Exception as exc:  # noqa: BLE001
        st.error(f"Unexpected error: {exc}")


def build_ai_insight_fields(result: dict[str, Any], document_text: str) -> dict[str, Any]:
    """Prepare AI insight card fields from processed output."""

    keywords = result.get("keywords", [])
    classification_score = float(result.get("classification_score", 0.0))
    summary = result.get("summary", "")

    readable_source = document_text if document_text.strip() else summary
    reading_time = estimate_reading_time_minutes(readable_source)

    return {
        "document_type": result.get("document_type", "unknown").upper(),
        "summary": summary,
        "top_keywords": keywords[:5],
        "confidence_score": classification_score,
        "reading_time": reading_time,
    }


def render_ai_insights_card(result: dict[str, Any], document_text: str) -> None:
    """Feature 1 + 3: Structured insights card with reading time estimator."""

    fields = build_ai_insight_fields(result, document_text)

    st.markdown(
        f"""
        ### AI Insights

        **Document Type:** {fields['document_type']}  
        **Summary:** {fields['summary']}  
        **Top Keywords:** {", ".join(fields['top_keywords']) if fields['top_keywords'] else "N/A"}  
        **Confidence Score:** {fields['confidence_score']:.2%}  
        **Estimated Reading Time:** {fields['reading_time']} minutes
        """
    )


def render_file_metadata(result: dict[str, Any]) -> None:
    """Feature 12: Show file metadata in a compact panel."""

    metadata = result.get("metadata", {})
    file_name = metadata.get("uploaded_file_name", "Unknown")
    file_size = int(metadata.get("uploaded_file_size_bytes", 0))
    page_count = metadata.get("uploaded_file_pages")

    st.markdown("### File Metadata")
    st.write(f"File Name: {file_name}")
    st.write(f"File Size: {file_size:,} bytes")
    st.write(f"Number of Pages: {page_count if page_count is not None else 'N/A'}")


def render_document_statistics(result: dict[str, Any], document_text: str) -> None:
    """Feature 4: Display key document statistics."""

    entities = result.get("entities", [])
    keywords = result.get("keywords", [])
    summary = result.get("summary", "")

    source_text = document_text if document_text.strip() else summary
    word_count = len(source_text.split())
    reading_time = estimate_reading_time_minutes(source_text)

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Word Count", f"{word_count:,}")
    col2.metric("Entities", len(entities))
    col3.metric("Keywords", len(keywords))
    col4.metric("Reading Time", f"{reading_time} min")


def render_color_coded_entities(entities: list[dict[str, Any]]) -> None:
    """Feature 6: Color-coded entity display by label category."""

    color_map = {
        "PERSON": "#3b82f6",
        "ORG": "#22c55e",
        "DATE": "#facc15",
    }

    if not entities:
        st.info("No entities detected.")
        return

    for entity in entities:
        label = str(entity.get("label", "UNKNOWN")).upper()
        color = color_map.get(label, "#9ca3af")
        text = entity.get("text", "")
        st.markdown(
            f"<span style='background-color:{color}; padding: 0.2rem 0.45rem; border-radius: 0.5rem; margin-right: 0.35rem;'>"
            f"{label}</span> {text}",
            unsafe_allow_html=True,
        )


def render_keyword_interaction(result: dict[str, Any], document_text: str) -> None:
    """Feature 9: Clickable keyword chips that filter relevant sentences."""

    keywords = result.get("keywords", [])
    if not keywords:
        st.info("No keywords available.")
        return

    st.markdown("### Interactive Keywords")

    keyword_columns = st.columns(min(4, max(1, len(keywords))))
    for idx, keyword in enumerate(keywords):
        with keyword_columns[idx % len(keyword_columns)]:
            if st.button(keyword, key=f"keyword_{idx}", use_container_width=True):
                st.session_state.selected_keyword = keyword

    selected = st.session_state.get("selected_keyword")
    if not selected:
        return

    st.markdown(f"**Filtered Sentences for:** {selected}")
    sentences = split_sentences(document_text)
    filtered = [sentence for sentence in sentences if selected.lower() in sentence.lower()]

    if not filtered:
        st.info("No matching sentence found for selected keyword.")
        return

    for sentence in filtered[:8]:
        st.write(f"- {sentence}")


def render_most_important_section(result: dict[str, Any], document_text: str) -> None:
    """Feature 10: Highlight an important section using first chunk heuristic."""

    chunks = split_sentences(document_text)
    if chunks:
        important = chunks[0]
    else:
        important = result.get("summary", "No section available.")

    st.markdown("### Most Important Section")
    st.success(important)


def render_download_report(result: dict[str, Any]) -> None:
    """Feature 8: Download JSON report with key outputs."""

    report = {
        "summary": result.get("summary", ""),
        "entities": result.get("entities", []),
        "keywords": result.get("keywords", []),
        "classification": {
            "label": result.get("classification_label", ""),
            "score": result.get("classification_score", 0.0),
        },
        "generated_at": datetime.utcnow().isoformat() + "Z",
    }

    st.download_button(
        "Download Report (JSON)",
        data=json.dumps(report, indent=2),
        file_name="cloudinsight_report.json",
        mime="application/json",
        use_container_width=True,
    )


def render_model_info_panel(advanced_mode: bool) -> None:
    """Feature 11 + 15: Model details and optional advanced diagnostics."""

    summarization_model = os.getenv("SUMMARIZATION_MODEL_NAME", "facebook/bart-large-cnn")
    embedding_model = os.getenv("EMBEDDING_MODEL_NAME", "sentence-transformers/all-MiniLM-L6-v2")
    ner_model = os.getenv("NER_MODEL_NAME", "en_core_web_sm")

    if not advanced_mode:
        return

    st.markdown("### Model Info")
    st.write(f"Summarization Model: {summarization_model}")
    st.write(f"Embedding Model: {embedding_model}")
    st.write(f"NER Model: {ner_model}")


def render_analytics_tab(result: dict[str, Any], api_base_url: str, advanced_mode: bool) -> None:
    """Feature 14: Basic analytics chart and optional debug outputs."""

    st.markdown("### Analytics")

    entities = result.get("entities", [])
    keywords = result.get("keywords", [])

    entity_counts = Counter(str(item.get("label", "UNKNOWN")) for item in entities)
    if entity_counts:
        st.subheader("Entity Counts")
        st.bar_chart(entity_counts)
    else:
        keyword_freq = Counter(keywords)
        if keyword_freq:
            st.subheader("Keyword Frequency")
            st.bar_chart(keyword_freq)
        else:
            st.info("No analytics data available yet.")

    if advanced_mode:
        st.markdown("### Debug Info")
        st.json(result)
        try:
            probe = call_search(api_base_url, query="overview", top_k=3)
            st.markdown("### Retrieval Probe")
            st.json(probe)
        except Exception as exc:  # noqa: BLE001
            st.warning(f"Search probe unavailable: {exc}")


def run_suggested_question(api_base_url: str, question: str) -> None:
    """Feature 2: Auto-fill and trigger chat responses for suggested prompts."""

    st.session_state.pending_question = question
    try:
        response = call_chat(api_base_url, question=question, top_k=5)
        st.session_state.chat_answer = response.get("answer", "")
        st.session_state.chat_results = response.get("results", [])
        st.session_state.question_history.append(question)
    except Exception as exc:  # noqa: BLE001
        st.error(f"Suggested question failed: {exc}")


def render_chat_tab(api_base_url: str) -> None:
    """Feature 2 + 13: Chat panel with suggested prompts and question history."""

    st.markdown("### Chat")

    st.markdown("#### Suggested Questions")
    sq_col1, sq_col2, sq_col3 = st.columns(3)
    with sq_col1:
        if st.button("Summarize this document", use_container_width=True):
            run_suggested_question(api_base_url, "Summarize this document")
    with sq_col2:
        if st.button("What are key findings?", use_container_width=True):
            run_suggested_question(api_base_url, "What are key findings?")
    with sq_col3:
        if st.button("Extract important entities", use_container_width=True):
            run_suggested_question(api_base_url, "Extract important entities")

    question = st.text_input(
        "Ask a question",
        value=st.session_state.get("pending_question", ""),
        key="chat_input_text",
    )
    top_k = st.slider("Retrieval depth", min_value=1, max_value=10, value=5)

    ask_clicked = st.button("Ask", type="primary")
    if ask_clicked and question.strip():
        try:
            response = call_chat(api_base_url, question=question.strip(), top_k=top_k)
            st.session_state.chat_answer = response.get("answer", "")
            st.session_state.chat_results = response.get("results", [])
            st.session_state.question_history.append(question.strip())
            st.session_state.pending_question = question.strip()
        except requests.HTTPError as exc:
            detail = ""
            try:
                detail = exc.response.json().get("detail", "")
            except Exception:
                detail = exc.response.text if exc.response is not None else str(exc)
            st.error(f"Chat failed: {detail}")
        except Exception as exc:  # noqa: BLE001
            st.error(f"Chat failed: {exc}")

    if st.session_state.chat_answer:
        st.markdown("#### Response")
        st.write(st.session_state.chat_answer)

    if st.session_state.chat_results:
        with st.expander("Supporting Chunks", expanded=False):
            for idx, item in enumerate(st.session_state.chat_results, start=1):
                st.write(f"{idx}. {item.get('text', '')}")

    st.markdown("#### Question History")
    history = st.session_state.get("question_history", [])
    if not history:
        st.caption("No previous questions yet.")
    else:
        for idx, item in enumerate(history[-10:], start=1):
            st.write(f"{idx}. {item}")


def render_tabs(api_base_url: str, advanced_mode: bool) -> None:
    """Render primary tabbed UI and feature modules."""

    result = st.session_state.get("processed_result")
    document_text = st.session_state.get("document_text", "")

    summary_tab, entities_tab, keywords_tab, analytics_tab, chat_tab = st.tabs(
        ["Summary", "Entities", "Keywords", "Analytics", "Chat"]
    )

    with summary_tab:
        if not result:
            st.info("Upload and process a document to view summary insights.")
        else:
            render_ai_insights_card(result, document_text)
            render_document_statistics(result, document_text)
            render_file_metadata(result)
            render_most_important_section(result, document_text)

            # Feature 5: Expandable detailed output blocks.
            with st.expander("Full Summary", expanded=True):
                st.write(result.get("summary", "No summary available."))

            render_download_report(result)

    with entities_tab:
        if not result:
            st.info("Process a document to view entities.")
        else:
            with st.expander("Entities", expanded=True):
                render_color_coded_entities(result.get("entities", []))

    with keywords_tab:
        if not result:
            st.info("Process a document to view keywords.")
        else:
            with st.expander("Keywords", expanded=True):
                keywords = result.get("keywords", [])
                if not keywords:
                    st.info("No keywords available.")
                else:
                    st.write(", ".join(keywords))

            render_keyword_interaction(result, document_text)

    with analytics_tab:
        if not result:
            st.info("Process a document to view analytics.")
        else:
            render_analytics_tab(result, api_base_url, advanced_mode)
            render_model_info_panel(advanced_mode)

    with chat_tab:
        render_chat_tab(api_base_url)


def main() -> None:
    """Entrypoint for the Streamlit app."""

    init_state()
    api_base_url, advanced_mode = render_header()
    render_upload_and_process(api_base_url)
    render_tabs(api_base_url, advanced_mode)


if __name__ == "__main__":
    main()
