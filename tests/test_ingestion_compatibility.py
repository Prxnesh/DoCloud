"""Compatibility tests for ingestion file-type support."""

from pathlib import Path

from services.ingestion.ingestion_service import DocumentIngestionService


def test_ingestion_service_supports_markdown_and_csv(tmp_path: Path) -> None:
    markdown_file = tmp_path / "sample.md"
    markdown_file.write_text("# Title\n\nSome content", encoding="utf-8")

    csv_file = tmp_path / "sample.csv"
    csv_file.write_text("name,value\nfoo,1\n", encoding="utf-8")

    ingestion = DocumentIngestionService()

    markdown_doc = ingestion.extract(markdown_file)
    csv_doc = ingestion.extract(csv_file)

    assert markdown_doc.document_type == "md"
    assert "Title" in markdown_doc.text
    assert csv_doc.document_type == "csv"
    assert "name,value" in csv_doc.text
