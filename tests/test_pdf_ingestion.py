import json
from pathlib import Path

import pytest

from docmanage.config import load_config, prepare_directories
from docmanage.pdf_ingestion import PdfIngestionError, ingest_pdf
from tests.pdf_helpers import create_pdf


def test_ingest_pdf_reads_pages_and_text_layer(tmp_path: Path) -> None:
    config = prepare_test_config(tmp_path)
    pdf_path = create_pdf(tmp_path / "sample.pdf", ["First page", None, "Third page"])

    result = ingest_pdf(config, str(pdf_path))

    assert result.page_count == 3
    assert result.text_layer_page_count == 2
    assert result.document_id.startswith("doc_")
    assert result.page_manifest_path.exists()

    first_page = result.pages[0]
    second_page = result.pages[1]

    assert first_page.page_id == f"{result.document_id}_page_0001"
    assert first_page.page_number == 1
    assert first_page.width == 300.0
    assert first_page.height == 400.0
    assert first_page.rotation == 0
    assert first_page.has_text_layer is True
    assert first_page.raw_text == "First page"
    assert first_page.word_count == 2
    assert second_page.has_text_layer is False
    assert second_page.raw_text == ""
    assert second_page.status == "no_text"


def test_ingest_pdf_accepts_registered_document_id(tmp_path: Path) -> None:
    config = prepare_test_config(tmp_path)
    pdf_path = create_pdf(tmp_path / "registered.pdf", ["Plain text"])

    first_result = ingest_pdf(config, str(pdf_path))
    second_result = ingest_pdf(config, first_result.document_id)

    assert second_result.document_id == first_result.document_id
    assert second_result.page_count == 1
    assert second_result.text_layer_page_count == 1


def test_ingest_pdf_creates_page_manifest(tmp_path: Path) -> None:
    config = prepare_test_config(tmp_path)
    pdf_path = create_pdf(tmp_path / "manifest.pdf", ["One", "Two"])

    result = ingest_pdf(config, str(pdf_path))
    manifest = json.loads(result.page_manifest_path.read_text(encoding="utf-8"))

    assert manifest["document_id"] == result.document_id
    assert manifest["original_name"] == "manifest.pdf"
    assert manifest["page_count"] == 2
    assert manifest["text_layer_page_count"] == 2
    assert len(manifest["pages"]) == 2
    assert manifest["pages"][0]["raw_text"] == "One"


def test_ingest_pdf_handles_pdf_without_text_layer(tmp_path: Path) -> None:
    config = prepare_test_config(tmp_path)
    pdf_path = create_pdf(tmp_path / "blank.pdf", [None])

    result = ingest_pdf(config, str(pdf_path))

    assert result.page_count == 1
    assert result.text_layer_page_count == 0
    assert result.pages[0].has_text_layer is False
    assert result.pages[0].char_count == 0
    assert result.pages[0].word_count == 0


def test_ingest_pdf_raises_for_non_pdf_file(tmp_path: Path) -> None:
    config = prepare_test_config(tmp_path)
    text_path = tmp_path / "notes.txt"
    text_path.write_text("sample", encoding="utf-8")

    with pytest.raises(PdfIngestionError, match="только PDF"):
        ingest_pdf(config, str(text_path))


def test_ingest_pdf_raises_for_broken_pdf(tmp_path: Path) -> None:
    config = prepare_test_config(tmp_path)
    pdf_path = tmp_path / "broken.pdf"
    pdf_path.write_bytes(b"%PDF-1.7\nbroken\n")

    with pytest.raises(PdfIngestionError, match="PDF поврежден"):
        ingest_pdf(config, str(pdf_path))


def prepare_test_config(tmp_path: Path):
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        "\n".join(
            [
                "project_name: docmanage-test",
                "run_mode: test",
                "data_dir: data",
                "artifacts_dir: artifacts",
                "temp_dir: tmp",
                "log_level: INFO",
            ]
        ),
        encoding="utf-8",
    )

    config = load_config(config_path)
    prepare_directories(config)
    return config
