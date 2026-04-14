import json
from pathlib import Path

import pytest

from docmanage.config import load_config, prepare_directories
from docmanage.documents import (
    DocumentRegistrationError,
    register_documents,
)


def test_register_documents_creates_manifest_and_record(tmp_path: Path) -> None:
    config = prepare_test_config(tmp_path)
    document_path = create_file(tmp_path / "invoice.pdf", b"%PDF-1.7\nsample\n")

    registered_documents, manifest_path = register_documents(config, [str(document_path)])

    assert len(registered_documents) == 1
    document = registered_documents[0]
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))

    assert manifest_path.exists()
    assert manifest["project_name"] == "docmanage-test"
    assert len(manifest["documents"]) == 1
    assert document.document_id.startswith("doc_")
    assert len(document.document_id) == 16
    assert document.original_name == "invoice.pdf"
    assert document.original_path == str(document_path)
    assert document.absolute_path == str(document_path.resolve())
    assert document.extension == ".pdf"
    assert document.document_kind == "pdf"
    assert document.file_size_bytes == document_path.stat().st_size
    assert len(document.sha256) == 64
    assert document.status == "registered"


def test_register_documents_adds_multiple_files(tmp_path: Path) -> None:
    config = prepare_test_config(tmp_path)
    first_path = create_file(tmp_path / "scan.png", b"png-data")
    second_path = create_file(tmp_path / "table.csv", b"a,b\n1,2\n")

    registered_documents, manifest_path = register_documents(
        config,
        [str(first_path), str(second_path)],
    )

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))

    assert len(registered_documents) == 2
    assert len(manifest["documents"]) == 2
    assert [document.document_kind for document in registered_documents] == [
        "image",
        "spreadsheet",
    ]


def test_register_documents_updates_existing_manifest(tmp_path: Path) -> None:
    config = prepare_test_config(tmp_path)
    first_path = create_file(tmp_path / "scan.png", b"png-data")
    second_path = create_file(tmp_path / "table.csv", b"a,b\n1,2\n")

    register_documents(config, [str(first_path)])
    register_documents(config, [str(second_path)])

    manifest = json.loads(config.manifest_path.read_text(encoding="utf-8"))

    assert len(manifest["documents"]) == 2
    assert [document["original_name"] for document in manifest["documents"]] == [
        "scan.png",
        "table.csv",
    ]


def test_register_documents_raises_for_missing_file(tmp_path: Path) -> None:
    config = prepare_test_config(tmp_path)

    with pytest.raises(DocumentRegistrationError, match="Файл не найден"):
        register_documents(config, [str(tmp_path / "missing.pdf")])


def test_register_documents_raises_for_empty_file_list(tmp_path: Path) -> None:
    config = prepare_test_config(tmp_path)

    with pytest.raises(DocumentRegistrationError, match="хотя бы один файл"):
        register_documents(config, [])


def test_register_documents_raises_for_unsupported_extension(tmp_path: Path) -> None:
    config = prepare_test_config(tmp_path)
    document_path = create_file(tmp_path / "notes.txt", b"sample")

    with pytest.raises(DocumentRegistrationError, match="Неподдерживаемое расширение"):
        register_documents(config, [str(document_path)])


def test_register_documents_raises_for_broken_manifest(tmp_path: Path) -> None:
    config = prepare_test_config(tmp_path)
    config.manifest_path.write_text("{broken", encoding="utf-8")
    document_path = create_file(tmp_path / "invoice.pdf", b"%PDF-1.7\nsample\n")

    with pytest.raises(DocumentRegistrationError, match="Manifest поврежден"):
        register_documents(config, [str(document_path)])


def test_register_documents_raises_for_duplicate_file(tmp_path: Path) -> None:
    config = prepare_test_config(tmp_path)
    document_path = create_file(tmp_path / "invoice.pdf", b"%PDF-1.7\nsample\n")

    register_documents(config, [str(document_path)])

    with pytest.raises(DocumentRegistrationError, match="уже зарегистрирован"):
        register_documents(config, [str(document_path)])


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


def create_file(path: Path, content: bytes) -> Path:
    path.write_bytes(content)
    return path
