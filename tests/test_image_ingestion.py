import json
from pathlib import Path

import pytest
from PIL import Image

from docmanage.config import load_config, prepare_directories
from docmanage.image_ingestion import ImageIngestionError, ingest_image
from tests.image_helpers import create_image


def test_ingest_image_processes_png(tmp_path: Path) -> None:
    config = prepare_test_config(tmp_path)
    image_path = create_image(
        tmp_path / "scan.png",
        size=(320, 480),
        mode="L",
        color=180,
    )

    result = ingest_image(config, str(image_path))

    assert result.document_id.startswith("doc_")
    assert result.page_count == 1
    assert result.page_manifest_path.exists()
    assert result.normalized_image_path.exists()

    page = result.pages[0]
    assert page.document_id == result.document_id
    assert page.page_id == f"{result.document_id}_page_0001"
    assert page.page_number == 1
    assert page.width == 320
    assert page.height == 480
    assert page.image_mode == "L"
    assert page.color_space == "grayscale"
    assert page.has_alpha is False
    assert page.file_format == "png"
    assert page.source_path == str(image_path.resolve())
    assert page.normalized_image_path == str(result.normalized_image_path)
    assert page.status == "image_loaded"


def test_ingest_image_processes_jpeg(tmp_path: Path) -> None:
    config = prepare_test_config(tmp_path)
    image_path = create_image(
        tmp_path / "photo.jpg",
        size=(640, 480),
        mode="RGB",
        color=(220, 210, 200),
    )

    result = ingest_image(config, str(image_path))

    assert result.page_count == 1
    assert result.pages[0].file_format == "jpeg"
    assert result.pages[0].image_mode == "RGB"
    assert result.pages[0].color_space == "rgb"


def test_ingest_image_accepts_registered_document_id(tmp_path: Path) -> None:
    config = prepare_test_config(tmp_path)
    image_path = create_image(tmp_path / "registered.jpeg")

    first_result = ingest_image(config, str(image_path))
    second_result = ingest_image(config, first_result.document_id)

    assert second_result.document_id == first_result.document_id
    assert second_result.pages[0].source_path == str(image_path.resolve())


def test_ingest_image_creates_page_manifest_and_normalized_copy(
    tmp_path: Path,
) -> None:
    config = prepare_test_config(tmp_path)
    image_path = create_image(
        tmp_path / "page.png",
        size=(200, 300),
        mode="RGBA",
        color=(100, 120, 140, 255),
    )

    result = ingest_image(config, str(image_path))
    manifest = json.loads(result.page_manifest_path.read_text(encoding="utf-8"))

    assert manifest["document_id"] == result.document_id
    assert manifest["original_name"] == "page.png"
    assert manifest["page_count"] == 1
    assert len(manifest["pages"]) == 1
    assert manifest["pages"][0]["normalized_image_path"] == str(result.normalized_image_path)

    with Image.open(result.normalized_image_path) as normalized_image:
        assert normalized_image.format == "PNG"
        assert normalized_image.size == (200, 300)


def test_ingest_image_raises_for_corrupted_file(tmp_path: Path) -> None:
    config = prepare_test_config(tmp_path)
    image_path = tmp_path / "broken.png"
    image_path.write_bytes(b"not-a-real-image")

    with pytest.raises(ImageIngestionError, match="не читается как изображение"):
        ingest_image(config, str(image_path))


def test_ingest_image_raises_for_unsupported_extension(tmp_path: Path) -> None:
    config = prepare_test_config(tmp_path)
    image_path = tmp_path / "scan.bmp"
    image_path.write_bytes(b"sample")

    with pytest.raises(ImageIngestionError, match="PNG, JPG и JPEG"):
        ingest_image(config, str(image_path))


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
