import json
from pathlib import Path

import pytest
from PIL import Image

from docmanage.config import load_config, prepare_directories
from docmanage.image_preprocessing import ImagePreprocessError, preprocess_image
from tests.image_helpers import create_document_like_image, create_image


def test_preprocess_image_creates_result_and_metadata(tmp_path: Path) -> None:
    config = prepare_test_config(tmp_path)
    image_path = create_document_like_image(tmp_path / "scan.png")

    result = preprocess_image(config, str(image_path))
    metadata = json.loads(result.metadata_path.read_text(encoding="utf-8"))

    assert result.document_id.startswith("doc_")
    assert result.preprocessed_image_path.exists()
    assert result.metadata_path.exists()
    assert metadata["document_id"] == result.document_id
    assert metadata["page_id"] == f"{result.document_id}_page_0001"
    assert metadata["grayscale_applied"] is True
    assert metadata["denoise_applied"] is True
    assert metadata["threshold_applied"] is True
    assert metadata["preprocessed_image_path"] == str(result.preprocessed_image_path)
    assert "threshold" in metadata["steps"]


def test_preprocess_image_works_with_grayscale_input(tmp_path: Path) -> None:
    config = prepare_test_config(tmp_path)
    image_path = create_image(tmp_path / "gray.png", mode="L", color=180)

    result = preprocess_image(config, str(image_path))

    assert result.metadata.grayscale_applied is False
    with Image.open(result.preprocessed_image_path) as image:
        assert image.mode == "L"


def test_preprocess_image_makes_binary_output(tmp_path: Path) -> None:
    config = prepare_test_config(tmp_path)
    image_path = create_document_like_image(tmp_path / "binary.png")

    result = preprocess_image(config, str(image_path))

    with Image.open(result.preprocessed_image_path) as image:
        values = set(image.convert("L").getdata())

    assert values.issubset({0, 255})


def test_preprocess_image_accepts_document_id(tmp_path: Path) -> None:
    config = prepare_test_config(tmp_path)
    image_path = create_document_like_image(tmp_path / "registered.jpg")

    first_result = preprocess_image(config, str(image_path))
    second_result = preprocess_image(config, first_result.document_id)

    assert second_result.document_id == first_result.document_id
    assert second_result.metadata.page_id == first_result.metadata.page_id


def test_preprocess_image_raises_for_corrupted_image(tmp_path: Path) -> None:
    config = prepare_test_config(tmp_path)
    image_path = tmp_path / "broken.png"
    image_path.write_bytes(b"broken")

    with pytest.raises(ImagePreprocessError, match="не читается как изображение"):
        preprocess_image(config, str(image_path))


def test_preprocess_image_raises_for_unsupported_extension(tmp_path: Path) -> None:
    config = prepare_test_config(tmp_path)
    image_path = tmp_path / "scan.bmp"
    image_path.write_bytes(b"sample")

    with pytest.raises(ImagePreprocessError, match="PNG, JPG и JPEG"):
        preprocess_image(config, str(image_path))


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
