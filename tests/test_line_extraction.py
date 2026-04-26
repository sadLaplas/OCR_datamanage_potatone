import json
from pathlib import Path

import pytest
from PIL import Image

from docmanage.config import load_config, prepare_directories
from docmanage.line_extraction import (
    LineExtractionError,
    LineExtractionParams,
    extract_page_lines,
)
from tests.image_helpers import create_document_like_image, create_image


def test_extract_page_lines_saves_crops_manifest_and_preview(tmp_path: Path) -> None:
    config = prepare_test_config(tmp_path)
    image_path = create_document_like_image(tmp_path / "page.png")
    output_dir = tmp_path / "lines"

    result = extract_page_lines(
        config,
        image_path,
        output_dir=output_dir,
        params=LineExtractionParams(min_line_height=6, padding=3),
    )
    manifest = json.loads(result.manifest_path.read_text(encoding="utf-8"))

    assert result.line_count == 8
    assert result.lines_dir == output_dir.resolve()
    assert result.manifest_path.exists()
    assert result.preview_path.exists()
    assert manifest["line_count"] == 8
    assert manifest["source_image"] == str(image_path.resolve())
    manifest_text = json.dumps(manifest, ensure_ascii=False)
    assert "recognized_text" not in manifest_text
    assert "prediction" not in manifest_text

    y_positions = [line.bbox.y_min for line in result.lines]
    assert y_positions == sorted(y_positions)

    for line in result.lines:
        assert line.image_path.exists()
        assert line.bbox.width > 0
        assert line.bbox.height > 0


def test_extract_page_lines_raises_for_blank_image(tmp_path: Path) -> None:
    config = prepare_test_config(tmp_path)
    image_path = create_image(tmp_path / "blank.png", size=(160, 120), color=(255, 255, 255))

    with pytest.raises(LineExtractionError, match="Строки не найдены"):
        extract_page_lines(config, image_path, output_dir=tmp_path / "lines")


def test_extract_page_lines_raises_for_unsupported_format(tmp_path: Path) -> None:
    config = prepare_test_config(tmp_path)
    image_path = tmp_path / "page.bmp"
    image = Image.new("RGB", (160, 120), (255, 255, 255))
    image.save(image_path, format="BMP")
    image.close()

    with pytest.raises(LineExtractionError, match="PNG, JPG и JPEG"):
        extract_page_lines(config, image_path, output_dir=tmp_path / "lines")


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
