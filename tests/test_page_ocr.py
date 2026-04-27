import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from docmanage.page_ocr import PageOcrError, run_page_ocr_from_manifest
from tests.image_helpers import create_image


def test_page_ocr_reads_manifest_and_saves_text_files(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    first_image = create_image(tmp_path / "line_bottom.png", size=(80, 24), mode="L", color=230)
    second_image = create_image(tmp_path / "line_top.png", size=(80, 24), mode="L", color=230)
    manifest_path = write_manifest(
        tmp_path,
        [
            {"line_id": "line_0002", "image_path": str(first_image), "bbox": [0, 40, 80, 60]},
            {"line_id": "line_0001", "image_path": str(second_image), "bbox": [0, 10, 80, 30]},
        ],
    )
    checkpoint_path = tmp_path / "model.pt"
    checkpoint_path.write_bytes(b"fake")
    patch_fake_inference(monkeypatch, checkpoint_path)

    result = run_page_ocr_from_manifest(
        manifest_path=manifest_path,
        checkpoint_path=checkpoint_path,
        output_dir=tmp_path / "out",
        show_progress=False,
    )
    payload = json.loads(result.json_path.read_text(encoding="utf-8"))

    assert result.line_count == 2
    assert result.recognized_line_count == 2
    assert result.failed_line_count == 0
    assert [line.line_id for line in result.lines] == ["line_0001", "line_0002"]
    assert payload["plain_text"] == "line_top\nline_bottom"
    assert payload["failed_lines"] == []
    assert result.text_path.read_text(encoding="utf-8") == "line_top\nline_bottom\n"


def test_page_ocr_writes_failed_lines_and_continues(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    existing_image = create_image(tmp_path / "line_ok.png", size=(80, 24), mode="L", color=230)
    missing_image = tmp_path / "line_missing.png"
    manifest_path = write_manifest(
        tmp_path,
        [
            {"line_id": "line_0001", "image_path": str(existing_image), "bbox": [0, 10, 80, 30]},
            {"line_id": "line_0002", "image_path": str(missing_image), "bbox": [0, 40, 80, 60]},
        ],
    )
    checkpoint_path = tmp_path / "model.pt"
    checkpoint_path.write_bytes(b"fake")
    patch_fake_inference(monkeypatch, checkpoint_path)

    result = run_page_ocr_from_manifest(
        manifest_path=manifest_path,
        checkpoint_path=checkpoint_path,
        output_dir=tmp_path / "out",
        show_progress=False,
    )
    payload = json.loads(result.json_path.read_text(encoding="utf-8"))

    assert result.recognized_line_count == 1
    assert result.failed_line_count == 1
    assert payload["plain_text"] == "line_ok"
    assert payload["failed_lines"][0]["line_id"] == "line_0002"
    assert payload["failed_lines"][0]["error"] == "Не нашел изображение строки"


def test_page_ocr_raises_for_missing_manifest(tmp_path: Path) -> None:
    with pytest.raises(PageOcrError, match="Manifest не найден"):
        run_page_ocr_from_manifest(
            manifest_path=tmp_path / "missing.json",
            checkpoint_path=tmp_path / "model.pt",
            show_progress=False,
        )


def test_page_ocr_raises_for_empty_lines(tmp_path: Path) -> None:
    manifest_path = write_manifest(tmp_path, [])
    checkpoint_path = tmp_path / "model.pt"
    checkpoint_path.write_bytes(b"fake")

    with pytest.raises(PageOcrError, match="В manifest нет строк"):
        run_page_ocr_from_manifest(
            manifest_path=manifest_path,
            checkpoint_path=checkpoint_path,
            show_progress=False,
        )


def test_page_ocr_raises_for_missing_checkpoint(tmp_path: Path) -> None:
    image_path = create_image(tmp_path / "line.png", size=(80, 24), mode="L", color=230)
    manifest_path = write_manifest(
        tmp_path,
        [{"line_id": "line_0001", "image_path": str(image_path), "bbox": [0, 10, 80, 30]}],
    )

    with pytest.raises(PageOcrError, match="Не нашел checkpoint"):
        run_page_ocr_from_manifest(
            manifest_path=manifest_path,
            checkpoint_path=tmp_path / "missing.pt",
            show_progress=False,
        )


def write_manifest(tmp_path: Path, lines: list[dict[str, object]]) -> Path:
    manifest_path = tmp_path / "line_manifest.json"
    payload = {
        "source_image": str(tmp_path / "page.png"),
        "line_count": len(lines),
        "lines": lines,
    }
    manifest_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return manifest_path


def patch_fake_inference(monkeypatch: pytest.MonkeyPatch, checkpoint_path: Path) -> None:
    monkeypatch.setattr(
        "docmanage.page_ocr.load_ocr_inference_session",
        lambda *_args, **_kwargs: SimpleNamespace(checkpoint_path=checkpoint_path.resolve()),
    )
    monkeypatch.setattr(
        "docmanage.page_ocr.run_ocr_line_inference_with_session",
        lambda image_path, session: SimpleNamespace(prediction=Path(image_path).stem),
    )
