import json
from pathlib import Path

import pytest

from docmanage.ocr_line_check import OcrLineCheckError, run_line_folder_check
from docmanage.ocr_inference import OcrInferenceResult
from tests.image_helpers import create_image


def test_line_folder_check_saves_predictions_without_ground_truth(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    images_dir = prepare_line_images(tmp_path)
    checkpoint_path = tmp_path / "best_model.pt"
    checkpoint_path.write_bytes(b"checkpoint")
    report_path = tmp_path / "report.json"
    monkeypatch.setattr(
        "docmanage.ocr_line_check.run_ocr_line_inference",
        fake_inference,
    )

    result = run_line_folder_check(
        images_dir=images_dir,
        checkpoint_path=checkpoint_path,
        output_path=report_path,
    )

    assert result.image_count == 2
    assert result.has_ground_truth is False
    assert result.average_cer is None
    assert result.exact_match_rate is None
    assert result.report_path.exists()

    payload = json.loads(report_path.read_text(encoding="utf-8"))
    assert payload["has_ground_truth"] is False
    assert payload["average_cer"] is None
    assert payload["predictions"][0]["target"] is None


def test_line_folder_check_counts_metrics_with_ground_truth(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    images_dir = prepare_line_images(tmp_path)
    checkpoint_path = tmp_path / "best_model.pt"
    checkpoint_path.write_bytes(b"checkpoint")
    ground_truth_path = tmp_path / "truth.jsonl"
    ground_truth_path.write_text(
        "\n".join(
            [
                json.dumps({"image": "line_001.png", "text": "текст"}, ensure_ascii=False),
                json.dumps({"image": "line_002.jpg", "text": "текст"}, ensure_ascii=False),
            ]
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr(
        "docmanage.ocr_line_check.run_ocr_line_inference",
        fake_inference,
    )

    result = run_line_folder_check(
        images_dir=images_dir,
        checkpoint_path=checkpoint_path,
        output_path=tmp_path / "report.json",
        ground_truth_path=ground_truth_path,
    )

    assert result.has_ground_truth is True
    assert result.average_cer == 0
    assert result.exact_match_rate == 1
    assert all(record.target == "текст" for record in result.records)


def test_line_folder_check_raises_for_empty_folder(tmp_path: Path) -> None:
    checkpoint_path = tmp_path / "best_model.pt"
    checkpoint_path.write_bytes(b"checkpoint")

    with pytest.raises(OcrLineCheckError, match="В папке нет изображений"):
        run_line_folder_check(
            images_dir=tmp_path,
            checkpoint_path=checkpoint_path,
            output_path=tmp_path / "report.json",
        )


def test_line_folder_check_raises_for_missing_ground_truth(tmp_path: Path) -> None:
    images_dir = prepare_line_images(tmp_path)
    checkpoint_path = tmp_path / "best_model.pt"
    checkpoint_path.write_bytes(b"checkpoint")

    with pytest.raises(OcrLineCheckError, match="Ground truth не найден"):
        run_line_folder_check(
            images_dir=images_dir,
            checkpoint_path=checkpoint_path,
            output_path=tmp_path / "report.json",
            ground_truth_path=tmp_path / "missing.jsonl",
        )


def test_line_folder_check_raises_when_ground_truth_misses_image(tmp_path: Path) -> None:
    images_dir = prepare_line_images(tmp_path)
    checkpoint_path = tmp_path / "best_model.pt"
    checkpoint_path.write_bytes(b"checkpoint")
    ground_truth_path = tmp_path / "truth.jsonl"
    ground_truth_path.write_text(
        json.dumps({"image": "line_001.png", "text": "текст"}, ensure_ascii=False),
        encoding="utf-8",
    )

    with pytest.raises(OcrLineCheckError, match="Не нашел разметку"):
        run_line_folder_check(
            images_dir=images_dir,
            checkpoint_path=checkpoint_path,
            output_path=tmp_path / "report.json",
            ground_truth_path=ground_truth_path,
        )


def prepare_line_images(tmp_path: Path) -> Path:
    images_dir = tmp_path / "lines"
    images_dir.mkdir()
    create_image(images_dir / "line_002.jpg", size=(180, 42), mode="L", color=245)
    create_image(images_dir / "line_001.png", size=(160, 38), mode="L", color=245)
    return images_dir


def fake_inference(image_path: Path, checkpoint_path: Path, **_kwargs) -> OcrInferenceResult:
    return OcrInferenceResult(
        image_path=Path(image_path).resolve(),
        checkpoint_path=Path(checkpoint_path).resolve(),
        prediction="текст",
        output_path=None,
        device="cpu",
    )
