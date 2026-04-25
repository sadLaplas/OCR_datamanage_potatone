from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

from .ocr_evaluation import character_error_rate
from .ocr_inference import (
    OcrInferenceError,
    run_ocr_line_inference,
)

SUPPORTED_LINE_IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg"}


class OcrLineCheckError(ValueError):
    pass


@dataclass(slots=True, frozen=True)
class LineCheckRecord:
    image_path: Path
    prediction: str
    target: str | None
    cer: float | None
    exact_match: bool | None

    def to_dict(self) -> dict[str, object]:
        return {
            "image_path": str(self.image_path),
            "prediction": self.prediction,
            "target": self.target,
            "cer": self.cer,
            "exact_match": self.exact_match,
        }


@dataclass(slots=True, frozen=True)
class LineCheckResult:
    images_dir: Path
    checkpoint_path: Path
    report_path: Path
    image_count: int
    has_ground_truth: bool
    average_cer: float | None
    exact_match_rate: float | None
    records: tuple[LineCheckRecord, ...]


def run_line_folder_check(
    images_dir: str | Path,
    checkpoint_path: str | Path,
    output_path: str | Path,
    ground_truth_path: str | Path | None = None,
    device_name: str = "cpu",
) -> LineCheckResult:
    actual_images_dir = resolve_images_dir(images_dir)
    image_paths = find_line_images(actual_images_dir)
    actual_checkpoint_path = resolve_checkpoint_path(checkpoint_path)
    ground_truth = load_ground_truth(ground_truth_path, actual_images_dir, image_paths)

    records: list[LineCheckRecord] = []
    for image_path in image_paths:
        try:
            prediction_result = run_ocr_line_inference(
                image_path=image_path,
                checkpoint_path=actual_checkpoint_path,
                device_name=device_name,
            )
        except OcrInferenceError as error:
            raise OcrLineCheckError(str(error)) from error

        target = ground_truth.get(image_path.name) if ground_truth is not None else None
        cer = character_error_rate(prediction_result.prediction, target) if target is not None else None
        exact_match = prediction_result.prediction == target if target is not None else None
        records.append(
            LineCheckRecord(
                image_path=image_path,
                prediction=prediction_result.prediction,
                target=target,
                cer=cer,
                exact_match=exact_match,
            )
        )

    average_cer = None
    exact_match_rate = None
    if ground_truth is not None:
        average_cer = sum(record.cer or 0.0 for record in records) / len(records)
        exact_match_rate = sum(bool(record.exact_match) for record in records) / len(records)

    actual_output_path = save_line_check_report(
        output_path=output_path,
        images_dir=actual_images_dir,
        checkpoint_path=actual_checkpoint_path,
        records=records,
        average_cer=average_cer,
        exact_match_rate=exact_match_rate,
        has_ground_truth=ground_truth is not None,
    )

    return LineCheckResult(
        images_dir=actual_images_dir,
        checkpoint_path=actual_checkpoint_path,
        report_path=actual_output_path,
        image_count=len(image_paths),
        has_ground_truth=ground_truth is not None,
        average_cer=average_cer,
        exact_match_rate=exact_match_rate,
        records=tuple(records),
    )


def resolve_images_dir(images_dir: str | Path) -> Path:
    path = Path(images_dir).expanduser().resolve()
    if not path.exists():
        raise OcrLineCheckError(f"Папка не найдена: {path}")
    if not path.is_dir():
        raise OcrLineCheckError(f"Путь должен быть папкой: {path}")
    return path


def find_line_images(images_dir: Path) -> list[Path]:
    image_paths = [
        path
        for path in images_dir.iterdir()
        if path.is_file() and path.suffix.lower() in SUPPORTED_LINE_IMAGE_EXTENSIONS
    ]
    image_paths.sort(key=lambda path: path.name)

    if not image_paths:
        raise OcrLineCheckError("В папке нет изображений.")
    return image_paths


def resolve_checkpoint_path(checkpoint_path: str | Path) -> Path:
    path = Path(checkpoint_path).expanduser().resolve()
    if not path.exists():
        raise OcrLineCheckError(f"Не нашел checkpoint: {path}")
    if not path.is_file():
        raise OcrLineCheckError(f"Путь к checkpoint указывает на папку: {path}")
    return path


def load_ground_truth(
    ground_truth_path: str | Path | None,
    images_dir: Path,
    image_paths: list[Path],
) -> dict[str, str] | None:
    if ground_truth_path is None:
        return None

    path = Path(ground_truth_path).expanduser().resolve()
    if not path.exists():
        raise OcrLineCheckError(f"Ground truth не найден: {path}")
    if not path.is_file():
        raise OcrLineCheckError(f"Ground truth должен быть файлом: {path}")

    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except OSError as error:
        raise OcrLineCheckError("Не получилось прочитать разметку.") from error

    image_names = {image_path.name for image_path in image_paths}
    ground_truth: dict[str, str] = {}
    for line in lines:
        if not line.strip():
            continue

        try:
            record = json.loads(line)
        except json.JSONDecodeError as error:
            raise OcrLineCheckError("Разметка повреждена.") from error

        image_name = str(record.get("image", "")).strip()
        text = record.get("text")
        if not image_name or not isinstance(text, str):
            raise OcrLineCheckError("В разметке нужны поля image и text.")
        if image_name not in image_names:
            raise OcrLineCheckError(f"В разметке есть лишнее изображение: {image_name}")
        ground_truth[image_name] = text

    for image_path in image_paths:
        if image_path.name not in ground_truth:
            raise OcrLineCheckError(f"Не нашел разметку для {image_path.name}")

    return ground_truth


def save_line_check_report(
    output_path: str | Path,
    images_dir: Path,
    checkpoint_path: Path,
    records: list[LineCheckRecord],
    average_cer: float | None,
    exact_match_rate: float | None,
    has_ground_truth: bool,
) -> Path:
    actual_output_path = Path(output_path).expanduser().resolve()
    payload = {
        "images_dir": str(images_dir),
        "checkpoint_path": str(checkpoint_path),
        "image_count": len(records),
        "has_ground_truth": has_ground_truth,
        "average_cer": average_cer,
        "exact_match_rate": exact_match_rate,
        "predictions": [record.to_dict() for record in records],
    }

    try:
        actual_output_path.parent.mkdir(parents=True, exist_ok=True)
        actual_output_path.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
    except OSError as error:
        raise OcrLineCheckError("Не удалось сохранить отчет.") from error

    return actual_output_path
