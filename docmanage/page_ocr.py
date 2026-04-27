from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from tqdm.auto import tqdm

from .ocr_inference import (
    OcrInferenceError,
    load_ocr_inference_session,
    run_ocr_line_inference_with_session,
)


class PageOcrError(ValueError):
    pass


@dataclass(slots=True, frozen=True)
class ManifestLine:
    line_id: str
    image_path: Path
    bbox: list[int]


@dataclass(slots=True, frozen=True)
class PageTextLine:
    line_id: str
    image_path: Path
    bbox: list[int]
    text: str

    def to_dict(self) -> dict[str, object]:
        return {
            "line_id": self.line_id,
            "image_path": str(self.image_path),
            "bbox": self.bbox,
            "text": self.text,
        }


@dataclass(slots=True, frozen=True)
class FailedPageLine:
    line_id: str
    image_path: Path
    bbox: list[int]
    error: str

    def to_dict(self) -> dict[str, object]:
        return {
            "line_id": self.line_id,
            "image_path": str(self.image_path),
            "bbox": self.bbox,
            "error": self.error,
        }


@dataclass(slots=True, frozen=True)
class PageOcrResult:
    source_manifest: Path
    source_image: str
    checkpoint_path: Path
    json_path: Path
    text_path: Path
    line_count: int
    recognized_line_count: int
    failed_line_count: int
    lines: tuple[PageTextLine, ...]
    failed_lines: tuple[FailedPageLine, ...]

    @property
    def plain_text(self) -> str:
        return "\n".join(line.text for line in self.lines)


def run_page_ocr_from_manifest(
    manifest_path: str | Path,
    checkpoint_path: str | Path,
    output_dir: str | Path | None = None,
    device_name: str = "cpu",
    show_progress: bool = True,
) -> PageOcrResult:
    actual_manifest_path = resolve_manifest_path(manifest_path)
    manifest_payload = load_manifest_payload(actual_manifest_path)
    source_image = str(manifest_payload.get("source_image", ""))
    manifest_lines = read_manifest_lines(manifest_payload, actual_manifest_path)
    actual_output_dir = resolve_output_dir(actual_manifest_path, output_dir)

    try:
        session = load_ocr_inference_session(checkpoint_path, device_name=device_name)
    except OcrInferenceError as error:
        raise PageOcrError(str(error)) from error

    recognized_lines: list[PageTextLine] = []
    failed_lines: list[FailedPageLine] = []
    progress = tqdm(
        manifest_lines,
        desc="Распознаю строки",
        unit="строка",
        disable=not show_progress,
    )

    for line in progress:
        if not line.image_path.exists():
            failed_lines.append(
                FailedPageLine(
                    line_id=line.line_id,
                    image_path=line.image_path,
                    bbox=line.bbox,
                    error="Не нашел изображение строки",
                )
            )
            continue
        if not line.image_path.is_file():
            failed_lines.append(
                FailedPageLine(
                    line_id=line.line_id,
                    image_path=line.image_path,
                    bbox=line.bbox,
                    error="Путь к строке указывает на папку",
                )
            )
            continue

        try:
            inference_result = run_ocr_line_inference_with_session(
                image_path=line.image_path,
                session=session,
            )
        except OcrInferenceError as error:
            failed_lines.append(
                FailedPageLine(
                    line_id=line.line_id,
                    image_path=line.image_path,
                    bbox=line.bbox,
                    error=str(error),
                )
            )
            continue

        recognized_lines.append(
            PageTextLine(
                line_id=line.line_id,
                image_path=line.image_path,
                bbox=line.bbox,
                text=inference_result.prediction,
            )
        )

    json_path = actual_output_dir / "page_text.json"
    text_path = actual_output_dir / "page_text.txt"
    result = PageOcrResult(
        source_manifest=actual_manifest_path,
        source_image=source_image,
        checkpoint_path=session.checkpoint_path,
        json_path=json_path,
        text_path=text_path,
        line_count=len(manifest_lines),
        recognized_line_count=len(recognized_lines),
        failed_line_count=len(failed_lines),
        lines=tuple(recognized_lines),
        failed_lines=tuple(failed_lines),
    )

    save_page_text_json(result)
    save_page_text_txt(result)
    return result


def resolve_manifest_path(manifest_path: str | Path) -> Path:
    path = Path(manifest_path).expanduser()

    if not path.exists():
        raise PageOcrError("Manifest не найден")
    if not path.is_file():
        raise PageOcrError("Путь к manifest указывает на папку")

    try:
        return path.resolve()
    except OSError as error:
        raise PageOcrError("Не удалось обработать путь manifest") from error


def load_manifest_payload(manifest_path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as error:
        raise PageOcrError("Manifest поврежден") from error
    except OSError as error:
        raise PageOcrError("Не получилось прочитать manifest") from error

    if not isinstance(payload, dict):
        raise PageOcrError("Manifest поврежден")
    return payload


def read_manifest_lines(
    payload: dict[str, Any],
    manifest_path: Path,
) -> list[ManifestLine]:
    raw_lines = payload.get("lines")
    if not isinstance(raw_lines, list):
        raise PageOcrError("В manifest нет lines")
    if not raw_lines:
        raise PageOcrError("В manifest нет строк")

    lines: list[ManifestLine] = []
    for line_index, raw_line in enumerate(raw_lines, start=1):
        if not isinstance(raw_line, dict):
            raise PageOcrError("Строка в manifest повреждена")

        line_id = read_line_id(raw_line, line_index)
        image_path = read_line_image_path(raw_line, manifest_path)
        bbox = read_bbox(raw_line)
        lines.append(ManifestLine(line_id=line_id, image_path=image_path, bbox=bbox))

    lines.sort(key=lambda line: line.bbox[1])
    return lines


def read_line_id(raw_line: dict[str, Any], line_index: int) -> str:
    value = raw_line.get("line_id")
    if isinstance(value, str) and value.strip():
        return value.strip()
    return f"line_{line_index:04d}"


def read_line_image_path(raw_line: dict[str, Any], manifest_path: Path) -> Path:
    value = raw_line.get("image_path")
    if not isinstance(value, str) or not value.strip():
        raise PageOcrError("В строке нет image_path")

    path = Path(value).expanduser()
    if not path.is_absolute():
        path = manifest_path.parent / path
    try:
        return path.resolve()
    except OSError as error:
        raise PageOcrError("Не удалось обработать путь строки") from error


def read_bbox(raw_line: dict[str, Any]) -> list[int]:
    value = raw_line.get("bbox")
    if not isinstance(value, list) or len(value) != 4:
        raise PageOcrError("bbox строки поврежден")

    try:
        bbox = [int(item) for item in value]
    except (TypeError, ValueError) as error:
        raise PageOcrError("bbox строки поврежден") from error

    x_min, y_min, x_max, y_max = bbox
    if x_max <= x_min or y_max <= y_min:
        raise PageOcrError("bbox строки поврежден")
    return bbox


def resolve_output_dir(manifest_path: Path, output_dir: str | Path | None) -> Path:
    if output_dir is None:
        path = manifest_path.parent
    else:
        path = Path(output_dir).expanduser().resolve()

    if path.exists() and not path.is_dir():
        raise PageOcrError("Папка результата указывает на файл")

    try:
        path.mkdir(parents=True, exist_ok=True)
    except OSError as error:
        raise PageOcrError("Не удалось создать папку результата") from error

    return path


def save_page_text_json(result: PageOcrResult) -> None:
    payload = {
        "source_manifest": str(result.source_manifest),
        "source_image": result.source_image,
        "checkpoint_path": str(result.checkpoint_path),
        "line_count": result.line_count,
        "recognized_line_count": result.recognized_line_count,
        "failed_line_count": result.failed_line_count,
        "recognized_at": current_timestamp(),
        "lines": [line.to_dict() for line in result.lines],
        "failed_lines": [line.to_dict() for line in result.failed_lines],
        "plain_text": result.plain_text,
    }

    try:
        result.json_path.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
    except OSError as error:
        raise PageOcrError("Не удалось сохранить page_text.json") from error


def save_page_text_txt(result: PageOcrResult) -> None:
    try:
        result.text_path.write_text(result.plain_text + "\n", encoding="utf-8")
    except OSError as error:
        raise PageOcrError("Не удалось сохранить page_text.txt") from error


def current_timestamp() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace(
        "+00:00", "Z"
    )
