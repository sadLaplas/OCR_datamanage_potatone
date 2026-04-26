from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime, timezone
from hashlib import sha256
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageOps, UnidentifiedImageError

from .config import AppConfig
from .image_ingestion import SUPPORTED_IMAGE_EXTENSIONS, SUPPORTED_IMAGE_FORMATS
from .image_preprocessing import apply_threshold, calculate_otsu_threshold


class LineExtractionError(ValueError):
    pass


@dataclass(slots=True, frozen=True)
class LineExtractionParams:
    min_line_height: int = 6
    padding: int = 4
    threshold: int | None = None


@dataclass(slots=True, frozen=True)
class LineBox:
    x_min: int
    y_min: int
    x_max: int
    y_max: int

    @property
    def width(self) -> int:
        return self.x_max - self.x_min

    @property
    def height(self) -> int:
        return self.y_max - self.y_min

    def to_list(self) -> list[int]:
        return [self.x_min, self.y_min, self.x_max, self.y_max]


@dataclass(slots=True, frozen=True)
class ExtractedLine:
    line_id: str
    image_path: Path
    bbox: LineBox

    def to_dict(self) -> dict[str, object]:
        return {
            "line_id": self.line_id,
            "image_path": str(self.image_path),
            "bbox": self.bbox.to_list(),
            "height": self.bbox.height,
            "width": self.bbox.width,
        }


@dataclass(slots=True, frozen=True)
class LineExtractionResult:
    source_image_path: Path
    document_id: str
    lines_dir: Path
    manifest_path: Path
    preview_path: Path
    line_count: int
    lines: tuple[ExtractedLine, ...]
    threshold_value: int
    skipped_regions_count: int


def extract_page_lines(
    config: AppConfig,
    source: str | Path,
    output_dir: str | Path | None = None,
    params: LineExtractionParams | None = None,
) -> LineExtractionResult:
    actual_params = params or LineExtractionParams()
    validate_params(actual_params)

    image_path = resolve_page_image_path(source)
    original_image, binary_image, threshold_value = load_images_for_lines(
        image_path,
        actual_params,
    )

    try:
        line_boxes, skipped_regions_count = find_line_boxes(
            binary_image,
            min_line_height=actual_params.min_line_height,
            padding=actual_params.padding,
        )
        if not line_boxes:
            raise LineExtractionError("Строки не найдены")

        document_id = guess_document_id(image_path)
        actual_output_dir = resolve_lines_dir(config, document_id, output_dir)
        lines = save_line_crops(original_image, line_boxes, actual_output_dir)
        manifest_path = actual_output_dir / "line_manifest.json"
        preview_path = actual_output_dir / "lines_preview.png"
        save_preview(original_image, line_boxes, preview_path)
        save_line_manifest(
            manifest_path=manifest_path,
            source_image_path=image_path,
            document_id=document_id,
            lines=lines,
            preview_path=preview_path,
            threshold_value=threshold_value,
            params=actual_params,
            skipped_regions_count=skipped_regions_count,
        )
    finally:
        original_image.close()
        binary_image.close()

    return LineExtractionResult(
        source_image_path=image_path,
        document_id=document_id,
        lines_dir=actual_output_dir,
        manifest_path=manifest_path,
        preview_path=preview_path,
        line_count=len(lines),
        lines=tuple(lines),
        threshold_value=threshold_value,
        skipped_regions_count=skipped_regions_count,
    )


def validate_params(params: LineExtractionParams) -> None:
    if params.min_line_height <= 0:
        raise LineExtractionError("min-line-height должен быть больше 0")
    if params.padding < 0:
        raise LineExtractionError("padding не может быть меньше 0")
    if params.threshold is not None and not 0 <= params.threshold <= 255:
        raise LineExtractionError("threshold должен быть от 0 до 255")


def resolve_page_image_path(source: str | Path) -> Path:
    path = Path(source).expanduser()

    if not path.exists():
        raise LineExtractionError("Файл не найден")
    if not path.is_file():
        raise LineExtractionError("Путь указывает на папку")
    if path.suffix.lower() not in SUPPORTED_IMAGE_EXTENSIONS:
        raise LineExtractionError("Поддерживаются только PNG, JPG и JPEG")

    try:
        return path.resolve()
    except OSError as error:
        raise LineExtractionError("Не удалось обработать путь") from error


def load_images_for_lines(
    image_path: Path,
    params: LineExtractionParams,
) -> tuple[Image.Image, Image.Image, int]:
    try:
        with Image.open(image_path) as source_image:
            source_format = (source_image.format or "").upper()
            if source_format not in SUPPORTED_IMAGE_FORMATS:
                raise LineExtractionError("Поддерживаются только PNG, JPG и JPEG")

            transposed_image = ImageOps.exif_transpose(source_image)
            original_image = transposed_image.convert("RGB")
    except LineExtractionError:
        raise
    except UnidentifiedImageError as error:
        raise LineExtractionError("Не получилось открыть изображение") from error
    except OSError as error:
        raise LineExtractionError("Изображение повреждено или не читается") from error

    if original_image.width < 20 or original_image.height < 20:
        original_image.close()
        raise LineExtractionError("Изображение слишком маленькое")

    grayscale_image = original_image.convert("L")
    if params.threshold is None:
        threshold_value = min(255, calculate_otsu_threshold(grayscale_image) + 1)
    else:
        threshold_value = params.threshold
    binary_image = apply_threshold(grayscale_image, threshold_value)
    grayscale_image.close()

    return original_image, binary_image, threshold_value


def find_line_boxes(
    binary_image: Image.Image,
    *,
    min_line_height: int,
    padding: int,
) -> tuple[list[LineBox], int]:
    pixels = np.asarray(binary_image)
    dark_mask = pixels < 128
    width = int(binary_image.width)
    height = int(binary_image.height)

    # Горизонтальная проекция показывает, где по высоте есть темные пиксели.
    row_dark_counts = dark_mask.sum(axis=1)
    min_dark_pixels = max(2, width // 100)
    raw_ranges = find_active_ranges(row_dark_counts, min_dark_pixels)

    # Буквы могут давать небольшие разрывы, поэтому близкие полосы склеиваются.
    merge_gap = max(2, min_line_height // 2)
    merged_ranges = merge_close_ranges(raw_ranges, merge_gap)

    line_boxes: list[LineBox] = []
    skipped_regions_count = 0
    for y_start, y_end in merged_ranges:
        if y_end - y_start + 1 < min_line_height:
            skipped_regions_count += 1
            continue

        # По найденной строке уточняем левую и правую границу текста.
        y_min = max(0, y_start - padding)
        y_max = min(height, y_end + 1 + padding)
        region_mask = dark_mask[y_start : y_end + 1, :]
        text_columns = np.flatnonzero(region_mask.any(axis=0))
        if len(text_columns) == 0:
            skipped_regions_count += 1
            continue

        x_min = max(0, int(text_columns[0]) - padding)
        x_max = min(width, int(text_columns[-1]) + 1 + padding)
        if x_max <= x_min or y_max <= y_min:
            skipped_regions_count += 1
            continue

        line_boxes.append(LineBox(x_min=x_min, y_min=y_min, x_max=x_max, y_max=y_max))

    line_boxes.sort(key=lambda box: box.y_min)
    return line_boxes, skipped_regions_count


def find_active_ranges(row_dark_counts: np.ndarray, min_dark_pixels: int) -> list[tuple[int, int]]:
    ranges: list[tuple[int, int]] = []
    range_start: int | None = None

    for row_index, dark_count in enumerate(row_dark_counts):
        if int(dark_count) >= min_dark_pixels:
            if range_start is None:
                range_start = row_index
            continue

        if range_start is not None:
            ranges.append((range_start, row_index - 1))
            range_start = None

    if range_start is not None:
        ranges.append((range_start, len(row_dark_counts) - 1))

    return ranges


def merge_close_ranges(ranges: list[tuple[int, int]], merge_gap: int) -> list[tuple[int, int]]:
    if not ranges:
        return []

    merged_ranges = [ranges[0]]
    for y_start, y_end in ranges[1:]:
        previous_start, previous_end = merged_ranges[-1]
        if y_start - previous_end <= merge_gap:
            merged_ranges[-1] = (previous_start, max(previous_end, y_end))
        else:
            merged_ranges.append((y_start, y_end))

    return merged_ranges


def save_line_crops(
    image: Image.Image,
    line_boxes: list[LineBox],
    output_dir: Path,
) -> list[ExtractedLine]:
    try:
        output_dir.mkdir(parents=True, exist_ok=True)
    except OSError as error:
        raise LineExtractionError("Не удалось сохранить строки") from error

    lines: list[ExtractedLine] = []
    for line_index, bbox in enumerate(line_boxes, start=1):
        line_id = f"line_{line_index:04d}"
        image_path = output_dir / f"{line_id}.png"
        crop = image.crop((bbox.x_min, bbox.y_min, bbox.x_max, bbox.y_max))

        # Сохраняем строки в том же порядке, в котором они идут сверху вниз.
        try:
            crop.save(image_path, format="PNG")
        except OSError as error:
            raise LineExtractionError("Не удалось сохранить строки") from error
        finally:
            crop.close()

        lines.append(ExtractedLine(line_id=line_id, image_path=image_path, bbox=bbox))

    return lines


def save_preview(image: Image.Image, line_boxes: list[LineBox], preview_path: Path) -> None:
    preview_image = image.copy()
    draw = ImageDraw.Draw(preview_image)

    for bbox in line_boxes:
        draw.rectangle(
            (bbox.x_min, bbox.y_min, bbox.x_max - 1, bbox.y_max - 1),
            outline=(220, 40, 40),
            width=2,
        )

    try:
        preview_path.parent.mkdir(parents=True, exist_ok=True)
        preview_image.save(preview_path, format="PNG")
    except OSError as error:
        raise LineExtractionError("Не удалось сохранить preview") from error
    finally:
        preview_image.close()


def save_line_manifest(
    *,
    manifest_path: Path,
    source_image_path: Path,
    document_id: str,
    lines: list[ExtractedLine],
    preview_path: Path,
    threshold_value: int,
    params: LineExtractionParams,
    skipped_regions_count: int,
) -> None:
    payload = {
        "source_image": str(source_image_path),
        "document_id": document_id,
        "created_at": current_timestamp(),
        "line_count": len(lines),
        "preview_path": str(preview_path),
        "parameters": {
            "threshold": threshold_value,
            "min_line_height": params.min_line_height,
            "padding": params.padding,
        },
        "skipped_regions_count": skipped_regions_count,
        "lines": [line.to_dict() for line in lines],
    }

    try:
        manifest_path.parent.mkdir(parents=True, exist_ok=True)
        manifest_path.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
    except OSError as error:
        raise LineExtractionError("Не удалось сохранить line_manifest") from error


def resolve_lines_dir(
    config: AppConfig,
    document_id: str,
    output_dir: str | Path | None,
) -> Path:
    if output_dir is None:
        return config.artifacts_dir / "documents" / document_id / "lines"

    path = Path(output_dir).expanduser().resolve()
    if path.exists() and not path.is_dir():
        raise LineExtractionError("Путь для строк указывает на файл")
    return path


def guess_document_id(image_path: Path) -> str:
    if image_path.parent.parent.name == "documents" and image_path.parent.name.startswith("doc_"):
        return image_path.parent.name

    try:
        digest = sha256(image_path.read_bytes()).hexdigest()
    except OSError as error:
        raise LineExtractionError("Не удалось прочитать изображение") from error

    return f"doc_{digest[:12]}"


def current_timestamp() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace(
        "+00:00", "Z"
    )
