from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

from PIL import Image, ImageFilter, UnidentifiedImageError

from .config import AppConfig
from .image_ingestion import ImageIngestionError, ingest_image


class ImagePreprocessError(ValueError):
    pass


@dataclass(slots=True, frozen=True)
class ImagePreprocessMetadata:
    document_id: str
    page_id: str
    source_image_path: str
    input_image_path: str
    preprocessed_image_path: str
    steps: list[str]
    grayscale_applied: bool
    denoise_applied: bool
    threshold_applied: bool
    deskew_applied: bool
    threshold_value: int
    deskew_angle: float
    width: int
    height: int
    status: str

    def to_dict(self) -> dict[str, object]:
        return {
            "document_id": self.document_id,
            "page_id": self.page_id,
            "source_image_path": self.source_image_path,
            "input_image_path": self.input_image_path,
            "preprocessed_image_path": self.preprocessed_image_path,
            "steps": self.steps,
            "grayscale_applied": self.grayscale_applied,
            "denoise_applied": self.denoise_applied,
            "threshold_applied": self.threshold_applied,
            "deskew_applied": self.deskew_applied,
            "threshold_value": self.threshold_value,
            "deskew_angle": self.deskew_angle,
            "width": self.width,
            "height": self.height,
            "status": self.status,
        }


@dataclass(slots=True, frozen=True)
class ImagePreprocessResult:
    document_id: str
    original_name: str
    preprocessed_image_path: Path
    metadata_path: Path
    metadata: ImagePreprocessMetadata


def preprocess_image(config: AppConfig, source: str) -> ImagePreprocessResult:
    try:
        ingestion_result = ingest_image(config, source)
    except ImageIngestionError as error:
        raise ImagePreprocessError(str(error)) from error

    page = ingestion_result.pages[0]
    input_image_path = ingestion_result.normalized_image_path
    preprocessed_image_path = build_preprocessed_image_path(
        config, ingestion_result.document_id
    )
    metadata_path = build_metadata_path(config, ingestion_result.document_id)

    metadata = run_preprocessing(
        document_id=ingestion_result.document_id,
        page_id=page.page_id,
        source_image_path=Path(page.source_path),
        input_image_path=input_image_path,
        preprocessed_image_path=preprocessed_image_path,
    )
    save_metadata(metadata_path, metadata)

    return ImagePreprocessResult(
        document_id=ingestion_result.document_id,
        original_name=ingestion_result.original_name,
        preprocessed_image_path=preprocessed_image_path,
        metadata_path=metadata_path,
        metadata=metadata,
    )


def run_preprocessing(
    *,
    document_id: str,
    page_id: str,
    source_image_path: Path,
    input_image_path: Path,
    preprocessed_image_path: Path,
) -> ImagePreprocessMetadata:
    try:
        with Image.open(input_image_path) as source_image:
            source_image.load()
            image = source_image.copy()
    except UnidentifiedImageError as error:
        raise ImagePreprocessError(
            f"Не получилось открыть изображение: {input_image_path}"
        ) from error
    except OSError as error:
        raise ImagePreprocessError(
            f"Не получилось прочитать изображение: {input_image_path}"
        ) from error

    if image.width < 10 or image.height < 10:
        image.close()
        raise ImagePreprocessError("Картинка слишком маленькая для обработки.")

    steps: list[str] = []
    grayscale_applied = image.mode != "L"
    if grayscale_applied:
        converted_image = image.convert("L")
        image.close()
        image = converted_image
        steps.append("grayscale")

    denoised_image = image.filter(ImageFilter.MedianFilter(size=3))
    image.close()
    image = denoised_image
    denoise_applied = True
    steps.append("denoise")

    threshold_value = calculate_otsu_threshold(image)
    threshold_image = apply_threshold(image, threshold_value)
    image.close()
    image = threshold_image
    threshold_applied = True
    steps.append("threshold")

    deskew_angle = estimate_deskew_angle(image)
    deskew_applied = abs(deskew_angle) >= 0.5
    if deskew_applied:
        rotated_image = image.rotate(
            deskew_angle,
            resample=Image.Resampling.BICUBIC,
            expand=True,
            fillcolor=255,
        )
        image.close()
        image = apply_threshold(rotated_image, 128)
        rotated_image.close()
        steps.append("deskew")

    save_preprocessed_image(image, preprocessed_image_path)
    width, height = image.size
    image.close()

    return ImagePreprocessMetadata(
        document_id=document_id,
        page_id=page_id,
        source_image_path=str(source_image_path),
        input_image_path=str(input_image_path),
        preprocessed_image_path=str(preprocessed_image_path),
        steps=steps,
        grayscale_applied=grayscale_applied,
        denoise_applied=denoise_applied,
        threshold_applied=threshold_applied,
        deskew_applied=deskew_applied,
        threshold_value=threshold_value,
        deskew_angle=round(deskew_angle, 2),
        width=width,
        height=height,
        status="done",
    )


def calculate_otsu_threshold(image: Image.Image) -> int:
    histogram = image.histogram()[:256]
    total = sum(histogram)
    sum_total = sum(index * value for index, value in enumerate(histogram))
    sum_background = 0
    weight_background = 0
    best_threshold = 128
    best_variance = -1.0

    for threshold, count in enumerate(histogram):
        weight_background += count
        if weight_background == 0:
            continue

        weight_foreground = total - weight_background
        if weight_foreground == 0:
            break

        sum_background += threshold * count
        mean_background = sum_background / weight_background
        mean_foreground = (sum_total - sum_background) / weight_foreground
        variance = (
            weight_background
            * weight_foreground
            * (mean_background - mean_foreground) ** 2
        )

        if variance > best_variance:
            best_variance = variance
            best_threshold = threshold

    return best_threshold


def apply_threshold(image: Image.Image, threshold_value: int) -> Image.Image:
    return image.point(lambda value: 0 if value < threshold_value else 255, mode="L")


def estimate_deskew_angle(image: Image.Image) -> float:
    if min(image.size) < 80:
        return 0.0

    sample = image.copy()
    sample.thumbnail((900, 900))
    base_score = projection_score(sample)
    best_score = base_score
    best_angle = 0.0

    for angle_step in range(-6, 7):
        angle = angle_step * 0.5
        if angle == 0:
            continue

        rotated_sample = sample.rotate(
            angle,
            resample=Image.Resampling.BICUBIC,
            expand=True,
            fillcolor=255,
        )
        score = projection_score(rotated_sample)
        rotated_sample.close()

        if score > best_score:
            best_score = score
            best_angle = angle

    sample.close()

    if best_angle == 0.0:
        return 0.0
    if base_score > 0 and best_score < base_score * 1.05:
        return 0.0

    return best_angle


def projection_score(image: Image.Image) -> float:
    pixels = image.load()
    width, height = image.size
    row_sums: list[int] = []

    for y_index in range(height):
        row_total = 0
        for x_index in range(width):
            row_total += 255 - pixels[x_index, y_index]
        row_sums.append(row_total)

    if len(row_sums) < 2:
        return 0.0

    return sum(
        (row_sums[index] - row_sums[index - 1]) ** 2
        for index in range(1, len(row_sums))
    )


def save_preprocessed_image(image: Image.Image, output_path: Path) -> None:
    try:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        image.save(output_path, format="PNG")
    except OSError as error:
        raise ImagePreprocessError(f"Не удалось сохранить результат: {output_path}") from error


def save_metadata(metadata_path: Path, metadata: ImagePreprocessMetadata) -> None:
    try:
        metadata_path.parent.mkdir(parents=True, exist_ok=True)
        metadata_path.write_text(
            json.dumps(metadata.to_dict(), ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
    except OSError as error:
        raise ImagePreprocessError(f"Не удалось сохранить метаданные: {metadata_path}") from error


def build_preprocessed_image_path(config: AppConfig, document_id: str) -> Path:
    return config.artifacts_dir / "documents" / document_id / "preprocessed_page.png"


def build_metadata_path(config: AppConfig, document_id: str) -> Path:
    return config.artifacts_dir / "documents" / document_id / "preprocess.json"
